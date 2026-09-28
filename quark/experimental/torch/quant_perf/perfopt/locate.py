#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Bottleneck localization via TraceLens's TreePerfAnalyzer.

Design ref: IMPL_SPEC §2.2.2. TraceLens is a pure parsing library: it never
collects a trace itself (perfopt/collect.py does that), it only turns one
into a ranked, roofline-annotated table of kernel-level bottlenecks.

Roofline fields (Roofline Bound, Pct Roofline) are populated only when
TraceLens can parse the input shapes from the trace args. When they are
absent (perf_params=None), we fall back to kernel time as the ranking
signal and mark bound_type as "UNKNOWN" -- GEAK still receives useful
context (kernel name, shape hint from repro_ir) even without roofline.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.perfopt._tracelens import load_tracelens as _load_tracelens
from quark.experimental.torch.quant_perf.perfopt.trace_evidence import (
    evidence_to_bottleneck,
    extract_kernel_evidence,
)

logger = logging.getLogger(__name__)


def _raw_kernel_bottlenecks(
    trace_path: str | Path,
    top_n: int,
) -> list[dict[str, Any]]:
    """Aggregate GPU kernel total time directly from traceEvents.

    Serves two roles:
    1. Fallback when TraceLens returns an empty df (trace format issues).
    2. Supplement for kernels TraceLens cannot build perf models for.

    TraceLens requires 'Input Dims' in the CPU op args to compute Kernel Time.
    FP8 quantization kernels (e.g. scaled_fp8_quant_kernel, wvSplitKQ) have no
    Input Dims in this trace format, so TraceLens assigns them kernel_time=0 and
    they disappear from the df even after enable_pseudo_ops+get_pseudo_op_mappings.
    This function bypasses that limitation by reading raw GPU kernel durations.

    Returns bottleneck dicts in the same schema as locate_bottlenecks.
    """
    evidence = extract_kernel_evidence(trace_path)
    top = sorted(
        evidence.values(),
        key=lambda row: -row.total_time_us,
    )[:top_n]
    return [evidence_to_bottleneck(row) for row in top]


def locate_bottlenecks(
    trace_path: str | Path,
    gpu: str = "MI355X",
    top_n: int = 10,
    gpu_arch_json_path: str | Path = "",
) -> list[dict[str, Any]]:
    tracelens = _load_tracelens()
    if tracelens is None:
        logger.warning("TraceLens is unavailable; falling back to raw kernel aggregation.")
        return _raw_kernel_bottlenecks(trace_path, top_n=top_n)

    try:
        if gpu_arch_json_path:
            arch = tracelens.resolve_gpu_arch(gpu_arch_json_path=str(gpu_arch_json_path))
        else:
            arch = tracelens.resolve_gpu_arch(gpu_arch_platform=gpu)
    except KeyError:
        logger.warning(
            "TraceLens has no architecture data for %s; falling back to raw kernel aggregation.",
            gpu,
        )
        return _raw_kernel_bottlenecks(trace_path, top_n=top_n)
    # enable_pseudo_ops=True: merges low-level GPU kernels into semantic ops
    # (e.g. MoE GEMM stages, FP8 GEMM, mxfp4 quant) so TraceLens can build
    # perf models for them. Without this, vllm::rocm_unquantized_gemm appears
    # as a single un-differentiated op regardless of whether it dispatches to
    # a BF16 or FP8 kernel, and high-frequency short-duration kernels like
    # scaled_fp8_quant (0.01ms × 774k calls = 4s total) are invisible.
    pa = tracelens.tree_perf_analyzer.from_file(str(trace_path), arch=arch, enable_pseudo_ops=True)
    pa.tree.apply_annotation(name_filters=["vllm::unified_attention_with_output"])

    # Load the vllm/aiter/mxfp4 pseudo-op perf model extensions.
    # These map op names like vllm::rocm_unquantized_gemm, aiter::moe_cktile2stages_gemm*,
    # aiter::mxfp4_moe_sort_hip etc. to their perf model classes, enabling
    # roofline analysis and proper kernel_time attribution for all quant schemes.
    #
    # pseudo_ops_perf_utils uses relative imports so cannot be loaded via
    # apply_extension(path) -- import it directly and update the map instead.
    if tracelens.get_pseudo_op_mappings is not None:
        try:
            pa.op_to_perf_model_class_map.update(tracelens.get_pseudo_op_mappings())
        except Exception as exc:
            logger.warning("TraceLens pseudo-op extension failed (%s); continuing without it", exc)

    df = pa.build_df_unified_perf_table()

    # Filter to ops with a perf model (i.e. GEMM, attention, MoE, etc.)
    # and sort by kernel time descending. Roofline columns may be absent
    # when TraceLens cannot parse shapes from the trace -- handle gracefully.
    # Find the kernel time column -- TraceLens may use different column names
    # depending on version or trace format.
    time_col = None
    for candidate in ("Kernel Time (µs)", "Kernel Time (us)", "kernel_time_us", "duration_us"):
        if candidate in df.columns:
            time_col = candidate
            break

    if "has_perf_model" in df.columns:
        has_perf = df[df["has_perf_model"]].copy()
    elif time_col:
        has_perf = df[df[time_col] > 0].copy()
    else:
        has_perf = df.copy()

    if df.empty or time_col is None or time_col not in has_perf.columns:
        logger.warning(
            "locate_bottlenecks: TraceLens returned no usable data "
            "(df.empty=%s, time_col=%s). Falling back to raw kernel aggregation.",
            df.empty,
            time_col,
        )
        return _raw_kernel_bottlenecks(trace_path, top_n=top_n)

    # Aggregate by op_name BEFORE taking top_n.
    # Sorting by single-instance kernel time is wrong for high-frequency,
    # short-duration kernels (e.g. FP8 quant kernels: 0.01 ms each but
    # 774k calls = 4+ seconds total). They never appear in a per-instance
    # top-N, so the differential filter never sees them. Instead, sum all
    # instances of the same op_name and rank by total time.
    #
    # For each op_name group, keep:
    #   - total kernel_time_us (sum)
    #   - union of kernel_names across all instances (critical: a single
    #     vllm op can dispatch to different GPU kernels in quantized vs
    #     baseline -- e.g. wvSplitKQ (FP8) vs wvSplitK (BF16) both map
    #     to rocm_unquantized_gemm -- so we must collect all variants)
    #   - representative row: the slowest single instance, for repro_ir
    #     and gemm_shape (best-effort, shapes may vary across instances)
    #   - roofline: from the representative row
    aggregated: dict[str, dict[str, Any]] = {}
    for _, row in has_perf.iterrows():
        op = row["name"]
        t = float(row[time_col])
        kernel_details = row.get("kernel_details") or []
        knames = [k["name"] for k in kernel_details if isinstance(k, dict)]

        bound = row.get("Roofline Bound")
        sol = row.get("Pct Roofline")

        if op not in aggregated:
            aggregated[op] = {
                "total_time_us": t,
                "kernel_names_set": set(knames),
                "rep_row": row,  # slowest single instance (for gemm_shape + repro_ir)
                "rep_time": t,
                # Collect all instances for statistical roofline
                "bound_counts": {bound: t} if bound is not None else {},
                "sol_samples": [(t, sol)] if sol is not None else [],
            }
        else:
            aggregated[op]["total_time_us"] += t
            aggregated[op]["kernel_names_set"].update(knames)
            if t > aggregated[op]["rep_time"]:
                aggregated[op]["rep_row"] = row
                aggregated[op]["rep_time"] = t
            if bound is not None:
                aggregated[op]["bound_counts"][bound] = aggregated[op]["bound_counts"].get(bound, 0) + t
            if sol is not None:
                aggregated[op]["sol_samples"].append((t, sol))

    # Supplement TraceLens results with raw kernel aggregation.
    # TraceLens's perf table only covers ops it has a perf model for; it misses
    # high-frequency short-duration kernels (e.g. FP8 quant/GEMM kernels with
    # 0.01 ms per call but 774k calls = 4+ s total). Extract these directly
    # from traceEvents and merge into the aggregated dict so they can compete
    # on total time for top_n slots.
    raw_bns = _raw_kernel_bottlenecks(trace_path, top_n=top_n * 5)
    for raw in raw_bns:
        kname = raw["op_name"]
        # Skip if already covered by TraceLens (either as op_name or kernel_name)
        already_covered = any(kname in agg["kernel_names_set"] or kname == op for op, agg in aggregated.items())
        if not already_covered:
            aggregated[kname] = {
                "total_time_us": raw["kernel_time_us"],
                "kernel_names_set": set(raw["kernel_names"]),
                "rep_row": None,
                "rep_time": raw["kernel_time_us"],
                "_raw": True,
            }

    # Sort aggregated ops by total time, take top_n
    top_ops = sorted(aggregated.items(), key=lambda x: -x[1]["total_time_us"])[:top_n]

    results: list[dict[str, Any]] = []
    for op_name, agg in top_ops:
        row = agg.get("rep_row")

        # roofline_bound: time-weighted mode across all instances.
        # MoE ops have variable expert routing per decode step, so M varies and
        # a single instance's bound can be misleading. The mode (heaviest by
        # total kernel time) represents the dominant regime across the profile window.
        #
        # sol_pct: time-weighted mean -- same rationale; averaging over all
        # instances reflects actual utilization better than the slowest outlier.
        roofline_bound = "UNKNOWN"
        sol_pct = None
        bound_counts = agg.get("bound_counts", {})
        if bound_counts:
            roofline_bound = max(bound_counts, key=bound_counts.get)
        sol_samples = agg.get("sol_samples", [])
        if sol_samples:
            total_w = sum(w for w, _ in sol_samples)
            if total_w > 0:
                sol_pct = sum(w * s for w, s in sol_samples) / total_w

        # GEMM shape from representative instance
        gemm_shape = {
            "M": row.get("param: M") if row is not None else None,
            "N": row.get("param: N") if row is not None else None,
            "K": row.get("param: K") if row is not None else None,
        }

        # Repro IR from representative instance
        repro = None
        if row is not None:
            with contextlib.suppress(Exception):
                repro = tracelens.event_replayer(pa.tree.get_UID2event(row["UID"]), lazy=True).get_repro_info()

        results.append(
            {
                "kernel_names": list(agg["kernel_names_set"]),
                "op_name": op_name,
                "gemm_shape": gemm_shape,
                "kernel_time_us": agg["total_time_us"],
                "roofline_bound": roofline_bound,
                "sol_pct": sol_pct,
                "repro_ir": repro,
            }
        )
    return results
