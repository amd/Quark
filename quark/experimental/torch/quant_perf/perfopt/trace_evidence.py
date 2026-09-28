#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Evidence-preserving GPU-kernel extraction from Kineto traces."""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FRAME_RE = re.compile(r"^(?P<path>.+?)\((?P<line>\d+)\):\s*(?P<function>.+)$")
_SKIP_LAUNCHER_FRAME_RE = re.compile(
    r"(?:"
    r"triton/runtime/|triton/backends/|triton/compiler/"
    r"|aiter/jit/|flydsl/compiler/"
    r"|torch/_dynamo/|torch/_inductor/"
    r"|torch/nn/modules/module\.py"
    r"|^<"
    r")"
)
_WRAPPER_FUNCTIONS = {
    "call",
    "inner",
    "wrapped",
    "wrapper",
    "_fn",
    "_inner",
    "_wrapped",
    "_wrapper",
}


@dataclass(frozen=True)
class KernelEvidence:
    device_kernel_name: str
    total_time_us: float
    call_count: int
    parent_op_names: tuple[str, ...] = ()
    primary_parent_op: str = ""
    external_ids: tuple[int, ...] = ()
    shape_cases: tuple[tuple[tuple[int, ...], ...], ...] = ()
    dtypes: tuple[str, ...] = ()
    source_file_hint: str = ""


def _load_trace(trace_path: str | Path) -> list[dict[str, Any]]:
    path = Path(trace_path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
    return events if isinstance(events, list) else []


def _normalize_shapes(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list | tuple):
        return ()
    shapes: list[tuple[int, ...]] = []
    for item in value:
        if not isinstance(item, list | tuple):
            continue
        try:
            shape = tuple(int(dim) for dim in item)
        except (TypeError, ValueError):
            continue
        if shape:
            shapes.append(shape)
    return tuple(shapes)


def _normalize_dtypes(value: Any) -> tuple[str, ...]:
    rows = value if isinstance(value, list | tuple) else [value]
    return tuple(dict.fromkeys(str(item) for item in rows if item not in (None, "")))


def _source_hint(args: dict[str, Any]) -> str:
    for key in ("source_file", "kernel_file", "file", "filename", "path"):
        value = args.get(key)
        if value:
            return str(value)
    return ""


def extract_kernel_evidence(
    trace_path: str | Path,
) -> dict[str, KernelEvidence]:
    """Aggregate GPU time while preserving CPU-op and shape provenance."""
    events = _load_trace(trace_path)
    op_by_external_id: dict[int, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "cpu_op":
            continue
        raw_args = event.get("args")
        args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        external_id = args.get("External id")
        if not isinstance(external_id, int):
            continue
        current = op_by_external_id.setdefault(
            external_id,
            {
                "name": str(event.get("name") or ""),
                "shapes": (),
                "dtypes": (),
                "source_file": "",
            },
        )
        current["name"] = current["name"] or str(event.get("name") or "")
        shapes = _normalize_shapes(args.get("Input Dims"))
        if shapes:
            current["shapes"] = shapes
        dtypes = _normalize_dtypes(args.get("Input type"))
        if dtypes:
            current["dtypes"] = dtypes
        current["source_file"] = current["source_file"] or _source_hint(args)

    aggregate: dict[str, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "kernel" or float(event.get("dur", 0.0) or 0.0) <= 0.0:
            continue
        name = str(event.get("name") or "")
        if not name:
            continue
        duration = float(event.get("dur", 0.0))
        row = aggregate.setdefault(
            name,
            {
                "total_time_us": 0.0,
                "call_count": 0,
                "parent_time": defaultdict(float),
                "external_ids": set(),
                "shape_cases": set(),
                "dtypes": set(),
                "source_files": set(),
            },
        )
        row["total_time_us"] += duration
        row["call_count"] += 1
        raw_args = event.get("args")
        args = raw_args if isinstance(raw_args, dict) else {}
        external_id = args.get("External id")
        if isinstance(external_id, int):
            row["external_ids"].add(external_id)
            op = op_by_external_id.get(external_id)
            if op:
                if op["name"]:
                    row["parent_time"][op["name"]] += duration
                if op["shapes"]:
                    row["shape_cases"].add(op["shapes"])
                row["dtypes"].update(op["dtypes"])
                if op["source_file"]:
                    row["source_files"].add(op["source_file"])

    result: dict[str, KernelEvidence] = {}
    for name, row in aggregate.items():
        parent_time = row["parent_time"]
        parent_names = tuple(
            name
            for name, _ in sorted(
                parent_time.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )
        result[name] = KernelEvidence(
            device_kernel_name=name,
            total_time_us=row["total_time_us"],
            call_count=row["call_count"],
            parent_op_names=parent_names,
            primary_parent_op=parent_names[0] if parent_names else "",
            external_ids=tuple(sorted(row["external_ids"])),
            shape_cases=tuple(sorted(row["shape_cases"])),
            dtypes=tuple(sorted(row["dtypes"])),
            source_file_hint=(sorted(row["source_files"])[0] if len(row["source_files"]) == 1 else ""),
        )
    return result


def _shape_from_matrix_operands(
    activation: tuple[int, ...],
    weight: tuple[int, ...],
) -> tuple[int, int, int] | None:
    if len(activation) != 2 or len(weight) != 2:
        return None
    m, k = activation
    if weight[0] == k:
        return m, weight[1], k
    if weight[1] == k:
        return m, weight[0], k
    return None


def _generic_case_alternatives(
    case: tuple[tuple[int, ...], ...],
) -> set[tuple[int, int, int]]:
    alternatives: set[tuple[int, int, int]] = set()
    two_d = [shape for shape in case if len(shape) == 2]
    for index, activation in enumerate(two_d):
        for weight in two_d[index + 1 :]:
            shape = _shape_from_matrix_operands(
                activation,
                weight,
            )
            if shape is not None and min(shape) > 0:
                alternatives.add(shape)
    return alternatives


def _gemm_shape_resolution(
    evidence: KernelEvidence,
) -> tuple[str, list[dict[str, int]], list[dict[str, int]]]:
    exact: set[tuple[int, int, int]] = set()
    ambiguous: set[tuple[int, int, int]] = set()
    operand_indices = {
        "aten::mm": (0, 1),
        "aten::_scaled_mm": (0, 1),
        "aten::addmm": (1, 2),
    }.get(evidence.primary_parent_op)
    for case in evidence.shape_cases:
        if operand_indices is not None:
            left, right = operand_indices
            if max(left, right) >= len(case):
                continue
            shape = _shape_from_matrix_operands(
                case[left],
                case[right],
            )
            if shape is not None and min(shape) > 0:
                exact.add(shape)
            continue
        alternatives = _generic_case_alternatives(case)
        if len(alternatives) == 1:
            exact.update(alternatives)
        elif len(alternatives) > 1:
            ambiguous.update(alternatives)
    if ambiguous:
        return "ambiguous", [], [{"M": m, "N": n, "K": k} for m, n, k in sorted(exact | ambiguous)]
    shapes = [{"M": m, "N": n, "K": k} for m, n, k in sorted(exact)]
    return ("exact" if shapes else "missing"), shapes, []


def _gemm_shape(evidence: KernelEvidence) -> dict[str, int | None]:
    _, shapes, _ = _gemm_shape_resolution(evidence)
    if len(shapes) == 1:
        return dict(shapes[0])
    return {"M": None, "N": None, "K": None}


def _gemm_shape_evidence(
    evidence: KernelEvidence,
    *,
    source: str,
    trace_path: str | Path,
) -> dict[str, Any]:
    status, shapes, alternatives = _gemm_shape_resolution(evidence)
    return {
        "status": status,
        "source": source,
        "trace_path": str(trace_path),
        "shapes": shapes,
        "alternatives": alternatives,
    }


def evidence_to_bottleneck(evidence: KernelEvidence) -> dict[str, Any]:
    _, gemm_shapes, _ = _gemm_shape_resolution(evidence)
    return {
        "kernel_names": [evidence.device_kernel_name],
        "device_kernel_name": evidence.device_kernel_name,
        "op_name": evidence.device_kernel_name,
        "parent_op_name": evidence.primary_parent_op,
        "parent_op_names": list(evidence.parent_op_names),
        "external_ids": list(evidence.external_ids),
        "call_count": evidence.call_count,
        "shape_cases": [[list(shape) for shape in case] for case in evidence.shape_cases],
        "dtypes": list(evidence.dtypes),
        "source_file": evidence.source_file_hint,
        "gemm_shape": _gemm_shape(evidence),
        "gemm_shapes": gemm_shapes,
        "kernel_time_us": evidence.total_time_us,
        "roofline_bound": "UNKNOWN",
        "sol_pct": None,
        "repro_ir": None,
    }


def enrich_bottlenecks_with_kernel_evidence(
    bottlenecks: list[dict[str, Any]],
    evidence_by_kernel: dict[str, KernelEvidence],
    *,
    source: str,
    trace_path: str | Path,
) -> None:
    """Merge exact-name metadata without replacing steady-state timings."""
    for bottleneck in bottlenecks:
        names = _bottleneck_kernel_names(bottleneck)
        matches = [evidence_by_kernel[name] for name in sorted(names) if name in evidence_by_kernel]
        if len(matches) != 1:
            if matches:
                bottleneck["gemm_shape_evidence"] = {
                    "status": "ambiguous",
                    "source": source,
                    "trace_path": str(trace_path),
                    "alternatives": [],
                }
            continue
        evidence = matches[0]
        metadata = evidence_to_bottleneck(evidence)
        for key in (
            "parent_op_name",
            "parent_op_names",
            "external_ids",
            "shape_cases",
            "dtypes",
            "source_file",
        ):
            if metadata.get(key) not in (None, "", [], {}):
                bottleneck[key] = metadata[key]
        bottleneck["gemm_shape"] = metadata["gemm_shape"]
        bottleneck["gemm_shapes"] = metadata["gemm_shapes"]
        bottleneck["gemm_shape_evidence"] = _gemm_shape_evidence(
            evidence,
            source=source,
            trace_path=trace_path,
        )


def compare_differential_evidence(
    quantized: dict[str, KernelEvidence],
    baseline: dict[str, KernelEvidence],
    *,
    slowdown_threshold: float = 1.15,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Select unique/slower quantized kernels without dropping their evidence."""
    candidates: list[dict[str, Any]] = []
    for name, evidence in quantized.items():
        baseline_time = baseline[name].total_time_us if name in baseline else 0.0
        if baseline_time == 0.0:
            differential_type = "unique_to_quantized"
        elif evidence.total_time_us > baseline_time * slowdown_threshold:
            differential_type = "slower_in_quantized"
        else:
            continue
        candidate = evidence_to_bottleneck(evidence)
        candidate.update(
            {
                "baseline_time_us": baseline_time,
                "differential_type": differential_type,
                "relative_slowdown": (evidence.total_time_us / baseline_time if baseline_time > 0.0 else None),
            }
        )
        candidates.append(candidate)
    candidates.sort(key=lambda row: -row["kernel_time_us"])
    return candidates[:top_n]


def _bottleneck_kernel_names(bottleneck: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in (
            bottleneck.get("op_name"),
            bottleneck.get("device_kernel_name"),
            *(bottleneck.get("kernel_names") or []),
        )
        if value
    }


def _innermost_launcher_frame(
    events: list[dict[str, Any]],
    *,
    pid: int,
    tid: int,
    timestamp: float,
) -> tuple[str, int, str] | None:
    frames = []
    for event in events:
        if event.get("cat") != "python_function":
            continue
        if int(event.get("pid") or 0) != pid:
            continue
        if int(event.get("tid") or 0) != tid:
            continue
        start = float(event.get("ts") or 0.0)
        duration = float(event.get("dur") or 0.0)
        if start <= timestamp <= start + duration:
            frames.append((start, duration, str(event.get("name") or "")))
    for _, _, name in sorted(
        frames,
        key=lambda item: (item[0], -item[1]),
        reverse=True,
    ):
        if _SKIP_LAUNCHER_FRAME_RE.search(name):
            continue
        match = _FRAME_RE.match(name)
        if not match:
            continue
        path = match.group("path").strip()
        function = match.group("function").strip()
        if not path.endswith(".py"):
            continue
        if function in _WRAPPER_FUNCTIONS:
            continue
        return path, int(match.group("line")), function
    return None


def enrich_bottlenecks_with_eager_launchers(
    bottlenecks: list[dict[str, Any]],
    trace_path: str | Path,
    *,
    max_samples_per_kernel: int = 3,
) -> None:
    """Attach eager Python-launcher evidence without treating it as source."""
    if not bottlenecks:
        return
    wanted = {name for bottleneck in bottlenecks for name in _bottleneck_kernel_names(bottleneck)}
    if not wanted:
        return
    try:
        events = _load_trace(trace_path)
    except (OSError, ValueError, TypeError):
        return

    kernels_by_correlation: dict[Any, set[str]] = defaultdict(set)
    graph_kernels: set[str] = set()
    runtime_by_correlation: dict[Any, dict[str, Any]] = {}
    for event in events:
        args = event.get("args")
        args = args if isinstance(args, dict) else {}
        correlation = args.get("correlation")
        if correlation is None:
            continue
        category = event.get("cat")
        name = str(event.get("name") or "")
        if category == "kernel" and name in wanted:
            kernels_by_correlation[correlation].add(name)
        elif category == "cuda_runtime" and "launch" in name.lower():
            runtime_by_correlation[correlation] = event

    votes: dict[str, defaultdict[tuple[str, int, str], int]] = {name: defaultdict(int) for name in wanted}
    launch_apis: dict[str, defaultdict[str, int]] = {name: defaultdict(int) for name in wanted}
    sample_counts: dict[str, int] = defaultdict(int)
    for correlation, kernel_names in kernels_by_correlation.items():
        runtime = runtime_by_correlation.get(correlation)
        if runtime is None:
            continue
        api = str(runtime.get("name") or "")
        if "graphlaunch" in api.lower():
            graph_kernels.update(kernel_names)
            continue
        for kernel_name in kernel_names:
            if sample_counts[kernel_name] >= max_samples_per_kernel:
                continue
            found = _innermost_launcher_frame(
                events,
                pid=int(runtime.get("pid") or 0),
                tid=int(runtime.get("tid") or 0),
                timestamp=float(runtime.get("ts") or 0.0),
            )
            if found is None:
                continue
            votes[kernel_name][found] += 1
            launch_apis[kernel_name][api] += 1
            sample_counts[kernel_name] += 1

    for bottleneck in bottlenecks:
        names = _bottleneck_kernel_names(bottleneck)
        combined: defaultdict[tuple[str, int, str], int] = defaultdict(int)
        combined_apis: defaultdict[str, int] = defaultdict(int)
        for name in names:
            for frame, count in votes.get(name, {}).items():
                combined[frame] += count
            for api, count in launch_apis.get(name, {}).items():
                combined_apis[api] += count
        if combined:
            ranked = sorted(
                combined.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                bottleneck["trace_parent_status"] = "eager_launcher_ambiguous"
                continue
            (path, line, function), count = ranked[0]
            bottleneck.update(
                {
                    "launcher_source_file": path,
                    "launcher_line": line,
                    "launcher_symbol": function,
                    "launcher_sample_count": count,
                    "launcher_launch_api": (
                        max(
                            combined_apis,
                            key=combined_apis.get,
                        )
                        if combined_apis
                        else ""
                    ),
                    "launcher_evidence_method": ("eager_trace_python_stack"),
                    "trace_parent_status": "eager_launcher_available",
                }
            )
        elif names & graph_kernels:
            bottleneck["trace_parent_status"] = "graph_replay_unavailable"
