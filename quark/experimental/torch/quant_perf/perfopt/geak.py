#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""GEAK invocation: the subprocess call, and the quantization-aware task
description that is Quark Quant-Perf's key improvement over a generic "optimize this
kernel" prompt.

Design ref: IMPL_SPEC §2.3.1 (run_geak) and §4.2 (build_geak_task /
get_quant_detail) -- both have a single authoritative definition here; do not
duplicate them elsewhere (see IMPL_SPEC §4.5's note on the drift this caused
once already).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.perfopt.guardrails import geak_execution_skip_reason
from quark.experimental.torch.quant_perf.perfopt.workload_contract import WorkloadContract


def run_geak(
    kernel_file: str,
    task: str,
    model: str,
    gpu_id: int,
    run_dir: str,
    *,
    session_dir: str = "",
    kernel_name: str = "",
    source_binding: dict[str, Any] | None = None,
    knowledge_ids: list[str] | None = None,
    budget: int = 3,
    timeout_s: float = 5400.0,
) -> dict[str, Any]:
    """Optimize ONE kernel via GEAK's single-kernel `kernel_workflow`.

    Replaces the former whole-model `interface/run_e2e.py` handoff, which
    self-profiled and self-selected kernels (ignoring Quark Quant-Perf's differential
    pick). Here Quark Quant-Perf's chosen kernel source (`kernel_file`) IS the target:
    kernel_workflow builds its own compile/correctness/benchmark COMMANDMENT
    from the source + a workload spec (parse_profile.py over this run's trace,
    so the perf harness times the real production shapes), optimizes it, and the
    driver runs the acceptance chain to completion or salvages the best
    verified patch when the timeout expires.
    """
    from quark.experimental.torch.quant_perf.perfopt.kernel_workflow import gen_workload_spec, run_kernel_workflow

    skip_reason = geak_execution_skip_reason(kernel_name, str((source_binding or {}).get("source_symbol") or ""))
    if skip_reason:
        return {
            "execution_status": "skipped",
            "watchdog_status": "unsupported_execution",
            "verified_speedup": None,
            "error": skip_reason,
        }

    if source_binding:
        binding_path = Path(run_dir) / "source_binding.json"
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        binding_path.write_text(
            json.dumps(source_binding, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # Workload spec from this run's collected trace, targeting THIS kernel, so
    # kernel_workflow benchmarks the exact (shape,dtype) cases it hits.
    workload_spec_path = ""
    if session_dir and kernel_name:
        out = str(Path(run_dir) / "workload.json")
        for trace_name in ("trace_shape_evidence", "trace"):
            workload_spec_path = (
                gen_workload_spec(
                    str(Path(session_dir) / trace_name),
                    kernel_name,
                    out,
                )
                or ""
            )
            if workload_spec_path:
                break

    workload_contract = WorkloadContract.from_spec(
        workload_spec_path,
        source_binding,
    )
    effective_task = task + (workload_contract.prompt_suffix() if workload_contract is not None else "")

    # kernel_workflow expects a kernel DIRECTORY (it copies the tree into its
    # workspace and runs `cd <dir> && git diff`); passing a bare file made the
    # engineer's cd fail and its git diff capture the WRONG repo. Hand it the
    # kernel's directory; the task hint + kernel_name focus the specific symbol.
    kernel_dir = os.path.dirname(kernel_file) if os.path.isfile(kernel_file) else kernel_file

    result = run_kernel_workflow(
        kernel_path=kernel_dir,
        task=effective_task,
        gpu_id=gpu_id,
        run_dir=run_dir,
        workload_contract=workload_contract,
        source_repo=str((source_binding or {}).get("source_repo") or ""),
        budget=budget,
        model=model or "claude-opus-4-8",
        hard_timeout_s=timeout_s,
    )
    if session_dir:
        from quark.experimental.torch.quant_perf.llm.audit import append_llm_call

        append_llm_call(
            session_dir,
            call_type="kernel_optimization",
            model=model or "claude-opus-4-8",
            round_id=0,
            prompt=effective_task,
            output=json.dumps(result, sort_keys=True, default=str),
            outcome=str(result.get("watchdog_status") or ("verified" if result.get("best_patch") else "no_patch")),
            knowledge_ids=knowledge_ids,
        )
    return result


def _match_layer_config(
    qcfg: dict[str, Any],
    op_name: str,
) -> dict[str, Any] | None:
    layer_quant = qcfg.get("layer_quant_config") or {}
    for pattern, layer_cfg in layer_quant.items():
        if pattern in op_name or op_name in pattern:
            return layer_cfg
    return None


def get_quant_detail(quant_ckpt_dir: str, op_name: str) -> str:
    """Reads the precise quantization format for this kernel's layer from
    Quark Quant-Perf's own exported quant_ckpt/config.json.

    Per-layer matching is best-effort (op_name -> layer-name pattern is not
    always reliable, especially for fused/composite kernels); falling back to
    global_quant_config is exact for direct_ptq (one global scheme) and only an
    approximation for mix_precision_search's per-linear case, which V1 accepts.
    """
    cfg = json.loads((Path(quant_ckpt_dir) / "config.json").read_text())
    qcfg = cfg.get("quantization_config", {})
    layer_cfg = _match_layer_config(qcfg, op_name) or qcfg.get("global_quant_config", qcfg)

    parts = [
        layer_cfg.get("dtype", "unknown"),
        layer_cfg.get("qscheme", ""),
        "symmetric" if layer_cfg.get("symmetric") else "asymmetric",
    ]
    if layer_cfg.get("group_size"):
        parts.append(f"group_size={layer_cfg['group_size']}")
    if layer_cfg.get("dynamic"):
        parts.append("dynamic activation")
    return ", ".join(p for p in parts if p)


def build_geak_task(
    bn: dict[str, Any],
    quant_strategy: str | None,
    quant_ckpt_dir: str,
    source_file: str | None = None,
    gpu_arch: str = "MI355X",
    knowledge_text: str | None = None,
) -> str:
    """Build the GEAK optimization task prompt.

    Structured after Hyperloom's build_prompt pattern:
    - kernel identity + quantization context
    - roofline bound + optimization hint
    - device kernel symbol (when dispatch shim promoted to device source)
    - source file content (first 12000 chars, as Hyperloom does)
    - captured shape and timing context
    - historical KB hint
    """
    bound = bn.get("roofline_bound", "UNKNOWN")
    sol_pct = bn.get("sol_pct")

    if bound == "COMPUTE_BOUND":
        hint = f"Focus on instruction throughput / tensor core utilization for {quant_strategy}."
    elif bound == "MEMORY_BOUND":
        hint = "Focus on HBM bandwidth; consider fusing with adjacent ops to reduce traffic."
    else:
        hint = "Roofline bound unknown; analyze memory access patterns and compute intensity."

    sol_line = f"currently at {sol_pct:.1f}% of roofline peak" if sol_pct is not None else "roofline % unavailable"
    quant_line = f"\nQuantization detail: {get_quant_detail(quant_ckpt_dir, bn['op_name'])}"
    kb_line = f"\n{knowledge_text}" if knowledge_text else ""
    shape = bn.get("gemm_shape") or {}
    shape_line = (
        "\nCaptured serving context: "
        f"M={shape.get('M')}, N={shape.get('N')}, K={shape.get('K')}, "
        f"kernel_time_us={bn.get('kernel_time_us')}."
    )
    parent_op = str(bn.get("parent_op_name") or "")
    raw_resolution = bn.get("source_resolution")
    resolution = raw_resolution if isinstance(raw_resolution, dict) else {}
    binding_lines = []
    if parent_op:
        binding_lines.append(f"Parent CPU operator: `{parent_op}`")
    if resolution.get("source_symbol"):
        binding_lines.append(f"Source symbol: `{resolution['source_symbol']}`")
    if resolution.get("builder_symbol"):
        binding_lines.append(f"Builder symbol: `{resolution['builder_symbol']}`")
    if resolution.get("launcher_source_file"):
        binding_lines.append(f"Launcher source: `{resolution['launcher_source_file']}`")
    if resolution.get("live_call_seam"):
        binding_lines.append(f"Live call seam: `{resolution['live_call_seam']}`")
    if resolution.get("compiler"):
        binding_lines.append(f"Compiler/source type: `{resolution['compiler']}`")
    if resolution.get("method") or resolution.get("confidence"):
        binding_lines.append(f"Resolution: `{resolution.get('method', '')}` / `{resolution.get('confidence', '')}`")
    binding_block = "\nSource binding evidence:\n- " + "\n- ".join(binding_lines) if binding_lines else ""

    # Source file content block (mirrors Hyperloom's source_block, capped at 12000 chars)
    source_block = ""
    if source_file and Path(source_file).exists():
        try:
            content = Path(source_file).read_text(encoding="utf-8", errors="replace")
            source_block = f"\nSource content ({source_file}):\n```\n{content[:12000]}\n```"
        except Exception:
            pass

    # Device kernel symbol block: when the op dispatches to a specific GPU kernel
    # (e.g. a Triton kernel resolved from TraceLens kernel_details), tell GEAK
    # which symbol to focus on in the source file.
    kernel_names = bn.get("kernel_names") or []
    device_symbol_block = ""
    if kernel_names and source_file:
        # Use the first kernel name as the primary device symbol
        primary_symbol = kernel_names[0]
        device_symbol_block = (
            f"\nDevice kernel symbol: `{primary_symbol}`\n"
            f"If the source file defines multiple kernels, focus on the symbol above; "
            f"preserve all other kernels verbatim."
        )

    return (
        f"Optimize kernel `{bn['op_name']}` for {quant_strategy} on {gpu_arch}.\n"
        f"kernel_url: {source_file or '(not resolved)'}\n"
        f"Roofline: {bound}, {sol_line}.\n"
        f"{hint}"
        f"{quant_line}"
        f"{kb_line}"
        f"{shape_line}"
        f"{binding_block}"
        f"{device_symbol_block}"
        f"{source_block}"
    )
