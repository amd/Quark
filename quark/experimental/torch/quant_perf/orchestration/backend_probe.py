#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Session-level orchestration for the MXFP4 GEMM backend probe."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.evaluation.gate import AccuracyGate
from quark.experimental.torch.quant_perf.orchestration.orchestrator import Orchestrator
from quark.experimental.torch.quant_perf.perfopt.backend_probe import (
    BackendProbeDecision,
    decide_micro_screen,
    decide_probe_backend,
    run_micro_probe,
)
from quark.experimental.torch.quant_perf.perfopt.backend_probe_inputs import (
    backend_probe_input_support_reason,
    load_backend_probe_inputs,
)
from quark.experimental.torch.quant_perf.pipeline.candidate_validation import (
    evaluate_candidate_accuracy,
    measure_quantized_candidate,
)
from quark.experimental.torch.quant_perf.runtime.backends import (
    configure_dense_mxfp4_backend,
    configure_runtime_env,
    is_mxfp4_model,
)
from quark.experimental.torch.quant_perf.session.spec import Checkpoint, SessionLock, Spec


def _backend_snapshot() -> dict[str, str | None]:
    keys = (
        "QUARK_QUANT_PERF_MXFP4_GEMM_BACKEND",
        "VLLM_ROCM_MXFP4_GEMM_BACKEND",
        "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM",
    )
    return {key: os.environ.get(key) for key in keys}


def _restore_backend(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _probe_supported(spec: Spec, quant_ckpt: Path) -> tuple[bool, str]:
    unsupported_reason = backend_probe_input_support_reason(spec.model_arch)
    if unsupported_reason:
        return False, unsupported_reason
    if not is_mxfp4_model(str(quant_ckpt)):
        return False, "checkpoint_is_not_mxfp4"
    return True, ""


def _persist_probe_result(
    session_dir: Path,
    ckpt: Checkpoint,
    result: dict[str, Any],
) -> dict[str, Any]:
    probe_dir = session_dir / "backend_probes" / datetime.now(UTC).strftime("mxfp4-gemm-%Y%m%d-%H%M%S")
    probe_dir.mkdir(parents=True, exist_ok=True)
    result_path = probe_dir / "result.json"
    result["artifact"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2))
    ckpt.state.setdefault("backend_probes", []).append(dict(result))
    ckpt.save()
    return result


def run_backend_probe(session_dir: Path) -> dict[str, Any]:
    session_dir = session_dir.resolve()
    ckpt = Checkpoint.load(session_dir)
    if ckpt is None:
        raise RuntimeError(f"no state.json found under {session_dir}")
    quant_ckpt = Path(str(ckpt.state.get("quant_ckpt_dir") or ""))
    if not (quant_ckpt / "config.json").is_file():
        raise RuntimeError("backend probe requires an existing quant checkpoint")
    spec = Spec.from_dict(
        dict(ckpt.state.get("run_spec") or {}),
        runtime_context=dict(ckpt.state.get("runtime_context") or {}),
    )
    spec = replace(spec, session_dir=str(session_dir))
    supported, unsupported_reason = _probe_supported(spec, quant_ckpt)
    if not supported:
        return _persist_probe_result(
            session_dir,
            ckpt,
            {
                "status": "unsupported",
                "created_at": datetime.now(UTC).isoformat(),
                "selected_backend": spec.mxfp4_gemm_backend,
                "reason": unsupported_reason,
            },
        )

    original_run_spec = copy.deepcopy(ckpt.state.get("run_spec"))
    snapshot = _backend_snapshot()
    orchestrator = Orchestrator()
    manager = None
    result: dict[str, Any] = {}
    with SessionLock(session_dir):
        try:
            manager = orchestrator._prepare_managed_workspaces(
                spec,
                ckpt,
            )
            configure_runtime_env(spec)
            configure_dense_mxfp4_backend("flydsl")
            probe_inputs = load_backend_probe_inputs(
                spec.model_arch,
                quant_ckpt,
            )
            micro_cases = run_micro_probe(probe_inputs)
            micro = decide_micro_screen(micro_cases)
            result = {
                "status": "complete",
                "created_at": datetime.now(UTC).isoformat(),
                "anchor_backend": "flydsl",
                "candidate_backend": "asm",
                "micro_cases": [case.to_dict() for case in micro_cases],
                "micro_speedup": micro.speedup,
                "micro_passed": micro.passed,
                "micro_reason": micro.reason,
                "selected_backend": "flydsl",
                "reason": micro.reason,
            }
            if micro.passed:
                gate = AccuracyGate(spec)
                gate._source_cache = float(ckpt.state.get("baseline_gsm8k") or 0.0)
                configure_dense_mxfp4_backend("flydsl")
                anchor_first = measure_quantized_candidate(
                    spec,
                    str(quant_ckpt),
                )
                configure_dense_mxfp4_backend("asm")
                accuracy_status, score, accuracy_reason = evaluate_candidate_accuracy(
                    spec,
                    str(quant_ckpt),
                )
                accuracy_gap = (
                    max(
                        0.0,
                        (gate._source_cache - float(score)) / max(gate._source_cache, 1e-9),
                    )
                    if accuracy_status == "passed" and score is not None
                    else None
                )
                accuracy_passed = bool(
                    accuracy_status == "passed" and accuracy_gap is not None and accuracy_gap <= spec.accuracy_gap
                )
                if accuracy_passed:
                    candidate_first = measure_quantized_candidate(
                        spec,
                        str(quant_ckpt),
                    )
                    candidate_second = measure_quantized_candidate(
                        spec,
                        str(quant_ckpt),
                    )
                    configure_dense_mxfp4_backend("flydsl")
                    anchor_second = measure_quantized_candidate(
                        spec,
                        str(quant_ckpt),
                    )
                    decision = decide_probe_backend(
                        micro_passed=True,
                        accuracy_passed=True,
                        anchor_first=anchor_first,
                        candidate_first=candidate_first,
                        candidate_second=candidate_second,
                        anchor_second=anchor_second,
                        keep_floor=spec.keep_floor,
                    )
                    measurement_result = {
                        "anchor_first": anchor_first.to_dict(),
                        "candidate_first": candidate_first.to_dict(),
                        "candidate_second": candidate_second.to_dict(),
                        "anchor_second": anchor_second.to_dict(),
                    }
                else:
                    decision = BackendProbeDecision(
                        "flydsl",
                        1.0,
                        spec.keep_floor,
                        "accuracy_failed",
                    )
                    measurement_result = {
                        "anchor_first": anchor_first.to_dict(),
                    }
                result.update(
                    {
                        "selected_backend": decision.selected_backend,
                        "reason": decision.reason,
                        "e2e_multiplier": decision.multiplier,
                        "effective_keep_floor": decision.effective_floor,
                        "accuracy_status": accuracy_status,
                        "accuracy_reason": accuracy_reason,
                        "accuracy_score": score,
                        "accuracy_gap": accuracy_gap,
                        "accuracy_passed": accuracy_passed,
                        **measurement_result,
                    }
                )
        finally:
            _restore_backend(snapshot)
            if manager is not None:
                manager.cleanup_terminal()
            if original_run_spec is not None:
                ckpt.state["run_spec"] = original_run_spec
        return _persist_probe_result(session_dir, ckpt, result)


def run_backend_probe_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="quark-quant-perf backend-probe")
    parser.add_argument("--session", required=True, dest="session_dir")
    args = parser.parse_args(argv)
    result = run_backend_probe(Path(args.session_dir).resolve())
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "complete" else 1
