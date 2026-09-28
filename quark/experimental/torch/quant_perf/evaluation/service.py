#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Managed accuracy-only evaluation for existing quantized checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.evaluation.gate import AccuracyGate
from quark.experimental.torch.quant_perf.evaluation.profile import resolve_eval_profile
from quark.experimental.torch.quant_perf.runtime.recovery import (
    build_accuracy_fingerprint,
    build_runtime_fingerprint,
)
from quark.experimental.torch.quant_perf.session.progress import write_progress
from quark.experimental.torch.quant_perf.session.spec import Spec
from quark.experimental.torch.quant_perf.workspace.git import get_head_sha

EVAL_STATE_SCHEMA_VERSION = 2


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _start_attempt(
    state: dict[str, Any],
    *,
    role: str,
    model: str,
    artifact: str,
    profile_hash: str,
    runtime_fingerprint: str,
    runtime_origins: dict[str, Any],
    status: str = "running",
) -> int:
    attempts = state.setdefault("attempts", [])
    attempt_id = (
        max(
            (int(attempt.get("attempt") or 0) for attempt in attempts if isinstance(attempt, dict)),
            default=0,
        )
        + 1
    )
    attempts.append(
        {
            "attempt": attempt_id,
            "role": role,
            "status": status,
            "started_at": _utc_now(),
            "ended_at": "",
            "model": model,
            "artifact": artifact,
            "profile_hash": profile_hash,
            "runtime_fingerprint": runtime_fingerprint,
            "runtime_origins": dict(runtime_origins),
            "score": None,
            "error": "",
        }
    )
    return attempt_id


def _finish_attempt(
    state: dict[str, Any],
    attempt_id: int,
    *,
    status: str,
    score: float | None = None,
    error: str = "",
) -> None:
    for attempt in reversed(state.get("attempts") or []):
        if int(attempt.get("attempt") or 0) != attempt_id:
            continue
        attempt.update(
            {
                "status": status,
                "ended_at": _utc_now(),
                "score": score,
                "error": error,
            }
        )
        return
    raise ValueError(f"evaluation attempt {attempt_id} is missing")


def _error_summary(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"[-4000:]


def _repo_commit(path: str) -> str:
    if not path or not Path(path).is_dir():
        return ""
    try:
        return get_head_sha(path)
    except OSError:
        return ""


def _runtime_fingerprints(
    spec: Spec,
    quant_model: str,
) -> tuple[str, str]:
    framework_commit = _repo_commit(spec.active_framework_repo)
    kernel_commit = _repo_commit(spec.active_kernel_repo)
    runtime_env = {**os.environ, **dict(spec.runtime_env)}
    baseline = build_runtime_fingerprint(
        spec,
        framework_commit=framework_commit,
        runtime_env=runtime_env,
        stage="managed_eval_baseline",
        effective_gpu_memory_utilization=(spec.vllm_gpu_memory_utilization),
    )
    quantized = build_accuracy_fingerprint(
        spec,
        quant_model,
        framework_commit=framework_commit,
        kernel_commit=kernel_commit,
        runtime_env=runtime_env,
    )
    return baseline, quantized


def _write_report(
    session_dir: Path,
    *,
    spec: Spec,
    quant_model: str,
    result: dict[str, Any],
) -> dict[str, str]:
    reports = session_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / "eval.json"
    md_path = reports / "eval.md"
    profile = spec.eval_profile
    if profile is None:
        raise ValueError("evaluation report requires a resolved Eval Profile")
    payload = {
        "schema_version": "quark.experimental.torch.quant_perf.eval.v1",
        "model": quant_model,
        "base_model": spec.base_model,
        "profile": profile.to_dict(),
        "accuracy": {
            "baseline": result["baseline"],
            "quantized": result["quantized"],
            "gap": result["gap"],
            "threshold": spec.accuracy_gap,
            "passed": result["passed"],
        },
        "artifacts": result["artifacts"],
        "runtime_drift": list(spec.eval_runtime_drift),
        "runtime_origins": dict(result["runtime_origins"]),
        "attempts": list(result["attempts"]),
    }
    _write_json(json_path, payload)
    md_path.write_text(
        "\n".join(
            [
                "# Quark Quant-Perf Evaluation Report",
                "",
                f"- Model: `{quant_model}`",
                f"- Base model: `{spec.base_model}`",
                f"- Profile: `{profile.profile_id}`",
                f"- Protocol hash: `{profile.profile_hash}`",
                f"- Settings source: `{profile.settings_source}`",
                f"- Baseline GSM8K: `{result['baseline']:.6f}`",
                f"- Quantized GSM8K: `{result['quantized']:.6f}`",
                f"- Relative gap: `{result['gap']:.6%}`",
                f"- Threshold: `{spec.accuracy_gap:.6%}`",
                f"- Passed: `{result['passed']}`",
                "",
            ]
        )
    )
    return {
        "eval_json": str(json_path),
        "eval_md": str(md_path),
    }


class EvaluationService:
    """Run baseline and quantized real accuracy under one frozen profile."""

    def run(
        self,
        spec: Spec,
        quant_model: str,
        *,
        runtime_origin_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session_dir = Path(spec.session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        state_path = session_dir / "eval_state.json"
        state = _read_json(state_path)
        attempts = list(state.get("attempts") or [])
        runtime_origins = dict(runtime_origin_evidence or {})

        if spec.eval_profile is None:
            thinking = {
                "enabled": True,
                "disabled": False,
            }.get(spec.eval_thinking_mode)
            spec = replace(
                spec,
                eval_profile=resolve_eval_profile(
                    spec.base_model,
                    discovery=spec.eval_discovery,
                    allow_llm=spec.eval_allow_llm,
                    overrides={
                        "task": spec.eval_task,
                        "num_fewshot": spec.eval_num_fewshot,
                        "prompting_strategy": (spec.eval_prompting_strategy),
                        "enable_thinking": thinking,
                        "max_gen_toks": spec.eval_max_gen_toks,
                    },
                    artifact_dir=session_dir / "evaluation" / "profile",
                ),
            )
        profile = spec.eval_profile
        if profile is None:
            raise RuntimeError("evaluation profile resolution returned no profile")

        baseline_fingerprint, quant_fingerprint = _runtime_fingerprints(
            spec,
            quant_model,
        )
        baseline_matches = state.get("baseline_fingerprint") == baseline_fingerprint and isinstance(
            state.get("baseline"), int | float
        )
        if not baseline_matches:
            state = {
                "schema_version": EVAL_STATE_SCHEMA_VERSION,
                "stage": "baseline",
                "profile_hash": profile.profile_hash,
                "base_model": spec.base_model,
                "quant_model": quant_model,
                "gsm8k_num_samples": spec.gsm8k_num_samples,
                "baseline_fingerprint": baseline_fingerprint,
                "quant_fingerprint": quant_fingerprint,
                "attempts": attempts,
                "runtime_origins": runtime_origins,
            }
            _write_json(state_path, state)
        elif state.get("quant_fingerprint") != quant_fingerprint:
            state.update(
                {
                    "stage": "quantized",
                    "quant_model": quant_model,
                    "quant_fingerprint": quant_fingerprint,
                    "quantized": None,
                    "gap": None,
                    "passed": None,
                    "quantized_artifact": "",
                    "error": "",
                    "runtime_origins": runtime_origins,
                }
            )
            _write_json(state_path, state)
        else:
            state["schema_version"] = EVAL_STATE_SCHEMA_VERSION
            state.setdefault("attempts", attempts)
            state["runtime_origins"] = runtime_origins
            _write_json(state_path, state)

        gate = AccuracyGate(spec)
        if isinstance(state.get("baseline"), int | float):
            gate._source_cache = float(state["baseline"])
            gate._baseline_artifact = str(state.get("baseline_artifact") or "")
            attempt_id = _start_attempt(
                state,
                role="baseline",
                model=spec.base_model,
                artifact=gate._baseline_artifact,
                profile_hash=profile.profile_hash,
                runtime_fingerprint=baseline_fingerprint,
                runtime_origins=runtime_origins,
                status="reused",
            )
            _finish_attempt(
                state,
                attempt_id,
                status="reused",
                score=gate._source_cache,
            )
            _write_json(state_path, state)
        else:
            write_progress(
                session_dir,
                stage="accuracy",
                stage_detail="running managed baseline real evaluation",
            )
            baseline_artifact = gate.reserve_artifact_dir("baseline")
            attempt_id = _start_attempt(
                state,
                role="baseline",
                model=spec.base_model,
                artifact=str(baseline_artifact),
                profile_hash=profile.profile_hash,
                runtime_fingerprint=baseline_fingerprint,
                runtime_origins=runtime_origins,
            )
            _write_json(state_path, state)
            try:
                gate.warm_baseline(output_dir=baseline_artifact)
            except Exception as exc:
                _finish_attempt(
                    state,
                    attempt_id,
                    status="failed",
                    error=_error_summary(exc),
                )
                state.update(
                    {
                        "stage": "failed",
                        "error": str(exc),
                    }
                )
                _write_json(state_path, state)
                write_progress(
                    session_dir,
                    stage="failed",
                    warning=f"baseline evaluation failed: {exc}",
                )
                raise
            _finish_attempt(
                state,
                attempt_id,
                status="passed",
                score=gate._source_cache,
            )
            state.update(
                {
                    "stage": "quantized",
                    "baseline": gate._source_cache,
                    "baseline_artifact": gate._baseline_artifact,
                    "baseline_fingerprint": baseline_fingerprint,
                    "error": "",
                }
            )
            _write_json(state_path, state)

        write_progress(
            session_dir,
            stage="accuracy",
            stage_detail="running managed quantized real evaluation",
        )
        quantized_artifact = gate.reserve_artifact_dir("quantized")
        attempt_id = _start_attempt(
            state,
            role="quantized",
            model=quant_model,
            artifact=str(quantized_artifact),
            profile_hash=profile.profile_hash,
            runtime_fingerprint=quant_fingerprint,
            runtime_origins=runtime_origins,
        )
        _write_json(state_path, state)
        try:
            accuracy = gate.eval_quantized(
                quant_model,
                output_dir=quantized_artifact,
            )
        except Exception as exc:
            _finish_attempt(
                state,
                attempt_id,
                status="failed",
                error=_error_summary(exc),
            )
            state.update(
                {
                    "stage": "failed",
                    "error": str(exc),
                }
            )
            _write_json(state_path, state)
            write_progress(
                session_dir,
                stage="failed",
                warning=f"quantized evaluation failed: {exc}",
            )
            raise
        _finish_attempt(
            state,
            attempt_id,
            status="passed",
            score=accuracy.quantized_gsm8k,
        )
        artifacts = dict(accuracy.artifacts)
        result: dict[str, Any] = {
            "status": ("success" if accuracy.passed else "accuracy_failed"),
            "baseline": accuracy.source_gsm8k,
            "quantized": accuracy.quantized_gsm8k,
            "gap": accuracy.gap,
            "passed": accuracy.passed,
            "artifacts": artifacts,
            "runtime_origins": runtime_origins,
            "attempts": list(state.get("attempts") or []),
        }
        state.update(
            {
                "stage": "done" if accuracy.passed else "failed",
                "baseline": accuracy.source_gsm8k,
                "quantized": accuracy.quantized_gsm8k,
                "gap": accuracy.gap,
                "passed": accuracy.passed,
                "error": "",
                "baseline_artifact": artifacts.get("baseline", ""),
                "quantized_artifact": artifacts.get(
                    "quantized",
                    "",
                ),
                "quant_fingerprint": quant_fingerprint,
            }
        )
        paths = _write_report(
            session_dir,
            spec=spec,
            quant_model=quant_model,
            result=result,
        )
        result["report_paths"] = paths
        state["report_paths"] = paths
        _write_json(state_path, state)
        write_progress(
            session_dir,
            stage=state["stage"],
            stage_detail="managed real evaluation complete",
            reports=paths,
        )
        return result
