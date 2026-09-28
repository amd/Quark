#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Shared real-model measurements for optimization candidates."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from quark.experimental.torch.quant_perf.evaluation.gate import AccuracyGate
from quark.experimental.torch.quant_perf.evaluation.throughput import ThroughputMeasurement
from quark.experimental.torch.quant_perf.pipeline.performance_policy import (
    RetestDisposition,
    aba_gain,
    abba_gain,
    decide_confirmed_retest,
    effective_keep_floor,
    screen_retest_mode,
)
from quark.experimental.torch.quant_perf.runtime.recovery import classify_failure
from quark.experimental.torch.quant_perf.session.spec import Checkpoint, PerfResult, Spec
from quark.experimental.torch.quant_perf.session.state import append_performance_measurement


def measure_quantized_candidate(
    spec: Spec,
    model_dir: str,
) -> ThroughputMeasurement:
    from quark.experimental.torch.quant_perf.evaluation.throughput import measure_throughput

    def run_once() -> ThroughputMeasurement:
        return measure_throughput(
            model_dir,
            gpu_id=spec.gpu_id,
            isl=spec.isl,
            osl=spec.osl,
            tp=spec.tp,
            concurrency=spec.bench_concurrency,
            gpu_memory_utilization=spec.vllm_gpu_memory_utilization,
            moe_backend=spec.vllm_moe_backend,
            max_num_seqs=spec.vllm_max_num_seqs,
            trust_remote_code=spec.vllm_trust_remote_code,
            runtime_python=spec.runtime_python,
            runtime_env=spec.runtime_env,
            kv_cache_dtype=spec.vllm_kv_cache_dtype,
        )

    measurement = run_once()
    return measurement if measurement.stable else run_once()


def measure_candidate_screen(
    spec: Spec,
    model_dir: str,
    runtime_env: dict[str, str],
) -> ThroughputMeasurement:
    from quark.experimental.torch.quant_perf.evaluation.throughput import measure_throughput

    return measure_throughput(
        model_dir,
        gpu_id=spec.gpu_id,
        isl=spec.isl,
        osl=spec.osl,
        num_prompts=2 * spec.bench_concurrency,
        tp=spec.tp,
        concurrency=spec.bench_concurrency,
        gpu_memory_utilization=spec.vllm_gpu_memory_utilization,
        timed_samples=2,
        max_timed_samples=2,
        moe_backend=spec.vllm_moe_backend,
        max_num_seqs=spec.vllm_max_num_seqs,
        trust_remote_code=spec.vllm_trust_remote_code,
        runtime_python=spec.runtime_python,
        runtime_env=runtime_env,
        kv_cache_dtype=spec.vllm_kv_cache_dtype,
    )


def evaluate_candidate_accuracy(
    spec: Spec,
    model_dir: str,
) -> tuple[str, float | None, str]:
    from quark.experimental.torch.quant_perf.evaluation.gsm8k import gsm8k_eval_offline

    last_error: Exception | None = None
    last_code = "unknown"
    for attempt in range(2):
        try:
            return (
                "passed",
                gsm8k_eval_offline(
                    model_dir,
                    gpu_id=spec.gpu_id,
                    num_questions=spec.gsm8k_num_samples,
                    tp=spec.tp,
                    gpu_memory_utilization=(spec.vllm_gpu_memory_utilization),
                    moe_backend=spec.vllm_moe_backend,
                    profile=spec.eval_profile,
                    trust_remote_code=spec.vllm_trust_remote_code,
                    max_num_seqs=spec.vllm_max_num_seqs,
                    runtime_python=spec.runtime_python,
                    runtime_env=spec.runtime_env,
                    kv_cache_dtype=spec.vllm_kv_cache_dtype,
                ),
                "",
            )
        except Exception as exc:
            last_error = exc
            diagnosis = classify_failure(
                exc,
                framework_repo=spec.active_framework_repo,
                kernel_repo=spec.active_kernel_repo,
            )
            last_code = diagnosis.code
            if attempt == 0 and diagnosis.recovery and diagnosis.recovery.get("action") == "retry_same":
                continue
            break
    if last_code == "cuda_graph_capture":
        return "needs_review", None, "graph_incompatible"
    return (
        "retryable_fault",
        None,
        f"validation_fault:{last_code}:{str(last_error)[-1000:]}",
    )


@dataclass(frozen=True)
class CandidateRetestResult:
    disposition: RetestDisposition
    mode: str
    multiplier: float
    effective_floor: float
    screen_gain: float
    anchor_measurements: tuple[ThroughputMeasurement, ...]
    candidate_measurements: tuple[ThroughputMeasurement, ...]
    candidate_active: bool


def run_candidate_retest(
    current_anchor: ThroughputMeasurement,
    candidate_first: ThroughputMeasurement,
    *,
    keep_floor: float,
    trace_rank: int | None,
    micro_speedup: float | None,
    deactivate: Callable[[], None],
    measure_anchor: Callable[[], ThroughputMeasurement],
    measure_candidate: Callable[[], ThroughputMeasurement],
) -> CandidateRetestResult:
    screen = screen_retest_mode(
        current_anchor,
        candidate_first,
        keep_floor=keep_floor,
        trace_rank=trace_rank,
        micro_speedup=micro_speedup,
    )
    anchor_measurements = [current_anchor]
    candidate_measurements = [candidate_first]
    mode = "screen"
    multiplier = candidate_first.median_tps / current_anchor.median_tps
    disposition = screen.disposition
    floor = screen.effective_floor
    candidate_active = True
    if disposition is RetestDisposition.CONFIRM_ABA:
        mode = "aba"
        # Strong gains already have a stable multi-sample candidate measurement.
        # Re-measure only the anchor to correct for temporal drift; borderline
        # and high-priority candidates take the more expensive ABBA path below.
        deactivate()
        candidate_active = False
        anchor_second = measure_anchor()
        anchor_measurements.append(anchor_second)
        floor = effective_keep_floor(
            keep_floor,
            current_anchor,
            candidate_first,
            anchor_second,
        )
        multiplier = aba_gain(
            current_anchor.median_tps,
            candidate_first.median_tps,
            anchor_second.median_tps,
        )
        disposition = decide_confirmed_retest(multiplier, floor)
    elif disposition is RetestDisposition.CONFIRM_ABBA:
        mode = "abba"
        candidate_second = measure_candidate()
        candidate_measurements.append(candidate_second)
        deactivate()
        candidate_active = False
        anchor_second = measure_anchor()
        anchor_measurements.append(anchor_second)
        floor = effective_keep_floor(
            keep_floor,
            current_anchor,
            candidate_first,
            candidate_second,
            anchor_second,
        )
        multiplier = abba_gain(
            current_anchor.median_tps,
            candidate_first.median_tps,
            candidate_second.median_tps,
            anchor_second.median_tps,
        )
        disposition = decide_confirmed_retest(multiplier, floor)
    elif disposition is RetestDisposition.CONFIRM_NEGATIVE:
        mode = "negative_confirm"
        candidate_second = measure_candidate()
        candidate_measurements.append(candidate_second)
        second_gain = candidate_second.median_tps / current_anchor.median_tps - 1.0
        disposition = (
            RetestDisposition.DROP_CONFIRMED
            if second_gain <= -screen.strong_threshold
            else RetestDisposition.NEEDS_REVIEW
        )
    return CandidateRetestResult(
        disposition=disposition,
        mode=mode,
        multiplier=multiplier,
        effective_floor=floor,
        screen_gain=screen.gain,
        anchor_measurements=tuple(anchor_measurements),
        candidate_measurements=tuple(candidate_measurements),
        candidate_active=candidate_active,
    )


def _restore_environment(
    snapshot: dict[str, str | None],
) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def validate_runtime_environment_candidates(
    perf: PerfResult,
    spec: Spec,
    *,
    quant_gain: float,
    quant_ckpt_dir: str,
    gate: AccuracyGate,
    ckpt: Checkpoint,
) -> tuple[dict[str, str], float, float, list[dict[str, Any]]]:
    """Apply the shared accuracy and E2E policy to runtime environments."""
    retained = {str(key): str(value) for key, value in (ckpt.state.get("retained_runtime_env") or {}).items()}
    for key, value in retained.items():
        os.environ[key] = value
    best_gain = quant_gain
    best_gap = ckpt.state.get("best_accuracy_gap") or 0.0
    baseline_score = ckpt.state.get("baseline_gsm8k") or gate._source_cache or 1e-9
    retained_artifacts = dict(perf.runtime_artifacts)
    retention_entries: list[dict[str, Any]] = []
    if not perf.runtime_candidates:
        perf.runtime_env = dict(retained)
        return retained, best_gain, best_gap, retention_entries
    current_anchor = None

    for candidate in perf.runtime_candidates:
        runtime_env = candidate.get("runtime_env") or {}
        if not isinstance(runtime_env, dict) or not runtime_env:
            continue
        normalized = {str(key): str(value) for key, value in runtime_env.items() if key and value}
        if not normalized:
            continue
        snapshot = {key: os.environ.get(key) for key in normalized}
        retained_before = dict(retained)
        artifacts_before = dict(retained_artifacts)
        gain_before = best_gain
        gap_before = best_gap
        name = str(candidate.get("name") or "runtime_candidate")
        if candidate.get("requires_screen"):
            screen_anchor_env = {
                **spec.runtime_env,
                **retained,
            }
            try:
                screen_anchor = measure_candidate_screen(
                    spec,
                    quant_ckpt_dir,
                    screen_anchor_env,
                )
                screen_candidate = measure_candidate_screen(
                    spec,
                    quant_ckpt_dir,
                    {
                        **screen_anchor_env,
                        **normalized,
                    },
                )
            except Exception as exc:
                ckpt.record_phase_event(
                    "vendor_gemm",
                    "retryable_fault",
                    candidate=name,
                    reason=f"screen_error:{str(exc)[-1000:]}",
                )
                ckpt.save()
                continue
            screen_floor = effective_keep_floor(
                spec.keep_floor,
                screen_anchor,
                screen_candidate,
            )
            screen_gain = screen_candidate.median_tps / screen_anchor.median_tps - 1.0
            if screen_anchor.stable and screen_candidate.stable and screen_gain <= -screen_floor:
                _restore_environment(snapshot)
                ckpt.record_phase_event(
                    "vendor_gemm",
                    "rejected",
                    candidate=name,
                    reason="screen_confirmed_no_gain",
                    gain=screen_gain,
                )
                ckpt.save()
                continue

        if current_anchor is None:
            try:
                current_anchor = measure_quantized_candidate(
                    spec,
                    quant_ckpt_dir,
                )
            except Exception as exc:
                ckpt.record_phase_event(
                    "vendor_gemm",
                    "retryable_fault",
                    reason=f"anchor_error:{str(exc)[-1000:]}",
                )
                ckpt.save()
                perf.runtime_env = dict(retained)
                return retained, best_gain, best_gap, retention_entries

        os.environ.update(normalized)
        try:
            accuracy_status, score, accuracy_reason = evaluate_candidate_accuracy(spec, quant_ckpt_dir)
            if accuracy_status != "passed" or score is None:
                _restore_environment(snapshot)
                ckpt.record_phase_event(
                    "vendor_gemm",
                    ("needs_review" if accuracy_status == "needs_review" else "retryable_fault"),
                    candidate=name,
                    reason=accuracy_reason,
                )
                ckpt.save()
                continue
            gap = max(
                0.0,
                (baseline_score - score) / max(baseline_score, 1e-9),
            )
            if gap > spec.accuracy_gap:
                _restore_environment(snapshot)
                ckpt.record_phase_event(
                    "vendor_gemm",
                    "rejected",
                    candidate=name,
                    reason="accuracy_regression",
                    accuracy_gap=gap,
                )
                ckpt.save()
                continue
            candidate_first = measure_quantized_candidate(
                spec,
                quant_ckpt_dir,
            )
        except Exception as exc:
            _restore_environment(snapshot)
            ckpt.record_phase_event(
                "vendor_gemm",
                "rejected",
                candidate=name,
                reason=f"validation_error: {exc}",
            )
            ckpt.save()
            continue

        try:
            retest = run_candidate_retest(
                current_anchor,
                candidate_first,
                keep_floor=spec.keep_floor,
                trace_rank=None,
                micro_speedup=candidate.get("micro_speedup"),
                deactivate=lambda snapshot=snapshot: _restore_environment(snapshot),
                measure_anchor=lambda: measure_quantized_candidate(
                    spec,
                    quant_ckpt_dir,
                ),
                measure_candidate=lambda: measure_quantized_candidate(
                    spec,
                    quant_ckpt_dir,
                ),
            )
        except Exception as exc:
            _restore_environment(snapshot)
            ckpt.record_phase_event(
                "vendor_gemm",
                "retryable_fault",
                candidate=name,
                reason=f"retest_error:{str(exc)[-1000:]}",
            )
            ckpt.save()
            continue

        anchor_measurements = list(retest.anchor_measurements)
        candidate_measurements = list(retest.candidate_measurements)
        gain = best_gain * retest.multiplier
        if retest.disposition is RetestDisposition.KEEP:
            if not retest.candidate_active:
                os.environ.update(normalized)
            retained.update(normalized)
            artifacts = candidate.get("artifacts") or {}
            if isinstance(artifacts, dict):
                retained_artifacts.update({str(key): str(value) for key, value in artifacts.items()})
            best_gain, best_gap = gain, gap
            current_anchor = candidate_measurements[-1]
            ckpt.state["retained_runtime_env"] = dict(retained)
            retention_entries.append(
                {
                    "kind": "runtime",
                    "candidate": name,
                    "runtime_snapshot": dict(snapshot),
                    "runtime_env_before": retained_before,
                    "runtime_env_after": dict(retained),
                    "runtime_artifacts_before": artifacts_before,
                    "runtime_artifacts_after": dict(retained_artifacts),
                    "gain_before": gain_before,
                    "gain_after": gain,
                    "accuracy_gap_before": gap_before,
                    "accuracy_gap_after": gap,
                    "active": True,
                }
            )
            append_performance_measurement(
                ckpt.state,
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "role": "vendor_gemm",
                    "candidate": name,
                    "mode": retest.mode,
                    "quantized_tps": current_anchor.median_tps,
                    "gain": gain,
                    "incremental_multiplier": retest.multiplier,
                    "effective_keep_floor": (retest.effective_floor),
                    "anchor_measurements": [item.to_dict() for item in anchor_measurements],
                    "candidate_measurements": [item.to_dict() for item in candidate_measurements],
                    "isl": spec.isl,
                    "osl": spec.osl,
                    "concurrency": spec.bench_concurrency,
                },
            )
            ckpt.record_phase_event(
                "vendor_gemm",
                "kept",
                candidate=name,
                gain=gain,
                accuracy_gap=gap,
            )
            ckpt.save()
        else:
            if retest.candidate_active:
                _restore_environment(snapshot)
            ckpt.record_phase_event(
                "vendor_gemm",
                ("rejected" if (retest.disposition is RetestDisposition.DROP_CONFIRMED) else "needs_review"),
                candidate=name,
                reason=(
                    "confirmed_no_gain"
                    if (retest.disposition is RetestDisposition.DROP_CONFIRMED)
                    else "within_measurement_noise"
                ),
                gain=gain,
                accuracy_gap=gap,
            )
            ckpt.save()

    perf.runtime_env = dict(retained)
    perf.runtime_artifacts = retained_artifacts
    return retained, best_gain, best_gap, retention_entries
