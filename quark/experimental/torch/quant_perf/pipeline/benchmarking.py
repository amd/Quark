#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Comparable throughput measurement and bounded runtime recovery."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from quark.experimental.torch.quant_perf.evaluation.gate import AccuracyGate
from quark.experimental.torch.quant_perf.pipeline.candidate_validation import measure_quantized_candidate
from quark.experimental.torch.quant_perf.repair.request import build_repair_request
from quark.experimental.torch.quant_perf.repair.service import RepairService
from quark.experimental.torch.quant_perf.runtime.recovery import (
    FailureDiagnosis,
    build_runtime_fingerprint,
    classify_failure,
)
from quark.experimental.torch.quant_perf.session.progress import write_progress
from quark.experimental.torch.quant_perf.session.spec import Checkpoint, Spec, StageError
from quark.experimental.torch.quant_perf.session.state import append_performance_measurement
from quark.experimental.torch.quant_perf.workspace.git import get_head_sha


class BenchmarkCoordinator:
    """Measure comparable throughput pairs and persist recovery evidence."""

    def __init__(self, repair_service: RepairService) -> None:
        self.repair_service = repair_service

    @staticmethod
    def _current_framework_commit(spec: Spec) -> str:
        repo = spec.active_framework_repo
        if not repo or not Path(repo).is_dir():
            return spec.framework_version
        try:
            return get_head_sha(repo) or spec.framework_version
        except OSError:
            return spec.framework_version

    def measure_baseline_and_quantized_throughput(
        self,
        spec: Spec,
        ckpt: Checkpoint,
        quant_ckpt_dir: str,
        *,
        allow_repair: bool = True,
    ) -> tuple[float, float, float]:
        """Measure one comparable baseline/quantized pair with recovery."""
        from quark.experimental.torch.quant_perf.evaluation.throughput import (
            throughput_benchmark,
            throughput_gain,
        )

        runtime = dict(ckpt.state.get("performance_runtime") or {})
        requested_util = spec.vllm_gpu_memory_utilization
        effective_util = float(
            runtime.get(
                "effective_gpu_memory_utilization",
                requested_util,
            )
        )
        retained_env = ckpt.state.setdefault(
            "retained_runtime_env",
            {},
        )
        for key, value in retained_env.items():
            os.environ[str(key)] = str(value)
        same_retry_codes: set[str] = set()
        framework_commit = self._current_framework_commit(spec)

        for attempt in range(1, 4):
            runtime_fingerprint = build_runtime_fingerprint(
                spec,
                framework_commit=framework_commit,
                runtime_env={**os.environ, **retained_env},
                stage="throughput",
                effective_gpu_memory_utilization=effective_util,
            )
            runtime.update(
                {
                    "requested_gpu_memory_utilization": requested_util,
                    "effective_gpu_memory_utilization": effective_util,
                    "runtime_env": dict(retained_env),
                    "attempt": attempt,
                    "fingerprint": runtime_fingerprint,
                }
            )
            ckpt.state["performance_runtime"] = runtime
            ckpt.save()

            measurements: dict[str, float] = {}
            for role, model_dir in (
                ("baseline", spec.base_model),
                ("quantized", quant_ckpt_dir),
            ):
                write_progress(
                    spec.session_dir,
                    stage_detail=(f"throughput benchmark: {role} (attempt {attempt}/3)"),
                )
                try:
                    measurements[role] = throughput_benchmark(
                        model_dir,
                        gpu_id=spec.gpu_id,
                        isl=spec.isl,
                        osl=spec.osl,
                        tp=spec.tp,
                        concurrency=spec.bench_concurrency,
                        gpu_memory_utilization=effective_util,
                        moe_backend=spec.vllm_moe_backend,
                        max_num_seqs=spec.vllm_max_num_seqs,
                        trust_remote_code=spec.vllm_trust_remote_code,
                        runtime_python=spec.runtime_python,
                        runtime_env=spec.runtime_env,
                        kv_cache_dtype=spec.vllm_kv_cache_dtype,
                    )
                    continue
                except Exception as exc:
                    failure_exc = exc
                    diagnosis = classify_failure(
                        exc,
                        framework_repo=spec.active_framework_repo,
                        kernel_repo=spec.active_kernel_repo,
                    )

                recovery = diagnosis.recovery
                before = {
                    "gpu_memory_utilization": effective_util,
                    "runtime_env": dict(retained_env),
                }
                applied = False
                if recovery and recovery.get("action") == "set_env":
                    key = str(recovery["key"])
                    value = str(recovery["value"])
                    if retained_env.get(key) != value:
                        retained_env[key] = value
                        os.environ[key] = value
                        applied = True
                elif recovery and recovery.get("action") == "increase_gpu_memory_utilization":
                    maximum = float(recovery["maximum"])
                    new_util = min(
                        maximum,
                        round(
                            effective_util + float(recovery["step"]),
                            2,
                        ),
                    )
                    if new_util > effective_util:
                        effective_util = new_util
                        applied = True
                elif recovery and recovery.get("action") == "retry_same":
                    if diagnosis.code not in same_retry_codes:
                        same_retry_codes.add(diagnosis.code)
                        applied = True

                ckpt.state["recovery_attempts"].append(
                    {
                        "attempt": (len(ckpt.state["recovery_attempts"]) + 1),
                        "stage": "throughput",
                        "role": role,
                        "failure_class": diagnosis.failure_class,
                        "code": diagnosis.code,
                        "action": (recovery.get("action") if recovery else "stop"),
                        "before": before,
                        "after": {
                            "gpu_memory_utilization": effective_util,
                            "runtime_env": dict(retained_env),
                        },
                        "outcome": "retry" if applied else "unresolved",
                        "error": str(failure_exc)[-4000:],
                    }
                )
                ckpt.state["performance_runtime"] = {
                    **runtime,
                    "effective_gpu_memory_utilization": effective_util,
                    "runtime_env": dict(retained_env),
                }
                ckpt.save()

                if not applied:
                    if allow_repair and diagnosis.repair_eligible:
                        fixed_pair = self._repair_and_remeasure_throughput(
                            spec,
                            ckpt,
                            quant_ckpt_dir,
                            error=str(failure_exc),
                            role=role,
                            diagnosis=diagnosis,
                        )
                        if fixed_pair is not None:
                            return fixed_pair
                    raise StageError(
                        "throughput",
                        f"{role} throughput failed ({diagnosis.code}): {str(failure_exc)[-2000:]}",
                    ) from failure_exc
                break
            else:
                baseline_tps = measurements["baseline"]
                quantized_tps = measurements["quantized"]
                return (
                    baseline_tps,
                    quantized_tps,
                    throughput_gain(quantized_tps, baseline_tps),
                )

        last = (ckpt.state.get("recovery_attempts") or [{}])[-1]
        raise StageError(
            "throughput",
            f"throughput recovery budget exhausted after {last.get('code', 'unknown')}",
        )

    def reuse_matching_quantized_throughput_pair(
        self,
        spec: Spec,
        ckpt: Checkpoint,
    ) -> tuple[float, float, float] | None:
        if ckpt.state.get("stage") != "perfopt":
            return None

        runtime = ckpt.state.get("performance_runtime")
        if not isinstance(runtime, dict) or not runtime.get("fingerprint"):
            return None
        effective_util = float(
            runtime.get(
                "effective_gpu_memory_utilization",
                spec.vllm_gpu_memory_utilization,
            )
        )
        retained_env = dict(ckpt.state.get("retained_runtime_env") or runtime.get("runtime_env") or {})
        current_fingerprint = build_runtime_fingerprint(
            spec,
            framework_commit=self._current_framework_commit(spec),
            runtime_env={**os.environ, **retained_env},
            stage="throughput",
            effective_gpu_memory_utilization=effective_util,
        )
        if runtime["fingerprint"] != current_fingerprint:
            return None

        accuracy_attempts = ckpt.state.get("accuracy_attempts") or []
        latest_accuracy_ts = str(accuracy_attempts[-1].get("ts") or "") if accuracy_attempts else ""
        measurements = ckpt.state.get("performance_measurements") or []
        for measurement in reversed(measurements):
            if measurement.get("role") != "quant_only":
                continue
            if (
                int(measurement.get("isl") or 0) != spec.isl
                or int(measurement.get("osl") or 0) != spec.osl
                or int(measurement.get("concurrency") or 0) != spec.bench_concurrency
            ):
                continue
            measurement_ts = str(measurement.get("ts") or "")
            if latest_accuracy_ts and measurement_ts and measurement_ts < latest_accuracy_ts:
                continue
            baseline_tps = float(measurement.get("baseline_tps") or 0.0)
            quantized_tps = float(measurement.get("quantized_tps") or 0.0)
            gain = float(measurement.get("gain") or 0.0)
            if baseline_tps > 0.0 and quantized_tps > 0.0 and gain > 0.0:
                return baseline_tps, quantized_tps, gain
        return None

    def measure_final_stack_with_abba(
        self,
        spec: Spec,
        ckpt: Checkpoint,
        quant_ckpt_dir: str,
    ) -> tuple[float, float, float, float]:
        """Measure the final stack in ABBA order to balance temporal drift."""
        from quark.experimental.torch.quant_perf.pipeline.performance_policy import (
            abba_gain,
            effective_keep_floor,
        )

        baseline_first = measure_quantized_candidate(
            spec,
            spec.base_model,
        )
        final_first = measure_quantized_candidate(spec, quant_ckpt_dir)
        final_second = measure_quantized_candidate(spec, quant_ckpt_dir)
        baseline_second = measure_quantized_candidate(
            spec,
            spec.base_model,
        )
        baseline_tps = (baseline_first.median_tps * baseline_second.median_tps) ** 0.5
        final_tps = (final_first.median_tps * final_second.median_tps) ** 0.5
        gain = abba_gain(
            baseline_first.median_tps,
            final_first.median_tps,
            final_second.median_tps,
            baseline_second.median_tps,
        )
        effective_floor = effective_keep_floor(
            spec.keep_floor,
            baseline_first,
            final_first,
            final_second,
            baseline_second,
        )
        append_performance_measurement(
            ckpt.state,
            {
                "ts": datetime.now(UTC).isoformat(),
                "role": "final_abba",
                "mode": "abba",
                "baseline_tps": baseline_tps,
                "final_tps": final_tps,
                "gain": gain,
                "effective_keep_floor": effective_floor,
                "baseline_measurements": [
                    baseline_first.to_dict(),
                    baseline_second.to_dict(),
                ],
                "final_measurements": [
                    final_first.to_dict(),
                    final_second.to_dict(),
                ],
                "isl": spec.isl,
                "osl": spec.osl,
                "concurrency": spec.bench_concurrency,
            },
        )
        ckpt.save()
        return baseline_tps, final_tps, gain, effective_floor

    def _repair_and_remeasure_throughput(
        self,
        spec: Spec,
        ckpt: Checkpoint,
        quant_ckpt_dir: str,
        *,
        error: str,
        role: str,
        diagnosis: FailureDiagnosis,
    ) -> tuple[float, float, float] | None:
        verified_pair: dict[str, tuple[float, float, float]] = {}

        def verify() -> tuple[bool, str]:
            gate = AccuracyGate(spec)
            profile = spec.eval_profile
            if profile is None:
                return False, "Eval Profile is missing"
            baseline = ckpt.state.get("baseline_gsm8k")
            if baseline is None:
                return False, "baseline GSM8K is missing"
            gate._source_cache = float(baseline)
            try:
                accuracy = gate.eval_quantized(quant_ckpt_dir)
            except Exception as exc:
                return False, f"post-repair accuracy errored: {exc}"
            ckpt.state.setdefault("accuracy_attempts", []).append(
                {
                    "attempt": (len(ckpt.state.get("accuracy_attempts") or []) + 1),
                    "ts": datetime.now(UTC).isoformat(),
                    "baseline": accuracy.source_gsm8k,
                    "quantized": accuracy.quantized_gsm8k,
                    "gap": accuracy.gap,
                    "passed": accuracy.passed,
                    "candidate": (
                        dict(ckpt.state["best_candidate"])
                        if isinstance(
                            ckpt.state.get("best_candidate"),
                            dict,
                        )
                        else None
                    ),
                    "reason": "post_performance_repair",
                    "profile_id": profile.profile_id,
                    "profile_hash": profile.profile_hash,
                    "artifacts": dict(accuracy.artifacts),
                }
            )
            ckpt.save()
            if not accuracy.passed:
                return False, (f"post-repair accuracy gap {accuracy.gap:.4f} exceeds {spec.accuracy_gap:.4f}")
            try:
                pair = self.measure_baseline_and_quantized_throughput(
                    spec,
                    ckpt,
                    quant_ckpt_dir,
                    allow_repair=False,
                )
            except Exception as exc:
                return False, f"post-repair throughput errored: {exc}"
            verified_pair["value"] = pair
            return True, ""

        request = build_repair_request(
            spec,
            failure_class="benchmark_execution",
            error=error,
            quant_ckpt_dir=quant_ckpt_dir,
            verifier_profile="benchmark_execution",
            quant_signature=(ckpt.state.get("performance_runtime") or {}).get(
                "fingerprint",
                ckpt.state.get("baseline_runtime_fingerprint", ""),
            ),
            verifier=verify,
            diagnosis=diagnosis,
        )
        if not self.repair_service.can_repair(request):
            return None
        repair = self.repair_service.repair(request)
        fixed = repair.status == "fixed"
        ckpt.state.setdefault("recovery_attempts", []).append(
            {
                "attempt": (len(ckpt.state.get("recovery_attempts") or []) + 1),
                "stage": "throughput",
                "role": role,
                "failure_class": diagnosis.failure_class,
                "code": "performance_repair",
                "action": "repair",
                "outcome": "fixed" if fixed else "failed",
                "error": error[-4000:],
            }
        )
        ckpt.save()
        return verified_pair.get("value") if fixed else None
