#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quark.experimental.torch.quant_perf.evaluation.gate import AccuracyGate
from quark.experimental.torch.quant_perf.evaluation.gsm8k import DependencyError
from quark.experimental.torch.quant_perf.repair.evidence import extract_failure_evidence
from quark.experimental.torch.quant_perf.repair.request import build_repair_request
from quark.experimental.torch.quant_perf.repair.service import RepairService
from quark.experimental.torch.quant_perf.repair.signatures import normalize_quant_signature
from quark.experimental.torch.quant_perf.runtime.recovery import build_runtime_fingerprint, classify_failure
from quark.experimental.torch.quant_perf.session.progress import write_progress
from quark.experimental.torch.quant_perf.session.spec import AccuracyResult, Checkpoint, Spec, StageError
from quark.experimental.torch.quant_perf.workspace.git import get_head_sha

if TYPE_CHECKING:
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore


class BaselineHealthStage:
    """Validate the unquantized model before quantization begins."""

    def __init__(
        self,
        repair_service: RepairService,
        experience_store: ExperienceStore | None = None,
    ) -> None:
        self.repair_service = repair_service
        self.experience_store = experience_store

    @staticmethod
    def baseline_reference_matches_current_runtime(
        spec: Spec,
        ckpt: Checkpoint,
        runtime_fingerprint: str,
    ) -> bool:
        reference = ckpt.state.get("baseline_reference")
        if not isinstance(reference, dict):
            return False
        profile_hash = spec.eval_profile.profile_hash if spec.eval_profile is not None else ""
        return (
            reference.get("score") == ckpt.state.get("baseline_gsm8k")
            and reference.get("profile_hash") == profile_hash
            and reference.get("gsm8k_num_samples") == spec.gsm8k_num_samples
            and reference.get("base_model") == str(Path(spec.base_model).resolve())
            and reference.get("runtime_fingerprint") == runtime_fingerprint
        )

    @staticmethod
    def _current_framework_commit(spec: Spec) -> str:
        repo = spec.active_framework_repo
        if not repo or not Path(repo).is_dir():
            return spec.framework_version
        try:
            return get_head_sha(repo) or spec.framework_version
        except OSError:
            return spec.framework_version

    @staticmethod
    def _effective_runtime_env(spec: Spec) -> dict[str, str]:
        return {**os.environ, **dict(spec.runtime_env)}

    def check_baseline_before_quantization(
        self,
        spec: Spec,
        ckpt: Checkpoint,
        gate: AccuracyGate,
        runtime_fingerprint: str,
        framework_commit: str,
    ) -> str:
        """Return an empty string when healthy, otherwise the diagnosis."""
        runtime_health = ckpt.state.get("baseline_runtime_health")
        if (
            isinstance(runtime_health, dict)
            and runtime_health.get("status") == "healthy"
            and runtime_health.get("fingerprint") == runtime_fingerprint
            and self.baseline_reference_matches_current_runtime(
                spec,
                ckpt,
                runtime_fingerprint,
            )
        ):
            baseline = ckpt.state.get("baseline_gsm8k")
            if baseline is None:
                gate.warm_baseline()
                ckpt.state["baseline_gsm8k"] = gate._source_cache
                ckpt.save()
            else:
                gate._source_cache = baseline
            return ""

        write_progress(
            spec.session_dir,
            stage="quantize",
            stage_detail=("checking baseline framework health before quantization"),
        )
        healthy, diagnosis, baseline = gate.check_baseline_health()
        same_retry_codes: set[str] = set()
        for _ in range(2):
            if healthy or not spec.fix_base_framework:
                break
            failure = classify_failure(
                diagnosis,
                framework_repo=spec.active_framework_repo,
                kernel_repo=spec.active_kernel_repo,
            )
            recovery = failure.recovery
            applied = False
            before = dict(ckpt.state.get("retained_runtime_env") or {})
            if recovery and recovery.get("action") == "set_env":
                key = str(recovery["key"])
                value = str(recovery["value"])
                if before.get(key) != value:
                    ckpt.state.setdefault(
                        "retained_runtime_env",
                        {},
                    )[key] = value
                    os.environ[key] = value
                    applied = True
            elif recovery and recovery.get("action") == "retry_same":
                if failure.code not in same_retry_codes:
                    same_retry_codes.add(failure.code)
                    applied = True
            if not applied:
                break
            ckpt.state.setdefault("recovery_attempts", []).append(
                {
                    "attempt": (len(ckpt.state.get("recovery_attempts") or []) + 1),
                    "stage": "baseline_health",
                    "role": "baseline",
                    "failure_class": failure.failure_class,
                    "code": failure.code,
                    "action": recovery["action"],
                    "before": {"runtime_env": before},
                    "after": {"runtime_env": dict(ckpt.state.get("retained_runtime_env") or {})},
                    "outcome": "retry",
                    "error": diagnosis[-4000:],
                }
            )
            runtime_fingerprint = build_runtime_fingerprint(
                spec,
                framework_commit=framework_commit,
                runtime_env=self._effective_runtime_env(spec),
                stage="baseline_health",
            )
            ckpt.state["baseline_runtime_fingerprint"] = runtime_fingerprint
            ckpt.save()
            gate._source_cache = None
            healthy, diagnosis, baseline = gate.check_baseline_health()

        failure = classify_failure(
            diagnosis,
            framework_repo=spec.active_framework_repo,
            kernel_repo=spec.active_kernel_repo,
        )
        if (
            not healthy
            and spec.fix_base_framework
            and spec.can_modify_framework
            and (failure.repair_eligible or failure.failure_class == "unknown")
        ):
            write_progress(
                spec.session_dir,
                stage_detail=("repairing baseline framework before quantization"),
            )
            repair = self.repair_service.repair(
                build_repair_request(
                    spec,
                    failure_class="load_run",
                    error=diagnosis,
                    quant_ckpt_dir=spec.base_model,
                    verifier_profile="load_inference",
                )
            )
            if repair.status == "fixed":
                gate._source_cache = None
                healthy, diagnosis, baseline = gate.check_baseline_health()
                if healthy and self.experience_store is not None:
                    framework_commit = self._current_framework_commit(spec)
                    runtime_fingerprint = build_runtime_fingerprint(
                        spec,
                        framework_commit=framework_commit,
                        runtime_env=self._effective_runtime_env(spec),
                        stage="baseline_health",
                    )
                    self.experience_store.record_baseline_health(
                        fingerprint=runtime_fingerprint,
                        framework=spec.framework,
                        framework_commit=framework_commit,
                        outcome="fixed",
                        failure_class="framework",
                        cache_policy="hard",
                        diagnosis=("fixed via baseline runtime repair"),
                    )

        if not healthy:
            failure = classify_failure(
                diagnosis,
                framework_repo=spec.active_framework_repo,
                kernel_repo=spec.active_kernel_repo,
            )
            ckpt.state["baseline_runtime_health"] = {
                "status": "unhealthy",
                "fingerprint": runtime_fingerprint,
                "failure_class": failure.failure_class,
                "cache_policy": failure.cache_policy,
                "diagnosis": diagnosis,
            }
            if self.experience_store is not None:
                self.experience_store.record_baseline_health(
                    fingerprint=runtime_fingerprint,
                    framework=spec.framework,
                    framework_commit=framework_commit,
                    outcome="failed",
                    failure_class=failure.failure_class,
                    cache_policy=failure.cache_policy,
                    diagnosis=diagnosis,
                    recovery=json.dumps(failure.recovery or {}),
                )
            ckpt.save()
            return diagnosis

        ckpt.state["baseline_gsm8k"] = gate._source_cache
        ckpt.state["baseline_reference"] = {
            "score": gate._source_cache,
            "profile_hash": ckpt.state.get("eval_profile_hash") or "",
            "gsm8k_num_samples": spec.gsm8k_num_samples,
            "base_model": str(Path(spec.base_model).resolve()),
            "runtime_fingerprint": runtime_fingerprint,
            "source": "measured",
        }
        ckpt.state["baseline_runtime_health"] = {
            "status": "healthy",
            "fingerprint": runtime_fingerprint,
            "source": "measured",
        }
        if self.experience_store is not None:
            self.experience_store.record_baseline_health(
                fingerprint=runtime_fingerprint,
                framework=spec.framework,
                framework_commit=framework_commit,
                outcome="healthy",
                failure_class="none",
                cache_policy="none",
                diagnosis="baseline health passed",
            )
        ckpt.record_phase_event(
            "baseline_health",
            "passed",
            baseline=baseline,
            framework=spec.framework,
        )
        ckpt.save()
        return ""


class AccuracyStage:
    MAX_REPAIR_ATTEMPTS = 3

    def __init__(self, repair_service: RepairService) -> None:
        self.repair_service = repair_service
        self.last_recovery: dict[str, Any] | None = None

    def evaluate_quantized_checkpoint(
        self,
        spec: Spec,
        gate: AccuracyGate,
        quant_ckpt_dir: str,
    ) -> AccuracyResult:
        self.last_recovery = None
        repair_attempts = 0
        resource_retry_attempted = False
        seen_failures: set[tuple[str, str]] = set()
        pending_error: Exception | None = None
        while True:
            try:
                if pending_error is None:
                    return gate.eval_quantized(quant_ckpt_dir)
                current_error = pending_error
                pending_error = None
                raise current_error
            except Exception as current_error:
                if isinstance(current_error, DependencyError):
                    raise
                diagnosis = classify_failure(
                    current_error,
                    framework_repo=spec.active_framework_repo,
                    kernel_repo=spec.active_kernel_repo,
                )
                if diagnosis.failure_class == "resource_contention":
                    if resource_retry_attempted:
                        raise StageError(
                            "land",
                            f"quantized accuracy resource retry failed: {current_error}",
                            code=diagnosis.code,
                            diagnostic=str(current_error),
                        ) from current_error
                    resource_retry_attempted = True
                    retry_utilization = max(
                        0.1,
                        round(
                            spec.vllm_gpu_memory_utilization - 0.01,
                            2,
                        ),
                    )
                    self.last_recovery = {
                        "stage": "accuracy",
                        "failure_class": diagnosis.failure_class,
                        "code": diagnosis.code,
                        "action": "decrease_gpu_memory_utilization",
                        "requested_gpu_memory_utilization": (spec.vllm_gpu_memory_utilization),
                        "effective_gpu_memory_utilization": (retry_utilization),
                        "outcome": "retrying",
                    }
                    try:
                        result = gate.eval_quantized(
                            quant_ckpt_dir,
                            gpu_memory_utilization=retry_utilization,
                        )
                        self.last_recovery["outcome"] = "passed"
                        return result
                    except Exception as retry_error:
                        self.last_recovery["outcome"] = "reclassified"
                        pending_error = retry_error
                        continue
                request = build_repair_request(
                    spec,
                    failure_class="load_run",
                    error=current_error,
                    quant_ckpt_dir=quant_ckpt_dir,
                    verifier_profile="load_inference",
                    quant_signature=normalize_quant_signature(spec.quant_strategy or ""),
                    diagnosis=diagnosis,
                )
                if not diagnosis.repair_eligible or not self.repair_service.can_repair(request):
                    raise StageError(
                        "land",
                        f"quantized accuracy load failed: {current_error}",
                        code=diagnosis.code,
                        diagnostic=str(current_error),
                    ) from current_error
                failure_key = (
                    diagnosis.code,
                    extract_failure_evidence(request).signature,
                )
                if failure_key in seen_failures or repair_attempts >= self.MAX_REPAIR_ATTEMPTS:
                    raise StageError(
                        "land",
                        f"quantized accuracy repair budget exhausted: {current_error}",
                        code=diagnosis.code,
                        diagnostic=str(current_error),
                    ) from current_error
                seen_failures.add(failure_key)
                repair_attempts += 1
                repair = self.repair_service.repair(request)
                if repair.status != "fixed":
                    raise StageError(
                        "land",
                        f"quantized accuracy repair failed: {current_error}",
                        code=diagnosis.code,
                        diagnostic=str(current_error),
                    ) from current_error
