#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Extract reusable runtime experience from terminal session evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quark.experimental.torch.quant_perf.perfopt.keep import normalize_kernel_sig
from quark.experimental.torch.quant_perf.session.spec import Spec
from quark.experimental.torch.quant_perf.session.state import SessionState

if TYPE_CHECKING:
    from .store import ExperienceStore


@dataclass(frozen=True)
class ExperienceCaptureSummary:
    quantization: int = 0
    repair: int = 0
    kernel_optimization: int = 0


def _fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _backend_context(spec: Spec) -> dict[str, str]:
    return {
        "mxfp4_moe": str(getattr(spec, "mxfp4_moe_backend", "") or ""),
        "mxfp4_gemm": str(getattr(spec, "mxfp4_gemm_backend", "") or ""),
        "w4a8_gemm": str(getattr(spec, "w4a8_gemm_backend", "") or ""),
    }


def _quant_signature(spec: Spec, state: SessionState) -> str:
    from quark.experimental.torch.quant_perf.repair.signatures import normalize_quant_signature

    return normalize_quant_signature(
        str(getattr(spec, "quant_strategy", "") or ""),
        state.get("best_candidate"),
    )


def _repair_is_verified(
    journey: dict[str, Any],
    state: SessionState,
    final_quant_signature: str,
) -> bool:
    verifier_results = list(journey.get("verifier_results") or [])
    screened = bool(
        journey.get("promoted")
        and journey.get("status") == "fixed"
        and any(result.get("passed") is True for result in verifier_results)
    )
    if not screened or journey.get("failure_class") != "accuracy_gap":
        return screened

    accuracy_validation = state.get("accuracy_validation")
    return bool(
        isinstance(accuracy_validation, dict)
        and accuracy_validation.get("passed") is True
        and str(journey.get("quant_signature") or "") == final_quant_signature
    )


class TerminalExperienceRecorder:
    """Persist terminal facts once so pipeline services never write history."""

    def __init__(self, store: ExperienceStore) -> None:
        self.store = store

    def capture(
        self,
        *,
        spec: Spec,
        state: SessionState,
    ) -> ExperienceCaptureSummary:
        session_id = str(state.get("session_id") or "unknown-session")
        quant_signature = _quant_signature(spec, state)
        quantization = self._capture_quantization(
            spec,
            state,
            session_id,
        )
        repair = self._capture_repair(
            spec,
            state,
            session_id,
            quant_signature,
        )
        kernel = self._capture_kernel_optimization(
            spec,
            state,
            session_id,
            quant_signature,
        )
        return ExperienceCaptureSummary(
            quantization=quantization,
            repair=repair,
            kernel_optimization=kernel,
        )

    def _capture_quantization(
        self,
        spec: Spec,
        state: SessionState,
        session_id: str,
    ) -> int:
        attempts = list(state.get("accuracy_attempts") or [])
        stack = {
            "arch_fingerprint": str(getattr(spec, "arch_fingerprint", "") or ""),
            "framework_version": str(getattr(spec, "framework_version", "") or ""),
            "kernel_version": str(getattr(spec, "kernel_version", "") or ""),
            "gpu_type": str(getattr(spec, "gpu_type", "") or ""),
            "tp": int(getattr(spec, "tp", 1) or 1),
            "backend": _backend_context(spec),
        }
        count = 0
        for index, attempt in enumerate(attempts, start=1):
            baseline = attempt.get("baseline")
            quantized = attempt.get("quantized")
            profile_hash = str(attempt.get("profile_hash") or "")
            verified = isinstance(baseline, int | float) and isinstance(quantized, int | float) and bool(profile_hash)
            payload = {
                "candidate": attempt.get("candidate"),
                "baseline_accuracy": baseline,
                "quantized_accuracy": quantized,
                "real_accuracy_gap": attempt.get("gap"),
                "eval_profile_hash": profile_hash,
                "artifacts": dict(attempt.get("artifacts") or {}),
                "stack": stack,
            }
            self.store.record_quantization_experience(
                record_id=(f"{session_id}:accuracy:{int(attempt.get('attempt') or index)}"),
                source_session_id=session_id,
                context_fingerprint=(
                    str((state.get("accuracy_validation") or {}).get("fingerprint") or "")
                    or _fingerprint({**stack, **payload})
                ),
                model_arch=str(getattr(spec, "model_arch", "") or ""),
                framework=str(getattr(spec, "framework", "") or ""),
                gpu_type=str(getattr(spec, "gpu_type", "") or ""),
                outcome=("passed" if attempt.get("passed") else "rejected"),
                verification_status=("verified" if verified else "observed"),
                payload=payload,
            )
            count += 1
        return count

    def _capture_repair(
        self,
        spec: Spec,
        state: SessionState,
        session_id: str,
        quant_signature: str,
    ) -> int:
        journeys = list(state.get("repair_journey") or [])
        count = 0
        for index, journey in enumerate(journeys, start=1):
            verifier_results = list(journey.get("verifier_results") or [])
            verified = _repair_is_verified(
                journey,
                state,
                quant_signature,
            )
            attempts = list(journey.get("attempts") or [])
            summary = ""
            for attempt in reversed(attempts):
                summary = str(attempt.get("tried") or attempt.get("summary") or "")
                if summary:
                    break
            payload = {
                "approach_summary": summary,
                "failure_evidence": dict(journey.get("failure_evidence") or {}),
                "changed_files": list(journey.get("changed_files") or []),
                "attempts": attempts,
                "verifier_results": verifier_results,
                "knowledge_ids": list(journey.get("knowledge_ids") or []),
                "target_role": str(journey.get("target_role") or ""),
            }
            failure_mode = {
                "benchmark_execution": "performance",
            }.get(
                str(journey.get("failure_class") or ""),
                str(journey.get("failure_class") or "load_run"),
            )
            context = {
                "framework": str(getattr(spec, "framework", "") or ""),
                "framework_version": str(getattr(spec, "framework_version", "") or ""),
                "arch_fingerprint": str(getattr(spec, "arch_fingerprint", "") or ""),
                "quant_signature": str(journey.get("quant_signature") or quant_signature or ""),
                "failure_mode": failure_mode,
                "error": str(journey.get("error") or ""),
            }
            self.store.record_repair_experience(
                record_id=f"{session_id}:repair:{index}",
                source_session_id=session_id,
                context_fingerprint=_fingerprint(context),
                framework=context["framework"],
                framework_version=context["framework_version"],
                arch_fingerprint=context["arch_fingerprint"],
                quant_signature=context["quant_signature"],
                failure_mode=failure_mode,
                error_signature=str(journey.get("error_signature") or ""),
                outcome="fixed" if verified else "failed",
                verification_status=("verified" if verified else "observed"),
                payload=payload,
            )
            count += 1
        return count

    def _capture_kernel_optimization(
        self,
        spec: Spec,
        state: SessionState,
        session_id: str,
        quant_signature: str,
    ) -> int:
        journeys = list(state.get("kernel_journey") or [])
        count = 0
        for index, journey in enumerate(journeys, start=1):
            kernel_signature = normalize_kernel_sig(str(journey.get("name") or "")) or str(
                journey.get("kernel_id") or ""
            )
            if not kernel_signature:
                continue
            e2e = dict(journey.get("e2e") or {})
            decision = str(e2e.get("decision") or "").upper()
            verified = bool(decision == "KEEP" and e2e.get("validated") is True)
            discovery = dict(journey.get("discovery") or {})
            source_mapping = dict(journey.get("source_mapping") or {})
            backend_attempts = list(journey.get("backend_attempts") or [])
            summary = ""
            for attempt in reversed(backend_attempts):
                summary = str(attempt.get("approach_summary") or attempt.get("summary") or "")
                if summary:
                    break
            payload = {
                "runtime_kernel_name": str(journey.get("name") or ""),
                "compiler": str(source_mapping.get("compiler") or ""),
                "source_symbol": str(source_mapping.get("source_symbol") or ""),
                "approach_summary": summary,
                "shape_cases": list(discovery.get("shape_cases") or []),
                "micro_speedup": journey.get("micro_speedup"),
                "e2e_gain": e2e.get("gain"),
                "accuracy_gap": e2e.get("accuracy_gap"),
                "reason": str(e2e.get("reason") or ""),
                "source_mapping": source_mapping,
                "framework_version": str(getattr(spec, "framework_version", "") or ""),
                "kernel_version": str(getattr(spec, "kernel_version", "") or ""),
                "gpu_type": str(getattr(spec, "gpu_type", "") or ""),
            }
            bound_type = str(discovery.get("roofline_bound") or "UNKNOWN")
            context = {
                "kernel_signature": kernel_signature,
                "bound_type": bound_type,
                "quant_signature": str(quant_signature),
                "payload": payload,
            }
            self.store.record_kernel_optimization_experience(
                record_id=f"{session_id}:kernel:{index}",
                source_session_id=session_id,
                context_fingerprint=_fingerprint(context),
                kernel_signature=kernel_signature,
                bound_type=bound_type,
                quant_signature=context["quant_signature"],
                outcome=(
                    "kept"
                    if verified
                    else "deferred"
                    if decision in {"NEEDS_REVIEW", "RETRYABLE_FAULT"}
                    else "rejected"
                ),
                verification_status=("verified" if verified else "observed"),
                payload=payload,
            )
            count += 1
        return count
