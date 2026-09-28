#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .types import KnowledgeContext, KnowledgeMatch, KnowledgeRecord

if TYPE_CHECKING:
    from .store import ExperienceStore


def _slug(value: str) -> str:
    """Normalize a value for use in a knowledge record ID.

    :param value: Source value.
    :return: Filesystem-safe identifier component.
    """
    return re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-") or "unknown"


def _exact(record: KnowledgeRecord, score: float = 100.0) -> KnowledgeMatch:
    """Wrap a record as an exact knowledge match.

    :param record: Knowledge record to wrap.
    :param score: Applicability score.
    :return: Exact knowledge match.
    """
    return KnowledgeMatch(
        item=record,
        match_type="exact",
        applicability_score=score,
    )


class ExperienceKnowledgeProvider:
    """Adapt verified runtime experience into advisory knowledge records."""

    def __init__(self, store: ExperienceStore) -> None:
        """Initialize the provider.

        :param store: Runtime experience store.
        """
        self.store = store

    def query(self, context: KnowledgeContext) -> list[KnowledgeMatch]:
        """Return experiential knowledge matching the requested domain.

        :param context: Knowledge query context.
        :return: Matching experiential knowledge.
        """
        if context.domain == "repair":
            return self._repair(context)
        if context.domain == "quantization":
            return self._quantization(context)
        if context.domain == "kernel_optimization":
            return self._kernel_optimization(context)
        return []

    def _repair(self, context: KnowledgeContext) -> list[KnowledgeMatch]:
        """Return prior repair guidance matching the current failure.

        :param context: Repair query context.
        :return: Matching repair guidance.
        """
        failure_mode = {
            "benchmark_execution": "performance",
        }.get(
            context.failure_class,
            context.failure_class
            or {
                "accuracy": "accuracy_gap",
                "benchmark": "performance",
            }.get(context.stage, "load_run"),
        )
        row = self.store.find_repair_guidance(
            framework=context.framework,
            framework_version=context.framework_version,
            arch_fingerprint=context.arch_fingerprint,
            quant_signature=context.quant_signature,
            failure_mode=failure_mode,
            error_signature=(context.error_signature if failure_mode == "load_run" else ""),
        )
        if not row:
            return []
        payload = dict(row.get("payload") or {})
        attempts = payload.get("attempts") or []
        related = row.get("match_type") == "related"
        guidance = []
        for attempt in attempts:
            tried = str(attempt.get("tried") or "")
            failed = str(attempt.get("failed_because") or "")
            if tried and failed and not related:
                guidance.append(f"Avoid repeating {tried}: {failed}")
        if row.get("outcome") == "fixed":
            summary = str(payload.get("approach_summary") or "A matching repair previously passed verification.")
        else:
            summary = str(payload.get("approach_summary") or "Prior repair attempts did not resolve this failure.")
        if related:
            summary = "Similar failure, not an exact match: " + summary
            guidance.extend(
                (
                    f"Previous failure: {row['error_signature']}",
                    f"Current failure: {context.error_signature}",
                    "Compare dimensions and configuration before reusing this approach; historical rejection does not rule it out here.",
                )
            )
        record = KnowledgeRecord(
            id=(f"experience.repair.{_slug(str(row['record_id']))}"),
            domain="repair",
            kind=("validated_playbook" if row.get("verification_status") == "verified" else "diagnostic_playbook"),
            status="validated",
            applicability={},
            match={},
            summary=summary,
            guidance=tuple(guidance),
            required_checks=(("load", "inference") if failure_mode == "load_run" else ("accuracy",)),
            evidence={
                "level": ("E2" if row.get("verification_status") == "verified" else "E1"),
                "outcome": str(row.get("outcome") or ""),
            },
            provenance={
                "kind": "runtime_experience",
                "source_session_id": str(row.get("source_session_id") or ""),
            },
        )
        return [KnowledgeMatch(record, "related", 50.0)] if related else [_exact(record)]

    def _quantization(
        self,
        context: KnowledgeContext,
    ) -> list[KnowledgeMatch]:
        """Return prior quantization outcomes for the current model and stack."""
        if not context.model_arch:
            return []
        rows = self.store.find_quantization_experience(
            model_arch=context.model_arch,
            framework=context.framework,
            gpu_type=context.gpu_type,
        )
        matches = []
        for row in rows:
            payload = dict(row.get("payload") or {})
            candidate = payload.get("candidate") or {}
            gap = payload.get("real_accuracy_gap")
            summary = f"Real exported evaluation {row.get('outcome', 'observed')} for candidate {candidate}"
            if isinstance(gap, int | float):
                summary += f" with accuracy gap {float(gap):.4f}."
            record = KnowledgeRecord(
                id=f"experience.quantization.{_slug(row['record_id'])}",
                domain="quantization",
                kind="reference",
                status="validated",
                applicability={},
                match={},
                summary=summary,
                guidance=("Use this observation for explanation or tie-breaking; never skip current real validation.",),
                required_checks=("post_export_accuracy",),
                evidence={
                    "level": ("E2" if row.get("verification_status") == "verified" else "E1"),
                    "outcome": str(row.get("outcome") or ""),
                },
                provenance={
                    "kind": "runtime_experience",
                    "source_session_id": str(row.get("source_session_id") or ""),
                },
            )
            matches.append(_exact(record, 50.0))
        return matches

    def _kernel_optimization(
        self,
        context: KnowledgeContext,
    ) -> list[KnowledgeMatch]:
        """Return prior kernel optimization outcomes for the current kernel."""
        from quark.experimental.torch.quant_perf.perfopt.keep import normalize_kernel_sig

        kernel = context.kernel_context
        signature = normalize_kernel_sig(str(kernel.get("op_name") or ""))
        bound_type = str(kernel.get("bound_type") or "")
        if not signature or not bound_type:
            return []
        rows = self.store.find_kernel_optimization_experience(
            kernel_signature=signature,
            bound_type=bound_type,
            quant_signature=context.quant_signature,
        )
        matches = []
        for row in rows:
            payload = dict(row.get("payload") or {})
            summary = str(payload.get("approach_summary") or f"Prior kernel candidate outcome: {row.get('outcome')}")
            record = KnowledgeRecord(
                id=f"experience.kernel.{_slug(row['record_id'])}",
                domain="kernel_optimization",
                kind="reference",
                status="validated",
                applicability={},
                match={},
                summary=summary,
                guidance=("Reproduce under the current shape and stack before keeping or rejecting a candidate.",),
                required_checks=(
                    "standalone_correctness",
                    "accuracy",
                    "e2e",
                ),
                evidence={
                    "level": ("E2" if row.get("verification_status") == "verified" else "E1"),
                    "outcome": str(row.get("outcome") or ""),
                },
                provenance={
                    "kind": "runtime_experience",
                    "source_session_id": str(row.get("source_session_id") or ""),
                },
            )
            matches.append(_exact(record, 50.0))
        return matches
