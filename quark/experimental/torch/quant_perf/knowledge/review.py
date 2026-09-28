#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Human-reviewed promotion of runtime experience into curated knowledge."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from .provider import _StrictSafeLoader, _validate_record_data
from .store import ExperienceStore


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-") or "item"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:10]


class KnowledgeReviewService:
    def __init__(self, store: ExperienceStore) -> None:
        self.store = store

    def list_reviewable(
        self,
        *,
        session_id: str = "",
        domain: str = "",
        include_reviewed: bool = False,
    ) -> list[dict[str, Any]]:
        return self.store.list_reviewable_experience(
            session_id=session_id,
            domain=domain,
            include_reviewed=include_reviewed,
        )

    def build_candidate(self, experience_key: str) -> dict[str, Any]:
        row = self.store.get_experience(experience_key)
        domain = str(row["domain"])
        payload = dict(row.get("payload") or {})
        if domain == "repair":
            return self._repair_candidate(row, payload)
        if domain == "quantization":
            return self._quantization_candidate(row, payload)
        if domain == "kernel_optimization":
            return self._kernel_candidate(row, payload)
        raise ValueError(f"unsupported knowledge domain: {domain}")

    def _base_candidate(
        self,
        row: dict[str, Any],
        *,
        record_id: str,
        kind: str,
        summary: str,
        guidance: list[str],
        applicability: dict[str, Any],
        match: dict[str, Any],
        required_checks: list[str],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "id": record_id,
            "domain": row["domain"],
            "kind": kind,
            "status": "proposed",
            "applicability": applicability,
            "match": match,
            "summary": summary,
            "guidance": guidance,
            "required_checks": required_checks,
            "evidence": {
                "level": ("E2" if row.get("verification_status") == "verified" else "E1"),
                "verification_status": str(row.get("verification_status") or ""),
                "outcome": str(row.get("outcome") or ""),
            },
            "provenance": {
                "kind": "runtime_experience",
                "experience_key": str(row["experience_key"]),
                "source_session_id": str(row.get("source_session_id") or ""),
                "context_fingerprint": str(row.get("context_fingerprint") or ""),
            },
        }

    def _repair_candidate(
        self,
        row: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        framework = str(row.get("framework") or "framework")
        failure_mode = str(row.get("failure_mode") or "failure")
        summary = str(payload.get("approach_summary") or "A prior repair attempt produced reusable evidence.")
        return self._base_candidate(
            row,
            record_id=(
                f"repair.runtime.{_slug(framework)}.{_slug(failure_mode)}.{_short_hash(row['experience_key'])}.v1"
            ),
            kind=(
                "verified_repair_recipe"
                if row.get("outcome") == "fixed" and row.get("verification_status") == "verified"
                else "diagnostic_playbook"
            ),
            summary=summary,
            guidance=[summary],
            applicability={
                "framework": [framework],
            },
            match={
                "failure_class": [failure_mode],
                "error_signature": ([str(row["error_signature"])] if row.get("error_signature") else []),
            },
            required_checks=(["load", "inference"] if failure_mode == "load_run" else ["accuracy"]),
        )

    def _quantization_candidate(
        self,
        row: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        candidate = payload.get("candidate") or {}
        outcome = str(row.get("outcome") or "observed")
        return self._base_candidate(
            row,
            record_id=(
                f"quantization.runtime."
                f"{_slug(str(row.get('model_arch') or 'model'))}."
                f"{_short_hash(row['experience_key'])}.v1"
            ),
            kind=("quantization_rule" if outcome == "passed" else "known_issue"),
            summary=(f"Real exported evaluation {outcome} for candidate {candidate}."),
            guidance=[
                "Keep this observation stack-specific unless independent evidence supports broader applicability.",
                "Never skip current real post-export evaluation.",
            ],
            applicability={
                "model_arch": [str(row.get("model_arch") or "")],
                "framework": [str(row.get("framework") or "")],
                "gpu_type": [str(row.get("gpu_type") or "")],
            },
            match={"keywords": [str(candidate)]},
            required_checks=["post_export_accuracy"],
        )

    def _kernel_candidate(
        self,
        row: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        signature = str(row.get("kernel_signature") or "kernel")
        outcome = str(row.get("outcome") or "observed")
        return self._base_candidate(
            row,
            record_id=(f"kernel.runtime.{_slug(signature)}.{_short_hash(row['experience_key'])}.v1"),
            kind=("kernel_principle" if outcome == "kept" else "known_issue"),
            summary=str(payload.get("approach_summary") or f"Kernel optimization outcome: {outcome}."),
            guidance=[
                "Reproduce the result under the current shape and stack.",
                "Require standalone correctness, accuracy, and E2E retention.",
            ],
            applicability={},
            match={
                "keywords": [
                    signature,
                    str(row.get("bound_type") or ""),
                ]
            },
            required_checks=[
                "standalone_correctness",
                "accuracy",
                "e2e",
            ],
        )

    def approve(
        self,
        *,
        candidate_path: str | Path,
        repo_root: str | Path,
    ) -> Path:
        candidate_file = Path(candidate_path)
        raw = yaml.load(
            candidate_file.read_text(encoding="utf-8"),
            Loader=_StrictSafeLoader,
        )
        if not isinstance(raw, dict) or raw.get("status") != "proposed":
            raise ValueError("knowledge approval requires status=proposed")
        evidence = dict(raw.get("evidence") or {})
        raw["status"] = (
            "verified"
            if raw.get("kind") == "verified_repair_recipe" and evidence.get("verification_status") == "verified"
            else "validated"
        )
        _validate_record_data(raw, path=candidate_file)
        root = Path(repo_root).resolve()
        records_root = root / "quark" / "experimental" / "torch" / "quant_perf" / "knowledge" / "records"
        if not records_root.is_dir():
            raise ValueError(f"Quark Quant-Perf knowledge records directory is missing: {records_root}")
        target = records_root / str(raw["domain"]) / f"{str(raw['id']).replace('/', '-')}.yaml"
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(raw, sort_keys=False),
            encoding="utf-8",
        )
        experience_key = str((raw.get("provenance") or {}).get("experience_key") or "")
        if experience_key:
            self.store.record_review_decision(
                experience_key=experience_key,
                domain=str(raw["domain"]),
                decision="approved",
            )
        return target

    def reject(self, experience_key: str, *, reason: str) -> None:
        row = self.store.get_experience(experience_key)
        self.store.record_review_decision(
            experience_key=experience_key,
            domain=str(row["domain"]),
            decision="rejected",
            reason=reason,
        )
