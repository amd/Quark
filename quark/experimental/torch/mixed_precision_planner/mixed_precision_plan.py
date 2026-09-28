#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ._artifact import Artifact
from ._serialization import StrictSchema
from .candidates import Candidate, Candidates
from .decision_space import DecisionSpace
from .errors import SchemaValidationError
from .evaluator import EvaluatorDescriptor, RuntimeQConfigAudit
from .mixed_precision_strategy import MixedPrecisionStrategy
from .ppl import PplResult


class CandidateEvaluationStatus(StrEnum):
    SUCCEEDED = "succeeded"
    QUANTIZE_ERROR = "quantize_error"
    EVAL_ERROR = "eval_error"


@dataclass(frozen=True, slots=True)
class CandidateEvaluation(StrictSchema):
    candidate_id: str
    status: CandidateEvaluationStatus
    ppl: PplResult | None
    degradation: float | None
    qconfig_hash: str | None
    runtime_audit: RuntimeQConfigAudit | None
    error: str | None

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("sha256:"):
            raise SchemaValidationError("Candidate evaluation id must be a SHA-256 value.")
        if self.status is CandidateEvaluationStatus.SUCCEEDED:
            if (
                self.ppl is None
                or self.degradation is None
                or not math.isfinite(self.degradation)
                or self.qconfig_hash is None
                or self.runtime_audit is None
                or self.error is not None
            ):
                raise SchemaValidationError("Successful candidate evaluation is incomplete.")
            if not self.qconfig_hash.startswith("sha256:"):
                raise SchemaValidationError("Successful candidate evaluation must record a QConfig hash.")
        elif (
            self.ppl is not None
            or self.degradation is not None
            or self.qconfig_hash is not None
            or self.runtime_audit is not None
            or not self.error
        ):
            raise SchemaValidationError("Failed candidate evaluation must contain only an error.")


@dataclass(frozen=True, slots=True)
class MixedPrecisionPlanPayload(StrictSchema):
    candidates_id: str
    baseline: PplResult
    evaluations: tuple[CandidateEvaluation, ...]
    selected_candidate: Candidate
    quality_gate_max_degradation: float
    selection_policy: str
    evaluator: EvaluatorDescriptor
    calibration_token_hash: str
    ppl_token_hash: str
    model_state_fingerprint: str
    model_behavior_fingerprint: str

    def __post_init__(self) -> None:
        if not self.candidates_id.startswith("sha256:"):
            raise SchemaValidationError("Plan candidates_id must be a SHA-256 value.")
        if (
            not math.isfinite(self.quality_gate_max_degradation)
            or self.quality_gate_max_degradation < 0
            or self.selection_policy != "lowest_effective_bits"
        ):
            raise SchemaValidationError("Invalid Plan quality gate or selection policy.")
        if (
            not self.calibration_token_hash.startswith("sha256:")
            or not self.ppl_token_hash.startswith("sha256:")
            or not self.model_state_fingerprint.startswith("sha256:")
            or not self.model_behavior_fingerprint.startswith("sha256:")
        ):
            raise SchemaValidationError("Plan token hashes must be SHA-256 values.")
        ids = [evaluation.candidate_id for evaluation in self.evaluations]
        if len(ids) != len(set(ids)):
            raise SchemaValidationError("Plan contains duplicate candidate evaluations.")
        if any(
            evaluation.runtime_audit is not None and evaluation.runtime_audit.backend != self.evaluator.backend
            for evaluation in self.evaluations
        ):
            raise SchemaValidationError("Plan evaluation audit backend differs from its evaluator.")
        for evaluation in self.evaluations:
            if evaluation.status is not CandidateEvaluationStatus.SUCCEEDED:
                continue
            assert evaluation.ppl is not None
            assert evaluation.degradation is not None
            assert evaluation.runtime_audit is not None
            expected_degradation = evaluation.ppl.ppl / self.baseline.ppl - 1.0
            if evaluation.ppl.token_count != self.baseline.token_count or not math.isclose(
                evaluation.degradation,
                expected_degradation,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise SchemaValidationError("Plan evaluation PPL or degradation differs from its baseline.")
            if (
                evaluation.runtime_audit.assignment_hash != evaluation.candidate_id
                or evaluation.runtime_audit.calibration_token_hash != self.calibration_token_hash
            ):
                raise SchemaValidationError("Plan evaluation runtime audit differs from its recorded inputs.")
        selected = next(
            (
                evaluation
                for evaluation in self.evaluations
                if evaluation.candidate_id == self.selected_candidate.candidate_id
            ),
            None,
        )
        if selected is None or selected.status is not CandidateEvaluationStatus.SUCCEEDED:
            raise SchemaValidationError("Selected candidate must have a successful evaluation.")
        assert selected.degradation is not None
        if selected.degradation > self.quality_gate_max_degradation:
            raise SchemaValidationError("Selected candidate does not pass the Plan quality gate.")


@dataclass(frozen=True, slots=True)
class MixedPrecisionPlan:
    artifact: Artifact
    payload: MixedPrecisionPlanPayload

    def __post_init__(self) -> None:
        self.validate_integrity()

    def validate_integrity(self) -> None:
        self.artifact.validate_integrity()
        if self.artifact.artifact_type != "mixed_precision_plan" or self.artifact.payload != self.payload.to_dict():
            raise SchemaValidationError("Mixed Precision Plan payload does not match its Artifact envelope.")

    @property
    def artifact_id(self) -> str:
        return self.artifact.artifact_id

    @classmethod
    def create(
        cls,
        payload: MixedPrecisionPlanPayload,
        *,
        decision_space: DecisionSpace,
        candidates: Candidates,
        strategy: MixedPrecisionStrategy,
    ) -> MixedPrecisionPlan:
        artifact = Artifact.create(
            "mixed_precision_plan",
            payload.to_dict(),
            model_fingerprint=decision_space.model_fingerprint,
            inputs={
                "plan_selection": strategy.plan_selection,
                "evaluator": payload.evaluator,
                "calibration_token_hash": payload.calibration_token_hash,
                "ppl_token_hash": payload.ppl_token_hash,
            },
            upstream={
                "decision_space": decision_space.artifact_id,
                "candidates": candidates.artifact_id,
            },
        )
        return cls(artifact=artifact, payload=payload)

    @classmethod
    def load(cls, path: str | Path) -> MixedPrecisionPlan:
        artifact = Artifact.load(path, expected_type="mixed_precision_plan")
        return cls(artifact=artifact, payload=MixedPrecisionPlanPayload.from_dict(artifact.payload))

    def save(self, path: str | Path) -> None:
        self.validate_integrity()
        self.artifact.save(path)


__all__ = [
    "CandidateEvaluation",
    "CandidateEvaluationStatus",
    "MixedPrecisionPlan",
    "MixedPrecisionPlanPayload",
]
