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
from .decision_space import DecisionSpace
from .errors import SchemaValidationError
from .mixed_precision_strategy import MixedPrecisionStrategy
from .sensitivity_profile import SensitivityProfile


class SearchTerminationReason(StrEnum):
    REQUESTED_CANDIDATE_COUNT_REACHED = "requested_candidate_count_reached"
    SEARCH_SPACE_EXHAUSTED = "search_space_exhausted"


@dataclass(frozen=True, slots=True)
class AssignmentEntry(StrictSchema):
    module_name: str
    scheme: str

    def __post_init__(self) -> None:
        if not self.module_name or not self.scheme:
            raise SchemaValidationError("Assignment module_name and scheme must not be empty.")


@dataclass(frozen=True, slots=True)
class Candidate(StrictSchema):
    candidate_id: str
    rank: int
    predicted_loss: float
    total_bits: int
    effective_bits: float
    assignment: tuple[AssignmentEntry, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("sha256:"):
            raise SchemaValidationError("candidate_id must be a SHA-256 value.")
        if self.rank <= 0:
            raise SchemaValidationError("Candidate rank must be positive.")
        if not math.isfinite(self.predicted_loss) or self.predicted_loss < 0:
            raise SchemaValidationError("Candidate predicted_loss must be finite and non-negative.")
        if self.total_bits <= 0:
            raise SchemaValidationError("Candidate total_bits must be positive.")
        if not math.isfinite(self.effective_bits) or self.effective_bits <= 0:
            raise SchemaValidationError("Candidate effective_bits must be finite and positive.")
        names = [entry.module_name for entry in self.assignment]
        if names != sorted(names) or len(names) != len(set(names)):
            raise SchemaValidationError("Candidate assignment must contain unique module names in sorted order.")


@dataclass(frozen=True, slots=True)
class CandidatesPayload(StrictSchema):
    decision_space_id: str
    sensitivity_profile_id: str
    model_state_fingerprint: str
    model_behavior_fingerprint: str
    solver_name: str
    solver_version: str
    termination_reason: SearchTerminationReason
    requested_candidates: int
    budget_effective_bits: float
    budget_limit_bits: float
    total_quantizable_params: int
    minimum_feasible_bits: int
    fixed_native_bits: int
    diversity_unit_ids: tuple[str, ...]
    min_diversity_differences: int
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        if (
            not self.decision_space_id.startswith("sha256:")
            or not self.sensitivity_profile_id.startswith("sha256:")
            or not self.model_state_fingerprint.startswith("sha256:")
            or not self.model_behavior_fingerprint.startswith("sha256:")
        ):
            raise SchemaValidationError("Candidate input ids must be SHA-256 values.")
        if self.solver_name != "pulp_cbc" or not self.solver_version:
            raise SchemaValidationError("MVP candidates must record PuLP/CBC.")
        if self.termination_reason not in (
            SearchTerminationReason.REQUESTED_CANDIDATE_COUNT_REACHED,
            SearchTerminationReason.SEARCH_SPACE_EXHAUSTED,
        ):
            raise SchemaValidationError("Invalid search termination reason.")
        if self.requested_candidates <= 0 or len(self.candidates) > self.requested_candidates:
            raise SchemaValidationError("Invalid requested candidate count.")
        if (
            not math.isfinite(self.budget_effective_bits)
            or self.budget_effective_bits <= 0
            or not math.isfinite(self.budget_limit_bits)
            or self.budget_limit_bits <= 0
            or self.total_quantizable_params <= 0
            or self.minimum_feasible_bits <= 0
            or self.fixed_native_bits < 0
        ):
            raise SchemaValidationError("Invalid candidate budget accounting.")
        if self.diversity_unit_ids != tuple(sorted(set(self.diversity_unit_ids))):
            raise SchemaValidationError("Candidate diversity unit ids must be sorted and unique.")
        if not 1 <= self.min_diversity_differences <= len(self.diversity_unit_ids):
            raise SchemaValidationError("Invalid minimum Candidate diversity difference count.")
        if (
            self.termination_reason is SearchTerminationReason.REQUESTED_CANDIDATE_COUNT_REACHED
            and len(self.candidates) != self.requested_candidates
        ):
            raise SchemaValidationError("Search did not return the requested candidate count.")
        if self.termination_reason is SearchTerminationReason.SEARCH_SPACE_EXHAUSTED and (
            not self.candidates or len(self.candidates) >= self.requested_candidates
        ):
            raise SchemaValidationError("Exhausted search must return fewer candidates than requested.")
        if [candidate.rank for candidate in self.candidates] != list(range(1, len(self.candidates) + 1)):
            raise SchemaValidationError("Candidate ranks must be contiguous and start at one.")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise SchemaValidationError("Candidate ids must be unique.")
        for candidate in self.candidates:
            if candidate.total_bits > self.budget_limit_bits + 1e-6:
                raise SchemaValidationError("Candidate exceeds the requested budget.")
            expected = candidate.total_bits / self.total_quantizable_params
            if not math.isclose(candidate.effective_bits, expected, rel_tol=0.0, abs_tol=1e-12):
                raise SchemaValidationError("Candidate effective_bits does not match total_bits.")


@dataclass(frozen=True, slots=True)
class Candidates:
    artifact: Artifact
    payload: CandidatesPayload

    def __post_init__(self) -> None:
        self.validate_integrity()

    def validate_integrity(self) -> None:
        self.artifact.validate_integrity()
        if self.artifact.artifact_type != "candidates" or self.artifact.payload != self.payload.to_dict():
            raise SchemaValidationError("Candidates payload does not match its Artifact envelope.")

    @property
    def artifact_id(self) -> str:
        return self.artifact.artifact_id

    @classmethod
    def create(
        cls,
        payload: CandidatesPayload,
        *,
        decision_space: DecisionSpace,
        sensitivity_profile: SensitivityProfile,
        strategy: MixedPrecisionStrategy,
    ) -> Candidates:
        artifact = Artifact.create(
            "candidates",
            payload.to_dict(),
            model_fingerprint=decision_space.model_fingerprint,
            inputs={"search": strategy.search, "seed": strategy.seed},
            upstream={
                "decision_space": decision_space.artifact_id,
                "sensitivity_profile": sensitivity_profile.artifact_id,
            },
        )
        return cls(artifact=artifact, payload=payload)

    @classmethod
    def load(cls, path: str | Path) -> Candidates:
        artifact = Artifact.load(path, expected_type="candidates")
        return cls(artifact=artifact, payload=CandidatesPayload.from_dict(artifact.payload))

    def save(self, path: str | Path) -> None:
        self.validate_integrity()
        self.artifact.save(path)


__all__ = [
    "AssignmentEntry",
    "Candidate",
    "Candidates",
    "CandidatesPayload",
    "SearchTerminationReason",
]
