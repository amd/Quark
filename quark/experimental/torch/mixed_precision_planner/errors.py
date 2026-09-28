#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


class MixedPrecisionError(Exception):
    """Base error for Torch mixed-precision planning."""


class SchemaValidationError(MixedPrecisionError, ValueError):
    """Raised when schema, assignment, artifact, or QConfig validation fails."""


class ArtifactCompatibilityError(MixedPrecisionError):
    """Raised when an artifact cannot be reused."""


class SearchError(MixedPrecisionError):
    """Raised when mixed-precision search fails or times out."""


class InfeasibleBudgetError(SearchError):
    """Raised when no assignment can satisfy the requested budget."""


class SolverUnavailableError(SearchError):
    """Raised when PuLP or CBC is unavailable."""


class ModelSourceError(MixedPrecisionError):
    """Raised when a checkpoint source cannot be resolved to an immutable snapshot."""


class PlanSelectionError(MixedPrecisionError):
    """Raised when candidate evaluation or selection cannot produce a plan."""


class EvaluatorUnavailableError(PlanSelectionError):
    """Raised when the selected evaluation backend cannot be constructed."""


class EvaluatorStateError(PlanSelectionError):
    """Raised when an evaluator can no longer guarantee isolated runtime state."""


class CandidateQuantizationError(PlanSelectionError):
    """Raised when an evaluator cannot prepare a candidate for evaluation."""


class CandidateEvaluationError(PlanSelectionError):
    """Raised when an evaluator cannot measure a prepared candidate."""


class NoCandidatePassedError(PlanSelectionError):
    """Raised when no candidate passes the configured quality gate."""


__all__ = [
    "ArtifactCompatibilityError",
    "CandidateEvaluationError",
    "CandidateQuantizationError",
    "EvaluatorStateError",
    "EvaluatorUnavailableError",
    "InfeasibleBudgetError",
    "MixedPrecisionError",
    "ModelSourceError",
    "NoCandidatePassedError",
    "PlanSelectionError",
    "SchemaValidationError",
    "SearchError",
    "SolverUnavailableError",
]
