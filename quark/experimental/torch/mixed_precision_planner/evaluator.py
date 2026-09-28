#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from torch import nn

from quark.torch.quantization.config.config import QConfig

from ._serialization import StrictSchema, sha256_json
from .candidates import Candidate
from .data import TokenDataset
from .decision_space import DecisionSpace
from .errors import SchemaValidationError
from .ppl import PplResult

ModelFactory = Callable[[], nn.Module]


@dataclass(frozen=True, slots=True)
class EvaluatorDescriptor(StrictSchema):
    """Stable identity for the runtime used to compare candidate quality."""

    backend: str
    protocol: str
    implementation_version: int
    runtime_version: str
    options: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.backend or not self.protocol or not self.runtime_version:
            raise SchemaValidationError("Evaluator descriptor strings must not be empty.")
        if self.implementation_version <= 0:
            raise SchemaValidationError("Evaluator implementation_version must be positive.")
        sha256_json(self.options)

    @property
    def descriptor_hash(self) -> str:
        return sha256_json(self)


@dataclass(frozen=True, slots=True)
class RuntimeQConfigAudit(StrictSchema):
    """Backend-neutral evidence that a candidate was applied to a runtime."""

    backend: str
    assignment_hash: str
    runtime_qconfig_hash: str
    resolved_quantized: tuple[str, ...]
    resolved_native: tuple[str, ...]
    calibration_token_hash: str
    calibrated_sequences: int
    calibrated_tokens: int

    def __post_init__(self) -> None:
        if not self.backend:
            raise SchemaValidationError("Runtime audit backend must not be empty.")
        for value in (self.assignment_hash, self.runtime_qconfig_hash, self.calibration_token_hash):
            if not value.startswith("sha256:"):
                raise SchemaValidationError("Runtime audit hashes must be SHA-256 values.")
        if len(set(self.resolved_quantized)) != len(self.resolved_quantized):
            raise SchemaValidationError("Runtime audit contains duplicate quantized targets.")
        if len(set(self.resolved_native)) != len(self.resolved_native):
            raise SchemaValidationError("Runtime audit contains duplicate native targets.")
        if set(self.resolved_quantized) & set(self.resolved_native):
            raise SchemaValidationError("Runtime target cannot be both quantized and native.")
        if self.calibrated_sequences < 0 or self.calibrated_tokens < 0:
            raise SchemaValidationError("Runtime calibration counts must be non-negative.")

    @property
    def audit_hash(self) -> str:
        return sha256_json(self)


@dataclass(frozen=True, slots=True)
class EvaluatorCandidateResult:
    ppl: PplResult
    qconfig_hash: str
    runtime_audit: RuntimeQConfigAudit


class CandidateEvaluator(Protocol):
    """Runtime boundary used by plan selection."""

    @property
    def descriptor(self) -> EvaluatorDescriptor: ...

    def evaluate_baseline(self, ppl_tokens: TokenDataset) -> PplResult: ...

    def evaluate_candidate(
        self,
        decision_space: DecisionSpace,
        candidate: Candidate,
        qconfig: QConfig,
        calibration_tokens: TokenDataset,
        ppl_tokens: TokenDataset,
    ) -> EvaluatorCandidateResult: ...

    def close(self) -> None: ...


__all__ = [
    "CandidateEvaluator",
    "EvaluatorCandidateResult",
    "EvaluatorDescriptor",
    "ModelFactory",
    "RuntimeQConfigAudit",
]
