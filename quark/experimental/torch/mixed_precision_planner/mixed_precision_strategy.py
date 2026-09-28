#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ._serialization import StrictSchema, atomic_write_json, load_json_object
from .errors import SchemaValidationError


class HardwareTarget(StrEnum):
    MI355 = "mi355"


class SearchGranularity(StrEnum):
    FINE = "fine"


class SensitivityMethod(StrEnum):
    WEIGHT_MSE = "weight_mse"


class SearchAlgorithm(StrEnum):
    LINEAR_PROGRAMMING = "linear_programming"


class DeploymentBackendName(StrEnum):
    NO_FUSION = "no_fusion"
    VLLM = "vllm"


class EvaluatorBackend(StrEnum):
    HF = "hf"
    VLLM = "vllm"


@dataclass(frozen=True, slots=True)
class HardwareConfig(StrictSchema):
    target: HardwareTarget


@dataclass(frozen=True, slots=True)
class DeploymentBackendConfig(StrictSchema):
    name: DeploymentBackendName


@dataclass(frozen=True, slots=True)
class DecisionSpaceConfig(StrictSchema):
    exclude_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not pattern for pattern in self.exclude_patterns):
            raise SchemaValidationError("exclude_patterns must not contain empty values.")
        if len(set(self.exclude_patterns)) != len(self.exclude_patterns):
            raise SchemaValidationError("exclude_patterns must not contain duplicates.")


@dataclass(frozen=True, slots=True)
class SensitivityProfileConfig(StrictSchema):
    methods: tuple[SensitivityMethod, ...]

    def __post_init__(self) -> None:
        if self.methods != (SensitivityMethod.WEIGHT_MSE,):
            raise SchemaValidationError("MVP sensitivity method must be ['weight_mse'].")


@dataclass(frozen=True, slots=True)
class CalibrationConfig(StrictSchema):
    dataset: str
    num_samples: int
    max_length: int
    batch_size: int
    revision: str | None = None

    def __post_init__(self) -> None:
        if self.dataset != "pileval":
            raise SchemaValidationError("MVP calibration dataset must be 'pileval'.")
        if self.revision == "":
            raise SchemaValidationError("Calibration revision must not be empty.")
        if not 1 <= self.num_samples <= 128:
            raise SchemaValidationError("Calibration num_samples must be between 1 and 128.")
        if not 1 <= self.max_length <= 2048:
            raise SchemaValidationError("Calibration max_length must be between 1 and 2048.")
        if self.batch_size != 1:
            raise SchemaValidationError("MVP calibration batch_size must be 1.")


@dataclass(frozen=True, slots=True)
class BudgetConfig(StrictSchema):
    metric: str
    value: float
    scope: str

    def __post_init__(self) -> None:
        if self.metric != "effective_bits" or self.scope != "quantizable":
            raise SchemaValidationError("MVP budget must use quantizable effective_bits.")
        if not math.isfinite(self.value) or self.value <= 0:
            raise SchemaValidationError("Budget value must be finite and positive.")


@dataclass(frozen=True, slots=True)
class DiversityConfig(StrictSchema):
    high_impact_fraction: float = 0.3
    min_high_impact_differences: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.high_impact_fraction) or not 0.0 < self.high_impact_fraction <= 1.0:
            raise SchemaValidationError("high_impact_fraction must be finite and in (0, 1].")
        if self.min_high_impact_differences <= 0:
            raise SchemaValidationError("min_high_impact_differences must be positive.")


@dataclass(frozen=True, slots=True)
class SearchConfig(StrictSchema):
    granularity: SearchGranularity
    algorithm: SearchAlgorithm
    budget: BudgetConfig
    num_candidates: int
    diversity: DiversityConfig = field(default_factory=DiversityConfig)

    def __post_init__(self) -> None:
        if not 1 <= self.num_candidates <= 20:
            raise SchemaValidationError("num_candidates must be between 1 and 20.")


@dataclass(frozen=True, slots=True)
class QualityGate(StrictSchema):
    metric: str
    dataset: str
    max_degradation: float
    num_chunks: int
    max_length: int
    revision: str | None = None

    def __post_init__(self) -> None:
        if self.metric != "ppl" or self.dataset != "wikitext2":
            raise SchemaValidationError("MVP quality gate must use Wikitext-2 PPL.")
        if self.revision == "":
            raise SchemaValidationError("Quality-gate revision must not be empty.")
        if not math.isfinite(self.max_degradation) or self.max_degradation < 0:
            raise SchemaValidationError("max_degradation must be finite and non-negative.")
        if not 1 <= self.num_chunks <= 32:
            raise SchemaValidationError("PPL num_chunks must be between 1 and 32.")
        if self.max_length != 2048:
            raise SchemaValidationError("MVP PPL max_length must be 2048.")


@dataclass(frozen=True, slots=True)
class PlanSelectionConfig(StrictSchema):
    quality_gate: QualityGate
    selection_policy: str
    evaluator: EvaluatorConfig

    def __post_init__(self) -> None:
        if self.selection_policy != "lowest_effective_bits":
            raise SchemaValidationError("MVP selection_policy must be 'lowest_effective_bits'.")


@dataclass(frozen=True, slots=True)
class EvaluatorConfig(StrictSchema):
    backend: EvaluatorBackend
    protocol: str

    def __post_init__(self) -> None:
        if self.protocol != "token_ppl_v1":
            raise SchemaValidationError("MVP evaluator protocol must be 'token_ppl_v1'.")


@dataclass(frozen=True, slots=True)
class OutputConfig(StrictSchema):
    dir: str

    def __post_init__(self) -> None:
        if not self.dir:
            raise SchemaValidationError("output.dir must not be empty.")


@dataclass(frozen=True, slots=True)
class MixedPrecisionStrategy(StrictSchema):
    schema_version: int
    hardware: HardwareConfig
    deployment_backend: DeploymentBackendConfig
    decision_space: DecisionSpaceConfig
    sensitivity_profile: SensitivityProfileConfig
    calibration: CalibrationConfig
    search: SearchConfig
    plan_selection: PlanSelectionConfig
    output: OutputConfig
    seed: int

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise SchemaValidationError(f"Unsupported strategy schema version: {self.schema_version}.")
        if isinstance(self.seed, bool) or not 1 <= self.seed <= 2_147_483_647:
            raise SchemaValidationError("seed must be an integer between 1 and 2147483647.")

    @classmethod
    def load(cls, path: str | Path) -> MixedPrecisionStrategy:
        return cls.from_dict(load_json_object(path))

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


__all__ = ["MixedPrecisionStrategy"]
