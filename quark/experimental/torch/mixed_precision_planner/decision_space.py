#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch import nn

from quark.torch.quantization.config.template import LLMTemplate
from quark.torch.utils.llm.model_preparation import preprocess_for_quantization

from ._artifact import Artifact
from ._serialization import StrictSchema
from .backend_capability import apply_deployment_constraints
from .errors import SchemaValidationError
from .hardware_capability import get_supported_schemes
from .mixed_precision_strategy import MixedPrecisionStrategy
from .quantizable_modules import (
    ModuleStatus,
    QuantizableModule,
    enumerate_linear_modules,
    model_structure_fingerprint,
)


@dataclass(frozen=True, slots=True)
class DecisionUnit(StrictSchema):
    unit_id: str
    members: tuple[str, ...]
    allowed_schemes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.unit_id or not self.members:
            raise SchemaValidationError("Decision unit identity and members must not be empty.")
        if self.members != tuple(sorted(set(self.members))):
            raise SchemaValidationError("Decision unit members must be sorted and unique.")
        if not self.allowed_schemes or self.allowed_schemes[0] != "native":
            raise SchemaValidationError("Every decision unit must include native as its first scheme.")


@dataclass(frozen=True, slots=True)
class DecisionSpacePayload(StrictSchema):
    model_type: str
    deployment_backend: str
    device: str
    modules: tuple[QuantizableModule, ...]
    decision_units: tuple[DecisionUnit, ...]
    supported_schemes: tuple[str, ...]
    template_excludes: tuple[str, ...]
    matched_user_patterns: dict[str, tuple[str, ...]]
    total_quantizable_params: int

    def __post_init__(self) -> None:
        if not self.model_type or not self.device:
            raise SchemaValidationError("Decision Space model type and device must not be empty.")
        if self.deployment_backend not in {"no_fusion", "vllm"}:
            raise SchemaValidationError(f"Unsupported Decision Space deployment backend: {self.deployment_backend!r}.")
        names = [module.name for module in self.modules]
        if names != sorted(set(names)):
            raise SchemaValidationError("Decision Space module names must be sorted and unique.")
        unit_ids = [unit.unit_id for unit in self.decision_units]
        if unit_ids != sorted(set(unit_ids)):
            raise SchemaValidationError("Decision unit ids must be sorted and unique.")
        searchable = {module.name for module in self.modules if module.status is ModuleStatus.SEARCHABLE}
        unit_members = {member for unit in self.decision_units for member in unit.members}
        if searchable != unit_members:
            raise SchemaValidationError("Every searchable module must belong to exactly one decision unit.")
        if len(unit_members) != sum(len(unit.members) for unit in self.decision_units):
            raise SchemaValidationError("A module belongs to more than one decision unit.")
        if any(unit.allowed_schemes != self.supported_schemes for unit in self.decision_units):
            raise SchemaValidationError("Every MVP decision unit must expose the same MI355 scheme set.")
        counted_storage: set[str] = set()
        expected_params = 0
        for module in self.modules:
            if module.status is ModuleStatus.TEMPLATE_EXCLUDED or module.storage_id in counted_storage:
                continue
            counted_storage.add(module.storage_id)
            expected_params += module.num_params
        if self.total_quantizable_params != expected_params or expected_params <= 0:
            raise SchemaValidationError("Decision Space total_quantizable_params does not match its module inventory.")


@dataclass(frozen=True, slots=True)
class DecisionSpace:
    artifact: Artifact
    payload: DecisionSpacePayload

    def __post_init__(self) -> None:
        self.validate_integrity()

    def validate_integrity(self) -> None:
        self.artifact.validate_integrity()
        if self.artifact.artifact_type != "decision_space" or self.artifact.payload != self.payload.to_dict():
            raise SchemaValidationError("Decision Space payload does not match its Artifact envelope.")

    @property
    def artifact_id(self) -> str:
        return self.artifact.artifact_id

    @property
    def model_fingerprint(self) -> str:
        assert self.artifact.model_fingerprint is not None
        return self.artifact.model_fingerprint

    @classmethod
    def create(
        cls,
        payload: DecisionSpacePayload,
        *,
        model_fingerprint: str,
        strategy: MixedPrecisionStrategy,
    ) -> DecisionSpace:
        artifact = Artifact.create(
            "decision_space",
            payload.to_dict(),
            model_fingerprint=model_fingerprint,
            inputs=_strategy_inputs(strategy),
        )
        return cls(artifact=artifact, payload=payload)

    @classmethod
    def load(cls, path: str | Path) -> DecisionSpace:
        artifact = Artifact.load(path, expected_type="decision_space")
        return cls(artifact=artifact, payload=DecisionSpacePayload.from_dict(artifact.payload))

    def save(self, path: str | Path) -> None:
        self.validate_integrity()
        self.artifact.save(path)

    def validate_compatibility(self, strategy: MixedPrecisionStrategy) -> None:
        self.validate_integrity()
        self.artifact.validate_compatibility(
            model_fingerprint=self.artifact.model_fingerprint,
            inputs=_strategy_inputs(strategy),
            upstream={},
        )


def _strategy_inputs(strategy: MixedPrecisionStrategy) -> dict[str, object]:
    return {
        "hardware": strategy.hardware,
        "deployment_backend": strategy.deployment_backend,
        "decision_space": strategy.decision_space,
        "granularity": strategy.search.granularity,
    }


def _model_type(model: nn.Module) -> str:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if not isinstance(model_type, str):
        raise SchemaValidationError("The model must expose config.model_type.")
    return model_type


def _total_quantizable_params(modules: tuple[QuantizableModule, ...]) -> int:
    counted_storage: set[str] = set()
    total = 0
    for module in modules:
        if module.status is ModuleStatus.TEMPLATE_EXCLUDED or module.storage_id in counted_storage:
            continue
        counted_storage.add(module.storage_id)
        total += module.num_params
    return total


def build_decision_space(model: nn.Module, strategy: MixedPrecisionStrategy) -> DecisionSpace:
    """Build a Decision Space using deployment constraints only."""
    model_type = _model_type(model)
    schemes = get_supported_schemes(strategy.hardware.target, model_type)

    preprocess_for_quantization(model)
    template = LLMTemplate.get(model_type)
    template_excludes = tuple(template.exclude_layers_name)
    inventory = enumerate_linear_modules(
        model,
        template_excludes=template_excludes,
        user_excludes=strategy.decision_space.exclude_patterns,
    )
    modules, deployment_groups = apply_deployment_constraints(
        inventory.modules,
        backend=strategy.deployment_backend.name.value,
    )

    devices = {
        str(module.weight.device)
        for _, module in model.named_modules(remove_duplicate=False)
        if isinstance(module, nn.Linear)
    }
    if len(devices) != 1:
        raise SchemaValidationError(f"MVP requires all Linear weights on one device, got {sorted(devices)}.")

    units = tuple(
        DecisionUnit(
            unit_id=group.unit_id,
            members=group.members,
            allowed_schemes=schemes,
        )
        for group in deployment_groups
        if all(
            next(module for module in modules if module.name == member).status is ModuleStatus.SEARCHABLE
            for member in group.members
        )
    )
    payload = DecisionSpacePayload(
        model_type=model_type,
        deployment_backend=strategy.deployment_backend.name.value,
        device=next(iter(devices)),
        modules=modules,
        decision_units=units,
        supported_schemes=schemes,
        template_excludes=template_excludes,
        matched_user_patterns=inventory.matched_user_patterns,
        total_quantizable_params=_total_quantizable_params(modules),
    )
    return DecisionSpace.create(
        payload,
        model_fingerprint=model_structure_fingerprint(model_type, modules),
        strategy=strategy,
    )


__all__ = [
    "DecisionSpace",
    "DecisionSpacePayload",
    "DecisionUnit",
    "build_decision_space",
]
