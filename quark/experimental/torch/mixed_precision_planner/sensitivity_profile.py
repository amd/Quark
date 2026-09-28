#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from quark.torch.quantization.config.template import LLMTemplate
from quark.torch.utils.llm.model_preparation import preprocess_for_quantization

from ._artifact import Artifact
from ._serialization import StrictSchema, sha256_json
from .backend_capability import apply_deployment_constraints
from .cost_accounting import WeightStorageCost, calculate_weight_storage
from .decision_space import DecisionSpace
from .errors import ArtifactCompatibilityError, SchemaValidationError
from .mixed_precision_strategy import MixedPrecisionStrategy
from .quantizable_modules import (
    ModuleStatus,
    QuantizableModule,
    enumerate_linear_modules,
    model_structure_fingerprint,
)
from .weight_mse import WeightMseScore, calculate_weight_mse


@dataclass(frozen=True, slots=True)
class SensitivityRecord(StrictSchema):
    unit_id: str
    scheme: str
    method: str
    method_version: int
    score: WeightMseScore
    cost: WeightStorageCost

    def __post_init__(self) -> None:
        if not self.unit_id or not self.scheme:
            raise SchemaValidationError("Sensitivity record unit_id and scheme must not be empty.")
        if self.method != "weight_mse" or self.method_version != 1:
            raise SchemaValidationError("MVP sensitivity records must use weight_mse version 1.")


@dataclass(frozen=True, slots=True)
class FixedNativeCost(StrictSchema):
    storage_id: str
    module_names: tuple[str, ...]
    cost: WeightStorageCost

    def __post_init__(self) -> None:
        if not self.storage_id or not self.module_names:
            raise SchemaValidationError("Fixed-native storage and module names must not be empty.")
        if self.module_names != tuple(sorted(set(self.module_names))):
            raise SchemaValidationError("Fixed-native module names must be sorted and unique.")


@dataclass(frozen=True, slots=True)
class SensitivityProfilePayload(StrictSchema):
    decision_space_id: str
    model_state_fingerprint: str
    model_behavior_fingerprint: str
    records: tuple[SensitivityRecord, ...]
    fixed_native_costs: tuple[FixedNativeCost, ...]
    total_quantizable_params: int

    def __post_init__(self) -> None:
        if (
            not self.decision_space_id.startswith("sha256:")
            or not self.model_state_fingerprint.startswith("sha256:")
            or not self.model_behavior_fingerprint.startswith("sha256:")
        ):
            raise SchemaValidationError("Profile input ids must be SHA-256 values.")
        pairs = [(record.unit_id, record.scheme) for record in self.records]
        if len(pairs) != len(set(pairs)):
            raise SchemaValidationError("Sensitivity Profile contains duplicate unit-scheme records.")
        fixed_storage_ids = [item.storage_id for item in self.fixed_native_costs]
        if len(fixed_storage_ids) != len(set(fixed_storage_ids)):
            raise SchemaValidationError("Sensitivity Profile contains duplicate fixed-native storage costs.")
        if self.total_quantizable_params <= 0:
            raise SchemaValidationError("Profile total_quantizable_params must be positive.")


@dataclass(frozen=True, slots=True)
class SensitivityProfile:
    artifact: Artifact
    payload: SensitivityProfilePayload

    def __post_init__(self) -> None:
        self.validate_integrity()

    def validate_integrity(self) -> None:
        self.artifact.validate_integrity()
        if self.artifact.artifact_type != "sensitivity_profile" or self.artifact.payload != self.payload.to_dict():
            raise SchemaValidationError("Sensitivity Profile payload does not match its Artifact envelope.")

    @property
    def artifact_id(self) -> str:
        return self.artifact.artifact_id

    @classmethod
    def create(
        cls,
        payload: SensitivityProfilePayload,
        *,
        decision_space: DecisionSpace,
        strategy: MixedPrecisionStrategy,
    ) -> SensitivityProfile:
        artifact = Artifact.create(
            "sensitivity_profile",
            payload.to_dict(),
            model_fingerprint=decision_space.model_fingerprint,
            inputs={
                "sensitivity_profile": strategy.sensitivity_profile,
                "model_state_fingerprint": payload.model_state_fingerprint,
                "model_behavior_fingerprint": payload.model_behavior_fingerprint,
            },
            upstream={"decision_space": decision_space.artifact_id},
        )
        return cls(artifact=artifact, payload=payload)

    @classmethod
    def load(cls, path: str | Path) -> SensitivityProfile:
        artifact = Artifact.load(path, expected_type="sensitivity_profile")
        return cls(artifact=artifact, payload=SensitivityProfilePayload.from_dict(artifact.payload))

    def save(self, path: str | Path) -> None:
        self.validate_integrity()
        self.artifact.save(path)


def model_state_fingerprint(model: nn.Module) -> str:
    """Hash the complete model state used for profiling and candidate evaluation."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise SchemaValidationError(f"Model state entry {name!r} is not a tensor.")
        value = tensor.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def _normalize_config_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _normalize_config_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_config_value(item) for item in value]
    return value


def model_behavior_fingerprint(model: nn.Module) -> str:
    """Hash the model and config classes plus behavior-relevant configuration."""
    config = getattr(model, "config", None)
    if config is None:
        raise SchemaValidationError("The model must expose config for behavior fingerprinting.")
    to_dict = getattr(config, "to_dict", None)
    config_payload = to_dict() if callable(to_dict) else vars(config)
    if not isinstance(config_payload, Mapping):
        raise SchemaValidationError("Model config serialization must return a mapping.")
    ignored_fields = {"_commit_hash", "_name_or_path", "name_or_path", "transformers_version"}
    normalized_config = {
        str(key): _normalize_config_value(value) for key, value in config_payload.items() if key not in ignored_fields
    }
    return sha256_json(
        {
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "config_class": f"{type(config).__module__}.{type(config).__qualname__}",
            "config": normalized_config,
        }
    )


def _fixed_native_costs(
    modules: tuple[QuantizableModule, ...],
    *,
    model_type: str,
) -> tuple[FixedNativeCost, ...]:
    grouped: dict[str, list[QuantizableModule]] = {}
    for module in modules:
        if module.status in (ModuleStatus.SEARCHABLE, ModuleStatus.TEMPLATE_EXCLUDED):
            continue
        grouped.setdefault(module.storage_id, []).append(module)

    result: list[FixedNativeCost] = []
    for storage_id, members in sorted(grouped.items()):
        representative = members[0]
        result.append(
            FixedNativeCost(
                storage_id=storage_id,
                module_names=tuple(sorted(module.name for module in members)),
                cost=calculate_weight_storage(
                    representative.weight_shape,
                    representative.source_dtype,
                    "native",
                    model_type=model_type,
                ),
            )
        )
    return tuple(result)


def _aggregate_score(scores: list[WeightMseScore]) -> WeightMseScore:
    squared_error = sum(score.squared_error for score in scores)
    signal_power = sum(score.signal_power for score in scores)
    relative_mse = squared_error / signal_power if signal_power else 0.0
    return WeightMseScore(
        squared_error=squared_error,
        signal_power=signal_power,
        relative_mse=relative_mse,
        weighted_score=sum(score.weighted_score for score in scores),
    )


def _aggregate_cost(costs: list[WeightStorageCost]) -> WeightStorageCost:
    weight_bits = sum(cost.weight_bits for cost in costs)
    metadata_bits = sum(cost.metadata_bits for cost in costs)
    num_params = sum(cost.num_params for cost in costs)
    total_bits = weight_bits + metadata_bits
    return WeightStorageCost(
        weight_bits=weight_bits,
        metadata_bits=metadata_bits,
        total_bits=total_bits,
        num_params=num_params,
        effective_bits=total_bits / num_params,
    )


def _fused_unit_weight(unit_members: tuple[str, ...], module_map: dict[str, nn.Linear]) -> torch.Tensor:
    weights = [module_map[member].weight for member in unit_members]
    input_features = {weight.shape[1] for weight in weights}
    dtypes = {weight.dtype for weight in weights}
    if len(input_features) != 1 or len(dtypes) != 1:
        raise SchemaValidationError(f"Fused Decision Unit members have incompatible weights: {unit_members}.")
    return torch.cat([weight.detach() for weight in weights], dim=0)


def _validate_profile_matrix(records: tuple[SensitivityRecord, ...], decision_space: DecisionSpace) -> None:
    expected = {
        (unit.unit_id, scheme) for unit in decision_space.payload.decision_units for scheme in unit.allowed_schemes
    }
    actual = {(record.unit_id, record.scheme) for record in records}
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise SchemaValidationError(f"Incomplete sensitivity matrix; missing={missing}, unknown={unknown}.")


def build_sensitivity_profile(
    model: nn.Module,
    decision_space: DecisionSpace,
    strategy: MixedPrecisionStrategy,
) -> SensitivityProfile:
    """Measure Weight-MSE and serialized weight cost for every unit-scheme pair."""
    decision_space.validate_compatibility(strategy)
    preprocess_for_quantization(model)
    template = LLMTemplate.get(decision_space.payload.model_type)
    inventory = enumerate_linear_modules(
        model,
        template_excludes=tuple(template.exclude_layers_name),
        user_excludes=strategy.decision_space.exclude_patterns,
    )
    resolved_modules, _ = apply_deployment_constraints(
        inventory.modules,
        backend=strategy.deployment_backend.name.value,
    )
    current_fingerprint = model_structure_fingerprint(decision_space.payload.model_type, inventory.modules)
    if current_fingerprint != decision_space.model_fingerprint or resolved_modules != decision_space.payload.modules:
        raise ArtifactCompatibilityError("Model structure does not match Decision Space.")

    module_map = {
        name: module for name, module in model.named_modules(remove_duplicate=False) if isinstance(module, nn.Linear)
    }
    records: list[SensitivityRecord] = []
    modules_by_name = {module.name: module for module in inventory.modules}
    for unit in decision_space.payload.decision_units:
        for scheme in unit.allowed_schemes:
            if decision_space.payload.deployment_backend == "vllm" and len(unit.members) > 1:
                fused_weight = _fused_unit_weight(unit.members, module_map)
                score = calculate_weight_mse(
                    fused_weight,
                    scheme,
                    total_quantizable_params=decision_space.payload.total_quantizable_params,
                    model_type=decision_space.payload.model_type,
                )
                cost = calculate_weight_storage(
                    tuple(fused_weight.shape),
                    modules_by_name[unit.members[0]].source_dtype,
                    scheme,
                    model_type=decision_space.payload.model_type,
                )
            else:
                scores = [
                    calculate_weight_mse(
                        module_map[member].weight,
                        scheme,
                        total_quantizable_params=decision_space.payload.total_quantizable_params,
                        model_type=decision_space.payload.model_type,
                    )
                    for member in unit.members
                ]
                costs = [
                    calculate_weight_storage(
                        modules_by_name[member].weight_shape,
                        modules_by_name[member].source_dtype,
                        scheme,
                        model_type=decision_space.payload.model_type,
                    )
                    for member in unit.members
                ]
                score = _aggregate_score(scores)
                cost = _aggregate_cost(costs)
            records.append(
                SensitivityRecord(
                    unit_id=unit.unit_id,
                    scheme=scheme,
                    method="weight_mse",
                    method_version=1,
                    score=score,
                    cost=cost,
                )
            )

    resolved_records = tuple(records)
    _validate_profile_matrix(resolved_records, decision_space)
    payload = SensitivityProfilePayload(
        decision_space_id=decision_space.artifact_id,
        model_state_fingerprint=model_state_fingerprint(model),
        model_behavior_fingerprint=model_behavior_fingerprint(model),
        records=resolved_records,
        fixed_native_costs=_fixed_native_costs(
            resolved_modules,
            model_type=decision_space.payload.model_type,
        ),
        total_quantizable_params=decision_space.payload.total_quantizable_params,
    )
    return SensitivityProfile.create(payload, decision_space=decision_space, strategy=strategy)


__all__ = [
    "FixedNativeCost",
    "SensitivityProfile",
    "SensitivityProfilePayload",
    "SensitivityRecord",
    "build_sensitivity_profile",
    "model_behavior_fingerprint",
    "model_state_fingerprint",
]
