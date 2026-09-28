#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import copy
import glob
from dataclasses import dataclass

from torch import nn

from quark.torch.quantization.config.config import QConfig, QLayerConfig
from quark.torch.quantization.model_transformation import setup_config_per_layer
from quark.torch.utils.llm.model_preparation import preprocess_for_quantization

from ._serialization import StrictSchema, sha256_json
from .backend_capability import apply_deployment_constraints
from .candidates import Candidate
from .decision_space import DecisionSpace
from .errors import SchemaValidationError
from .hardware_capability import get_scheme_config
from .plan_checks import validate_assignment
from .quantizable_modules import enumerate_linear_modules

_DTYPE_NAMES = {"torch.float16": "float16", "torch.bfloat16": "bfloat16"}


@dataclass(frozen=True, slots=True)
class QConfigAudit(StrictSchema):
    assignment_hash: str
    qconfig_hash: str
    resolved_quantized: tuple[str, ...]
    resolved_native: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.assignment_hash.startswith("sha256:") or not self.qconfig_hash.startswith("sha256:"):
            raise SchemaValidationError("QConfig audit hashes must be SHA-256 values.")


def build_qconfig(decision_space: DecisionSpace, candidate: Candidate) -> QConfig:
    """Compile an exact module assignment into a Quark QConfig."""
    decision_space.validate_integrity()
    validate_assignment(decision_space, candidate)
    assignment = {entry.module_name: entry.scheme for entry in candidate.assignment}
    layer_quant_config = {
        module_name: get_scheme_config(decision_space.payload.model_type, scheme)
        for module_name, scheme in sorted(assignment.items())
        if scheme != "native"
    }
    excludes = sorted(module_name for module_name, scheme in assignment.items() if scheme == "native")
    return QConfig(
        global_quant_config=QLayerConfig(),
        layer_quant_config=layer_quant_config,
        layer_type_quant_config={},
        exclude=excludes,
        kv_cache_quant_config={},
        kv_cache_group=[],
        shared_scale_groups=[],
    )


def qconfig_semantic_dict(qconfig: QConfig) -> dict[str, object]:
    value = qconfig.to_dict()
    excludes = set(value["exclude"])
    value["exclude"] = sorted(
        pattern for pattern in excludes if not (pattern.endswith(".*") and pattern[:-2] in excludes)
    )
    return value


def qconfig_semantic_hash(qconfig: QConfig) -> str:
    return sha256_json(qconfig_semantic_dict(qconfig))


def _validate_model_binding(model: nn.Module, decision_space: DecisionSpace) -> dict[str, nn.Module]:
    preprocess_for_quantization(model)
    named_modules = dict(model.named_modules(remove_duplicate=False))
    parameter_devices = {str(parameter.device) for parameter in model.parameters()}
    if parameter_devices != {decision_space.payload.device}:
        raise SchemaValidationError(
            f"Model device placement changed; expected {decision_space.payload.device!r}, "
            f"got {sorted(parameter_devices)}."
        )
    inventory = enumerate_linear_modules(
        model,
        template_excludes=decision_space.payload.template_excludes,
        user_excludes=tuple(decision_space.payload.matched_user_patterns),
    )
    resolved_modules, _ = apply_deployment_constraints(
        inventory.modules,
        backend=decision_space.payload.deployment_backend,
    )
    if resolved_modules != decision_space.payload.modules:
        raise SchemaValidationError("Model module topology or storage aliases changed from Decision Space.")
    linear_names = {name for name, module in named_modules.items() if isinstance(module, nn.Linear)}
    expected_names = {module.name for module in decision_space.payload.modules}
    if linear_names != expected_names:
        raise SchemaValidationError(
            f"Model Linear inventory changed; missing={sorted(expected_names - linear_names)}, "
            f"unknown={sorted(linear_names - expected_names)}."
        )
    for module_info in decision_space.payload.modules:
        module = named_modules[module_info.name]
        assert isinstance(module, nn.Linear)
        dtype_name = _DTYPE_NAMES.get(str(module.weight.dtype))
        if tuple(module.weight.shape) != module_info.weight_shape or dtype_name != module_info.source_dtype:
            raise SchemaValidationError(f"Model binding changed for {module_info.name!r}.")
    return named_modules


def audit_qconfig(
    model: nn.Module,
    decision_space: DecisionSpace,
    candidate: Candidate,
    qconfig: QConfig,
) -> QConfigAudit:
    """Resolve QConfig through Quark and prove it matches the assignment exactly."""
    decision_space.validate_integrity()
    assignment_hash = validate_assignment(decision_space, candidate)
    if qconfig.global_quant_config != QLayerConfig():
        raise SchemaValidationError("Mixed-precision QConfig global_quant_config must be empty.")
    if qconfig.layer_type_quant_config or qconfig.kv_cache_quant_config:
        raise SchemaValidationError("Mixed-precision QConfig must not use layer-type or KV-cache fallback.")
    if any(glob.has_magic(name) for name in qconfig.layer_quant_config):
        raise SchemaValidationError("Mixed-precision layer_quant_config keys must be exact names.")

    named_modules = _validate_model_binding(model, decision_space)
    resolved_config = copy.deepcopy(qconfig)
    module_configs: dict[str, object] = {}
    setup_config_per_layer(resolved_config, named_modules, module_configs)

    assignment = {entry.module_name: entry.scheme for entry in candidate.assignment}
    resolved_quantized: list[str] = []
    resolved_native: list[str] = []
    global_fallback: list[str] = []
    for module_name, scheme in sorted(assignment.items()):
        if scheme == "native":
            if module_name in module_configs or module_name not in resolved_config.exclude:
                raise SchemaValidationError(f"Native module {module_name!r} was not excluded exactly.")
            resolved_native.append(module_name)
            continue

        actual = module_configs.get(module_name)
        expected = get_scheme_config(decision_space.payload.model_type, scheme)
        if actual is None:
            raise SchemaValidationError(f"Quantized module {module_name!r} has no resolved QLayerConfig.")
        if actual == resolved_config.global_quant_config:
            global_fallback.append(module_name)
        if actual != expected:
            raise SchemaValidationError(f"Resolved QLayerConfig for {module_name!r} does not match {scheme!r}.")
        resolved_quantized.append(module_name)

    if global_fallback:
        raise SchemaValidationError(f"Modules depend on global fallback: {global_fallback}.")
    return QConfigAudit(
        assignment_hash=assignment_hash,
        qconfig_hash=qconfig_semantic_hash(qconfig),
        resolved_quantized=tuple(resolved_quantized),
        resolved_native=tuple(resolved_native),
    )


def compile_qconfig(
    model: nn.Module,
    decision_space: DecisionSpace,
    candidate: Candidate,
) -> tuple[QConfig, QConfigAudit]:
    qconfig = build_qconfig(decision_space, candidate)
    return qconfig, audit_qconfig(model, decision_space, candidate, qconfig)


__all__ = [
    "QConfigAudit",
    "audit_qconfig",
    "build_qconfig",
    "compile_qconfig",
    "qconfig_semantic_dict",
    "qconfig_semantic_hash",
]
