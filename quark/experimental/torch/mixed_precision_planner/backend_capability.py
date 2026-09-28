#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from dataclasses import dataclass, replace

from .errors import SchemaValidationError
from .quantizable_modules import ModuleStatus, QuantizableModule

_VLLM_FUSIONS = {
    "q_proj": ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    "k_proj": ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    "v_proj": ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    "gate_proj": ("gate_up_proj", ("gate_proj", "up_proj")),
    "up_proj": ("gate_up_proj", ("gate_proj", "up_proj")),
}


@dataclass(frozen=True, slots=True)
class DeploymentGroup:
    unit_id: str
    members: tuple[str, ...]


def _group_id(name: str, backend: str) -> tuple[str, tuple[str, ...] | None]:
    if backend == "no_fusion":
        return name, None
    if backend != "vllm":
        raise SchemaValidationError(f"Unsupported deployment backend: {backend!r}.")

    parent, separator, leaf = name.rpartition(".")
    fusion = _VLLM_FUSIONS.get(leaf)
    if fusion is None:
        return name, None
    fused_leaf, expected_leaves = fusion
    prefix = f"{parent}." if separator else ""
    return prefix + fused_leaf, tuple(prefix + expected for expected in expected_leaves)


def build_deployment_groups(
    modules: tuple[QuantizableModule, ...],
    *,
    backend: str,
) -> tuple[DeploymentGroup, ...]:
    """Group canonical HF modules according to deployment fusion constraints."""
    names = {module.name for module in modules}
    grouped: dict[str, list[str]] = {}
    expected_by_target: dict[str, tuple[str, ...]] = {}
    for module in modules:
        unit_id, expected = _group_id(module.name, backend)
        grouped.setdefault(unit_id, []).append(module.name)
        if expected is not None:
            expected_by_target[unit_id] = expected

    incomplete = {
        target: sorted(set(expected) - names)
        for target, expected in expected_by_target.items()
        if set(expected) - names
    }
    if incomplete:
        raise SchemaValidationError(f"Fusion constraint groups are incomplete: {incomplete}.")

    return tuple(
        DeploymentGroup(unit_id=target, members=tuple(sorted(members))) for target, members in sorted(grouped.items())
    )


def apply_deployment_constraints(
    modules: tuple[QuantizableModule, ...],
    *,
    backend: str,
) -> tuple[tuple[QuantizableModule, ...], tuple[DeploymentGroup, ...]]:
    """Propagate fixed-native status across every deployment fusion group."""
    groups = build_deployment_groups(modules, backend=backend)
    by_name = {module.name: module for module in modules}
    forced_members = {
        member
        for group in groups
        if any(by_name[member].status is not ModuleStatus.SEARCHABLE for member in group.members)
        for member in group.members
    }
    resolved = tuple(
        replace(module, status=ModuleStatus.FORCED_NATIVE_DEPLOYMENT)
        if module.name in forced_members and module.status is ModuleStatus.SEARCHABLE
        else module
        for module in modules
    )
    return resolved, groups


__all__ = [
    "DeploymentGroup",
    "apply_deployment_constraints",
    "build_deployment_groups",
]
