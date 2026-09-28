#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from ._serialization import sha256_json
from .candidates import Candidate
from .decision_space import DecisionSpace
from .errors import SchemaValidationError
from .quantizable_modules import ModuleStatus


def validate_assignment(decision_space: DecisionSpace, candidate: Candidate) -> str:
    """Validate exact coverage, forced-native rules, and unit capabilities."""
    assignment = {entry.module_name: entry.scheme for entry in candidate.assignment}
    expected_names = {module.name for module in decision_space.payload.modules}
    missing = sorted(expected_names - set(assignment))
    unknown = sorted(set(assignment) - expected_names)
    if missing or unknown:
        raise SchemaValidationError(f"Assignment coverage mismatch; missing={missing}, unknown={unknown}.")

    allowed_by_module = {
        member: set(unit.allowed_schemes) for unit in decision_space.payload.decision_units for member in unit.members
    }
    for unit in decision_space.payload.decision_units:
        schemes = {assignment[member] for member in unit.members}
        if len(schemes) != 1:
            raise SchemaValidationError(
                f"Decision unit {unit.unit_id!r} members must use one deployment-compatible scheme."
            )
    for module in decision_space.payload.modules:
        scheme = assignment[module.name]
        if module.status is ModuleStatus.SEARCHABLE:
            if scheme not in allowed_by_module[module.name]:
                raise SchemaValidationError(f"Module {module.name!r} uses unsupported scheme {scheme!r}.")
        elif scheme != "native":
            raise SchemaValidationError(f"Module {module.name!r} is {module.status.value} and must remain native.")

    assignment_hash = sha256_json([entry.to_dict() for entry in candidate.assignment])
    if candidate.candidate_id != assignment_hash:
        raise SchemaValidationError("candidate_id does not match the canonical assignment.")
    return assignment_hash


__all__ = ["validate_assignment"]
