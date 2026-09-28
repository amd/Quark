#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import importlib
import math
import os
import shutil
from dataclasses import replace
from types import ModuleType
from typing import Any

from ._serialization import sha256_json
from .candidates import AssignmentEntry, Candidate, Candidates, CandidatesPayload, SearchTerminationReason
from .decision_space import DecisionSpace
from .errors import (
    ArtifactCompatibilityError,
    InfeasibleBudgetError,
    SchemaValidationError,
    SearchError,
    SolverUnavailableError,
)
from .mixed_precision_strategy import MixedPrecisionStrategy
from .quantizable_modules import ModuleStatus
from .sensitivity_profile import SensitivityProfile, SensitivityRecord, _validate_profile_matrix

_LOSS_OBJECTIVE_SCALE = 10**15


def _load_pulp() -> ModuleType:
    try:
        return importlib.import_module("pulp")
    except ImportError as exc:
        raise SolverUnavailableError("PuLP and CBC are required for mixed-precision search.") from exc


def _cbc_path() -> str | None:
    system_path = shutil.which("cbc")
    if system_path is not None:
        return system_path
    try:
        coin_api = importlib.import_module("pulp.apis.coin_api")
    except ImportError:
        return None
    bundled_path = getattr(coin_api, "pulp_cbc_path", None)
    if isinstance(bundled_path, str) and os.path.isfile(bundled_path) and os.access(bundled_path, os.X_OK):
        return bundled_path
    return None


def _validate_inputs(decision_space: DecisionSpace, profile: SensitivityProfile) -> None:
    decision_space.validate_integrity()
    profile.validate_integrity()
    if profile.payload.decision_space_id != decision_space.artifact_id:
        raise ArtifactCompatibilityError("Sensitivity Profile does not belong to this Decision Space.")
    if profile.artifact.model_fingerprint != decision_space.model_fingerprint:
        raise ArtifactCompatibilityError("Sensitivity Profile model does not match Decision Space.")
    if profile.payload.total_quantizable_params != decision_space.payload.total_quantizable_params:
        raise ArtifactCompatibilityError("Sensitivity Profile parameter denominator does not match Decision Space.")
    if profile.artifact.upstream != {"decision_space": decision_space.artifact_id}:
        raise ArtifactCompatibilityError("Sensitivity Profile upstream does not match Decision Space.")
    if not decision_space.payload.decision_units:
        raise SchemaValidationError("Decision Space has no searchable units.")
    _validate_profile_matrix(profile.payload.records, decision_space)
    expected_fixed = {
        module.storage_id
        for module in decision_space.payload.modules
        if module.status not in (ModuleStatus.SEARCHABLE, ModuleStatus.TEMPLATE_EXCLUDED)
    }
    actual_fixed = {item.storage_id for item in profile.payload.fixed_native_costs}
    if actual_fixed != expected_fixed:
        raise ArtifactCompatibilityError("Sensitivity Profile fixed-native storage coverage is incomplete.")


def _record_map(profile: SensitivityProfile) -> dict[tuple[str, str], SensitivityRecord]:
    return {(record.unit_id, record.scheme): record for record in profile.payload.records}


def _diversity_unit_ids(
    decision_space: DecisionSpace,
    records: dict[tuple[str, str], SensitivityRecord],
    high_impact_fraction: float,
) -> tuple[str, ...]:
    """Select deterministic high-impact units by their available serialized-bit range."""
    impacts = []
    for unit in decision_space.payload.decision_units:
        costs = [records[(unit.unit_id, scheme)].cost.total_bits for scheme in unit.allowed_schemes]
        impacts.append((unit.unit_id, max(costs) - min(costs)))
    impacts.sort(key=lambda item: (-item[1], item[0]))
    cutoff = max(1, int(round(len(impacts) * high_impact_fraction)))
    return tuple(sorted(unit_id for unit_id, _impact in impacts[:cutoff]))


def _fixed_assignment(decision_space: DecisionSpace) -> dict[str, str]:
    return {
        module.name: "native"
        for module in decision_space.payload.modules
        if module.status is not ModuleStatus.SEARCHABLE
    }


def _make_candidate(
    selected: dict[str, str],
    *,
    decision_space: DecisionSpace,
    records: dict[tuple[str, str], SensitivityRecord],
    fixed_native_bits: int,
    rank: int,
) -> Candidate:
    assignment = _fixed_assignment(decision_space)
    units_by_id = {unit.unit_id: unit for unit in decision_space.payload.decision_units}
    for unit_id, scheme in selected.items():
        for member in units_by_id[unit_id].members:
            assignment[member] = scheme
    entries = tuple(
        AssignmentEntry(module_name=module_name, scheme=scheme) for module_name, scheme in sorted(assignment.items())
    )
    expected_modules = {module.name for module in decision_space.payload.modules}
    if set(assignment) != expected_modules:
        raise SchemaValidationError("Candidate assignment does not cover every Decision Space module.")

    predicted_loss = sum(records[(unit_id, scheme)].score.weighted_score for unit_id, scheme in selected.items())
    total_bits = fixed_native_bits + sum(
        records[(unit_id, scheme)].cost.total_bits for unit_id, scheme in selected.items()
    )
    effective_bits = total_bits / decision_space.payload.total_quantizable_params
    candidate_id = sha256_json([entry.to_dict() for entry in entries])
    return Candidate(
        candidate_id=candidate_id,
        rank=rank,
        predicted_loss=predicted_loss,
        total_bits=total_bits,
        effective_bits=effective_bits,
        assignment=entries,
    )


def _candidate_sort_key(candidate: Candidate) -> tuple[float, int, tuple[tuple[str, str], ...]]:
    return (
        candidate.predicted_loss,
        candidate.total_bits,
        tuple((entry.module_name, entry.scheme) for entry in candidate.assignment),
    )


def search_candidates(
    decision_space: DecisionSpace,
    profile: SensitivityProfile,
    strategy: MixedPrecisionStrategy,
    *,
    timeout_seconds: int = 60,
) -> Candidates:
    """Solve the fixed-budget one-hot problem with pairwise high-impact Top-K diversity."""
    _validate_inputs(decision_space, profile)
    decision_space.validate_compatibility(strategy)
    if timeout_seconds <= 0:
        raise SchemaValidationError("Solver timeout must be positive.")

    records = _record_map(profile)
    diversity_unit_ids = _diversity_unit_ids(
        decision_space,
        records,
        strategy.search.diversity.high_impact_fraction,
    )
    min_diversity_differences = strategy.search.diversity.min_high_impact_differences
    if min_diversity_differences > len(diversity_unit_ids):
        raise SchemaValidationError(
            f"min_high_impact_differences={min_diversity_differences} exceeds "
            f"the {len(diversity_unit_ids)} selected high-impact units."
        )
    diversity_unit_set = set(diversity_unit_ids)
    fixed_native_bits = sum(item.cost.total_bits for item in profile.payload.fixed_native_costs)
    minimum_feasible_bits = fixed_native_bits + sum(
        min(records[(unit.unit_id, scheme)].cost.total_bits for scheme in unit.allowed_schemes)
        for unit in decision_space.payload.decision_units
    )
    budget_limit_bits = strategy.search.budget.value * decision_space.payload.total_quantizable_params
    if budget_limit_bits + 1e-6 < minimum_feasible_bits:
        minimum_effective = minimum_feasible_bits / decision_space.payload.total_quantizable_params
        raise InfeasibleBudgetError(
            f"Budget {strategy.search.budget.value:.6g} effective bits is infeasible; "
            f"minimum is {minimum_effective:.6g}."
        )

    pulp = _load_pulp()
    cbc_path = _cbc_path()
    if cbc_path is None:
        raise SolverUnavailableError("PuLP is installed but the CBC executable is unavailable.")
    solver = pulp.COIN_CMD(
        path=cbc_path,
        msg=False,
        timeLimit=timeout_seconds,
        threads=1,
        options=[f"randomSeed {strategy.seed}", f"randomCbcSeed {strategy.seed}"],
    )
    if not solver.available():
        raise SolverUnavailableError("PuLP is installed but the CBC executable is unavailable.")

    candidates: list[Candidate] = []
    no_good_solutions: list[tuple[tuple[str, str], ...]] = []
    for _ in range(strategy.search.num_candidates):
        problem = pulp.LpProblem("quark_mixed_precision", pulp.LpMinimize)
        variables: dict[tuple[str, str], Any] = {}
        for unit_index, unit in enumerate(decision_space.payload.decision_units):
            unit_variables = []
            for scheme_index, scheme in enumerate(unit.allowed_schemes):
                variable = problem.add_variable(f"x_{unit_index}_{scheme_index}", lowBound=0, upBound=1, cat="Binary")
                variables[(unit.unit_id, scheme)] = variable
                unit_variables.append(variable)
            problem += pulp.lpSum(unit_variables) == 1

        loss_objective = pulp.lpSum(
            round(records[pair].score.weighted_score * _LOSS_OBJECTIVE_SCALE) * variable
            for pair, variable in variables.items()
        )
        storage_objective = fixed_native_bits + pulp.lpSum(
            records[pair].cost.total_bits * variable for pair, variable in variables.items()
        )
        problem.setObjective(loss_objective)
        problem += storage_objective <= budget_limit_bits
        for solution in no_good_solutions:
            problem += pulp.lpSum(variables[pair] for pair in solution) <= len(solution) - 1
            diversity_pairs = [pair for pair in solution if pair[0] in diversity_unit_set]
            problem += (
                pulp.lpSum(variables[pair] for pair in diversity_pairs)
                <= len(diversity_pairs) - min_diversity_differences
            )

        try:
            problem.solve(solver)
        except Exception as exc:
            raise SearchError(f"CBC failed: {exc}") from exc
        status = str(pulp.LpStatus.get(problem.status, problem.status))
        if status == "Infeasible":
            if not candidates:
                raise InfeasibleBudgetError("CBC found no assignment satisfying the requested budget.")
            break
        if status != "Optimal":
            raise SearchError(f"CBC did not finish optimally: {status}.")

        optimal_loss = round(float(pulp.value(loss_objective) or 0.0))
        loss_constraint = f"lex_loss_{len(candidates)}"
        problem += loss_objective <= optimal_loss + 0.5, loss_constraint
        problem.setObjective(storage_objective)
        try:
            problem.solve(solver)
        except Exception as exc:
            raise SearchError(f"CBC storage tie-break failed: {exc}") from exc
        storage_status = str(pulp.LpStatus.get(problem.status, problem.status))
        if storage_status != "Optimal":
            raise SearchError(f"CBC storage tie-break failed: {storage_status}.")

        # A fixed seed and one CBC thread make this storage-optimal solution reproducible. Avoid a third
        # exact canonical MILP here: it can time out on the 196 decision units in Qwen3-0.6B.
        selected: dict[str, str] = {}
        for unit in decision_space.payload.decision_units:
            chosen = [
                (scheme, variables[(unit.unit_id, scheme)])
                for scheme in unit.allowed_schemes
                if float(pulp.value(variables[(unit.unit_id, scheme)]) or 0.0) > 0.5
            ]
            if len(chosen) != 1:
                raise SearchError(f"CBC returned an invalid one-hot assignment for {unit.unit_id!r}.")
            scheme, _ = chosen[0]
            selected[unit.unit_id] = scheme

        candidate = _make_candidate(
            selected,
            decision_space=decision_space,
            records=records,
            fixed_native_bits=fixed_native_bits,
            rank=len(candidates) + 1,
        )
        if candidate.total_bits > budget_limit_bits + 1e-6:
            raise SearchError("CBC candidate failed exact post-solve budget verification.")
        candidates.append(candidate)
        no_good_solutions.append(tuple(selected.items()))

    ordered = sorted(candidates, key=_candidate_sort_key)
    ranked = tuple(replace(candidate, rank=index) for index, candidate in enumerate(ordered, start=1))
    complete = len(ranked) == strategy.search.num_candidates
    payload = CandidatesPayload(
        decision_space_id=decision_space.artifact_id,
        sensitivity_profile_id=profile.artifact_id,
        model_state_fingerprint=profile.payload.model_state_fingerprint,
        model_behavior_fingerprint=profile.payload.model_behavior_fingerprint,
        solver_name="pulp_cbc",
        solver_version=f"pulp-{getattr(pulp, '__version__', 'unknown')}",
        termination_reason=(
            SearchTerminationReason.REQUESTED_CANDIDATE_COUNT_REACHED
            if complete
            else SearchTerminationReason.SEARCH_SPACE_EXHAUSTED
        ),
        requested_candidates=strategy.search.num_candidates,
        budget_effective_bits=strategy.search.budget.value,
        budget_limit_bits=budget_limit_bits,
        total_quantizable_params=decision_space.payload.total_quantizable_params,
        minimum_feasible_bits=minimum_feasible_bits,
        fixed_native_bits=fixed_native_bits,
        diversity_unit_ids=diversity_unit_ids,
        min_diversity_differences=min_diversity_differences,
        candidates=ranked,
    )
    if not ranked:
        raise SearchError("CBC returned no candidates.")
    if any(not math.isfinite(candidate.predicted_loss) for candidate in ranked):
        raise SearchError("CBC returned a non-finite objective.")
    return Candidates.create(
        payload,
        decision_space=decision_space,
        sensitivity_profile=profile,
        strategy=strategy,
    )


__all__ = ["search_candidates"]
