#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pulp
import pytest
import torch
from torch import nn

import quark.experimental.torch.mixed_precision_planner.linear_programming as linear_programming
from quark.experimental.torch.mixed_precision_planner import (
    Candidates,
    MixedPrecisionStrategy,
    SensitivityProfile,
)
from quark.experimental.torch.mixed_precision_planner._artifact import Artifact
from quark.experimental.torch.mixed_precision_planner.candidates import (
    AssignmentEntry,
    SearchTerminationReason,
)
from quark.experimental.torch.mixed_precision_planner.cost_accounting import calculate_weight_storage
from quark.experimental.torch.mixed_precision_planner.decision_space import build_decision_space
from quark.experimental.torch.mixed_precision_planner.errors import (
    ArtifactCompatibilityError,
    InfeasibleBudgetError,
    SchemaValidationError,
    SearchError,
    SolverUnavailableError,
)
from quark.experimental.torch.mixed_precision_planner.linear_programming import search_candidates
from quark.experimental.torch.mixed_precision_planner.mixed_precision_strategy import DiversityConfig
from quark.experimental.torch.mixed_precision_planner.plan_checks import validate_assignment
from quark.experimental.torch.mixed_precision_planner.quantizable_modules import ModuleStatus
from quark.experimental.torch.mixed_precision_planner.sensitivity_profile import build_sensitivity_profile
from quark.experimental.torch.mixed_precision_planner.weight_mse import calculate_weight_mse


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)


class TinyQwen3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer()])
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.to(dtype=torch.bfloat16)


class Qwen3LikeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.k_proj = nn.Linear(8, 4, bias=False)
        self.self_attn.v_proj = nn.Linear(8, 4, bias=False)
        self.self_attn.o_proj = nn.Linear(8, 8, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(8, 16, bias=False)
        self.mlp.up_proj = nn.Linear(8, 16, bias=False)
        self.mlp.down_proj = nn.Linear(16, 8, bias=False)


class Qwen3LikeModel(nn.Module):
    def __init__(self, *, num_layers: int = 28) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(Qwen3LikeLayer() for _ in range(num_layers))
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.to(dtype=torch.bfloat16)


def make_strategy(
    *,
    budget: float = 12.0,
    num_candidates: int = 3,
    deployment_backend: str = "no_fusion",
    evaluator_backend: str = "hf",
    exclude_patterns: list[str] | None = None,
) -> MixedPrecisionStrategy:
    return MixedPrecisionStrategy.from_dict(
        {
            "schema_version": 1,
            "hardware": {"target": "mi355"},
            "deployment_backend": {"name": deployment_backend},
            "decision_space": {"exclude_patterns": exclude_patterns or []},
            "sensitivity_profile": {"methods": ["weight_mse"]},
            "calibration": {
                "dataset": "pileval",
                "num_samples": 16,
                "max_length": 512,
                "batch_size": 1,
            },
            "search": {
                "granularity": "fine",
                "algorithm": "linear_programming",
                "budget": {"metric": "effective_bits", "value": budget, "scope": "quantizable"},
                "num_candidates": num_candidates,
            },
            "plan_selection": {
                "evaluator": {"backend": evaluator_backend, "protocol": "token_ppl_v1"},
                "quality_gate": {
                    "metric": "ppl",
                    "dataset": "wikitext2",
                    "max_degradation": 0.02,
                    "num_chunks": 1,
                    "max_length": 2048,
                },
                "selection_policy": "lowest_effective_bits",
            },
            "output": {"dir": "./run"},
            "seed": 42,
        }
    )


def build_inputs() -> tuple[object, object]:
    torch.manual_seed(5)
    model = TinyQwen3()
    strategy = make_strategy()
    space = build_decision_space(model, strategy)
    profile = build_sensitivity_profile(model, space, strategy)
    return space, profile


def test_evaluator_choice_does_not_change_search_artifacts() -> None:
    torch.manual_seed(7)
    model = TinyQwen3()
    hf_strategy = make_strategy(num_candidates=1, evaluator_backend="hf")
    vllm_strategy = make_strategy(num_candidates=1, evaluator_backend="vllm")

    hf_space = build_decision_space(model, hf_strategy)
    vllm_space = build_decision_space(model, vllm_strategy)
    assert hf_space.payload == vllm_space.payload
    assert hf_space.artifact_id == vllm_space.artifact_id

    hf_profile = build_sensitivity_profile(model, hf_space, hf_strategy)
    vllm_profile = build_sensitivity_profile(model, vllm_space, vllm_strategy)
    assert hf_profile.payload == vllm_profile.payload
    assert hf_profile.artifact_id == vllm_profile.artifact_id
    hf_candidates = search_candidates(hf_space, hf_profile, hf_strategy)
    vllm_candidates = search_candidates(
        vllm_space,
        vllm_profile,
        vllm_strategy,
    )
    assert hf_candidates.payload == vllm_candidates.payload
    assert hf_candidates.artifact_id == vllm_candidates.artifact_id


def test_pulp_top_k_search_is_complete_deterministic_and_round_trips(tmp_path: Path) -> None:
    space, profile = build_inputs()
    strategy = make_strategy(budget=16.0, num_candidates=3)
    first = search_candidates(space, profile, strategy)
    second = search_candidates(space, profile, strategy)

    assert first.payload.termination_reason is SearchTerminationReason.REQUESTED_CANDIDATE_COUNT_REACHED
    assert len(first.payload.candidates) == 3
    assert [candidate.candidate_id for candidate in first.payload.candidates] == [
        candidate.candidate_id for candidate in second.payload.candidates
    ]
    assert all(candidate.effective_bits <= 16.0 for candidate in first.payload.candidates)
    assert all(
        [entry.module_name for entry in candidate.assignment]
        == ["lm_head", "model.layers.0.k_proj", "model.layers.0.q_proj"]
        for candidate in first.payload.candidates
    )
    assert first.payload.diversity_unit_ids == ("model.layers.0.k_proj",)
    assert first.payload.min_diversity_differences == 1
    skeleton_schemes = {
        next(entry.scheme for entry in candidate.assignment if entry.module_name == "model.layers.0.k_proj")
        for candidate in first.payload.candidates
    }
    assert skeleton_schemes == {"native", "fp8", "ptpc_fp8"}

    path = tmp_path / "candidates.json"
    first.save(path)
    assert Candidates.load(path) == first


def test_search_rejects_impossible_high_impact_diversity() -> None:
    space, profile = build_inputs()
    strategy = make_strategy(num_candidates=2)
    strategy = replace(
        strategy,
        search=replace(
            strategy.search,
            diversity=DiversityConfig(high_impact_fraction=0.3, min_high_impact_differences=2),
        ),
    )
    with pytest.raises(SchemaValidationError, match="min_high_impact_differences"):
        search_candidates(space, profile, strategy)


def test_top_k_enforces_pairwise_high_impact_differences() -> None:
    space, profile = build_inputs()
    strategy = make_strategy(budget=16.0, num_candidates=3)
    strategy = replace(
        strategy,
        search=replace(
            strategy.search,
            diversity=DiversityConfig(high_impact_fraction=1.0, min_high_impact_differences=2),
        ),
    )
    result = search_candidates(space, profile, strategy)

    assert result.payload.diversity_unit_ids == (
        "model.layers.0.k_proj",
        "model.layers.0.q_proj",
    )
    assignments = [
        {entry.module_name: entry.scheme for entry in candidate.assignment} for candidate in result.payload.candidates
    ]
    for index, assignment in enumerate(assignments):
        for other in assignments[index + 1 :]:
            differences = sum(assignment[unit_id] != other[unit_id] for unit_id in result.payload.diversity_unit_ids)
            assert differences >= result.payload.min_diversity_differences


def test_vllm_deployment_constraints_expand_one_scheme_to_every_hf_member() -> None:
    strategy = make_strategy(
        num_candidates=1,
        deployment_backend="vllm",
        evaluator_backend="hf",
    )
    model = Qwen3LikeModel(num_layers=1)
    space = build_decision_space(model, strategy)
    profile = build_sensitivity_profile(model, space, strategy)
    candidate = search_candidates(space, profile, strategy).payload.candidates[0]
    assignment = {entry.module_name: entry.scheme for entry in candidate.assignment}

    assert len(space.payload.decision_units) == 4
    assert len(profile.payload.records) == 4 * 3
    qkv_native = next(
        record
        for record in profile.payload.records
        if record.unit_id == "model.layers.0.self_attn.qkv_proj" and record.scheme == "native"
    )
    qkv_fp8 = next(
        record
        for record in profile.payload.records
        if record.unit_id == "model.layers.0.self_attn.qkv_proj" and record.scheme == "fp8"
    )
    qkv_unit = next(unit for unit in space.payload.decision_units if unit.unit_id == qkv_fp8.unit_id)
    fused_qkv_weight = torch.cat(
        [dict(model.named_modules())[member].weight for member in qkv_unit.members],
        dim=0,
    )
    fused_qkv_score = calculate_weight_mse(
        fused_qkv_weight,
        "fp8",
        total_quantizable_params=space.payload.total_quantizable_params,
        model_type="qwen3",
    )
    fused_qkv_cost = calculate_weight_storage(
        tuple(fused_qkv_weight.shape),
        "bfloat16",
        "fp8",
        model_type="qwen3",
    )
    assert qkv_native.cost.num_params == 128
    assert qkv_native.cost.total_bits == 128 * 16
    assert qkv_fp8.score.squared_error == pytest.approx(fused_qkv_score.squared_error)
    assert qkv_fp8.score.signal_power == pytest.approx(fused_qkv_score.signal_power)
    assert qkv_fp8.score.weighted_score == pytest.approx(fused_qkv_score.weighted_score)
    assert qkv_fp8.cost == fused_qkv_cost
    assert len(candidate.assignment) == 8
    for layer_index in range(1):
        prefix = f"model.layers.{layer_index}"
        assert (
            len(
                {
                    assignment[f"{prefix}.self_attn.q_proj"],
                    assignment[f"{prefix}.self_attn.k_proj"],
                    assignment[f"{prefix}.self_attn.v_proj"],
                }
            )
            == 1
        )
        assert assignment[f"{prefix}.mlp.gate_proj"] == assignment[f"{prefix}.mlp.up_proj"]

    incompatible = list(candidate.assignment)
    q_index = next(
        index for index, entry in enumerate(incompatible) if entry.module_name == "model.layers.0.self_attn.q_proj"
    )
    replacement_scheme = "native" if incompatible[q_index].scheme != "native" else "fp8"
    incompatible[q_index] = replace(incompatible[q_index], scheme=replacement_scheme)
    with pytest.raises(SchemaValidationError, match="deployment-compatible"):
        validate_assignment(space, replace(candidate, assignment=tuple(incompatible)))


def test_vllm_forced_native_group_is_costed_and_expanded() -> None:
    strategy = make_strategy(
        budget=12.0,
        num_candidates=1,
        deployment_backend="vllm",
        evaluator_backend="hf",
        exclude_patterns=["*.k_proj"],
    )
    model = Qwen3LikeModel(num_layers=1)
    space = build_decision_space(model, strategy)
    profile = build_sensitivity_profile(model, space, strategy)
    candidate = search_candidates(space, profile, strategy).payload.candidates[0]

    qkv_names = {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    }
    fixed_names = {module_name for cost in profile.payload.fixed_native_costs for module_name in cost.module_names}
    assignment = {entry.module_name: entry.scheme for entry in candidate.assignment}
    assert fixed_names == qkv_names
    assert {assignment[name] for name in qkv_names} == {"native"}

    incomplete_profile = SensitivityProfile.create(
        replace(profile.payload, fixed_native_costs=profile.payload.fixed_native_costs[1:]),
        decision_space=space,
        strategy=strategy,
    )
    with pytest.raises(ArtifactCompatibilityError, match="fixed-native"):
        linear_programming._validate_inputs(space, incomplete_profile)


def test_search_scales_to_qwen3_decision_count() -> None:
    torch.manual_seed(6)
    model = Qwen3LikeModel()
    strategy = make_strategy(num_candidates=1)
    space = build_decision_space(model, strategy)
    profile = build_sensitivity_profile(model, space, strategy)
    result = search_candidates(space, profile, strategy)

    assert len(space.payload.decision_units) == 196
    assert len(result.payload.candidates[0].assignment) == 197
    assert result.payload.termination_reason is SearchTerminationReason.REQUESTED_CANDIDATE_COUNT_REACHED


def test_search_returns_partial_when_diverse_space_is_exhausted() -> None:
    space, profile = build_inputs()
    result = search_candidates(space, profile, make_strategy(num_candidates=5))
    assert result.payload.termination_reason is SearchTerminationReason.SEARCH_SPACE_EXHAUSTED
    assert len(result.payload.candidates) == 2


def test_lexicographic_tie_break_prefers_lowest_storage() -> None:
    model = TinyQwen3()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.zero_()
    strategy = make_strategy(budget=16.0, num_candidates=1)
    space = build_decision_space(model, strategy)
    profile = build_sensitivity_profile(model, space, strategy)
    result = search_candidates(space, profile, strategy)
    assert result.payload.candidates[0].effective_bits == pytest.approx(8.25)


def test_lexicographic_loss_is_not_traded_for_storage() -> None:
    space, profile = build_inputs()
    first_unit = space.payload.decision_units[0].unit_id
    records = []
    for record in profile.payload.records:
        if record.scheme == "ptpc_fp8":
            score = replace(record.score, weighted_score=1.0)
        elif record.scheme == "fp8" and record.unit_id == first_unit:
            score = replace(record.score, weighted_score=5e-13)
        elif record.scheme == "fp8":
            score = replace(record.score, weighted_score=0.0)
        else:
            score = record.score
        records.append(replace(record, score=score))
    strategy = make_strategy(budget=12.125, num_candidates=1)
    adjusted = SensitivityProfile.create(
        replace(profile.payload, records=tuple(records)),
        decision_space=space,
        strategy=strategy,
    )
    result = search_candidates(space, adjusted, strategy)
    assert result.payload.candidates[0].effective_bits == pytest.approx(12.125)


def test_search_rejects_incomplete_loaded_profile() -> None:
    space, profile = build_inputs()
    strategy = make_strategy()
    incomplete = SensitivityProfile.create(
        replace(profile.payload, records=profile.payload.records[:-1]),
        decision_space=space,
        strategy=strategy,
    )
    with pytest.raises(SchemaValidationError, match="Incomplete sensitivity matrix"):
        search_candidates(space, incomplete, strategy)


def test_search_rejects_infeasible_budget() -> None:
    space, profile = build_inputs()
    with pytest.raises(InfeasibleBudgetError, match="minimum"):
        search_candidates(space, profile, make_strategy(budget=8.0))


def test_search_rejects_invalid_timeout_and_missing_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    space, profile = build_inputs()
    strategy = make_strategy()
    with pytest.raises(SchemaValidationError, match="timeout"):
        search_candidates(space, profile, strategy, timeout_seconds=0)

    monkeypatch.setattr(linear_programming, "_cbc_path", lambda: None)
    with pytest.raises(SolverUnavailableError, match="CBC"):
        search_candidates(space, profile, strategy)


def test_optional_solver_discovery_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_module(_name: str) -> object:
        raise ImportError

    monkeypatch.setattr(linear_programming.importlib, "import_module", missing_module)
    with pytest.raises(SolverUnavailableError, match="PuLP"):
        linear_programming._load_pulp()
    monkeypatch.setattr(linear_programming.shutil, "which", lambda _name: None)
    assert linear_programming._cbc_path() is None


def test_system_cbc_path_is_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(linear_programming.shutil, "which", lambda _name: "/usr/bin/cbc")
    assert linear_programming._cbc_path() == "/usr/bin/cbc"


def test_search_rejects_unavailable_coin_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    space, profile = build_inputs()
    monkeypatch.setattr(linear_programming, "_cbc_path", lambda: "/bin/true")
    monkeypatch.setattr(pulp, "COIN_CMD", lambda **_kwargs: SimpleNamespace(available=lambda: False))
    with pytest.raises(SolverUnavailableError, match="CBC"):
        search_candidates(space, profile, make_strategy())


def test_search_input_compatibility_checks() -> None:
    space, profile = build_inputs()
    strategy = make_strategy()
    wrong_space_profile = SensitivityProfile.create(
        replace(profile.payload, decision_space_id="sha256:other"),
        decision_space=space,
        strategy=strategy,
    )
    with pytest.raises(ArtifactCompatibilityError, match="belong"):
        linear_programming._validate_inputs(space, wrong_space_profile)

    other_artifact = Artifact.create(
        "sensitivity_profile",
        profile.payload.to_dict(),
        model_fingerprint="other",
    )
    with pytest.raises(ArtifactCompatibilityError, match="model"):
        linear_programming._validate_inputs(space, SensitivityProfile(other_artifact, profile.payload))
    with pytest.raises(ArtifactCompatibilityError, match="denominator"):
        wrong_denominator = SensitivityProfile.create(
            replace(
                profile.payload,
                total_quantizable_params=profile.payload.total_quantizable_params + 1,
            ),
            decision_space=space,
            strategy=strategy,
        )
        linear_programming._validate_inputs(
            space,
            wrong_denominator,
        )

    no_search_modules = tuple(
        replace(module, status=ModuleStatus.TEMPLATE_EXCLUDED) for module in space.payload.modules
    )
    with pytest.raises(SchemaValidationError, match="total_quantizable_params"):
        replace(space.payload, modules=no_search_modules, decision_units=())


def test_search_maps_non_optimal_and_solver_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    space, profile = build_inputs()
    strategy = make_strategy()

    def not_solved(problem: object, _solver: object) -> int:
        problem.status = pulp.LpStatusNotSolved  # type: ignore[attr-defined]
        return problem.status  # type: ignore[attr-defined,no-any-return]

    monkeypatch.setattr(pulp.LpProblem, "solve", not_solved)
    with pytest.raises(SearchError, match="Not Solved"):
        search_candidates(space, profile, strategy)

    def fail(_problem: object, _solver: object) -> int:
        raise RuntimeError("solver crashed")

    monkeypatch.setattr(pulp.LpProblem, "solve", fail)
    with pytest.raises(SearchError, match="solver crashed"):
        search_candidates(space, profile, strategy)


def test_search_maps_solver_infeasible_and_tie_break_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    space, profile = build_inputs()
    strategy = make_strategy()

    def infeasible(problem: object, _solver: object) -> int:
        problem.status = pulp.LpStatusInfeasible  # type: ignore[attr-defined]
        return problem.status  # type: ignore[attr-defined,no-any-return]

    monkeypatch.setattr(pulp.LpProblem, "solve", infeasible)
    with pytest.raises(InfeasibleBudgetError, match="CBC"):
        search_candidates(space, profile, strategy)

    monkeypatch.undo()
    monkeypatch.setattr(pulp, "value", lambda _value: 0.0)
    with pytest.raises(SearchError, match="tie-break"):
        search_candidates(space, profile, strategy)


def test_search_post_solve_budget_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    space, profile = build_inputs()
    monkeypatch.setattr(
        linear_programming,
        "_make_candidate",
        lambda *_args, **_kwargs: SimpleNamespace(total_bits=10**30),
    )
    with pytest.raises(SearchError, match="post-solve budget"):
        search_candidates(space, profile, make_strategy())


def test_make_candidate_requires_complete_assignment() -> None:
    space, profile = build_inputs()
    records = linear_programming._record_map(profile)
    selected = {space.payload.decision_units[0].unit_id: "fp8"}
    with pytest.raises(SchemaValidationError, match="cover"):
        linear_programming._make_candidate(
            selected,
            decision_space=space,
            records=records,
            fixed_native_bits=0,
            rank=1,
        )


def test_candidates_payload_validation() -> None:
    space, profile = build_inputs()
    result = search_candidates(space, profile, make_strategy(budget=16.0))
    payload = result.payload

    with pytest.raises(SchemaValidationError, match="ranks"):
        replace(payload, candidates=(replace(payload.candidates[0], rank=2),) + payload.candidates[1:])
    with pytest.raises(SchemaValidationError, match="budget"):
        replace(payload, budget_limit_bits=0.0)
    with pytest.raises(SchemaValidationError, match="requested"):
        replace(payload, requested_candidates=0)
    with pytest.raises(SchemaValidationError, match="Exhausted"):
        replace(payload, termination_reason=SearchTerminationReason.SEARCH_SPACE_EXHAUSTED)
    with pytest.raises(SchemaValidationError, match="diversity unit"):
        replace(payload, diversity_unit_ids=payload.diversity_unit_ids + payload.diversity_unit_ids)
    with pytest.raises(SchemaValidationError, match="diversity difference"):
        replace(payload, min_diversity_differences=0)


def test_candidate_value_validation() -> None:
    space, profile = build_inputs()
    result = search_candidates(space, profile, make_strategy())
    candidate = result.payload.candidates[0]

    with pytest.raises(SchemaValidationError, match="module_name"):
        AssignmentEntry("", "native")
    invalid_changes = [
        ({"candidate_id": "bad"}, "SHA-256"),
        ({"rank": 0}, "rank"),
        ({"predicted_loss": -1.0}, "predicted_loss"),
        ({"total_bits": 0}, "total_bits"),
        ({"effective_bits": 0.0}, "effective_bits"),
        ({"assignment": tuple(reversed(candidate.assignment))}, "sorted"),
        ({"assignment": (candidate.assignment[0], candidate.assignment[0])}, "unique"),
    ]
    for changes, message in invalid_changes:
        with pytest.raises(SchemaValidationError, match=message):
            replace(candidate, **changes)


def test_candidate_envelope_validation() -> None:
    space, profile = build_inputs()
    result = search_candidates(space, profile, make_strategy(budget=16.0))
    payload = result.payload
    candidates = payload.candidates

    invalid_changes = [
        ({"decision_space_id": "bad"}, "input ids"),
        ({"solver_name": "other"}, "PuLP"),
        ({"candidates": candidates[:-1]}, "requested candidate count"),
        (
            {
                "candidates": (
                    candidates[0],
                    replace(candidates[1], candidate_id=candidates[0].candidate_id),
                    candidates[2],
                )
            },
            "ids",
        ),
        ({"budget_limit_bits": candidates[0].total_bits - 1}, "exceeds"),
        (
            {
                "candidates": (replace(candidates[0], effective_bits=candidates[0].effective_bits + 1.0),)
                + candidates[1:]
            },
            "does not match",
        ),
        (
            {"termination_reason": cast(SearchTerminationReason, "unknown")},
            "termination reason",
        ),
    ]
    for changes, message in invalid_changes:
        with pytest.raises(SchemaValidationError, match=message):
            replace(payload, **changes)

    assert result.artifact_id == result.artifact.artifact_id
