#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from pathlib import Path

import pytest

from quark.experimental.torch.mixed_precision_planner import MixedPrecisionStrategy
from quark.experimental.torch.mixed_precision_planner.errors import SchemaValidationError
from quark.experimental.torch.mixed_precision_planner.mixed_precision_strategy import (
    BudgetConfig,
    CalibrationConfig,
    DecisionSpaceConfig,
    DiversityConfig,
    EvaluatorBackend,
    EvaluatorConfig,
    OutputConfig,
    PlanSelectionConfig,
    QualityGate,
)


def strategy_dict() -> dict[str, object]:
    return {
        "schema_version": 1,
        "hardware": {"target": "mi355"},
        "deployment_backend": {"name": "no_fusion"},
        "decision_space": {"exclude_patterns": []},
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
            "budget": {
                "metric": "effective_bits",
                "value": 8.5,
                "scope": "quantizable",
            },
            "num_candidates": 5,
        },
        "plan_selection": {
            "evaluator": {"backend": "hf", "protocol": "token_ppl_v1"},
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


def test_strategy_round_trip(tmp_path: Path) -> None:
    strategy = MixedPrecisionStrategy.from_dict(strategy_dict())
    assert strategy.search.diversity == DiversityConfig()
    path = tmp_path / "strategy.json"
    strategy.save(path)
    assert MixedPrecisionStrategy.load(path) == strategy


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("hardware", "target"), "mi300"),
        (("deployment_backend", "name"), "other"),
        (("sensitivity_profile", "methods"), ["prior"]),
        (("calibration", "dataset"), "wikitext2"),
        (("search", "granularity"), "coarse"),
        (("search", "algorithm"), "enumeration"),
        (("search", "num_candidates"), 0),
        (("plan_selection", "selection_policy"), "best_quality"),
    ],
)
def test_strategy_rejects_unsupported_values(path: tuple[str, str], value: object) -> None:
    data = strategy_dict()
    section = data[path[0]]
    assert isinstance(section, dict)
    section[path[1]] = value
    with pytest.raises(SchemaValidationError):
        MixedPrecisionStrategy.from_dict(data)


def test_strategy_rejects_unknown_field_and_version() -> None:
    data = strategy_dict()
    data["unknown"] = True
    with pytest.raises(SchemaValidationError, match="unknown fields"):
        MixedPrecisionStrategy.from_dict(data)

    data = strategy_dict()
    data["schema_version"] = 2
    with pytest.raises(SchemaValidationError):
        MixedPrecisionStrategy.from_dict(data)

    for invalid_seed in (-1, 0, 2_147_483_648):
        data = strategy_dict()
        data["seed"] = invalid_seed
        with pytest.raises(SchemaValidationError, match="seed"):
            MixedPrecisionStrategy.from_dict(data)


def test_strategy_keeps_deployment_and_evaluator_backends_independent() -> None:
    data = strategy_dict()
    deployment = data["deployment_backend"]
    plan_selection = data["plan_selection"]
    assert isinstance(deployment, dict) and isinstance(plan_selection, dict)
    plan_selection["evaluator"] = {"backend": "vllm", "protocol": "token_ppl_v1"}
    strategy = MixedPrecisionStrategy.from_dict(data)
    assert strategy.deployment_backend.name.value == "no_fusion"
    assert strategy.plan_selection.evaluator.backend.value == "vllm"

    deployment["name"] = "vllm"
    plan_selection["evaluator"] = {"backend": "hf", "protocol": "token_ppl_v1"}
    strategy = MixedPrecisionStrategy.from_dict(data)
    assert strategy.deployment_backend.name.value == "vllm"
    assert strategy.plan_selection.evaluator.backend.value == "hf"

    plan_selection["evaluator"] = {"backend": "hf", "protocol": "other"}
    with pytest.raises(SchemaValidationError, match="protocol"):
        MixedPrecisionStrategy.from_dict(data)


def test_strategy_component_validation() -> None:
    invalid_factories = [
        lambda: DecisionSpaceConfig(("",)),
        lambda: DecisionSpaceConfig(("x", "x")),
        lambda: CalibrationConfig("other", 1, 16, 1),
        lambda: CalibrationConfig("pileval", 0, 16, 1),
        lambda: CalibrationConfig("pileval", 1, 0, 1),
        lambda: CalibrationConfig("pileval", 1, 16, 2),
        lambda: CalibrationConfig("pileval", 1, 16, 1, ""),
        lambda: BudgetConfig("other", 1.0, "quantizable"),
        lambda: BudgetConfig("effective_bits", 0.0, "quantizable"),
        lambda: DiversityConfig(0.0, 1),
        lambda: DiversityConfig(1.01, 1),
        lambda: DiversityConfig(0.3, 0),
        lambda: QualityGate("other", "wikitext2", 0.02, 1, 2048),
        lambda: QualityGate("ppl", "other", 0.02, 1, 2048),
        lambda: QualityGate("ppl", "wikitext2", -0.1, 1, 2048),
        lambda: QualityGate("ppl", "wikitext2", 0.02, 0, 2048),
        lambda: QualityGate("ppl", "wikitext2", 0.02, 1, 1024),
        lambda: QualityGate("ppl", "wikitext2", 0.02, 1, 2048, ""),
        lambda: PlanSelectionConfig(
            QualityGate("ppl", "wikitext2", 0.02, 1, 2048),
            "other",
            EvaluatorConfig(EvaluatorBackend.HF, "token_ppl_v1"),
        ),
        lambda: OutputConfig(""),
    ]
    for factory in invalid_factories:
        with pytest.raises(SchemaValidationError):
            factory()
