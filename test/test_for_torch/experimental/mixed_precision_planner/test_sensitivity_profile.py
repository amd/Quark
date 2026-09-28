#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from quark.experimental.torch.mixed_precision_planner import (
    MixedPrecisionStrategy,
    SensitivityProfile,
)
from quark.experimental.torch.mixed_precision_planner.cost_accounting import (
    WeightStorageCost,
    calculate_weight_storage,
)
from quark.experimental.torch.mixed_precision_planner.decision_space import build_decision_space
from quark.experimental.torch.mixed_precision_planner.errors import (
    ArtifactCompatibilityError,
    SchemaValidationError,
)
from quark.experimental.torch.mixed_precision_planner.hardware_capability import get_scheme_config
from quark.experimental.torch.mixed_precision_planner.sensitivity_profile import (
    SensitivityProfilePayload,
    build_sensitivity_profile,
    model_behavior_fingerprint,
    model_state_fingerprint,
)
from quark.experimental.torch.mixed_precision_planner.weight_mse import (
    WeightMseScore,
    calculate_weight_mse,
    quantize_weight_for_profile,
)
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear


class TinyLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)


class TinyQwen3(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer(width)])
        self.lm_head = nn.Linear(width, 16, bias=False)
        self.to(dtype=torch.bfloat16)


def make_strategy() -> MixedPrecisionStrategy:
    return MixedPrecisionStrategy.from_dict(
        {
            "schema_version": 1,
            "hardware": {"target": "mi355"},
            "deployment_backend": {"name": "no_fusion"},
            "decision_space": {"exclude_patterns": ["*.k_proj"]},
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
                "budget": {"metric": "effective_bits", "value": 8.5, "scope": "quantizable"},
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
    )


def test_weight_storage_cost_uses_packed_shapes_and_scale_metadata() -> None:
    native = calculate_weight_storage((4, 4), "bfloat16", "native", model_type="qwen3")
    fp8 = calculate_weight_storage((4, 4), "bfloat16", "fp8", model_type="qwen3")
    ptpc = calculate_weight_storage((4, 4), "bfloat16", "ptpc_fp8", model_type="qwen3")

    assert (native.weight_bits, native.metadata_bits, native.effective_bits) == (256, 0, 16.0)
    assert (fp8.weight_bits, fp8.metadata_bits, fp8.effective_bits) == (128, 16, 9.0)
    assert (ptpc.weight_bits, ptpc.metadata_bits, ptpc.effective_bits) == (128, 64, 12.0)


@pytest.mark.parametrize("scheme", ["fp8", "ptpc_fp8"])
def test_weight_qdq_matches_quant_linear(scheme: str) -> None:
    torch.manual_seed(7)
    linear = nn.Linear(8, 8, bias=False).to(dtype=torch.bfloat16)
    original = linear.weight.detach().clone()

    quant_linear = QuantLinear.from_float(linear, get_scheme_config("qwen3", scheme))
    assert isinstance(quant_linear, QuantLinear)
    expected = quant_linear.get_quant_weight(quant_linear.weight)
    actual = quantize_weight_for_profile(linear.weight, scheme, model_type="qwen3")

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(linear.weight, original)


def test_weight_mse_semantics() -> None:
    weight = torch.linspace(-1, 1, 16, dtype=torch.bfloat16).reshape(4, 4)
    native = calculate_weight_mse(weight, "native", total_quantizable_params=32, model_type="qwen3")
    fp8 = calculate_weight_mse(weight, "fp8", total_quantizable_params=32, model_type="qwen3")
    all_zero = calculate_weight_mse(
        torch.zeros_like(weight),
        "fp8",
        total_quantizable_params=16,
        model_type="qwen3",
    )

    assert native.relative_mse == native.weighted_score == 0.0
    assert fp8.relative_mse > 0
    assert fp8.weighted_score == pytest.approx(fp8.relative_mse / 2)
    assert all_zero.relative_mse == 0.0


def test_build_sensitivity_profile_and_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = TinyQwen3()
    strategy = make_strategy()
    space = build_decision_space(model, strategy)
    profile = build_sensitivity_profile(model, space, strategy)

    assert len(profile.payload.records) == 3
    assert {(record.unit_id, record.scheme) for record in profile.payload.records} == {
        ("model.layers.0.q_proj", "native"),
        ("model.layers.0.q_proj", "fp8"),
        ("model.layers.0.q_proj", "ptpc_fp8"),
    }
    assert profile.payload.fixed_native_costs[0].module_names == ("model.layers.0.k_proj",)
    assert profile.payload.total_quantizable_params == 2 * 64
    assert profile.artifact.upstream == {"decision_space": space.artifact_id}

    path = tmp_path / "sensitivity_profile.json"
    profile.save(path)
    assert SensitivityProfile.load(path) == profile


def test_profile_rejects_model_structure_mismatch() -> None:
    strategy = make_strategy()
    space = build_decision_space(TinyQwen3(width=8), strategy)
    with pytest.raises(ArtifactCompatibilityError, match="structure"):
        build_sensitivity_profile(TinyQwen3(width=4), space, strategy)


def test_model_fingerprints_bind_behavior_and_reject_extra_state() -> None:
    first = TinyQwen3()
    second = TinyQwen3()
    second.load_state_dict(first.state_dict())
    second.config.rope_theta = 1_000_000.0
    assert model_state_fingerprint(first) == model_state_fingerprint(second)
    assert model_behavior_fingerprint(first) != model_behavior_fingerprint(second)

    class ExtraStateModel(TinyQwen3):
        def get_extra_state(self) -> dict[str, str]:
            return {"mode": "different"}

        def set_extra_state(self, _state: object) -> None:
            pass

    with pytest.raises(SchemaValidationError, match="not a tensor"):
        model_state_fingerprint(ExtraStateModel())


def test_cost_and_profile_value_validation() -> None:
    with pytest.raises(SchemaValidationError, match="shape"):
        calculate_weight_storage((), "bfloat16", "native", model_type="qwen3")
    with pytest.raises(SchemaValidationError, match="source dtype"):
        calculate_weight_storage((4, 4), "float32", "native", model_type="qwen3")
    with pytest.raises(SchemaValidationError, match="scheme"):
        calculate_weight_storage((4, 4), "bfloat16", "mxfp4", model_type="qwen3")
    with pytest.raises(SchemaValidationError, match="total_bits"):
        WeightStorageCost(8, 8, 8, 1, 8.0)
    with pytest.raises(SchemaValidationError, match="Weight-MSE"):
        WeightMseScore(-1.0, 1.0, 1.0, 1.0)
    with pytest.raises(SchemaValidationError, match="positive"):
        calculate_weight_mse(torch.ones(1), "native", total_quantizable_params=0, model_type="qwen3")

    model = TinyQwen3()
    strategy = make_strategy()
    payload = build_sensitivity_profile(model, build_decision_space(model, strategy), strategy).payload
    with pytest.raises(SchemaValidationError, match="duplicate"):
        replace(payload, records=payload.records + (payload.records[0],))
    with pytest.raises(SchemaValidationError, match="positive"):
        replace(payload, total_quantizable_params=0)
    assert isinstance(payload, SensitivityProfilePayload)
