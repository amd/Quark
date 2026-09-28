#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from quark.experimental.torch.mixed_precision_planner import (
    DecisionSpace,
    MixedPrecisionStrategy,
)
from quark.experimental.torch.mixed_precision_planner.decision_space import (
    DecisionSpacePayload,
    DecisionUnit,
    build_decision_space,
)
from quark.experimental.torch.mixed_precision_planner.errors import ArtifactCompatibilityError, SchemaValidationError
from quark.experimental.torch.mixed_precision_planner.hardware_capability import get_supported_schemes
from quark.experimental.torch.mixed_precision_planner.mixed_precision_strategy import HardwareTarget
from quark.experimental.torch.mixed_precision_planner.quantizable_modules import ModuleStatus


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMLP()


class TinyDenseModel(nn.Module):
    def __init__(self, *, model_type: str = "qwen3", dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer()])
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.to(dtype=dtype)


def make_strategy(
    *,
    exclude_patterns: list[str] | None = None,
    deployment_backend: str = "no_fusion",
    evaluator_backend: str = "hf",
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
                "budget": {"metric": "effective_bits", "value": 8.5, "scope": "quantizable"},
                "num_candidates": 5,
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


def test_build_decision_space_and_round_trip(tmp_path: Path) -> None:
    space = build_decision_space(TinyDenseModel(), make_strategy(exclude_patterns=["*.k_proj"]))
    by_name = {module.name: module for module in space.payload.modules}

    assert space.payload.model_type == "qwen3"
    assert space.payload.deployment_backend == "no_fusion"
    assert space.payload.supported_schemes == ("native", "fp8", "ptpc_fp8")
    assert by_name["lm_head"].status is ModuleStatus.TEMPLATE_EXCLUDED
    assert by_name["model.layers.0.self_attn.k_proj"].status is ModuleStatus.FORCED_NATIVE_USER
    assert len(space.payload.decision_units) == 6
    assert space.payload.total_quantizable_params == 144
    assert space.payload.matched_user_patterns == {
        "*.k_proj": ("model.layers.0.self_attn.k_proj",),
    }

    path = tmp_path / "decision_space.json"
    space.save(path)
    loaded = DecisionSpace.load(path)
    assert loaded == space
    loaded.validate_compatibility(make_strategy(exclude_patterns=["*.k_proj"]))
    with pytest.raises(ArtifactCompatibilityError, match="inputs"):
        loaded.validate_compatibility(make_strategy())
    loaded.validate_compatibility(make_strategy(exclude_patterns=["*.k_proj"], evaluator_backend="vllm"))
    loaded.payload.matched_user_patterns["tampered"] = ()
    with pytest.raises(SchemaValidationError, match="Artifact envelope"):
        loaded.validate_integrity()


def test_model_fingerprint_is_independent_of_user_excludes() -> None:
    first = build_decision_space(TinyDenseModel(), make_strategy())
    second = build_decision_space(TinyDenseModel(), make_strategy(exclude_patterns=["*.k_proj"]))
    assert first.model_fingerprint == second.model_fingerprint
    assert first.artifact_id != second.artifact_id


def test_evaluator_does_not_change_no_fusion_decision_space() -> None:
    hf_space = build_decision_space(TinyDenseModel(), make_strategy(evaluator_backend="hf"))
    vllm_space = build_decision_space(TinyDenseModel(), make_strategy(evaluator_backend="vllm"))
    assert hf_space.payload == vllm_space.payload
    assert hf_space.artifact_id == vllm_space.artifact_id
    assert hf_space.artifact.input_fingerprints == vllm_space.artifact.input_fingerprints
    assert len(vllm_space.payload.decision_units) == 7
    assert all(len(unit.members) == 1 for unit in vllm_space.payload.decision_units)


def test_vllm_deployment_constraints_apply_with_hf_evaluator() -> None:
    strategy = make_strategy(deployment_backend="vllm", evaluator_backend="hf")
    space = build_decision_space(TinyDenseModel(), strategy)
    units = {unit.unit_id: unit for unit in space.payload.decision_units}
    assert units["model.layers.0.self_attn.qkv_proj"].members == (
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    )
    assert space.payload.deployment_backend == "vllm"


def test_shared_storage_is_forced_native_and_counted_once() -> None:
    model = TinyDenseModel()
    model.model.layers[0].self_attn.o_proj.weight = model.model.layers[0].self_attn.q_proj.weight
    space = build_decision_space(model, make_strategy())
    by_name = {module.name: module for module in space.payload.modules}

    q_proj = by_name["model.layers.0.self_attn.q_proj"]
    o_proj = by_name["model.layers.0.self_attn.o_proj"]
    assert q_proj.status is ModuleStatus.FORCED_NATIVE_ALIAS
    assert o_proj.status is ModuleStatus.FORCED_NATIVE_ALIAS
    assert q_proj.storage_id == o_proj.storage_id
    assert space.payload.total_quantizable_params == 128


def test_partial_storage_views_are_rejected() -> None:
    model = TinyDenseModel()
    shared = torch.zeros((8, 4), dtype=torch.bfloat16)
    model.model.layers[0].self_attn.q_proj.weight = nn.Parameter(shared[:4])
    model.model.layers[0].self_attn.o_proj.weight = nn.Parameter(shared[4:])
    with pytest.raises(SchemaValidationError, match="partial"):
        build_decision_space(model, make_strategy())


def test_decision_space_rejects_invalid_models_and_patterns() -> None:
    with pytest.raises(SchemaValidationError, match="did not match"):
        build_decision_space(TinyDenseModel(), make_strategy(exclude_patterns=["*.missing"]))
    with pytest.raises(SchemaValidationError, match="BF16 or FP16"):
        build_decision_space(TinyDenseModel(dtype=torch.float32), make_strategy())
    with pytest.raises(SchemaValidationError, match="dense model types"):
        build_decision_space(TinyDenseModel(model_type="qwen2"), make_strategy())
    with pytest.raises(SchemaValidationError, match="dense model types"):
        build_decision_space(TinyDenseModel(model_type="qwen3_moe"), make_strategy())
    with pytest.raises(SchemaValidationError, match="config.model_type"):
        build_decision_space(nn.Linear(4, 4).to(torch.bfloat16), make_strategy())

    no_linears = nn.Module()
    no_linears.config = SimpleNamespace(model_type="qwen3")
    with pytest.raises(SchemaValidationError, match="no nn.Linear"):
        build_decision_space(no_linears, make_strategy())

    invalid_weight = TinyDenseModel()
    invalid_weight.model.layers[0].self_attn.q_proj.weight = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    with pytest.raises(SchemaValidationError, match="rank 2"):
        build_decision_space(invalid_weight, make_strategy())

    meta_model = TinyDenseModel()
    meta_model.to(device="meta")
    with pytest.raises(SchemaValidationError, match="materialized"):
        build_decision_space(meta_model, make_strategy())


def test_hardware_capability_rejects_invalid_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SchemaValidationError, match="hardware"):
        get_supported_schemes(cast(HardwareTarget, "mi300"), "llama")
    with pytest.raises(SchemaValidationError, match="dense model types"):
        get_supported_schemes(HardwareTarget.MI355, "qwen2")

    monkeypatch.setattr(
        "quark.experimental.torch.mixed_precision_planner.hardware_capability.LLMTemplate.get_supported_schemes",
        lambda _self: ["fp8"],
    )
    with pytest.raises(SchemaValidationError, match="ptpc_fp8"):
        get_supported_schemes(HardwareTarget.MI355, "qwen3")


def test_dense_llama_remains_supported() -> None:
    space = build_decision_space(TinyDenseModel(model_type="llama"), make_strategy())
    assert space.payload.model_type == "llama"
    assert space.payload.supported_schemes == ("native", "fp8", "ptpc_fp8")


def test_decision_unit_validates_identity_and_members() -> None:
    unit = DecisionUnit("fused", ("first", "second"), ("native", "fp8", "ptpc_fp8"))
    assert unit.members == ("first", "second")
    with pytest.raises(SchemaValidationError, match="sorted"):
        DecisionUnit("unit", ("second", "first"), ("native", "fp8", "ptpc_fp8"))
    with pytest.raises(SchemaValidationError, match="native"):
        DecisionUnit("unit", ("unit",), ("fp8",))


def test_decision_space_payload_invariants() -> None:
    payload = build_decision_space(TinyDenseModel(), make_strategy()).payload
    with pytest.raises(SchemaValidationError, match="deployment backend"):
        replace(payload, deployment_backend="other")
    with pytest.raises(SchemaValidationError, match="sorted and unique"):
        replace(payload, modules=payload.modules + (payload.modules[0],))
    with pytest.raises(SchemaValidationError, match="exactly one"):
        replace(payload, decision_units=payload.decision_units[:-1])
    with pytest.raises(SchemaValidationError, match="sorted and unique"):
        replace(payload, decision_units=payload.decision_units + (payload.decision_units[0],))
    with pytest.raises(SchemaValidationError, match="scheme set"):
        changed_unit = replace(payload.decision_units[0], allowed_schemes=("native", "fp8"))
        replace(payload, decision_units=(changed_unit,) + payload.decision_units[1:])
    with pytest.raises(SchemaValidationError, match="total_quantizable_params"):
        replace(payload, total_quantizable_params=0)

    assert isinstance(payload, DecisionSpacePayload)
