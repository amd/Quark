#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from quark.experimental.torch.mixed_precision_planner import MixedPrecisionStrategy
from quark.experimental.torch.mixed_precision_planner._serialization import sha256_json
from quark.experimental.torch.mixed_precision_planner.candidates import AssignmentEntry, Candidate
from quark.experimental.torch.mixed_precision_planner.decision_space import build_decision_space
from quark.experimental.torch.mixed_precision_planner.errors import SchemaValidationError
from quark.experimental.torch.mixed_precision_planner.hardware_capability import get_scheme_config
from quark.experimental.torch.mixed_precision_planner.linear_programming import search_candidates
from quark.experimental.torch.mixed_precision_planner.plan_checks import validate_assignment
from quark.experimental.torch.mixed_precision_planner.qconfig_builder import (
    QConfigAudit,
    _validate_model_binding,
    audit_qconfig,
    build_qconfig,
    compile_qconfig,
)
from quark.experimental.torch.mixed_precision_planner.sensitivity_profile import build_sensitivity_profile
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig
from quark.torch.quantization.config.config_verification import ConfigVerifier


class TinyLayer(nn.Module):
    def __init__(self, width: int = 8) -> None:
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
                "budget": {"metric": "effective_bits", "value": 12.0, "scope": "quantizable"},
                "num_candidates": 3,
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


def build_inputs() -> tuple[TinyQwen3, object, Candidate]:
    torch.manual_seed(17)
    model = TinyQwen3()
    strategy = make_strategy()
    space = build_decision_space(model, strategy)
    profile = build_sensitivity_profile(model, space, strategy)
    candidate = search_candidates(space, profile, strategy).payload.candidates[0]
    return model, space, candidate


def _candidate(entries: tuple[AssignmentEntry, ...], source: Candidate) -> Candidate:
    return Candidate(
        candidate_id=sha256_json([entry.to_dict() for entry in entries]),
        rank=source.rank,
        predicted_loss=source.predicted_loss,
        total_bits=source.total_bits,
        effective_bits=source.effective_bits,
        assignment=entries,
    )


def test_compile_qconfig_is_exact_and_round_trip_safe() -> None:
    model, space, candidate = build_inputs()
    qconfig, audit = compile_qconfig(model, space, candidate)
    assignment = {entry.module_name: entry.scheme for entry in candidate.assignment}

    assert qconfig.global_quant_config == QLayerConfig()
    assert set(qconfig.layer_quant_config) == {name for name, scheme in assignment.items() if scheme != "native"}
    assert all("*" not in name for name in qconfig.layer_quant_config)
    assert audit.resolved_quantized == tuple(sorted(qconfig.layer_quant_config))
    assert audit.resolved_native == tuple(sorted(name for name, scheme in assignment.items() if scheme == "native"))
    ModelQuantizer(qconfig)

    reloaded = QConfig.from_dict(qconfig.to_dict())
    assert audit_qconfig(model, space, candidate, reloaded).assignment_hash == candidate.candidate_id
    reloaded_again = QConfig.from_dict(reloaded.to_dict())
    assert set(reloaded_again.exclude) == set(reloaded.exclude)
    assert audit.qconfig_hash == audit_qconfig(model, space, candidate, reloaded_again).qconfig_hash


def test_empty_global_config_does_not_hide_dynamic_ptpc_activation() -> None:
    qconfig = QConfig(
        global_quant_config=QLayerConfig(),
        layer_quant_config={"layer": get_scheme_config("qwen3", "ptpc_fp8")},
    )
    assert ConfigVerifier(qconfig).is_act_dynamic


def test_assignment_validation_rejects_coverage_capability_and_hash_errors() -> None:
    _, space, candidate = build_inputs()
    entries = list(candidate.assignment)
    assert validate_assignment(space, candidate) == candidate.candidate_id

    with pytest.raises(SchemaValidationError, match="coverage"):
        validate_assignment(space, _candidate(tuple(entries[:-1]), candidate))

    unknown = tuple(sorted(entries + [AssignmentEntry("unknown", "native")], key=lambda entry: entry.module_name))
    with pytest.raises(SchemaValidationError, match="unknown"):
        validate_assignment(space, _candidate(unknown, candidate))

    quant_index = next(index for index, entry in enumerate(entries) if entry.scheme != "native")
    unsupported = entries.copy()
    unsupported[quant_index] = replace(unsupported[quant_index], scheme="mxfp4")
    with pytest.raises(SchemaValidationError, match="unsupported"):
        validate_assignment(space, _candidate(tuple(unsupported), candidate))

    native_index = next(index for index, entry in enumerate(entries) if entry.module_name == "lm_head")
    forced_quant = entries.copy()
    forced_quant[native_index] = replace(forced_quant[native_index], scheme="fp8")
    with pytest.raises(SchemaValidationError, match="must remain native"):
        validate_assignment(space, _candidate(tuple(forced_quant), candidate))

    with pytest.raises(SchemaValidationError, match="candidate_id"):
        validate_assignment(space, replace(candidate, candidate_id="sha256:wrong"))


def test_qconfig_audit_rejects_fallback_patterns_and_wrong_recipe() -> None:
    model, space, candidate = build_inputs()
    qconfig = build_qconfig(space, candidate)
    quantized_name = next(entry.module_name for entry in candidate.assignment if entry.scheme != "native")
    expected_scheme = next(entry.scheme for entry in candidate.assignment if entry.module_name == quantized_name)
    wrong_scheme = "ptpc_fp8" if expected_scheme == "fp8" else "fp8"

    with pytest.raises(SchemaValidationError, match="global_quant_config"):
        audit_qconfig(
            model,
            space,
            candidate,
            replace(qconfig, global_quant_config=get_scheme_config("qwen3", "fp8")),
        )
    with pytest.raises(SchemaValidationError, match="exact names"):
        broad = replace(qconfig, layer_quant_config={"*q_proj": get_scheme_config("qwen3", expected_scheme)})
        audit_qconfig(model, space, candidate, broad)
    with pytest.raises(SchemaValidationError, match="does not match"):
        wrong = replace(
            qconfig,
            layer_quant_config={**qconfig.layer_quant_config, quantized_name: get_scheme_config("qwen3", wrong_scheme)},
        )
        audit_qconfig(model, space, candidate, wrong)
    with pytest.raises(SchemaValidationError, match="not excluded"):
        audit_qconfig(model, space, candidate, replace(qconfig, exclude=[]))
    with pytest.raises(SchemaValidationError, match="layer-type"):
        audit_qconfig(
            model,
            space,
            candidate,
            replace(qconfig, layer_type_quant_config={nn.Linear: get_scheme_config("qwen3", "fp8")}),
        )


def test_qconfig_audit_rejects_model_binding_changes() -> None:
    model, space, candidate = build_inputs()
    qconfig = build_qconfig(space, candidate)

    with pytest.raises(SchemaValidationError, match="topology or storage aliases"):
        changed = TinyQwen3()
        changed.extra = nn.Linear(8, 8, bias=False).to(dtype=torch.bfloat16)
        audit_qconfig(changed, space, candidate, qconfig)
    with pytest.raises(SchemaValidationError, match="topology or storage aliases"):
        audit_qconfig(TinyQwen3(width=4), space, candidate, qconfig)

    aliased = TinyQwen3()
    aliased.model.layers[0].k_proj.weight = aliased.model.layers[0].q_proj.weight
    strategy = make_strategy()
    alias_space = build_decision_space(aliased, strategy)
    untied = TinyQwen3()
    untied.load_state_dict(aliased.state_dict())
    with pytest.raises(SchemaValidationError, match="topology or storage aliases"):
        _validate_model_binding(untied, alias_space)


def test_qconfig_audit_value_validation() -> None:
    with pytest.raises(SchemaValidationError, match="hashes"):
        QConfigAudit("bad", "sha256:qconfig", (), ())
