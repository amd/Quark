#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from quark.experimental.torch.quant_perf.evaluation.throughput import ThroughputMeasurement
from quark.experimental.torch.quant_perf.perfopt.backend_probe import (
    BackendMicroCase,
    basic_output_valid,
    decide_micro_screen,
    decide_probe_backend,
)
from quark.experimental.torch.quant_perf.perfopt.backend_probe_inputs import (
    backend_probe_input_support_reason,
    load_backend_probe_inputs,
    load_gemma4_probe_inputs,
)


def _measurement(tps: float, mad: float = 0.002) -> ThroughputMeasurement:
    return ThroughputMeasurement(
        samples_tps=(tps, tps, tps),
        median_tps=tps,
        mad_tps=tps * mad,
        relative_mad=mad,
        warmup_tps=tps,
        stable=True,
    )


def test_micro_screen_requires_gain_and_rejects_single_shape_regression():
    passing = [
        BackendMicroCase("gate_up_m32", 100.0, 97.0, True),
        BackendMicroCase("gate_up_m64", 100.0, 98.0, True),
        BackendMicroCase("down_m32", 100.0, 98.0, True),
        BackendMicroCase("down_m64", 100.0, 98.0, True),
    ]
    regressing = [
        *passing[:-1],
        BackendMicroCase("down_m64", 100.0, 106.0, True),
    ]

    assert decide_micro_screen(passing).passed is True
    assert decide_micro_screen(regressing).passed is False
    assert decide_micro_screen(regressing).reason == "shape_regression"


def test_micro_correctness_only_requires_matching_finite_outputs():
    flydsl = torch.tensor([[1.0, 2.0]])
    asm = torch.tensor([[1.2, 1.8]])
    nan = torch.tensor([[float("nan"), 1.0]])

    assert basic_output_valid(flydsl, asm) is True
    assert basic_output_valid(flydsl, nan) is False
    assert basic_output_valid(flydsl, torch.ones(2, 2)) is False


def test_probe_selects_asm_only_after_accuracy_and_abba_gain():
    result = decide_probe_backend(
        micro_passed=True,
        accuracy_passed=True,
        anchor_first=_measurement(1000.0),
        candidate_first=_measurement(1020.0),
        candidate_second=_measurement(1022.0),
        anchor_second=_measurement(1001.0),
        keep_floor=0.01,
    )

    assert result.selected_backend == "asm"
    assert result.reason == "confirmed_e2e_gain"
    assert result.multiplier > 1.01


def test_probe_falls_back_to_flydsl_on_noise_or_accuracy_failure():
    noisy = decide_probe_backend(
        micro_passed=True,
        accuracy_passed=True,
        anchor_first=_measurement(1000.0),
        candidate_first=_measurement(1005.0),
        candidate_second=_measurement(1004.0),
        anchor_second=_measurement(1001.0),
        keep_floor=0.01,
    )
    inaccurate = decide_probe_backend(
        micro_passed=True,
        accuracy_passed=False,
        anchor_first=_measurement(1000.0),
        candidate_first=_measurement(1100.0),
        candidate_second=_measurement(1100.0),
        anchor_second=_measurement(1000.0),
        keep_floor=0.01,
    )

    assert noisy.selected_backend == "flydsl"
    assert noisy.reason == "within_measurement_noise"
    assert inaccurate.selected_backend == "flydsl"
    assert inaccurate.reason == "accuracy_failed"


def test_load_gemma4_probe_inputs_merges_gate_and_up(tmp_path):
    model = tmp_path / "quant"
    model.mkdir()
    save_file(
        {
            "model.language_model.layers.0.mlp.gate_proj.weight": (torch.ones((3, 2), dtype=torch.uint8)),
            "model.language_model.layers.0.mlp.gate_proj.weight_scale": (torch.ones((3, 1), dtype=torch.uint8)),
            "model.language_model.layers.0.mlp.up_proj.weight": (torch.full((3, 2), 2, dtype=torch.uint8)),
            "model.language_model.layers.0.mlp.up_proj.weight_scale": (torch.full((3, 1), 2, dtype=torch.uint8)),
            "model.language_model.layers.0.mlp.down_proj.weight": (torch.ones((2, 3), dtype=torch.uint8)),
            "model.language_model.layers.0.mlp.down_proj.weight_scale": (torch.ones((2, 1), dtype=torch.uint8)),
        },
        model / "model.safetensors",
    )

    inputs = load_gemma4_probe_inputs(model)
    weights = inputs.projections

    assert tuple(weights["gate_up"]["weight"].shape) == (6, 2)
    assert tuple(weights["gate_up"]["scale"].shape) == (6, 1)
    assert tuple(weights["down"]["weight"].shape) == (2, 3)


def test_backend_probe_inputs_fail_explicitly_for_unsupported_architecture(
    tmp_path,
):
    reason = backend_probe_input_support_reason("qwen3_5_moe")

    assert reason == "unsupported_model_arch:qwen3_5_moe"
    with pytest.raises(RuntimeError, match="unsupported_model_arch"):
        load_backend_probe_inputs("qwen3_5_moe", tmp_path)
