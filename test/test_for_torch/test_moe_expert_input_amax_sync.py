#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from unittest.mock import patch

import torch
import torch.nn as nn

from quark.common.utils.testing_utils import capture_quark_logs
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Int8PerTensorSpec, QConfig, QLayerConfig
from quark.torch.quantization.utils import (
    _compute_observer_amax_from_quantizer,
    sync_moe_expert_input_quantizer_qparams,
)


class DummyObserver:
    """Observer stub with configurable min/max tensors and qparams behavior."""

    def __init__(
        self,
        min_value: torch.Tensor | None = None,
        max_value: torch.Tensor | None = None,
        return_qparams: bool = True,
    ) -> None:
        if min_value is not None:
            self.min_val = min_value
        if max_value is not None:
            self.max_val = max_value
        self.return_qparams = return_qparams

    def _calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.return_qparams:
            return None

        minimum_tensor = getattr(self, "min_val", None)
        maximum_tensor = getattr(self, "max_val", None)
        if minimum_tensor is None or maximum_tensor is None:
            return None

        return maximum_tensor.clone(), minimum_tensor.clone()


class DummyQuantizer(nn.Module):
    """Quantizer stub that mimics the buffers updated during MoE synchronization."""

    def __init__(self, observer: DummyObserver | None = None, symmetric: bool | None = True) -> None:
        super().__init__()
        self.observer = observer
        self.symmetric = symmetric
        self.scale = torch.tensor([1.0], dtype=torch.float32)
        self.zero_point = torch.tensor([0.0], dtype=torch.float32)
        self.updated_buffers: dict[str, torch.Tensor] = {}

    def update_buffer(self, name: str, tensor_value: torch.Tensor, device: torch.device) -> None:
        updated_tensor = tensor_value.to(device)
        setattr(self, name, updated_tensor)
        self.updated_buffers[name] = updated_tensor


class DummyProjection(nn.Module):
    """Projection wrapper that optionally exposes an input quantizer."""

    def __init__(self, input_quantizer: nn.Module | None = None) -> None:
        super().__init__()
        if input_quantizer is not None:
            self._input_quantizer = input_quantizer


class DummyExpert(nn.Module):
    """MoE expert stub with the projection names expected by the sync helper."""

    def __init__(self, gate_input_quantizer: nn.Module, down_input_quantizer: nn.Module | None = None) -> None:
        super().__init__()
        self.gate_proj = DummyProjection(gate_input_quantizer)
        self.up_proj = DummyProjection()
        self.down_proj = DummyProjection(down_input_quantizer)


class DummyExpandedNameExpert(nn.Module):
    """MoE expert stub that exposes alternative projection names such as ``w1``."""

    def __init__(self, w1_input_quantizer: nn.Module) -> None:
        super().__init__()
        self.w1 = DummyProjection(w1_input_quantizer)


class DummyMlp(nn.Module):
    """MLP stub that exposes experts under ``mlp.experts``."""

    def __init__(self, experts: list[DummyExpert]) -> None:
        super().__init__()
        self.experts = nn.ModuleList(experts)


class DummyLayer(nn.Module):
    """Layer stub that matches the MoE naming convention used by Quark."""

    def __init__(self, experts: list[DummyExpert]) -> None:
        super().__init__()
        self.mlp = DummyMlp(experts)


class DummyInnerModel(nn.Module):
    """Inner model stub that exposes ``layers`` for the regex prefix."""

    def __init__(self, experts: list[DummyExpert]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(experts)])


class DummyMoeModel(nn.Module):
    """Top-level model stub for exercising MoE quantizer synchronization."""

    def __init__(self, experts: list[DummyExpert]) -> None:
        super().__init__()
        self.model = DummyInnerModel(experts)
        self.non_moe_quantizer = DummyQuantizer(
            DummyObserver(
                min_value=torch.tensor([-1.0], dtype=torch.float32),
                max_value=torch.tensor([1.0], dtype=torch.float32),
            )
        )


def test_compute_observer_amax_from_quantizer_handles_invalid_states() -> None:
    """Verify that invalid observer states are ignored and valid ranges return max absolute value."""
    no_observer_module = nn.Linear(2, 2)
    assert _compute_observer_amax_from_quantizer(no_observer_module) is None

    missing_minimum_module = DummyQuantizer(DummyObserver(max_value=torch.tensor([3.0], dtype=torch.float32)))
    assert _compute_observer_amax_from_quantizer(missing_minimum_module) is None

    non_scalar_range_module = DummyQuantizer(
        DummyObserver(
            min_value=torch.tensor([-2.0, -1.0], dtype=torch.float32),
            max_value=torch.tensor([1.0, 4.0], dtype=torch.float32),
        )
    )
    assert _compute_observer_amax_from_quantizer(non_scalar_range_module) is None

    uninitialized_observer_module = DummyQuantizer(
        DummyObserver(
            min_value=torch.tensor(float("inf")),
            max_value=torch.tensor(float("-inf")),
        )
    )
    assert _compute_observer_amax_from_quantizer(uninitialized_observer_module) is None

    valid_module = DummyQuantizer(
        DummyObserver(
            min_value=torch.tensor([-5.0], dtype=torch.float32),
            max_value=torch.tensor([3.0], dtype=torch.float32),
        )
    )
    assert _compute_observer_amax_from_quantizer(valid_module) == 5.0


def test_sync_moe_expert_input_quantizer_qparams_synchronizes_each_substage() -> None:
    """Verify that MoE expert input synchronization groups quantizers by projection and substage."""
    expert_zero = DummyExpert(
        gate_input_quantizer=nn.ModuleList(
            [
                DummyQuantizer(
                    DummyObserver(
                        min_value=torch.tensor([-2.0], dtype=torch.float32),
                        max_value=torch.tensor([2.0], dtype=torch.float32),
                    )
                ),
                DummyQuantizer(
                    DummyObserver(
                        min_value=torch.tensor([-7.0], dtype=torch.float32),
                        max_value=torch.tensor([7.0], dtype=torch.float32),
                    )
                ),
            ]
        ),
        down_input_quantizer=DummyQuantizer(DummyObserver(max_value=torch.tensor([4.0], dtype=torch.float32))),
    )
    expert_one = DummyExpert(
        gate_input_quantizer=nn.ModuleList(
            [
                DummyQuantizer(
                    DummyObserver(
                        min_value=torch.tensor([-5.0], dtype=torch.float32),
                        max_value=torch.tensor([5.0], dtype=torch.float32),
                    )
                ),
                DummyQuantizer(
                    DummyObserver(
                        min_value=torch.tensor([-3.0], dtype=torch.float32),
                        max_value=torch.tensor([3.0], dtype=torch.float32),
                    )
                ),
            ]
        )
    )
    expert_two = DummyExpert(gate_input_quantizer=nn.ModuleList([DummyQuantizer(observer=None)]))
    expert_three = DummyExpert(
        gate_input_quantizer=nn.ModuleList(
            [DummyQuantizer(DummyObserver(min_value=torch.tensor([-4.0], dtype=torch.float32)))]
        )
    )
    expert_four = DummyExpert(
        gate_input_quantizer=nn.ModuleList(
            [
                DummyQuantizer(
                    DummyObserver(
                        min_value=torch.tensor([-6.0], dtype=torch.float32),
                        max_value=torch.tensor([6.0], dtype=torch.float32),
                        return_qparams=False,
                    )
                )
            ]
        )
    )
    model = DummyMoeModel([expert_zero, expert_one, expert_two, expert_three, expert_four])

    synchronized_module_count = sync_moe_expert_input_quantizer_qparams(model)

    expert_zero_stage_zero = expert_zero.gate_proj._input_quantizer[0]
    expert_one_stage_zero = expert_one.gate_proj._input_quantizer[0]
    expert_zero_stage_one = expert_zero.gate_proj._input_quantizer[1]
    expert_one_stage_one = expert_one.gate_proj._input_quantizer[1]
    qparams_none_quantizer = expert_four.gate_proj._input_quantizer[0]
    unmatched_down_quantizer = expert_zero.down_proj._input_quantizer

    assert synchronized_module_count == 4

    assert torch.allclose(expert_zero_stage_zero.observer.min_val, torch.tensor([-6.0], dtype=torch.float32))
    assert torch.allclose(expert_zero_stage_zero.observer.max_val, torch.tensor([6.0], dtype=torch.float32))
    assert torch.allclose(expert_one_stage_zero.observer.min_val, torch.tensor([-6.0], dtype=torch.float32))
    assert torch.allclose(expert_one_stage_zero.observer.max_val, torch.tensor([6.0], dtype=torch.float32))
    assert torch.allclose(expert_zero_stage_one.observer.min_val, torch.tensor([-7.0], dtype=torch.float32))
    assert torch.allclose(expert_zero_stage_one.observer.max_val, torch.tensor([7.0], dtype=torch.float32))
    assert torch.allclose(expert_one_stage_one.observer.min_val, torch.tensor([-7.0], dtype=torch.float32))
    assert torch.allclose(expert_one_stage_one.observer.max_val, torch.tensor([7.0], dtype=torch.float32))

    assert torch.allclose(expert_zero_stage_zero.scale, torch.tensor([6.0], dtype=torch.float32))
    assert torch.allclose(expert_zero_stage_zero.zero_point, torch.tensor([-6.0], dtype=torch.float32))
    assert torch.allclose(expert_zero_stage_one.scale, torch.tensor([7.0], dtype=torch.float32))
    assert torch.allclose(expert_zero_stage_one.zero_point, torch.tensor([-7.0], dtype=torch.float32))

    assert qparams_none_quantizer.updated_buffers == {}
    assert torch.allclose(qparams_none_quantizer.scale, torch.tensor([1.0], dtype=torch.float32))
    assert torch.allclose(unmatched_down_quantizer.scale, torch.tensor([1.0], dtype=torch.float32))


def test_sync_moe_expert_input_quantizer_qparams_supports_expanded_projection_names_with_substages() -> None:
    """Verify that expanded expert projection names still preserve substage synchronization."""
    expert_zero = DummyExpandedNameExpert(
        w1_input_quantizer=nn.ModuleList(
            [
                DummyQuantizer(
                    DummyObserver(
                        min_value=torch.tensor([-2.0], dtype=torch.float32),
                        max_value=torch.tensor([2.0], dtype=torch.float32),
                    )
                ),
                DummyQuantizer(
                    DummyObserver(
                        min_value=torch.tensor([-3.0], dtype=torch.float32),
                        max_value=torch.tensor([3.0], dtype=torch.float32),
                    )
                ),
            ]
        )
    )
    expert_one = DummyExpandedNameExpert(
        w1_input_quantizer=nn.ModuleList(
            [
                DummyQuantizer(
                    DummyObserver(
                        min_value=torch.tensor([-5.0], dtype=torch.float32),
                        max_value=torch.tensor([5.0], dtype=torch.float32),
                    )
                ),
                DummyQuantizer(
                    DummyObserver(
                        min_value=torch.tensor([-7.0], dtype=torch.float32),
                        max_value=torch.tensor([7.0], dtype=torch.float32),
                    )
                ),
            ]
        )
    )
    model = DummyMoeModel([expert_zero, expert_one])

    synchronized_module_count = sync_moe_expert_input_quantizer_qparams(model)

    expert_zero_stage_zero = expert_zero.w1._input_quantizer[0]
    expert_one_stage_zero = expert_one.w1._input_quantizer[0]
    expert_zero_stage_one = expert_zero.w1._input_quantizer[1]
    expert_one_stage_one = expert_one.w1._input_quantizer[1]

    assert synchronized_module_count == 4
    assert torch.allclose(expert_zero_stage_zero.scale, torch.tensor([5.0], dtype=torch.float32))
    assert torch.allclose(expert_one_stage_zero.scale, torch.tensor([5.0], dtype=torch.float32))
    assert torch.allclose(expert_zero_stage_one.scale, torch.tensor([7.0], dtype=torch.float32))
    assert torch.allclose(expert_one_stage_one.scale, torch.tensor([7.0], dtype=torch.float32))
    assert torch.allclose(expert_zero_stage_zero.zero_point, torch.tensor([-5.0], dtype=torch.float32))
    assert torch.allclose(expert_zero_stage_one.zero_point, torch.tensor([-7.0], dtype=torch.float32))


def test_sync_moe_expert_input_quantizer_qparams_skips_asymmetric_quantizers() -> None:
    """Verify that asymmetric MoE input quantizers are left unchanged during synchronization."""
    symmetric_quantizer = DummyQuantizer(
        DummyObserver(
            min_value=torch.tensor([-2.0], dtype=torch.float32),
            max_value=torch.tensor([2.0], dtype=torch.float32),
        )
    )
    asymmetric_quantizer = DummyQuantizer(
        DummyObserver(
            min_value=torch.tensor([-6.0], dtype=torch.float32),
            max_value=torch.tensor([4.0], dtype=torch.float32),
        ),
        symmetric=False,
    )
    model = DummyMoeModel(
        [
            DummyExpert(gate_input_quantizer=symmetric_quantizer),
            DummyExpert(gate_input_quantizer=asymmetric_quantizer),
        ]
    )

    with capture_quark_logs() as captured_output:
        synchronized_module_count = sync_moe_expert_input_quantizer_qparams(model)

    output_content = captured_output.getvalue()

    assert synchronized_module_count == 1
    assert torch.allclose(symmetric_quantizer.observer.min_val, torch.tensor([-2.0], dtype=torch.float32))
    assert torch.allclose(symmetric_quantizer.observer.max_val, torch.tensor([2.0], dtype=torch.float32))
    assert torch.allclose(asymmetric_quantizer.observer.min_val, torch.tensor([-6.0], dtype=torch.float32))
    assert torch.allclose(asymmetric_quantizer.observer.max_val, torch.tensor([4.0], dtype=torch.float32))
    assert torch.allclose(asymmetric_quantizer.scale, torch.tensor([1.0], dtype=torch.float32))
    assert torch.allclose(asymmetric_quantizer.zero_point, torch.tensor([0.0], dtype=torch.float32))
    assert asymmetric_quantizer.updated_buffers == {}
    assert "Skipped MoE expert input amax synchronization for 1 asymmetric quantizer(s):" in output_content
    assert "model.layers.0.mlp.experts.1.gate_proj._input_quantizer" in output_content


def test_do_post_calib_optimization_logs_synchronized_quantizer_count() -> None:
    """Verify that eager post-calibration optimization reports synchronized MoE quantizer count."""
    quantization_spec = Int8PerTensorSpec(is_dynamic=False).to_quantization_spec()
    quantization_config = QConfig(
        global_quant_config=QLayerConfig(weight=quantization_spec),
        sync_moe_expert_input_amax=True,
    )
    quantizer = ModelQuantizer(quantization_config)
    model = nn.Linear(2, 2)

    with (
        patch("builtins.breakpoint"),
        patch("quark.torch.quantization.api.sync_moe_expert_input_quantizer_qparams", return_value=3),
        capture_quark_logs() as captured_output,
    ):
        optimized_model = quantizer._do_post_calib_optimization(model)

    output_content = captured_output.getvalue()

    assert optimized_model is model
    assert "Synchronized MoE expert input amax across 3 quantizers." in output_content
