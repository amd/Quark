#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for the shared diffusers <-> Quark integration helpers.

Covers ``qconfig_needs_activation_calibration`` and
``quantize_diffusion_model_in_place`` from
``quark.torch.utils.diffusers.integration``.  These back both the in-tree
``quark.integrations.diffusers`` quantizer and the upstream
``diffusers.quantizers.quark`` quantizer, so the contract is locked here
independently of either plugin.
"""

import pytest
import torch
import torch.nn as nn

from quark.torch.quantization.config.config import (
    Int8PerTensorSpec,
    QConfig,
    QLayerConfig,
)
from quark.torch.quantization.nn.modules.mixin import QuantMixin
from quark.torch.utils.diffusers import (
    qconfig_needs_activation_calibration,
    quantize_diffusion_model_in_place,
)

_INT8_SPEC = Int8PerTensorSpec(
    observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
).to_quantization_spec()

_INT8_DYNAMIC_ACT_SPEC = Int8PerTensorSpec(
    observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=True
).to_quantization_spec()


def _weight_only_config() -> QConfig:
    return QConfig(global_quant_config=QLayerConfig(weight=_INT8_SPEC))


def _w8a8_config() -> QConfig:
    """Static activation quant (needs calibration)."""
    return QConfig(global_quant_config=QLayerConfig(weight=_INT8_SPEC, input_tensors=_INT8_SPEC))


def _dynamic_w8a8_config() -> QConfig:
    """Dynamic activation quant (no calibration needed -- xDiT-style)."""
    return QConfig(global_quant_config=QLayerConfig(weight=_INT8_SPEC, input_tensors=_INT8_DYNAMIC_ACT_SPEC))


class _TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_weight_only_config_does_not_need_calibration() -> None:
    """A QConfig with only a weight quantizer needs no activation calibration."""
    assert qconfig_needs_activation_calibration(_weight_only_config()) is False


def test_static_input_quantizer_needs_calibration() -> None:
    """A static input (activation) quantizer needs calibration."""
    assert qconfig_needs_activation_calibration(_w8a8_config()) is True


def test_static_output_quantizer_needs_calibration() -> None:
    """A static output quantizer needs calibration."""
    qconfig = QConfig(global_quant_config=QLayerConfig(weight=_INT8_SPEC, output_tensors=_INT8_SPEC))
    assert qconfig_needs_activation_calibration(qconfig) is True


def test_dynamic_activation_does_not_need_calibration() -> None:
    """Dynamic activation quant computes scales at runtime -- no calibration (xDiT-style)."""
    assert qconfig_needs_activation_calibration(_dynamic_w8a8_config()) is False


def test_none_layer_config_is_skipped() -> None:
    """``None`` layer-config entries are skipped (e.g. an unset global config)."""
    assert qconfig_needs_activation_calibration(QConfig(global_quant_config=None)) is False


def test_quantize_in_place_weight_only_succeeds() -> None:
    """Weight-only on-the-fly quantization replaces linears with quant modules."""
    model = _TinyLinearModel()
    quantize_diffusion_model_in_place(model, _weight_only_config())

    assert any(isinstance(m, QuantMixin) for m in model.modules()), (
        "weight-only on-the-fly quantization should produce QuantMixin modules"
    )

    # Frozen + inference-ready: a forward pass must still run and stay finite.
    with torch.no_grad():
        out = model(torch.randn(2, 16))
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_quantize_in_place_dynamic_activation_succeeds() -> None:
    """Dynamic-activation on-the-fly quantization (xDiT-style) needs no calibration."""
    model = _TinyLinearModel()
    quantize_diffusion_model_in_place(model, _dynamic_w8a8_config())

    assert any(isinstance(m, QuantMixin) for m in model.modules()), (
        "dynamic-activation on-the-fly quantization should produce QuantMixin modules"
    )

    with torch.no_grad():
        out = model(torch.randn(2, 16))
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_quantize_in_place_static_activation_config_raises() -> None:
    """Static activation-quantized configs are rejected with an actionable message."""
    model = _TinyLinearModel()
    with pytest.raises(NotImplementedError, match="static activation"):
        quantize_diffusion_model_in_place(model, _w8a8_config())
