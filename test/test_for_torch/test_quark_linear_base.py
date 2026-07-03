#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Tests for QuarkLinearBase, quark_linear_serialization, and the updated QParamsLinear hierarchy."""

from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.export.nn.modules.quark_linear_base import QuarkLinearBase
from quark.torch.export.utils import apply_export_state_dict_mappings, fix_loaded_state_dict_mismatch
from quark.torch.quantization.config.config import QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FP8_PER_TENSOR_SPEC = QTensorConfig(
    dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
)


# ---------------------------------------------------------------------------
# QuarkLinearBase tests
# ---------------------------------------------------------------------------


class ConcreteQuarkLinear(nn.Linear, QuarkLinearBase):
    """Minimal concrete subclass for testing the ABC."""

    def __init__(self, linear: nn.Linear, custom_mode: str = "quark", **kwargs: Any):
        super().__init__(
            linear.in_features, linear.out_features, bias=linear.bias is not None, device=linear.weight.device
        )
        self._custom_mode = custom_mode
        self.weight_quantizer = None
        self.input_quantizer = None
        self.output_quantizer = None
        self.bias_quantizer = None
        with torch.no_grad():
            self.weight.copy_(linear.weight)
            if linear.bias is not None:
                self.bias.copy_(linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


def test_quark_linear_base_is_abstract():
    """QuarkLinearBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        QuarkLinearBase()  # type: ignore[abstract]


def test_from_module_creates_instance():
    """from_module class method builds a ConcreteQuarkLinear from nn.Linear."""
    linear = nn.Linear(16, 32, bias=True, device=DEVICE)
    instance = ConcreteQuarkLinear.from_module(linear, custom_mode="quark")
    assert isinstance(instance, ConcreteQuarkLinear)
    assert isinstance(instance, QuarkLinearBase)
    assert instance.in_features == 16
    assert instance.out_features == 32
    assert instance.bias is not None


def test_from_module_no_bias():
    """from_module works for linear layers without bias."""
    linear = nn.Linear(8, 4, bias=False, device=DEVICE)
    instance = ConcreteQuarkLinear.from_module(linear)
    assert instance.bias is None


def test_preprocess_postprocess_weight_noop():
    """Default preprocess/postprocess_weight are no-ops."""
    linear = nn.Linear(4, 4, device=DEVICE)
    instance = ConcreteQuarkLinear.from_module(linear)
    instance.preprocess_weight()
    instance.postprocess_weight()


def test_forward_contract():
    """Concrete subclass forward produces expected output."""
    linear = nn.Linear(8, 4, bias=True, device=DEVICE)
    instance = ConcreteQuarkLinear.from_module(linear)
    x = torch.randn(2, 8, device=DEVICE)
    out = instance(x)
    assert out.shape == (2, 4)


def test_native_inference_enabled_default():
    """_native_inference_enabled class variable defaults to False."""
    assert not QuarkLinearBase._native_inference_enabled
    linear = nn.Linear(4, 4, device=DEVICE)
    instance = ConcreteQuarkLinear.from_module(linear)
    assert not instance._native_inference_enabled


# ---------------------------------------------------------------------------
# QParamsLinear inherits from QuarkLinearBase
# ---------------------------------------------------------------------------


def test_qparamslinear_is_quark_linear_base():
    """QParamsLinear inherits from QuarkLinearBase."""
    assert issubclass(QParamsLinear, QuarkLinearBase)


def test_qparamslinear_from_module():
    """QParamsLinear.from_module works via inherited QuarkLinearBase.from_module."""
    linear = nn.Linear(16, 32, bias=True, dtype=torch.float32, device=DEVICE)
    quant_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    qpl = QParamsLinear.from_module(linear, custom_mode="fp8", pack_method=None, quant_config=quant_config)
    assert isinstance(qpl, QParamsLinear)
    assert isinstance(qpl, QuarkLinearBase)
    assert qpl.in_features == 16
    assert qpl.out_features == 32


def test_qparamslinear_preprocess_postprocess_noop():
    """Default preprocess/postprocess_weight on QParamsLinear are no-ops (inherited)."""
    linear = nn.Linear(8, 4, dtype=torch.float32, device=DEVICE)
    quant_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    qpl = QParamsLinear.from_module(linear, custom_mode="fp8", pack_method=None, quant_config=quant_config)
    original_bytes = qpl.weight.data.view(torch.int8).clone()
    qpl.preprocess_weight()
    assert torch.equal(qpl.weight.data.view(torch.int8), original_bytes)
    qpl.postprocess_weight()
    assert torch.equal(qpl.weight.data.view(torch.int8), original_bytes)


# ---------------------------------------------------------------------------
# quark_linear_serialization tests
# ---------------------------------------------------------------------------


def test_apply_export_state_dict_mappings_no_override():
    """When QPARAMSLINEAR_OVERRIDES_STATE_DICT is False, keys pass through unchanged."""
    module = Mock()
    module.weight_quantizer = None

    destination = {"prefix.weight_quantizer.scale": torch.tensor(1.0), "prefix.weight": torch.randn(4, 4)}

    with patch("quark.torch.export.utils.QPARAMSLINEAR_OVERRIDES_STATE_DICT", False):
        apply_export_state_dict_mappings(module, destination, prefix="prefix.")

    assert "prefix.weight_quantizer.scale" in destination
    assert "prefix.weight" in destination


def test_apply_export_state_dict_mappings_with_override():
    """When QPARAMSLINEAR_OVERRIDES_STATE_DICT is True, keys are remapped."""
    module = Mock()
    module._custom_mode = "fp8"
    module.weight_quantizer = None

    scale_val = torch.tensor(1.0)
    destination = {"prefix.weight_quantizer.scale": scale_val, "prefix.weight": torch.randn(4, 4)}

    with (
        patch("quark.torch.export.utils.QPARAMSLINEAR_OVERRIDES_STATE_DICT", True),
        patch(
            "quark.torch.export.utils._fix_state_dict_key_on_save",
            side_effect=lambda k: (k.replace("weight_quantizer.scale", "weight_scale"), True),
        ),
    ):
        apply_export_state_dict_mappings(module, destination, prefix="prefix.")

    assert "prefix.weight_scale" in destination
    assert destination["prefix.weight_scale"] is scale_val


def test_apply_export_state_dict_mappings_awq_mode():
    """AWQ mode remaps keys through AWQ_SAVE_MAP."""
    module = Mock()
    module._custom_mode = "awq"
    module.weight_quantizer = None

    weight_val = torch.randn(4, 4)
    destination = {"prefix.weight": weight_val}

    with (
        patch("quark.torch.export.utils.QPARAMSLINEAR_OVERRIDES_STATE_DICT", True),
        patch(
            "quark.torch.export.utils._fix_state_dict_key_on_save",
            side_effect=lambda k: (k, True),
        ),
        patch(
            "quark.torch.export.utils.AWQ_SAVE_MAP",
            {"weight": "qweight"},
        ),
    ):
        apply_export_state_dict_mappings(module, destination, prefix="prefix.")

    assert "prefix.qweight" in destination
    assert "prefix.weight" not in destination


def test_apply_export_state_dict_mappings_mx_fp4():
    """MX export with fp4 element dtype splits weight into scale + payload."""
    module = Mock()
    module.weight_quantizer = Mock()
    module.weight_quantizer.qspec.dtype.value = "mx"
    module.weight_quantizer.qspec.mx_element_dtype.value = "fp4"
    module.weight = torch.randn(4, 17)

    destination: dict[str, torch.Tensor] = {"prefix.weight": module.weight.clone()}

    with patch("quark.torch.export.utils.QPARAMSLINEAR_OVERRIDES_STATE_DICT", False):
        apply_export_state_dict_mappings(module, destination, prefix="prefix.")

    assert "prefix.weight" in destination
    assert "prefix.weight_scale" in destination
    assert destination["prefix.weight"].shape == (4, 16)
    assert destination["prefix.weight_scale"].dtype == torch.uint8


def test_apply_export_state_dict_mappings_mx_non_fp4():
    """MX export with non-fp4 element dtype uses reshape_shape=25."""
    module = Mock()
    module.weight_quantizer = Mock()
    module.weight_quantizer.qspec.dtype.value = "mx"
    module.weight_quantizer.qspec.mx_element_dtype.value = "int8"
    module.weight = torch.randn(4, 25)

    destination: dict[str, torch.Tensor] = {"prefix.weight": module.weight.clone()}

    with patch("quark.torch.export.utils.QPARAMSLINEAR_OVERRIDES_STATE_DICT", False):
        apply_export_state_dict_mappings(module, destination, prefix="prefix.")

    assert "prefix.weight" in destination
    assert "prefix.weight_scale" in destination
    assert destination["prefix.weight"].shape == (4, 24)
    assert destination["prefix.weight_scale"].dtype == torch.uint8


def test_fix_loaded_state_dict_mismatch_no_override():
    """When QPARAMSLINEAR_OVERRIDES_STATE_DICT is False, state_dict passes through."""
    module = Mock()
    state_dict = {"layer.weight_scale": torch.tensor(1.0)}

    with patch("quark.torch.export.utils.QPARAMSLINEAR_OVERRIDES_STATE_DICT", False):
        result = fix_loaded_state_dict_mismatch(module, state_dict, prefix="layer.")

    assert result is state_dict
    assert "layer.weight_scale" in result


def test_fix_loaded_state_dict_mismatch_with_override():
    """When QPARAMSLINEAR_OVERRIDES_STATE_DICT is True, delegates to _fix_loaded_weights_key_mismatch."""
    module = Mock()
    module._custom_mode = "fp8"

    original_dict = {"layer.weight_scale": torch.tensor(1.0)}
    remapped_dict = {"layer.weight_quantizer.scale": torch.tensor(1.0)}

    with (
        patch("quark.torch.export.utils.QPARAMSLINEAR_OVERRIDES_STATE_DICT", True),
        patch(
            "quark.torch.export.utils._fix_loaded_weights_key_mismatch",
            return_value=remapped_dict,
        ) as mock_fix,
    ):
        result = fix_loaded_state_dict_mismatch(module, original_dict, prefix="layer.")

    mock_fix.assert_called_once_with(original_dict, weight_format="real_quantized", custom_mode="fp8")
    assert "layer.weight_quantizer.scale" in result


def test_fix_loaded_state_dict_mismatch_awq_mode():
    """AWQ mode remaps keys through AWQ_LOAD_MAP."""
    module = Mock()
    module._custom_mode = "awq"

    state_dict = {"layer.scales": torch.tensor(1.0)}

    with (
        patch("quark.torch.export.utils.QPARAMSLINEAR_OVERRIDES_STATE_DICT", True),
        patch(
            "quark.torch.export.utils._fix_loaded_weights_key_mismatch",
            return_value=state_dict,
        ),
        patch(
            "quark.torch.export.utils.AWQ_LOAD_MAP",
            {"scales": "weight_quantizer.scale"},
        ),
    ):
        result = fix_loaded_state_dict_mismatch(module, state_dict, prefix="layer.")

    assert "layer.weight_quantizer.scale" in result
    assert "layer.scales" not in result


# ---------------------------------------------------------------------------
# QuarkLinearBase state_dict / _load_from_state_dict (moved from QParamsLinear)
# ---------------------------------------------------------------------------


def test_base_state_dict_calls_postprocess_preprocess():
    """QuarkLinearBase.state_dict wraps with postprocess_weight / preprocess_weight."""
    linear = nn.Linear(8, 4, device=DEVICE)
    instance = ConcreteQuarkLinear.from_module(linear)

    call_order: list[str] = []
    original_postprocess = instance.postprocess_weight
    original_preprocess = instance.preprocess_weight

    def tracking_postprocess():
        call_order.append("postprocess")
        return original_postprocess()

    def tracking_preprocess():
        call_order.append("preprocess")
        return original_preprocess()

    instance.postprocess_weight = tracking_postprocess
    instance.preprocess_weight = tracking_preprocess

    instance.state_dict()

    assert call_order == ["postprocess", "preprocess"]


def test_base_state_dict_preprocess_called_on_error():
    """preprocess_weight is called even if the parent state_dict raises."""
    linear = nn.Linear(8, 4, device=DEVICE)
    instance = ConcreteQuarkLinear.from_module(linear)

    preprocess_called: list[bool] = []
    original_preprocess = instance.preprocess_weight

    def tracking_preprocess():
        preprocess_called.append(True)
        return original_preprocess()

    instance.preprocess_weight = tracking_preprocess

    with (
        patch.object(nn.Module, "state_dict", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        instance.state_dict()

    assert preprocess_called, "preprocess_weight should be called in the finally block"


def test_base_state_dict_returns_weight_and_bias():
    """QuarkLinearBase.state_dict includes weight (and bias when present)."""
    linear = nn.Linear(8, 4, bias=True, device=DEVICE)
    instance = ConcreteQuarkLinear.from_module(linear)

    sd = instance.state_dict()
    assert "weight" in sd
    assert "bias" in sd
    assert torch.equal(sd["weight"], instance.weight.data)
    assert torch.equal(sd["bias"], instance.bias.data)


def test_base_state_dict_no_bias():
    """state_dict omits bias key when the layer has no bias."""
    linear = nn.Linear(8, 4, bias=False, device=DEVICE)
    instance = ConcreteQuarkLinear.from_module(linear)

    sd = instance.state_dict()
    assert "weight" in sd
    assert "bias" not in sd


def test_base_state_dict_calls_apply_export_mappings():
    """state_dict delegates to apply_export_state_dict_mappings."""
    linear = nn.Linear(4, 4, device=DEVICE)
    instance = ConcreteQuarkLinear.from_module(linear)

    with patch("quark.torch.export.nn.modules.quark_linear_base.apply_export_state_dict_mappings") as mock_apply:
        sd = instance.state_dict()
        mock_apply.assert_called_once_with(instance, sd, "")


def test_base_state_dict_destination_param():
    """When destination dict is provided, state_dict merges into it."""
    linear = nn.Linear(4, 4, bias=False, device=DEVICE)
    instance = ConcreteQuarkLinear.from_module(linear)

    dest: dict[str, torch.Tensor] = {"existing_key": torch.tensor(42.0)}
    result = instance.state_dict(destination=dest, prefix="layer.")

    assert "existing_key" in result
    assert "layer.weight" in result
    assert result is dest


def test_base_load_from_state_dict_delegates():
    """_load_from_state_dict calls fix_loaded_state_dict_mismatch then super()."""
    linear = nn.Linear(4, 4, bias=False, device=DEVICE)
    instance = ConcreteQuarkLinear.from_module(linear)

    new_weight = torch.randn(4, 4, device=DEVICE)
    sd = {"weight": new_weight}

    instance.load_state_dict(sd, strict=True)

    assert torch.equal(instance.weight.data, new_weight)


def test_base_load_from_state_dict_calls_fix_mismatch():
    """_load_from_state_dict invokes fix_loaded_state_dict_mismatch."""
    linear = nn.Linear(4, 4, bias=False, device=DEVICE)
    instance = ConcreteQuarkLinear.from_module(linear)

    sd = {"weight": torch.randn(4, 4, device=DEVICE)}

    with patch(
        "quark.torch.export.nn.modules.quark_linear_base.fix_loaded_state_dict_mismatch", return_value=sd
    ) as mock_fix:
        instance.load_state_dict(sd, strict=True)
        mock_fix.assert_called_once()


def test_base_mro_puts_quark_before_module():
    """QuarkLinearBase sits before nn.Module in the MRO for concrete subclasses."""
    mro = ConcreteQuarkLinear.__mro__
    quark_idx = mro.index(QuarkLinearBase)
    module_idx = mro.index(nn.Module)
    assert quark_idx < module_idx, (
        f"QuarkLinearBase (index {quark_idx}) must precede nn.Module (index {module_idx}) in MRO"
    )


# ---------------------------------------------------------------------------
# QParamsLinear inherits state_dict from QuarkLinearBase
# ---------------------------------------------------------------------------


def test_qparamslinear_state_dict_calls_postprocess_preprocess():
    """QParamsLinear.state_dict (inherited) wraps with postprocess_weight / preprocess_weight."""
    linear = nn.Linear(8, 4, dtype=torch.float32, device=DEVICE)
    quant_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    qpl = QParamsLinear.from_module(linear, custom_mode="fp8", pack_method=None, quant_config=quant_config)

    call_order: list[str] = []
    original_postprocess = qpl.postprocess_weight
    original_preprocess = qpl.preprocess_weight

    def tracking_postprocess():
        call_order.append("postprocess")
        return original_postprocess()

    def tracking_preprocess():
        call_order.append("preprocess")
        return original_preprocess()

    qpl.postprocess_weight = tracking_postprocess
    qpl.preprocess_weight = tracking_preprocess

    qpl.state_dict()

    assert call_order == ["postprocess", "preprocess"]


def test_qparamslinear_state_dict_preprocess_called_on_error():
    """preprocess_weight is called even if state_dict raises (QParamsLinear)."""
    linear = nn.Linear(8, 4, dtype=torch.float32, device=DEVICE)
    quant_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    qpl = QParamsLinear.from_module(linear, custom_mode="fp8", pack_method=None, quant_config=quant_config)

    preprocess_called: list[bool] = []

    original_preprocess = qpl.preprocess_weight

    def tracking_preprocess():
        preprocess_called.append(True)
        return original_preprocess()

    qpl.preprocess_weight = tracking_preprocess

    with (
        patch.object(nn.Module, "state_dict", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        qpl.state_dict()

    assert preprocess_called, "preprocess_weight should be called in the finally block"


def test_qparamslinear_state_dict_is_inherited():
    """QParamsLinear.state_dict resolves to QuarkLinearBase.state_dict."""
    assert "state_dict" not in QParamsLinear.__dict__, (
        "state_dict should be inherited from QuarkLinearBase, not defined on QParamsLinear"
    )
    assert QParamsLinear.state_dict is QuarkLinearBase.state_dict


def test_qparamslinear_mro_puts_quark_before_module():
    """QuarkLinearBase sits before nn.Module in QParamsLinear's MRO."""
    mro = QParamsLinear.__mro__
    quark_idx = mro.index(QuarkLinearBase)
    module_idx = mro.index(nn.Module)
    assert quark_idx < module_idx
