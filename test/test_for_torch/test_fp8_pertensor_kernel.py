#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from unittest.mock import Mock, patch

import torch
from torch import nn

import quark.torch.export.nn.modules.qparamslinear as _qparamslinear_module
from quark.common.utils.testing_utils import PatchEverywhere
from quark.torch.export.constants import _check_scaled_mm_available_dev
from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.quantization.config.config import QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FP8_PER_TENSOR_SPEC = QTensorConfig(
    dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
)


def _make_fp8_qparams_linear(with_bias: bool = True) -> QParamsLinear:
    """Create a QParamsLinear configured for FP8 per-tensor weight + input."""
    quantization_config = QLayerConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
    float_module = nn.Linear(in_features=64, out_features=64, bias=with_bias, dtype=torch.float32).to(DEVICE)
    return QParamsLinear.from_module(
        float_module, custom_mode="fp8", pack_method=None, quant_config=quantization_config
    )


# We simulate GPU-less devices
def test_check_scaled_mm_available_dev():
    with patch("torch.cuda.is_available", return_value=False):
        _ = _check_scaled_mm_available_dev()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.version.cuda", new=None),
        patch("torch.version.hip", new="5.6.0"),
        patch("subprocess.run") as mock_subprocess,
    ):
        mock_subprocess.return_value = Mock(returncode=0, stdout="gfx940")
        _ = _check_scaled_mm_available_dev()

    with patch("torch.cuda.get_device_capability", return_value=(9, 0)), patch("torch.version.cuda", new=True):
        _ = _check_scaled_mm_available_dev()

    # The ci machine does not have the hardware to support the torch._scaled_mm function,
    # so the following functions are designed to test as many functions as possible and increase coverage
    FP8_PER_TENSOR_SPEC = QTensorConfig(
        dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
    )
    W_FP8_A_FP8_PER_TENSOR_CONFIG = QLayerConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    float_module = nn.Linear(in_features=512, out_features=512, bias=True, dtype=dtype).to(device)
    qparam_linear = QParamsLinear.from_module(
        float_module,
        custom_mode="fp8",
        pack_method=None,
        quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG,
    )
    input = torch.randn([512, 512], dtype=dtype, device=device)

    with PatchEverywhere("SCALED_MM_AVAILABLE_DEV", "hip", module_name_prefix="quark"):
        try:
            _ = qparam_linear(input)
        except ValueError as e:
            assert "Bias is not supported when out_dtype is set to Float32" in str(e)
        else:
            raise ValueError("Expected ValueError was not raised.")
        #
        input = input.to(torch.float16)
        with patch("torch._scaled_mm", return_value=torch.randn(512, 512, dtype=torch.float32)) as mock_scaled_mm:
            output = qparam_linear(input)
            mock_scaled_mm.assert_called_once()
            assert output is not None
        #
        qparam_linear.bias.data = qparam_linear.bias.data.to(torch.float16)
        with patch(
            "torch._scaled_mm", return_value=(torch.randn(512, 512, dtype=torch.float32), torch.tensor(1.0))
        ) as mock_scaled_mm:
            output = qparam_linear(input)
            mock_scaled_mm.assert_called_once()
            assert output is not None
        #
        qparam_linear.bias = None
        with patch(
            "torch._scaled_mm", return_value=(torch.randn(512, 512, dtype=torch.float32), torch.tensor(1.0))
        ) as mock_scaled_mm:
            output = qparam_linear(input)
            mock_scaled_mm.assert_called_once()
            assert output is not None
        #
        qparam_linear.weight_quantizer = None
        with patch(
            "torch._scaled_mm", return_value=(torch.randn(512, 512, dtype=torch.float32), torch.tensor(1.0))
        ) as mock_scaled_mm:
            output = qparam_linear(input)
            mock_scaled_mm.assert_not_called()


# ---------------------------------------------------------------------------
# DTensor FP8 forward path
# ---------------------------------------------------------------------------


class _FakeDTensor:
    """Lightweight stand-in for DTensor that delegates to a real Tensor."""

    def __init__(self, real_tensor: torch.Tensor, device_mesh: Mock) -> None:
        self._real = real_tensor
        self.device_mesh = device_mesh
        self.dtype = real_tensor.dtype
        self.shape = real_tensor.shape

    def view(self, *shape: int) -> torch.Tensor:
        return self._real.view(*shape)

    def __truediv__(self, other: object) -> torch.Tensor:
        return self._real / other


def test_forward_fp8_dtensor_path():
    """Cover _forward_fp8_dtensor with distribute_tensor, hip normalization, and bias addition."""
    qparams_linear = _make_fp8_qparams_linear(with_bias=True)

    input_tensor = torch.randn([64, 64], dtype=torch.float16, device=DEVICE)
    fake_dtensor_input = _FakeDTensor(input_tensor, Mock())

    qparams_linear._quant_dict = {
        "input_quantizer": qparams_linear.input_quantizer,
        "weight_quantizer": qparams_linear.weight_quantizer,
    }

    scaled_mm_output = (torch.randn(64, 64, dtype=torch.float16, device=DEVICE), torch.tensor(1.0))

    # With hip normalization + bias
    with (
        PatchEverywhere("SCALED_MM_AVAILABLE_DEV", "hip", module_name_prefix="quark"),
        patch.object(_qparamslinear_module, "distribute_tensor", side_effect=lambda tensor, **kwargs: tensor),
        patch.object(
            _qparamslinear_module,
            "e4m3fn_to_e4m3fnuz",
            side_effect=lambda tensor, tensor_scale: (tensor, tensor_scale),
        ),
        patch("torch._scaled_mm", return_value=scaled_mm_output) as mock_scaled_mm,
    ):
        output = qparams_linear._forward_fp8_dtensor(
            fake_dtensor_input, qparams_linear.bias.to(torch.float16), torch.float16
        )
        mock_scaled_mm.assert_called_once()
        assert output is not None
        assert len(output) == 2

    # Without hip normalization, no bias
    qparams_linear.bias = None
    with (
        PatchEverywhere("SCALED_MM_AVAILABLE_DEV", None, module_name_prefix="quark"),
        patch.object(_qparamslinear_module, "distribute_tensor", side_effect=lambda tensor, **kwargs: tensor),
        patch(
            "torch._scaled_mm", return_value=torch.randn(64, 64, dtype=torch.float16, device=DEVICE)
        ) as mock_scaled_mm,
    ):
        output = qparams_linear._forward_fp8_dtensor(fake_dtensor_input, None, torch.float16)
        mock_scaled_mm.assert_called_once()
        assert output is not None


def test_forward_fp8_dtensor_quant_dict_restoration():
    """Cover _forward_fp8_dtensor restoring quantizers from _quant_dict when they are None."""
    qparams_linear = _make_fp8_qparams_linear(with_bias=False)

    saved_input_quantizer = qparams_linear.input_quantizer
    saved_weight_quantizer = qparams_linear.weight_quantizer
    qparams_linear._quant_dict = {
        "input_quantizer": saved_input_quantizer,
        "weight_quantizer": saved_weight_quantizer,
    }
    qparams_linear.input_quantizer = None
    qparams_linear.weight_quantizer = None

    fake_dtensor_input = _FakeDTensor(torch.randn([64, 64], dtype=torch.float16, device=DEVICE), Mock())
    scaled_mm_output = (torch.randn(64, 64, dtype=torch.float16, device=DEVICE), torch.tensor(1.0))

    with (
        PatchEverywhere("SCALED_MM_AVAILABLE_DEV", None, module_name_prefix="quark"),
        patch.object(_qparamslinear_module, "distribute_tensor", side_effect=lambda tensor, **kwargs: tensor),
        patch("torch._scaled_mm", return_value=scaled_mm_output),
    ):
        output = qparams_linear._forward_fp8_dtensor(fake_dtensor_input, None, torch.float16)
        assert output is not None
        assert qparams_linear.input_quantizer is saved_input_quantizer
        assert qparams_linear.weight_quantizer is saved_weight_quantizer


def test_forward_fp8_dispatches_dtensor_branch():
    """Cover _forward_fp8 dispatching to _forward_fp8_dtensor when input is DTensor."""
    qparams_linear = _make_fp8_qparams_linear(with_bias=False)
    fake_result = (torch.randn(64, 64, dtype=torch.float16, device=DEVICE), [64, 64])

    with (
        patch.object(qparams_linear, "_forward_fp8_dtensor", return_value=fake_result) as mock_dtensor_forward,
        patch.object(qparams_linear, "_prepare_fp8_bias", return_value=None),
        patch.object(qparams_linear, "can_use_fp8_kernel", return_value=True),
        patch.object(_qparamslinear_module, "DTensor", _FakeDTensor),
    ):
        fake_input = _FakeDTensor(torch.randn(64, 64, dtype=torch.float16, device=DEVICE), Mock())
        output = qparams_linear(fake_input)
        mock_dtensor_forward.assert_called_once()
        assert output is not None
