
#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
from torch import nn
import quark
from quark.torch.export.api import check_scaled_mm_available_dev
from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.quantization.config.config import QuantizationSpec, QuantizationConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
from unittest.mock import patch, Mock


# We simulate GPU-less devices
def test_check_scaled_mm_available_dev():
    with patch("torch.cuda.is_available", return_value=False):
        result = check_scaled_mm_available_dev()

    with patch("torch.cuda.is_available", return_value=True), patch("torch.version.cuda", new=None), patch("torch.version.hip", new="5.6.0"), patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = Mock(returncode=0, stdout="gfx940")
        result = check_scaled_mm_available_dev()

    with patch("torch.cuda.get_device_capability", return_value=(9, 0)), patch("torch.version.cuda", new=True):
        result = check_scaled_mm_available_dev()

    # The ci machine does not have the hardware to support the torch._scaled_mm function,
    # so the following functions are designed to test as many functions as possible and increase coverage
    FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                           qscheme=QSchemeType.per_tensor,
                                           observer_cls=PerTensorMinMaxObserver,
                                           is_dynamic=False)
    W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                                       weight=FP8_PER_TENSOR_SPEC)
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
    quark.torch.export.nn.modules.qparamslinear.SCALED_MM_AVAILABLE_DEV = "hip"
    #
    try:
        out = qparam_linear(input)
    except ValueError as e:
        assert "Bias is not supported when out_dtype is set to Float32" in str(e)
    else:
        raise ValueError("Expected ValueError was not raised.")
    #
    input = input.to(torch.float16)
    with patch('torch._scaled_mm', return_value=torch.randn(512, 512, dtype=torch.float32)) as mock_scaled_mm:
        output = qparam_linear(input)
        mock_scaled_mm.assert_called_once()
        assert output is not None
    #
    qparam_linear.bias.data = qparam_linear.bias.data.to(torch.float16)
    with patch('torch._scaled_mm', return_value=(torch.randn(512, 512, dtype=torch.float32), torch.tensor(1.0))) as mock_scaled_mm:
        output = qparam_linear(input)
        mock_scaled_mm.assert_called_once()
        assert output is not None
    #
    qparam_linear.bias = None
    with patch('torch._scaled_mm', return_value=(torch.randn(512, 512, dtype=torch.float32), torch.tensor(1.0))) as mock_scaled_mm:
        output = qparam_linear(input)
        mock_scaled_mm.assert_called_once()
        assert output is not None
    #
    qparam_linear.weight_quantizer = None
    with patch('torch._scaled_mm', return_value=(torch.randn(512, 512, dtype=torch.float32), torch.tensor(1.0))) as mock_scaled_mm:
        output = qparam_linear(input)
        mock_scaled_mm.assert_not_called()
