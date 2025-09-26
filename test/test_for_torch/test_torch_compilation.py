#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quark.shares.utils.testing_utils import torch_device
from quark.torch.quantization.config.config import Config, QuantizationConfig, QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import (
    PerBlockMXObserver,
    PerChannelMinMaxObserver,
    PerGroupMinMaxObserver,
    PerTensorMinMaxObserver,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sum(x)
        return x


def quantize_and_compile_model(device, QUANT_SPEC):
    model = SimpleModel().to(device)
    input_data = torch.randn(100, 10).to(device)
    calib_dataloader = DataLoader(input_data, batch_size=10, shuffle=True)
    QUANT_CONFIG = QuantizationConfig(weight=QUANT_SPEC, input_tensors=QUANT_SPEC)
    quant_config = Config(global_quant_config=QUANT_CONFIG)

    from quark.torch import ModelQuantizer

    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(model, calib_dataloader)

    input_data = torch.randn(10, 10).to(device)
    output1 = quant_model(input_data)

    # Freeze quantized model
    frozen_quant_model = quantizer.freeze(quant_model)

    # custom backend to present graph breaks without further compilation
    from torch._functorch.aot_autograd import aot_module_simplified

    def custom_backend(gm, sample_inputs):
        def custom_compiler(gm, sample_inputs):
            print("Model graph with custom_backend:")
            # gm.print_readable() # enable this line if you want to see readable graph with node dtype
            print(gm.graph)
            # <implement your backend here>
            return gm

        # helper function to run AOTAutograd manually
        return aot_module_simplified(
            gm,
            sample_inputs,
            # decompositions=decompositions,
            fw_compiler=custom_compiler,
        )

    frozen_quant_model = torch.compile(backend=custom_backend)(frozen_quant_model)
    output2 = frozen_quant_model(input_data)

    # Test the result is same after compilation
    assert torch.equal(output1, output2)


def test_int_per_tensor_for_torch_compile():
    QUANT_SPEC = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    quantize_and_compile_model(torch_device, QUANT_SPEC)


def test_int_per_channel_for_torch_compile():
    QUANT_SPEC = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_channel,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        ch_axis=1,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    quantize_and_compile_model(torch_device, QUANT_SPEC)


def test_int_per_group_for_torch_compile():
    QUANT_SPEC = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_group,
        observer_cls=PerGroupMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        ch_axis=1,
        group_size=2,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    quantize_and_compile_model(torch_device, QUANT_SPEC)


@pytest.mark.parametrize(
    "dtype",
    [
        (Dtype.fp8_e4m3),
        (Dtype.fp8_e5m2),
    ],
)
def test_fp8_per_tensor_for_torch_compile(dtype):
    QUANT_SPEC = QuantizationSpec(
        dtype=dtype,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    quantize_and_compile_model(torch_device, QUANT_SPEC)


@pytest.mark.parametrize(
    "dtype,element_dtype",
    [
        (Dtype.mx6, None),
        (Dtype.mx9, None),
        (Dtype.mx, Dtype.int8),
        (Dtype.mx, Dtype.fp8_e4m3),
        (Dtype.mx, Dtype.fp8_e5m2),
        (Dtype.mx, Dtype.fp6_e2m3),
        (Dtype.mx, Dtype.fp6_e3m2),
        (Dtype.mx, Dtype.fp4),
    ],
)
def test_mx_for_torch_compile(dtype, element_dtype):
    QUANT_SPEC = QuantizationSpec(
        dtype=dtype,
        mx_element_dtype=element_dtype,
        qscheme=QSchemeType.per_group,
        observer_cls=PerBlockMXObserver,
        ch_axis=-1,
        group_size=16,
        round_method=RoundType.half_even,
        is_dynamic=False,
        scale_calculation_mode="floor",
    )
    quantize_and_compile_model(torch_device, QUANT_SPEC)
