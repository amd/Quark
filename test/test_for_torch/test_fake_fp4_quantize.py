#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
from functools import reduce
import torch
import torch.nn as nn
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.observer.observer import PerChannelMinMaxObserver, PerBlockMXObserver
from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig, FP4PerGroupSpec
from quark.torch.quantization.utils import calculate_qmin_qmax, get_dtype_params, reshape_to_blocks
from quark.torch.kernel.hw_emulation.hw_emulation_interface import fake_quantize_mx, fake_quantize_non_mx
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

from quark.torch import ModelQuantizer
from torch.utils.data import Dataset, DataLoader
import quark.torch.kernel  # noqa
from torch import ops  # type: ignore[attr-defined]


class ToyModel(nn.Module):

    def __init__(self, in_features, out_features):
        super(ToyModel, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.fc(x)
        return x


input_tensor = torch.ones(1, 4096, 4096)


class MyDataset(Dataset):

    def __init__(self):
        return

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return input_tensor


def is_power_of_two(tensor):
    log2_values = torch.log2(tensor)
    return torch.all(log2_values == log2_values.round())



@pytest.mark.parametrize("dtype,dim,scale_format,scale_calculation_mode", [
    (Dtype.fp4, 0, "e8m0", "even"),
    (Dtype.fp4, 0, "e8m0", "floor"),
    (Dtype.fp4, 0, "e8m0", "ceil"),
    (Dtype.fp4, 0, "e4m3", ""),
    (Dtype.fp4, 0, "float32", ""),
    (Dtype.fp4, -1, "e8m0", "even"),
    (Dtype.fp4, -1, "e8m0", "floor"),
    (Dtype.fp4, -1, "e8m0", "ceil"),
    (Dtype.fp4, -1, "e4m3", ""),
    (Dtype.fp4, -1, "float32", ""),
    (Dtype.fp4, -2, "e8m0", "even"),
    (Dtype.fp4, -2, "e8m0", "floor"),
    (Dtype.fp4, -2, "e8m0", "ceil"),
    (Dtype.fp4, -2, "e4m3", ""),
    (Dtype.fp4, -2, "float32", ""),

])
def test_fp4_per_channel_scaled_fake_quantize(dtype, dim, scale_format, scale_calculation_mode):
    if len(scale_calculation_mode) == 0:
        scale_calculation_mode = None

    tensor_shape = (4096, 4096)
    x: torch.Tensor = torch.randn(tensor_shape, dtype=torch.float32)

    fp4_per_channel_spec = QuantizationSpec(dtype=dtype,
                                            qscheme=QSchemeType.per_channel,
                                            observer_cls=PerChannelMinMaxObserver,
                                            ch_axis=dim,
                                            scale_format=scale_format,
                                            scale_calculation_mode=scale_calculation_mode,
                                            is_dynamic=False)
    quantizer_per_channel = ScaledFakeQuantize(fp4_per_channel_spec)
    fp4_per_channel_scaled_fake_quantize = quantizer_per_channel(x.clone())

    fp4_per_group_spec = QuantizationSpec(dtype=dtype,
                                          qscheme=QSchemeType.per_group,
                                          observer_cls=PerBlockMXObserver,
                                          group_size=tensor_shape[-1],
                                          ch_axis=dim,
                                          scale_format=scale_format,
                                          scale_calculation_mode=scale_calculation_mode,
                                          is_dynamic=False)
    quantizer_per_group = ScaledFakeQuantize(fp4_per_group_spec)
    fp4_per_group_scaled_fake_quantize = quantizer_per_group(x.transpose(0, 1).clone())
    fp4_per_group_scaled_fake_quantize = fp4_per_group_scaled_fake_quantize.transpose(0, 1)

    if scale_format == "e8m0":
        diff = fp4_per_channel_scaled_fake_quantize - fp4_per_group_scaled_fake_quantize
        assert (torch.count_nonzero(diff).item() / reduce(lambda x, y: x * y, tensor_shape)) < 0.1
    else:
        assert torch.allclose(fp4_per_channel_scaled_fake_quantize, fp4_per_group_scaled_fake_quantize)


@pytest.mark.parametrize("dtype,group_size", [
    (Dtype.fp4, 32),
    (Dtype.fp4, 16),
    (Dtype.fp8_e4m3, 32),
    (Dtype.fp8_e4m3, 16),
    (Dtype.fp8_e5m2, 32),
    (Dtype.fp8_e5m2, 16),
])
def test_fp_per_group_weight_quantization_qparams(dtype, group_size):

    FP_WEIGHT_PER_GROUP_SPEC = QuantizationSpec(dtype=dtype,
                                                qscheme=QSchemeType.per_group,
                                                observer_cls=PerBlockMXObserver,
                                                ch_axis=-1,
                                                group_size=group_size,
                                                scale_format="e8m0",
                                                scale_calculation_mode="floor",
                                                is_dynamic=False)
    FP_WEIGHT_PER_GROUP_CONFIG = QuantizationConfig(weight=FP_WEIGHT_PER_GROUP_SPEC)
    model = ToyModel(in_features=4096, out_features=4096)
    model.fc.weight = torch.nn.Parameter(torch.randn([4096, 4096]))
    model(input_tensor)
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    quant_config = Config(global_quant_config=FP_WEIGHT_PER_GROUP_CONFIG)
    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(model, dataloader)
    assert quant_model.fc._weight_quantizer.scale.shape[1] == int(model.fc.weight.shape[0] / group_size)


@pytest.mark.parametrize("dtype", [
    (Dtype.fp4),
])
def test_mxfp_per_group_scaled_fake_quantize(dtype):
    tensor_shape = (4096, 4096)
    x: torch.Tensor = torch.randn(tensor_shape, dtype=torch.float32)

    _, _, emax = get_dtype_params(dtype)
    block_size, axis = 32, 1
    quant_min, quant_max = calculate_qmin_qmax(dtype)

    block_x = reshape_to_blocks(x.clone(), block_size, axis)
    scale, _ = torch.max(torch.abs(block_x), dim=axis + 1, keepdim=True)
    scale = torch.pow(2, torch.floor(torch.log2(scale)) - emax)
    zero_point = torch.zeros_like(scale, dtype=torch.int32)

    qx_scaled_fake_quantize = ops.quark.scaled_fake_quantize(dtype.value, x.clone(), scale, zero_point, -1, 32, quant_min, quant_max,
                                                             0, QSchemeType.per_group.value, 'None')

    scale = torch.ones((2, 3), dtype=torch.float32)
    qx_fake_quantize_mx = fake_quantize_mx(input_tensor=x.clone(),
                                           scale=scale,
                                           mx_element_dtype=dtype,
                                           axis=-1,
                                           block_size=32,
                                           scale_calculation_mode="floor")

    assert torch.allclose(qx_scaled_fake_quantize, qx_fake_quantize_mx)


@pytest.mark.parametrize("dtype, block_size", [
    (Dtype.fp4, 8),
    (Dtype.fp4, 16),
    (Dtype.fp4, 32)
])
def test_non_mxfp4_per_group_scaled_fake_quantize(dtype, block_size):
    tensor_shape = (4096, 4096)
    x: torch.Tensor = torch.randn(tensor_shape, dtype=torch.float32)

    quant_min, quant_max = calculate_qmin_qmax(dtype)
    block_x = reshape_to_blocks(x.clone(), block_size, 1)

    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    scale = torch.div(amax, quant_max)
    zero_point = torch.zeros_like(scale, dtype=torch.int32)

    qx_scaled_fake_quantize = ops.quark.scaled_fake_quantize(dtype.value, x.clone().unsqueeze(0), scale, zero_point, -1, block_size, quant_min, quant_max,
                                                             0, QSchemeType.per_group.value, 'None')

    qx_fake_quantize_non_mx = fake_quantize_non_mx(input_tensor=x.clone(),
                                                   element_dtype=dtype,
                                                   axis=-1,
                                                   block_size=block_size)

    assert torch.allclose(qx_scaled_fake_quantize, qx_fake_quantize_non_mx)


@pytest.mark.parametrize("dtype,scale_calculation_mode", [
    (Dtype.fp4, "even"),
    (Dtype.fp4, "floor"),
    (Dtype.fp4, "ceil"),
])
def test_mxfp4_per_group_scaled_and_non_scaled_fake_quantize(dtype, scale_calculation_mode):
    tensor_shape = (4096, 4096)
    x: torch.Tensor = torch.randn(tensor_shape, dtype=torch.float32)

    block_size, axis = 32, -1

    spec = FP4PerGroupSpec(ch_axis=axis,
                           group_size=block_size,
                           scale_format="e8m0",
                           scale_calculation_mode=scale_calculation_mode,
                           is_dynamic=False).to_quantization_spec()
    quantizer = ScaledFakeQuantize(spec)
    scaled_fake_quantize = quantizer(x.clone())

    non_scaled_fake_quantize = fake_quantize_mx(input_tensor=x.clone(),
                                                axis=axis,
                                                block_size=block_size,
                                                mx_element_dtype=dtype,
                                                scale_calculation_mode=scale_calculation_mode)

    assert torch.allclose(scaled_fake_quantize, non_scaled_fake_quantize)


@pytest.mark.parametrize("scale_format", [
    ("e4m3"),
    ("float32")
])
def test_fp4_per_group_scale(scale_format):
    tensor_shape = (4096, 4096)
    x: torch.Tensor = torch.randn(tensor_shape, dtype=torch.float32)

    block_size, axis = 32, -1
    spec = FP4PerGroupSpec(ch_axis=axis,
                           group_size=block_size,
                           scale_format=scale_format,
                           is_dynamic=False).to_quantization_spec()
    quantizer = ScaledFakeQuantize(spec)
    quantizer(x.clone())
