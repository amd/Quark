#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
import torch
import torch.nn as nn
from torch import ops  # type: ignore[attr-defined]
from torch.utils.data import DataLoader, Dataset

import quark.torch.kernel  # noqa
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.observer.observer import PerChannelMinMaxObserver


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_features=4, out_features=num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


input_tensor = torch.ones(1, 4, 4)


class MyDataset(Dataset):
    def __init__(self):
        return

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return input_tensor


@pytest.mark.parametrize(
    "dtype,dim",
    [
        (Dtype.fp8_e4m3, 0),
        (Dtype.fp8_e4m3, 1),
        (Dtype.fp8_e4m3, -1),
        (Dtype.fp8_e4m3, -2),
        (Dtype.fp8_e5m2, 0),
        (Dtype.fp8_e5m2, 1),
        (Dtype.fp8_e5m2, -1),
        (Dtype.fp8_e5m2, -2),
    ],
)
def test_weight_scale_dim(dtype, dim):
    FP8_WEIGHT_PER_CHANNEL_SPEC = QTensorConfig(
        dtype=dtype,
        qscheme=QSchemeType.per_channel,
        observer_cls=PerChannelMinMaxObserver,
        ch_axis=dim,
        is_dynamic=False,
    )
    TEST_WEIGHT = QLayerConfig(weight=FP8_WEIGHT_PER_CHANNEL_SPEC)
    model = SimpleCNN()
    model.fc.weight = torch.nn.Parameter(torch.ones([2, 4]) * 0.1)
    model(input_tensor)
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    quant_config = QConfig(global_quant_config=TEST_WEIGHT)
    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(model, dataloader)
    assert quant_model.fc._weight_quantizer.scale.shape[0] == model.fc.weight.shape[dim]


@pytest.mark.parametrize(
    "dtype,ch_axis",
    [
        (Dtype.fp8_e4m3, 0),
        (Dtype.fp8_e4m3, 1),
        (Dtype.fp8_e4m3, 2),
        (Dtype.fp8_e4m3, 3),
        (Dtype.fp8_e4m3, 4),
        (Dtype.fp8_e5m2, 0),
        (Dtype.fp8_e5m2, 1),
        (Dtype.fp8_e5m2, 2),
        (Dtype.fp8_e5m2, 3),
        (Dtype.fp8_e5m2, 4),
    ],
)
def test_fp8_per_channel_kernel(dtype, ch_axis):
    test_dim = [3, 4, 5, 6, 7]
    x: torch.Tensor = torch.ones(test_dim)

    dim_size = test_dim[ch_axis]
    scale = torch.ones(dim_size)
    scale[0] = 1000

    # Set default empty value not use for fp8 per channel
    zero_point = torch.Tensor([])
    group_size = 1
    round_mode = 0
    quant_min = 0
    quant_max = 0

    out = ops.quark.scaled_fake_quantize(
        dtype.value,
        x,
        scale,
        zero_point,
        ch_axis,
        group_size,
        quant_min,
        quant_max,
        round_mode,
        QSchemeType.per_channel.value,
        "None",
    )
    slice_arr_0 = [slice(None, None) for a in x.shape]
    slice_arr_0[ch_axis] = slice(0, 1)
    slice_arr_a = [slice(None, None) for a in x.shape]
    slice_arr_b = [slice(None, None) for a in x.shape]
    for i in range(1, test_dim[ch_axis] - 1):
        slice_arr_a[ch_axis] = slice(i, i + 1)
        slice_arr_b[ch_axis] = slice(i + 1, i + 2)
        assert torch.allclose(out[slice_arr_0], out[slice_arr_b]) is False
        assert torch.allclose(out[slice_arr_a], out[slice_arr_b]) is True
