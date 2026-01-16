#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from quark.torch.kernel.hw_emulation.hw_emulation_interface import fake_quantize_mx
from quark.torch.quantization.config.config import FP6E2M3PerGroupSpec, FP6E3M2PerGroupSpec
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase


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


@pytest.mark.parametrize("dtype,scale_calculation_mode", [(Dtype.fp6_e2m3, "even"), (Dtype.fp6_e3m2, "even")])
def test_fp6_per_group_scaled_and_non_scaled_fake_quantize(dtype, scale_calculation_mode):
    tensor_shape = (4096, 4096)
    x: torch.Tensor = torch.randn(tensor_shape, dtype=torch.float32)

    block_size, axis = 32, -1

    select_spec = FP6E2M3PerGroupSpec if dtype == Dtype.fp6_e2m3 else FP6E3M2PerGroupSpec
    spec = select_spec(
        ch_axis=axis,
        group_size=block_size,
        scale_format="e8m0",
        scale_calculation_mode=scale_calculation_mode,
        is_dynamic=False,
    ).to_quantization_spec()
    quantizer = FakeQuantizeBase.get_fake_quantize(spec)
    scaled_fake_quantize = quantizer(x.clone())

    non_scaled_fake_quantize = fake_quantize_mx(
        input_tensor=x.clone(),
        axis=axis,
        block_size=block_size,
        mx_element_dtype=dtype,
        scale_calculation_mode=scale_calculation_mode,
    )

    assert torch.allclose(scaled_fake_quantize, non_scaled_fake_quantize)
