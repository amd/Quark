#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver

INT8_PER_TENSOR_SPEC = QTensorConfig(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)

DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG = QLayerConfig(
    input_tensors=INT8_PER_TENSOR_SPEC,
    weight=INT8_PER_TENSOR_SPEC,
    bias=INT8_PER_TENSOR_SPEC,
    output_tensors=INT8_PER_TENSOR_SPEC,
)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


input_tensor = torch.randn(1, 64, 64)


def test_net():
    class MyDataset(Dataset):
        def __init__(self):
            return

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return input_tensor

    model = SimpleCNN(num_classes=10)
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    quant_config = QConfig(global_quant_config=DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG)
    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(model, dataloader)
    assert (
        hasattr(quant_model.conv, "_input_quantizer")
        and hasattr(quant_model.conv, "_weight_quantizer")
        and hasattr(quant_model.conv, "_bias_quantizer")
        and hasattr(quant_model.conv, "_output_quantizer")
    )


if __name__ == "__main__":
    model = SimpleCNN(num_classes=10)
    model(input_tensor)
    test_net()
