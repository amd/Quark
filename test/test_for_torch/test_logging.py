#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from quark.shares.utils.log import ScreenLogger
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Config, QuantizationConfig, QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver

logger = ScreenLogger(__name__)
INT8_PER_TENSOR_SPEC = QuantizationSpec(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)

DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(
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
    # set log lovel
    log_level = 1
    quant_config = Config(global_quant_config=DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG, log_severity_level=log_level)

    # After initialization of ModelQuantizer, log level is successfully set.
    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(model, dataloader)

    # check log debug
    logger.debug("Failed to check Log Debug.")

    # check log info
    logger.info("Successfully checked Logger Info.", allow_duplicate=False)

    # check log warning
    logger.warning("Successfully checked Logger Warning.", allow_duplicate=True)

    # check log error
    try:
        logger.error("Checking Logger Error...", allow_duplicate=True)
        print("Falied to check Logger Error.")
    except SystemExit:
        print("Successfully checked Logger Error.")

    # check log critical
    try:
        logger.critical("Checking Logger Critical...")
        print("Falied to check Logger Critical.")
    except SystemExit:
        print("Successfully checked Logger Critical.")
        pass

    # check log exception
    try:
        x = 1 / 0
    except Exception as e:
        logger.exception("Checking Logger Exception: " + str(e))
    print("Successfully checked Logger Exception.")

    # check debug
    quantizer.config.log_severity_level = 0
    quantizer.set_logging_level()
    logger.debug("Successfully checked Log Debug.")

    # checkwarning
    quantizer.config.log_severity_level = 2
    quantizer.set_logging_level()
    logger.warning("Successfully checked Log Warning.")

    # check error
    quantizer.config.log_severity_level = 3
    quantizer.set_logging_level()
    try:
        logger.error("Checking Log Error.")
    except SystemExit:
        print("Successfully checked Logger Error.")

    # check critical
    quantizer.config.log_severity_level = 4
    quantizer.set_logging_level()
    try:
        logger.critical("Checking Log Critical.")
    except SystemExit:
        print("Successfully checked Logger Critical.")


if __name__ == "__main__":
    test_net()
