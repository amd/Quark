#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any
import torch
from torch import nn
from .mixin import QuantMixin
from quark.torch.quantization.config.config import QuantizationConfig
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ["QuantLeakyReLU"]


class QuantLeakyReLU(nn.LeakyReLU, QuantMixin):
    '''
    Align with NPU's hw contrain, see more refer to NNDCT's class DPULeakyReLU
    '''

    def __init__(
            self,
            negative_slope: float = 0.01,
            inplace: bool = False,
            # args about quantization
            quant_config: QuantizationConfig = QuantizationConfig(),
            device: torch.device = torch.device("cpu"),
            **kwargs: Any) -> None:
        super().__init__(negative_slope, inplace)
        self.negative_slope = 0.1015625  # if need more information, refer to NPU related hw team
        self.init_quantizer(quant_config, device, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quant_input = self.get_quant_input(input)
        output = super().forward(quant_input)
        quant_output: torch.Tensor = self.get_quant_output(output)
        return quant_output
