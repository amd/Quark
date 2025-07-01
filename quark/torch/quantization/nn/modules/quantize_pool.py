#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any, Optional, List
import torch
from torch import nn
import math
from torch.nn.common_types import _size_2_t
from .mixin import QuantMixin
from torch.nn.common_types import _size_any_opt_t
from quark.torch.quantization.config.config import QuantizationConfig
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ["QuantAvgPool2d", "QuantAdaptiveAvgPool2d"]


class QuantAvgPool2d(nn.AvgPool2d, QuantMixin):
    """Quantized version of nn.AvgPool2d
    """

    def __init__(
            self,
            kernel_size: _size_2_t,
            stride: Optional[_size_2_t] = None,
            padding: _size_2_t = 0,
            ceil_mode: bool = False,
            count_include_pad: bool = True,
            divisor_override: Optional[int] = None,
            # args about quantization
            quant_config: QuantizationConfig = QuantizationConfig(),
            device: torch.device = torch.device("cpu"),
            **kwargs: Any) -> None:
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
        self.init_quantizer(quant_config, device, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quant_input = self.get_quant_input(input)
        output = super().forward(quant_input)
        '''
        Align NPU etc. hw constrain
        '''
        scale = 1.0
        assert isinstance(self.kernel_size, List)
        if self.kernel_size == [3, 3]:
            scale = 9.0 * 7.0 / 64.0
        elif self.kernel_size == [5, 5]:
            scale = 25.0 * 10.0 / 256.0
        elif self.kernel_size in [[6, 6], [3, 6], [6, 3]]:
            scale = 36.0 * 7.0 / 256.0
        elif self.kernel_size == [7, 7]:
            scale = 49.0 * 21.0 / 1024.0
        elif self.kernel_size == [14, 14]:
            scale = 196.0 * 21.0 / 4096.0
        else:
            rec = int(self.kernel_size[0]) * int(self.kernel_size[1])
            max_factor = math.ceil(math.log(rec * 128, 2))
            diff = 1.0
            multi_factor = 0.0
            shift_factor = 0.0
            for shift_factor_ in range(max_factor):
                factor = round((2**shift_factor_) / rec)
                diff_ = abs(factor / (2**shift_factor_) - 1 / rec)
                if diff_ < diff:
                    multi_factor = factor
                    diff = diff_
                    shift_factor = shift_factor_
            scale = rec * multi_factor / (2**shift_factor)

        output = output * scale
        quant_output: torch.Tensor = self.get_quant_output(output)
        return quant_output


class QuantAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, QuantMixin):

    def __init__(
            self,
            output_size: _size_any_opt_t,
            # args about quantization
            quant_config: QuantizationConfig = QuantizationConfig(),
            device: torch.device = torch.device("cpu"),
            **kwargs: Any) -> None:
        super(nn.AdaptiveAvgPool2d, self).__init__(output_size)
        self.init_quantizer(quant_config, device, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quant_input = self.get_quant_input(input)
        output = super().forward(quant_input)
        '''
        Align NPU etc. hw constrain
        '''

        if (isinstance(self.output_size, (tuple, list))
                and tuple(self.output_size) != (1, 1)) or (isinstance(self.output_size, int) and self.output_size != 1):
            print("[WARNING] For AdaptiveAvgPooling, NPU only supports output_size=1")

        scale = 1.0
        if input.shape[2] == 3 and input.shape[3] == 3:
            scale = 9.0 * 7.0 / 64.0
        elif input.shape[2] == 5 and input.shape[3] == 5:
            scale = 25.0 * 10.0 / 256.0
        elif (input.shape[2] == 6 and input.shape[3] == 6) or (input.shape[2] == 3
                                                               and input.shape[3] == 6) or (input.shape[2] == 6
                                                                                            and input.shape[3] == 3):
            scale = 36.0 * 7.0 / 256.0
        elif input.shape[2] == 7 and input.shape[3] == 7:
            scale = 49.0 * 21.0 / 1024.0
        elif input.shape[2] == 14 and input.shape[3] == 14:
            scale = 196.0 * 21.0 / 4096.0
        else:
            rec = int(input.shape[2]) * int(input.shape[3])
            max_factor = math.ceil(math.log(rec * 128, 2))
            diff = 1.0
            multi_factor = 0.0
            shift_factor = 0.0
            for shift_factor_ in range(max_factor):
                factor = round((2**shift_factor_) / rec)
                diff_ = abs(factor / (2**shift_factor_) - 1 / rec)
                if diff_ < diff:
                    multi_factor = factor
                    diff = diff_
                    shift_factor = shift_factor_
            scale = rec * multi_factor / (2**shift_factor)

        output = output * scale
        quant_output: torch.Tensor = self.get_quant_output(output)
        return quant_output
