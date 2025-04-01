#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any, Optional
import torch
from torch import nn
from torch.nn import functional as F
from .mixin import QuantMixin
from quark.torch.quantization.config.config import QuantizationConfig
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ["QuantLinear"]


class QuantLinear(nn.Linear, QuantMixin):
    """Quantized version of nn.Linear

    """

    def __init__(self, in_features: int, out_features: int, device: torch.device, bias: bool,
                 quant_config: QuantizationConfig, **kwargs: Any) -> None:
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        if not bias:
            # if bias is None Modify user settings
            quant_config.bias = None
        self.init_quantizer(quant_config, device, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        quant_input = self.get_quant_input(args[0])
        quant_weight = self.get_quant_weight(self.weight)
        quant_bias = self.get_quant_bias(self.bias)
        output = F.linear(quant_input, quant_weight, bias=quant_bias)
        quant_output: torch.Tensor = self.get_quant_output(output)

        return quant_output

    # In the original __init__ function of torch.nn.Linear,
    # the reset_parameters function is called, which takes up a lot of time.
    # This is the reason why inplace ops replacement is slow.
    # Therefore, overload this function in this class to skip the parameter
    # allocation operation, reducing the time of inplace ops replacement.
    def reset_parameters(self) -> None:
        pass

    @classmethod
    def from_float(cls,
                   float_module: nn.Module,
                   layer_quant_config: QuantizationConfig,
                   reload: bool = False,
                   weight_tensor: Optional[torch.Tensor] = None,
                   bias_tensor: Optional[torch.Tensor] = None) -> nn.Linear:
        bias = False if (float_module.bias is None) and (reload is False or bias_tensor is None) else True
        quant_linear = cls(float_module.in_features,
                           float_module.out_features,
                           float_module.weight.device,
                           bias,
                           layer_quant_config,
                           reload=reload)
        if reload is True and weight_tensor is not None:
            quant_linear.weight.data = weight_tensor.to(float_module.weight.device)
        else:
            quant_linear.weight = float_module.weight

        if reload is True and bias_tensor is not None:
            quant_linear.bias.data = bias_tensor.to(float_module.weight.device)
        else:
            quant_linear.bias = float_module.bias
        return quant_linear
