#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any, List
from torch.nn.common_types import _size_2_t
from quark.torch.quantization.config.config import QuantizationConfig
from torch import Tensor
from torch.nn.modules.utils import _pair
from .mixin import QuantMixin
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ["QuantConv2d", "QuantConvTranspose2d"]


class _QuantizedConvNd(nn.modules.conv._ConvNd, QuantMixin):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 quant_config: QuantizationConfig,
                 reload: bool = False,
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding,
                         groups, bias, padding_mode)
        if not bias:
            # if bias is None Modify user settings
            quant_config.bias = None
        self.init_quantizer(quant_config, device, reload=reload)

    # In the original __init__ function of torch.nn._ConvNd,
    # the reset_parameters function is called, which takes up a lot of time.
    # This is the reason why inplace ops replacement is slow.
    # Therefore, overload this function in this class to skip the parameter
    # allocation operation, reducing the time of inplace ops replacement.
    def reset_parameters(self) -> None:
        pass


class _QuantizedConv(_QuantizedConvNd):

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        quant_input = self.get_quant_input(args[0])
        quant_weight = self.get_quant_weight(self.weight)
        quant_bias = self.get_quant_bias(self.bias)
        output = self._conv_forward(quant_input, quant_weight, quant_bias)
        quant_output: torch.Tensor = self.get_quant_output(output)
        return quant_output

    @classmethod
    def from_float(cls,
                   float_module: nn.Module,
                   quant_config: QuantizationConfig,
                   reload: bool = False,
                   weight_tensor: Optional[torch.Tensor] = None,
                   bias_tensor: Optional[torch.Tensor] = None) -> nn.Module:

        quant_conv = cls(float_module.in_channels,
                         float_module.out_channels,
                         float_module.kernel_size,
                         stride=float_module.stride,
                         padding=float_module.padding,
                         dilation=float_module.dilation,
                         output_padding=_pair(0),
                         groups=float_module.groups,
                         bias=float_module.bias is not None,
                         padding_mode=float_module.padding_mode,
                         quant_config=quant_config,
                         reload=reload,
                         device=float_module.weight.device)  # type: ignore
        if reload is True and weight_tensor is not None:
            quant_conv.weight.data = weight_tensor.to(float_module.weight.device)
        else:
            quant_conv.weight = float_module.weight

        if reload is True and bias_tensor is not None and quant_conv.bias is not None:
            quant_conv.bias.data = bias_tensor.to(float_module.weight.device)
        else:
            quant_conv.bias = float_module.bias
        return quant_conv


class QuantConv2d(_QuantizedConv):
    """Quantized version of nn.Conv2d

    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 output_padding: _size_2_t = 0,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 quant_config: QuantizationConfig = QuantizationConfig(),
                 reload: bool = False,
                 device: torch.device = torch.device("cpu")) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        output_padding = _pair(output_padding)
        dilation = _pair(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, output_padding,
                         groups, bias, padding_mode, quant_config, reload, device)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> torch.Tensor:
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight, bias,
                            self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class _QuantizedConvTransposeNd(_QuantizedConvNd, nn.modules.conv._ConvTransposeNd):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 dim: int,
                 quant_config: QuantizationConfig,
                 reload: bool = False,
                 device: torch.device = torch.device("cpu")) -> None:

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding,
                         groups, bias, padding_mode, quant_config, reload, device)
        self.dim = dim  # 1->1d, 2-> 2d, 3->3d
        self._conv_transpose_fn = [F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d][dim - 1]

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose{}d'.format(self.dim))
        assert isinstance(self.padding, tuple)

        quant_input = self.get_quant_input(input)
        quant_weight = self.get_quant_weight(self.weight)
        quant_bias = self.get_quant_bias(self.bias)
        num_spatial_dims = self.dim
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation)  # type: ignore[arg-type]
        output = self._conv_transpose_fn(quant_input, quant_weight, quant_bias, self.stride, self.padding,
                                         output_padding, self.groups, self.dilation)
        quant_output: torch.Tensor = self.get_quant_output(output)
        return quant_output

    @classmethod
    def from_float(cls,
                   float_module: nn.Module,
                   quant_config: QuantizationConfig,
                   reload: bool = False,
                   weight_tensor: Optional[torch.Tensor] = None,
                   bias_tensor: Optional[torch.Tensor] = None) -> nn.Module:

        quant_conv = cls(float_module.in_channels,
                         float_module.out_channels,
                         float_module.kernel_size,
                         stride=float_module.stride,
                         padding=float_module.padding,
                         dilation=float_module.dilation,
                         output_padding=float_module.output_padding,
                         groups=float_module.groups,
                         bias=float_module.bias is not None,
                         padding_mode=float_module.padding_mode,
                         quant_config=quant_config,
                         reload=reload,
                         device=float_module.weight.device)  # type: ignore
        if reload is True and weight_tensor is not None:
            quant_conv.weight.data = weight_tensor.to(float_module.weight.device)
        else:
            quant_conv.weight = float_module.weight

        if reload is True and bias_tensor is not None and quant_conv.bias is not None:
            quant_conv.bias.data = bias_tensor.to(float_module.weight.device)
        else:
            quant_conv.bias = float_module.bias
        return quant_conv


class QuantConvTranspose2d(_QuantizedConvTransposeNd):
    """Quantized version of nn.ConvTranspose2d
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 output_padding: _size_2_t = 0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: _size_2_t = 1,
                 padding_mode: str = 'zeros',
                 quant_config: QuantizationConfig = QuantizationConfig(),
                 reload: bool = False,
                 device: torch.device = torch.device("cpu")) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, True, output_padding,
                         groups, bias, padding_mode, 2, quant_config, reload, device)
