#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Any
from typing_extensions import Self
from .mixin import QuantMixin
from quark.torch.quantization.config.config import QuantizationConfig

__all__ = ["QuantizedConvBatchNorm2d", "update_bn_stats", "freeze_bn_stats", "fuse_conv_bn", "clear_non_native_bias"]
_BN_CLASS_MAP = {
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}


class _ConvBnNd(nn.modules.conv._ConvNd, QuantMixin):

    _version = 2

    def __init__(
            self,
            # ConvNd args
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
            # BatchNormNd args
            dim: int = 2,
            eps: float = 1e-05,
            momentum: float = 0.1,
            freeze_bn_stats: bool = False,
            # quant args
            quant_config: QuantizationConfig = QuantizationConfig(),
    ):
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation,
                                         transposed, output_padding, groups, False, padding_mode)

        self.bn_frozen = freeze_bn_stats if self.training else True
        self.dim = dim
        self.bn = _BN_CLASS_MAP[dim](out_channels, eps, momentum, True, True)
        # TODO haoliang
        self.init_quantizer(quant_config, self.weight.device)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn_stats:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

        self.conv_bn_fused = False

    def reset_running_stats(self) -> None:
        self.bn.reset_running_stats()

    def reset_bn_parameters(self) -> None:
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)
        # note: below is actully for conv, not BN
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)  # type: ignore
            bound = 1 / math.sqrt(float(fan_in))
            init.uniform_(self.bias, -bound, bound)

    def batch_stats(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the batch mean and variance of x and updates the BatchNorm's running mean and average.

      Args:
        x (torch.Tensor): input batch.
        bias (torch.Tensor): the bias that is to be applied to the batch.

      Returns:
        (mean, variance)

      Note:
        In case of `nn.Linear`, x may be of shape (N, C, L) or (N, L)
        where N is batch size, C is number of channels, L is the features size.
        The batch norm computes the stats over C in the first case or L on the second case.
        The batch normalization layer is
        (`nn.BatchNorm1d`)

        In case of `nn.Conv2d`, x is of shape (N, C, H, W)
        where H,W are the image dimensions, and the batch norm computes the stats over C.
        The batch normalization layer is
        (`nn.BatchNorm2d`)

        In case of `nn.Conv3d`, x is of shape (N, C, D, H, W)
        where H,W are the image dimensions, D is additional channel dimension,
        and the batch norm computes the stats over C.
        The batch normalization layer is
        (`nn.BatchNorm3d`)
    """
        assert self.bn.num_batches_tracked is not None
        channel_size = self.bn.num_features
        self.bn.num_batches_tracked.add_(1)

        # Calculate current batch stats
        batch_mean = x.transpose(0, 1).contiguous().view(channel_size, -1).mean(1)
        # BatchNorm currently uses biased variance (without Bessel's correction) as was discussed at
        #
        # also see the source code itself:
        batch_var = x.transpose(0, 1).contiguous().view(channel_size, -1).var(1, unbiased=False)

        # Update running stats
        with torch.no_grad():
            biased_batch_mean = batch_mean + (bias if bias is not None else 0)
            # However - running_var is updated using unbiased variance!
            n = x.numel() / channel_size
            corrected_var = batch_var * (n / float(n - 1))
            momentum = self.bn.momentum
            if momentum is None:
                # momentum is None - we compute a cumulative moving average
                momentum = 1. / float(self.bn.num_batches_tracked)
            if self.bn.running_mean is not None:
                self.bn.running_mean.mul_(1 - momentum).add_(momentum * biased_batch_mean)
            if self.bn.running_var is not None:
                self.bn.running_var.mul_(1 - momentum).add_(momentum * corrected_var)

        return batch_mean, batch_var

    def reset_parameters(self) -> None:
        super(_ConvBnNd, self).reset_parameters()

    def merge_bn_to_conv(self) -> None:
        with torch.no_grad():
            # Use the same implementation in nndct_shared/optimzation/fuse_conv_bn.py
            # to make sure the test accruacy is same as the deployable model.
            gamma = self.bn.weight.detach().cpu().numpy()
            beta = self.bn.bias.detach().cpu().numpy()
            running_var = self.bn.running_var.detach().cpu().numpy() if self.bn.running_var is not None else 1
            running_mean = self.bn.running_mean.detach().cpu().numpy() if self.bn.running_mean is not None else 0
            epsilon = self.bn.eps

            scale = gamma / np.sqrt(running_var + epsilon)  # type: ignore[operator]
            offset = beta - running_mean * scale

            weight = self.weight.detach().cpu().numpy()
            # Conv2d
            if self.dim == 2 and not self.transposed:
                # OIHW -> IHWO -> OIHW
                weight = np.multiply(weight.transpose(1, 2, 3, 0), scale).transpose(3, 0, 1, 2)
            # ConvTranspose2d
            elif self.dim == 2 and self.transposed:
                # IOHW -> IHWO -> IOHW
                weight = np.multiply(weight.transpose(0, 2, 3, 1), scale).transpose(0, 3, 1, 2)
            # Conv3D
            elif self.dim == 3 and not self.transposed:
                weight = np.multiply(weight.transpose(1, 2, 3, 4, 0), scale).transpose(4, 0, 1, 2, 3)
            # ConvTranspose3d
            elif self.dim == 3 and self.transposed:
                weight = np.multiply(weight.transpose(2, 3, 4, 0, 1), scale).transpose(3, 4, 0, 1, 2)
            else:
                raise RuntimeError('Unsupported combinations: (dim={}, transposed={})'.format(
                    self.dim, self.transposed))
            self.weight.copy_(torch.from_numpy(weight))

            bias = self.bias.detach().cpu().numpy() if self.bias is not None else 0
            bias = torch.from_numpy(bias * scale + offset).to(self.weight.device)
            if self.bias is not None:
                self.bias.copy_(bias)
            else:
                self.register_parameter("bias", None)
                self.bias = nn.Parameter(bias)
        self.conv_bn_fused = True

    def update_bn_stats(self) -> None:
        self.bn_frozen = False

    def freeze_bn_stats(self) -> None:
        self.bn_frozen = True

    def clear_non_native_bias(self) -> None:
        if self.bias is None:
            print('[WARNING] No bias to unmerge')
            return

        with torch.no_grad():
            gamma = self.bn.weight.detach().cpu().numpy()
            beta = self.bn.bias.detach().cpu().numpy()
            running_var = self.bn.running_var.detach().cpu().numpy() if self.bn.running_var is not None else 1
            epsilon = self.bn.eps

            scale = gamma / np.sqrt(running_var + epsilon)  # type: ignore[operator]

            bias = self.bias.detach().cpu().numpy()
            beta = torch.from_numpy(bias * scale + beta)
            self.bn.bias.copy_(beta)
            self.bias = None

    def broadcast_correction(self, c: torch.Tensor) -> torch.Tensor:
        """Broadcasts a correction factor to the output for elementwise operations.

    Two tensors are "broadcastable" if the following rules hold:
      - Each tensor has at least one dimension.
      - When iterating over the dimension sizes, starting at the trailing
        dimension, the dimension sizes must either be equal,
        one of them is 1, or one of them does not exist.
    """
        expected_output_dim = self.dim + 2
        view_fillers_dim = expected_output_dim - c.dim() - 1
        view_filler = (1, ) * view_fillers_dim
        expected_view_shape = c.shape + view_filler
        return c.view(*expected_view_shape)

    def broadcast_correction_weight(self, c: torch.Tensor) -> torch.Tensor:
        """Broadcasts a correction factor to the weight."""
        if c.dim() != 1:
            raise ValueError("Correction factor needs to have a single dimension")

        expected_weight_dim = self.dim + 2
        view_fillers_dim = expected_weight_dim - c.dim()
        view_filler = (1, ) * view_fillers_dim
        expected_view_shape = c.shape + view_filler
        return c.view(*expected_view_shape)

    def forward(self, x: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        """
    See https://arxiv.org/pdf/1806.08342.pdf section 3.2.2.
    bn(conv(x)) = (conv(x) - E(conv(x))) * gamma / std(conv(x)) + beta
                = (x*W + B - E(x*W + B)) * gamma / sqrt(E((x*W + B - E(x*W + B))^2)) + beta
                = (x*W - E(x*W)) * gamma / std(x*W) + beta
    """
        x = self.get_quant_input(x)
        gamma, beta = self.bn.weight, self.bn.bias
        if self.conv_bn_fused:
            quantized_weight = self.get_quant_weight(self.weight)
            quantized_bias = self.get_quant_bias(self.bias) if self.bias is not None else self.bias
            x = self._conv_forward(x, quantized_weight, quantized_bias)
            return self.get_quant_output(x)

        if self.training and not self.bn_frozen:
            with torch.no_grad():
                batch_mean, batch_var = self.batch_stats(self._conv_forward(x, self.weight, None), self.bias)
                recip_sigma_batch = torch.rsqrt(batch_var + self.bn.eps)
                running_sigma = torch.sqrt(self.bn.running_var +
                                           self.bn.eps) if self.bn.running_var is not None else torch.tensor(
                                               1, device=self.weight.device)

            w_corrected = self.weight * self.broadcast_correction_weight(gamma / running_sigma)
            w_quantized = self.get_quant_weight(w_corrected)

            recip_c = self.broadcast_correction(running_sigma * recip_sigma_batch)
            bias_corrected = beta - gamma * batch_mean * recip_sigma_batch
            bias_quantized = self.broadcast_correction(self.get_quant_bias(bias_corrected))

            y = self._conv_forward(x, w_quantized, None)
            y.mul_(recip_c).add_(bias_quantized)
        else:
            with torch.no_grad():
                recip_running_sigma = torch.rsqrt(self.bn.running_var +
                                                  self.bn.eps) if self.bn.running_var is not None else torch.tensor(
                                                      1, device=self.weight.device)
            w_corrected = self.weight * self.broadcast_correction_weight(gamma * recip_running_sigma)
            w_quantized = self.get_quant_weight(w_corrected)

            mean_corrected = self.bn.running_mean if self.bias is None else self.bn.running_mean - self.bias
            bias_corrected = beta - gamma * mean_corrected * recip_running_sigma
            bias_quantized = self.get_quant_bias(bias_corrected)
            y = self._conv_forward(x, w_quantized, bias_quantized)
        return self.get_quant_output(y)

    def get_fused_float_param(self) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
        if self.conv_bn_fused:
            if self.bias is None:
                raise ValueError(
                    f'{self.__class__.__name__} \'s bias should not be None under conv_bn_fused, please check')
            return torch.nn.Parameter(self.weight, self.weight.requires_grad), torch.nn.Parameter(
                self.bias, self.bias.requires_grad)

        gamma, beta = self.bn.weight, self.bn.bias
        with torch.no_grad():
            recip_running_sigma = torch.rsqrt(self.bn.running_var +
                                              self.bn.eps) if self.bn.running_var is not None else torch.tensor(
                                                  1, device=self.weight.device)
            w_corrected = self.weight * self.broadcast_correction_weight(gamma * recip_running_sigma)

            mean_corrected = self.bn.running_mean if self.bias is None else self.bn.running_mean - self.bias
            # mean_corrected = self.bn.running_mean - (self.bias if self.bias is not None else 0)
            bias_corrected = beta - gamma * mean_corrected * recip_running_sigma
        return torch.nn.Parameter(w_corrected,
                                  w_corrected.requires_grad), torch.nn.Parameter(bias_corrected,
                                                                                 bias_corrected.requires_grad)

    def train(self, mode: bool = True) -> Self:
        """Batchnorm's training behavior is using the self.training flag. Prevent
    changing it if BN is frozen. This makes sure that calling `model.train()`
    on a model with a frozen BN will behave properly.
    """
        self.training = mode
        if not self.bn_frozen:
            for module in self.children():
                module.train(mode)
        return self

    @property
    def is_quantized(self) -> bool:
        return True

    @classmethod
    def from_float(cls, conv: nn.Module, bn: nn.Module, quant_config: Optional[QuantizationConfig],
                   **kwargs: Any) -> nn.Module:
        """Create a qat module from a float module.
        Args:
            conv: The float module to be quantized.
                Must be one of type [nn.Conv2d, nn.Conv3d]
            bn: The float module to be quantized.
                Must be one of type [nn.BatchNorm2d, nn.BatchNorm3d]
            quant_config: QuantizationConfig
        """
        convbn = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bn.track_running_stats,  # bn's track_running_stats will decide conv's bias
            conv.padding_mode,
            bn.eps,
            bn.momentum,
            False,
            quant_config)  # type: ignore
        convbn.weight = conv.weight
        convbn.bias = conv.bias
        convbn.bn.weight = bn.weight
        convbn.bn.bias = bn.bias
        convbn.bn.running_mean = bn.running_mean
        convbn.bn.running_var = bn.running_var
        convbn.bn.num_batches_tracked = bn.num_batches_tracked
        convbn.bn.eps = bn.eps
        return convbn


class QuantizedConvBatchNorm2d(_ConvBnNd, nn.Conv2d):
    """A QuantizedConvBatchNorm2d module is a module fused from
    Conv2d and BatchNorm2d attached with FakeQuantizer modules for weight and
    batchnorm stuffs used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Implementation details: https://arxiv.org/pdf/1806.08342.pdf section 3.2.2

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantizer modules initialized
    to default.
#     """

    def __init__(
        self,
        # conv config
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        # BatchNorm2d args
        eps: float = 1e-05,
        momentum: float = 0.1,
        freeze_bn_stats: bool = False,
        # quant config
        quant_config: Optional[QuantizationConfig] = QuantizationConfig()
    ) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _ConvBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            2,  # dim
            eps,
            momentum,
            freeze_bn_stats,
            quant_config)  # type: ignore [arg-type]

    def _conv_forward(self,
                      input: torch.Tensor,
                      weight: torch.Tensor,
                      bias: Optional[torch.Tensor] = None,
                      output_size: Optional[List[int]] = None) -> torch.Tensor:
        assert output_size is None
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight, bias,
                            self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)


_FUSED_CLS = [QuantizedConvBatchNorm2d]


def update_bn_stats(mod: nn.Module) -> None:
    if type(mod) in _FUSED_CLS:
        mod.update_bn_stats()


def freeze_bn_stats(mod: nn.Module) -> None:
    if type(mod) in _FUSED_CLS:
        mod.freeze_bn_stats()


def fuse_conv_bn(mod: nn.Module) -> None:
    if type(mod) in _FUSED_CLS:
        mod.merge_bn_to_conv()


def clear_non_native_bias(mod: nn.Module) -> None:
    if type(mod) in _FUSED_CLS:
        mod.clear_non_native_bias()
