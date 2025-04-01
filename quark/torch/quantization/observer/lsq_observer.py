#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from quark.torch.quantization.observer.observer import UniformScalingObserver
from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.utils import get_num_bits, calculate_qmin_qmax
import math
import torch
from typing import Tuple, Optional


class LSQObserver(UniformScalingObserver):

    def __init__(self, qspec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__(qspec, device)
        _bitwidth = get_num_bits(qspec.dtype)
        assert isinstance(_bitwidth, int)
        self.ch_axis = qspec.ch_axis

        if not qspec.symmetric:
            self.quant_min = 0
            self.quant_max = 2**_bitwidth - 1
        else:
            self.quant_min, self.quant_max = calculate_qmin_qmax(qspec.dtype)

        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float, device=device))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int, device=device))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps], device=device))

        self.register_buffer('initialized', torch.tensor([0], dtype=torch.uint8, device=device))

    def forward(self, x: torch.Tensor) -> None:
        if self.training and self.initialized == 0:
            x_detached = x.detach()
            if self.ch_axis:
                if not isinstance(self.ch_axis, int):
                    raise RuntimeError('An integer ch_axis must be speficied for per_channel quantization, '
                                       'but given {}'.format(self.ch_axis))
                num_channels = x.size(self.ch_axis)
                self.scale = torch.nn.Parameter(torch.ones(num_channels, dtype=torch.float, device=self.scale.device))
                zero_point = getattr(self, "zero_point")
                setattr(self, "zero_point", torch.zeros(num_channels, dtype=zero_point.dtype, device=zero_point.device))
                eps = getattr(self, "eps")
                setattr(
                    self, "eps",
                    torch.tensor(num_channels * [torch.finfo(torch.float32).eps], dtype=eps.dtype, device=eps.device))
                if self.ch_axis < 0:
                    self.ch_axis += x_detached.dim()
                dims = [i for i in range(x_detached.dim()) if i != self.ch_axis]
                scale = 2 * x_detached.abs().mean(dims) / math.sqrt(self.quant_max)
                zero_point = torch.zeros([x_detached.size(self.ch_axis)], device=x.device)
            else:
                scale = 2 * x_detached.abs().mean() / math.sqrt(self.quant_max)
                zero_point = torch.zeros(1, device=x.device)

            self.scale.data.copy_(scale)
            self.zero_point.data.copy_(zero_point)

    def _calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.scale, self.zero_point
