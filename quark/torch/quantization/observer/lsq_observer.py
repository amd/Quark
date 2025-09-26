#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import QuantizationSpec
from quark.shares.utils.log import ScreenLogger
from quark.torch.quantization.observer.observer import UniformScalingObserver
from quark.torch.quantization.utils import calculate_qmin_qmax, get_num_bits

logger = ScreenLogger(__name__)


class LSQObserver(UniformScalingObserver):
    def __init__(self, qspec: QuantizationSpec, device: torch.device | None = None) -> None:
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
        self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int, device=device))
        self.register_buffer("eps", torch.tensor([torch.finfo(torch.float32).eps], device=device), persistent=False)

        self.register_buffer("initialized", torch.tensor([0], dtype=torch.uint8, device=device), persistent=False)

    def forward(self, x: torch.Tensor) -> None:
        if self.training and self.initialized == 0:
            x_detached = x.detach()
            if self.ch_axis:
                if not isinstance(self.ch_axis, int):
                    raise RuntimeError(
                        f"An integer ch_axis must be speficied for per_channel quantization, but given {self.ch_axis}"
                    )
                num_channels = x.size(self.ch_axis)
                self.scale = torch.nn.Parameter(torch.ones(num_channels, dtype=torch.float, device=self.scale.device))
                zero_point = self.zero_point  # type: ignore[has-type]
                self.zero_point = torch.zeros(num_channels, dtype=zero_point.dtype, device=zero_point.device)
                eps = self.eps
                self.eps = torch.tensor(
                    num_channels * [torch.finfo(torch.float32).eps], dtype=eps.dtype, device=eps.device
                )
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

    def _calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scale, self.zero_point
