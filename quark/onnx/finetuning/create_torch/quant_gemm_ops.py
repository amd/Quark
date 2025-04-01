#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch

from .quant_base_ops import QuantizeWrapper
from torch.nn import Linear
from typing import Any


class QGemm(QuantizeWrapper, Linear):

    def __init__(self, transA: int = 0, transB: int = 0, **kwargs: Any) -> None:
        # These parameters are defined by ourself
        w_alpha = kwargs.pop('w_alpha', 1.0)
        b_beta = kwargs.pop('b_beta', 1.0)

        # The alpha and beta's implement is in this
        QuantizeWrapper.__init__(self, w_alpha=w_alpha, b_beta=b_beta, **kwargs)
        Linear.__init__(self, **kwargs)

        self.transA = transA
        self.transB = transB

    def forward_impl(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:

        if self.transA != 0:
            A = input.transpose(-1, -2)
        else:
            A = input

        if self.transB != 0:
            B = weight.transpose(-1, -2)
        else:
            B = weight

        # Here we could use linear or matmul
        # return torch.nn.functional.linear(A, B, bias=None)
        return torch.matmul(A, B)
