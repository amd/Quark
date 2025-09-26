#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

import torch
from torch.nn import Linear

from .quant_base_ops import QuantizeWrapper


class QMatMul(QuantizeWrapper, Linear):  # type: ignore
    def __init__(self, **kwargs: Any) -> None:
        QuantizeWrapper.__init__(self, **kwargs)
        Linear.__init__(self, bias=False, **kwargs)

    def forward_impl(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        A = input
        B = weight

        return torch.matmul(A, B)
