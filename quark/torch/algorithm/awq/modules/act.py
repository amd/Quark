#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import cast
import torch
import torch.nn as nn


class ScaledActivation(nn.Module):

    def __init__(self, module: nn.Module, scales: torch.Tensor) -> None:
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.act(x) / self.scales.view(1, 1, -1).to(x.device))
