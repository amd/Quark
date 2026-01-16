#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseAlgoProcessor(ABC):
    @abstractmethod
    def __init__(
        self,
        model: nn.Module,
        quant_algo_config: Any,
        calib_data: DataLoader[torch.Tensor]
        | DataLoader[list[dict[str, torch.Tensor]]]
        | DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        pass

    @abstractmethod
    def apply(self) -> None:
        pass
