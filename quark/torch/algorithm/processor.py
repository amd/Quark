#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union, List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseAlgoProcessor(ABC):

    @abstractmethod
    def __init__(
        self, model: nn.Module, quant_algo_config: Any, calib_data: Union[DataLoader[torch.Tensor],
                                                                          DataLoader[List[Dict[str, torch.Tensor]]],
                                                                          DataLoader[Dict[str, torch.Tensor]]]
    ) -> None:
        pass

    @abstractmethod
    def apply(self) -> None:
        pass
