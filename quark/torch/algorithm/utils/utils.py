#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any, Dict, Optional, Tuple, List
import gc
import torch
import torch.nn as nn


class TensorData(torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]):

    def __init__(self, data: List[torch.Tensor], targets: List[torch.Tensor], device: torch.device) -> None:
        self.data = data
        self.targets = targets
        self.device = device

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index]
        y = self.targets[index]
        return x.to(self.device), y.to(self.device)

    def __len__(self) -> int:
        return len(self.targets)


def clear_memory(weight: Optional[torch.Tensor] = None) -> None:
    if weight is not None:
        del weight
    gc.collect()
    torch.cuda.empty_cache()


def get_device_map(model: nn.Module, is_accelerate: Optional[bool]) -> Dict[str, Any]:
    device_map = {"": model.device}
    if is_accelerate:
        device_map = model.hf_device_map
    return device_map


def set_device_map(model: nn.Module, device_map: Dict[str, Any]) -> nn.Module:
    if len(device_map) == 1 and "" in device_map.keys():
        model = model.to(device_map[""])
    else:
        for name, module in model.named_modules(remove_duplicate=False):
            if name in device_map:
                module.to(torch.device(device_map[name])) if isinstance(device_map[name], int) else model.to(
                    device_map[name])
    return model
