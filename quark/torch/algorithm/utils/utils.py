#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import gc
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class TensorData(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data: list[torch.Tensor], targets: list[torch.Tensor], device: torch.device) -> None:
        self.data = data
        self.targets = targets
        self.device = device

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index]
        y = self.targets[index]
        return x.to(self.device), y.to(self.device)

    def __len__(self) -> int:
        return len(self.targets)


def clear_memory(weight: torch.Tensor | None = None) -> None:
    if weight is not None:
        del weight
    QUARK_AWQ_MEMORY_OPTIMIZATION = os.environ.get("QUARK_AWQ_MEMORY_OPTIMIZATION", None) == "1"
    # When memory recycling is turned on in QUARK_AWQ_MEMORY_OPTIMIZATION mode
    if QUARK_AWQ_MEMORY_OPTIMIZATION:
        gc.collect()
        torch.cuda.empty_cache()


def get_device_map(model: nn.Module, is_accelerate: bool | None) -> dict[str, Any]:
    device_map = {"": model.device}
    if is_accelerate:
        device_map = model.hf_device_map
    return device_map


def set_device_map(model: nn.Module, device_map: dict[str, Any]) -> nn.Module:
    if len(device_map) == 1 and "" in device_map.keys():
        model = model.to(device_map[""])
    else:
        for name, module in model.named_modules(remove_duplicate=False):
            if name in device_map:
                # if cpu or disk, you can't move them
                if device_map[name] == "cpu" or device_map[name] == "disk":
                    break
                module.to(torch.device(device_map[name])) if isinstance(device_map[name], int) else model.to(
                    device_map[name]
                )
    return model


def get_num_attn_heads_from_model(model: nn.Module) -> tuple[int, int]:
    num_attention_heads, num_key_value_heads = -1, -1
    if hasattr(model, "config"):
        if hasattr(model.config, "num_attention_heads") and hasattr(
            model.config, "num_key_value_heads"
        ):  # llm: llama, qwen, deepseek, chatglm, grok, dbrx, ...
            num_attention_heads = model.config.num_attention_heads
            num_key_value_heads = model.config.num_key_value_heads
        elif hasattr(model.config, "text_config"):  # vlm: llama4, mllama
            if hasattr(model.config.text_config, "num_attention_heads") and hasattr(
                model.config.text_config, "num_key_value_heads"
            ):
                num_attention_heads = model.config.text_config.num_attention_heads
                num_key_value_heads = model.config.text_config.num_key_value_heads

    return num_attention_heads, num_key_value_heads


def is_attention_module(model: object) -> bool:
    return "attention" in type(model).__name__.lower()
