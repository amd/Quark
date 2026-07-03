#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from collections.abc import Iterator

import torch
from torch import nn

from quark.common.utils.log import ScreenLogger
from quark.experimental.torch.algorithm.blockwise_joint_tuning.quantize.learnable_linear import (
    ExperimentalLearnableQuantizedLinear,
)

logger = ScreenLogger(__name__)


def set_quant_state(model: nn.Module, weight_quant: bool = False) -> None:
    for module in model.modules():
        if isinstance(module, ExperimentalLearnableQuantizedLinear):
            module.enable_weight_quant(weight_quant)


def set_op_by_name(layer: nn.Module, name: str, new_module: nn.Module) -> None:
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for idx in range(len(levels) - 1):
            level = levels[idx]
            try:
                # Prefer attribute traversal because nn.Module stores children
                # in _modules and supports numeric keys such as "0".
                mod_ = getattr(mod_, level)
            except AttributeError:
                if not level.isdigit():
                    logger.error(
                        f"Failed to resolve module path '{name}': '{level}' is neither an attribute of "
                        f"{type(mod_).__name__} nor a numeric submodule index."
                    )
                    raise
                mod_ = mod_[int(level)]
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def trainable_parameters(model: nn.Module) -> Iterator[torch.nn.Parameter]:
    params = []
    for _, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
    return iter(params)


def trainable_parameters_num(model: nn.Module) -> int:
    total = 0
    for _, p in model.named_parameters():
        if p.requires_grad:
            total += p.numel()
    return total


@torch.no_grad()
def quant_inplace(model: nn.Module) -> None:
    for _, module in model.named_modules():
        if isinstance(module, ExperimentalLearnableQuantizedLinear):
            module.weight.data = module.weight_quantizer(module.weight.data)
