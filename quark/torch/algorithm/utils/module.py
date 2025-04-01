#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations
from typing import Any, Union, Dict, List, Tuple, cast, TypeVar
import torch
import torch.nn as nn
from functools import reduce


def get_named_linears(module: nn.Module) -> Dict[str, nn.Linear]:
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_named_quant_linears(module: nn.Module) -> Dict[str, nn.Linear]:
    from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def get_moe_layers(module: nn.Module) -> Dict[str, nn.Linear]:
    return {name: m for name, m in module.named_modules() if "MoeBlock" in m.__class__.__name__}


def get_op_by_name(module: nn.Module, op_name: str) -> nn.Module:
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return cast(nn.Module, m)
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def get_op_name(module: nn.Module, op: nn.Module) -> str:
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return cast(str, name)
    raise ValueError(f"Cannot find op {op} in module {module}")


NestedStrListTuple = Union[List[Tuple[str, Union[Tuple[str, ...], torch.Tensor], torch.Tensor]],
                           Tuple[str, Union[Tuple[str, ...], torch.Tensor], torch.Tensor], object]


def append_str_prefix(x: NestedStrListTuple, prefix: str) -> Any:
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x


def set_op_by_name(layer: Union[nn.Module, nn.ModuleList], name: str, new_module: nn.Module) -> None:
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit() and isinstance(mod_, nn.ModuleList):
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_device(obj: Union[torch.Tensor, nn.Module]) -> torch.device:
    if isinstance(obj, torch.Tensor):
        return obj.device
    elif isinstance(obj, nn.Module):
        return next(obj.parameters()).device
    else:
        raise TypeError("obj must be a torch.Tensor or nn.Module")


def get_dtype(obj: Union[torch.Tensor, nn.Module]) -> torch.dtype:
    if isinstance(obj, torch.Tensor):
        return obj.dtype
    elif isinstance(obj, nn.Module):
        return next(obj.parameters()).dtype
    else:
        raise TypeError("obj must be a torch.Tensor or nn.Module")


T = TypeVar('T', torch.Tensor, torch.nn.Module)


def move_to_device(obj: T, device: torch.device) -> T:
    if get_device(obj) != device:
        obj = obj.to(device)

    return obj


def get_nested_attr_from_module(obj: nn.Module, attr_path: str) -> Any:
    """
    Retrieves the value of a nested attribute based on a given attribute path string.

    Parameters:
    - obj: The starting object.
    - attr_path: The string representing the attribute path, such as "model.decoder.layers".

    Returns:
    - The value of the nested attribute.
    """

    return reduce(getattr, attr_path.split('.'), obj)
