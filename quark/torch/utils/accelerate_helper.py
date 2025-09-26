#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, Union

import torch
import torch.nn as nn

from quark.shares.utils.import_utils import is_accelerate_available
from quark.torch.utils import getattr_recursive, setattr_recursive

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook
    from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset, find_tied_parameters
elif TYPE_CHECKING:
    from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset


def untie_parameters(model: nn.Module) -> nn.Module:
    """
    Unties parameters from the PyTorch ``model``.

    Some parameters of the model may share the underlying data. This is e.g. the case with token embedding weight, and lm head weight in transformer models.

    This function makes it so that no weights are shared.
    """
    if not is_accelerate_available():
        raise ImportError(
            "The function `untie_parameters` requires the package `accelerate`, but it was not found in the environment. Please install it with `pip install accelerate.`"
        )

    tied_params = find_tied_parameters(model)

    for weight_group in tied_params:
        for param_name in weight_group:
            param = getattr_recursive(model, param_name)
            if isinstance(param, torch.nn.Parameter):
                setattr_recursive(model, param_name, torch.nn.Parameter(param.clone()))
            else:
                setattr_recursive(model, param_name, param.clone())

    return model


class OffloadParameter:
    def __init__(self, module_list: Union[nn.Module, list[nn.Module]]):
        if isinstance(module_list, nn.Module):
            self.module_list: list[nn.Module] = [module_list]
        else:
            self.module_list = module_list

    def __enter__(self) -> None:
        for module in self.module_list:
            for name, sub_module in module.named_modules():
                if hasattr(sub_module, "_hf_hook"):
                    sub_module._hf_hook.pre_forward(sub_module)

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any | None) -> None:
        pass


def offload_to_weights_map(
    weights_map: Union["PrefixedDataset", dict[str, torch.Tensor], "OffloadedWeightsLoader"],
    key: str,
    value: torch.Tensor,
    offload_device: Union[torch.device, Literal["disk"]] | None = None,
) -> None:
    if not is_accelerate_available():
        raise ImportError(
            "The function `offload_to_weights_map` requires the package `accelerate`, but it was not found in the environment. Please install it with `pip install accelerate.`"
        )

    if offload_device == "disk":
        raise ValueError(f"Cannot offload to disk with type {type(weights_map)}")

    if isinstance(weights_map, PrefixedDataset):
        dataset = weights_map.dataset
        key = f"{weights_map.prefix}{key}"
        offload_to_weights_map(dataset, key, value, offload_device)

    elif isinstance(weights_map, OffloadedWeightsLoader):
        if key not in weights_map.all_keys:
            weights_map.all_keys.append(key)

        if len(weights_map.index) <= 0:
            offload_to_weights_map(weights_map.state_dict, key, value, offload_device)

    elif isinstance(weights_map, dict):
        if offload_device is None:
            if key in weights_map:
                offload_device = weights_map[key].device
            else:
                tens = next(iter(weights_map.values()), None)
                if tens is None:
                    raise ValueError("Cannot infer offload device from empty weights_map")
                offload_device = tens.device

        weights_map[key] = value.to(device=offload_device)

    else:
        raise NotImplementedError(f"Updating offload data not implemented for weights_map of type {type(weights_map)}")


def update_offload_parameter(
    module: torch.nn.Module,
    name: str,
    data: torch.Tensor,
    offload_device: Union[torch.device, Literal["disk"]] | None = None,
) -> None:
    param: torch.nn.Parameter = getattr(module, name)
    if param.device != torch.device("meta") and data is not param.data:
        param.data.copy_(data)

    if hasattr(module, "_hf_hook") and isinstance(module._hf_hook, AlignDevicesHook) and module._hf_hook.offload:
        weights_map = module._hf_hook.weights_map
        offload_to_weights_map(weights_map, name, data, offload_device)
