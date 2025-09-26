#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import fnmatch
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.utils.module import get_device, get_nested_attr_from_module
from quark.torch.algorithm.utils.utils import clear_memory

logger = ScreenLogger(__name__)


def cache_model_inps(
    model: nn.Module, modules: nn.ModuleList, samples: DataLoader[torch.Tensor]
) -> tuple[nn.ModuleList, dict[str, Any], list[torch.Tensor]]:
    inps: list[torch.Tensor] = []
    layer_args: list[Union[torch.Tensor, None]] = []
    layer_kwargs: dict[str, Any] = {}

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(
            self,
            module: nn.Module,
            inps: list[torch.Tensor],
            layer_args: list[Union[torch.Tensor, None]],
            layer_kwargs: dict[str, Any],
        ) -> None:
            super().__init__()
            self.module = module
            self.inps = inps
            self.layer_args = layer_args
            self.layer_kwargs = layer_kwargs

        # in case need module's attribute is explicitly needed
        def __getattr__(self, name: str) -> Any:
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, *args: torch.Tensor, **kwargs: Any) -> None:
            # assume first input to forward is hidden states
            if len(args) > 0:
                hidden_states = args[0]
                if len(self.layer_args) == 0:
                    self.layer_args.extend(
                        args[1:]
                    )  # For attention_mask and rotary_pos_emb, the value of the new input is always same, so it is kept once
            else:
                first_key = list(kwargs.keys())[0]
                hidden_states = kwargs.pop(first_key)

            self.inps.append(hidden_states)
            self.layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs

    cur_layer_device = (
        get_device(modules[0])
        if not get_device(modules[0]) == torch.device("meta")
        else modules[0]._hf_hook.execution_device
    )
    required_kwargs = inspect.signature(modules[0].forward).parameters
    modules[0] = Catcher(modules[0], inps, layer_args, layer_kwargs)
    for sample in samples:
        if isinstance(sample, torch.Tensor):
            try:
                model(sample.to(cur_layer_device), use_cache=False)
            except ValueError:  # work with early exit
                pass
        else:
            try:
                model(**{key: val.to(cur_layer_device) for key, val in sample.items()})
            except ValueError:  # work with early exit
                pass
    del samples
    modules[0] = modules[0].module  # restore

    clear_memory()

    arg_idx = 0

    for k, v in required_kwargs.items():
        if k == "hidden_states" or k in layer_kwargs or v.kind == v.VAR_KEYWORD:
            # `layer_args` here holds the positional arguments from position one, so
            # `arg_idx` is not incremented here.
            continue
        elif arg_idx < len(layer_args):  # pragma: no cover
            layer_kwargs[k] = layer_args[arg_idx]
            arg_idx += 1
        else:
            break

    return modules, layer_kwargs, inps


def move_embed(
    model: nn.Module, embedding_layer_name_list: list[str], device: Union[dict[str, torch.device], torch.device]
) -> None:
    for embedding_layer_name in embedding_layer_name_list:
        embedding_layer = get_nested_attr_from_module(model, embedding_layer_name)
        if isinstance(device, dict):
            embedding_layer = embedding_layer.to(device[embedding_layer_name])
        else:
            embedding_layer = embedding_layer.to(device)


def get_layers_for_scaling(
    module: nn.Module, input_feat: dict[str, Any], module_kwargs: dict[str, Any], scaling_layers: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    def get_dense_layers(
        module: nn.Module,
        input_feat: dict[str, Any],
        module_kwargs: dict[str, Any],
        layer: dict[str, Any],
        layers: list[dict[str, Any]],
        has_kwargs: bool,
    ) -> bool:
        if layer["inp"] in input_feat:  # hooked inputs
            linear_layers = []
            for i in range(len(layer["layers"])):
                linear_layers.append(get_nested_attr_from_module(module, layer["layers"][i]))

            layer_dict = dict(
                prev_op=get_nested_attr_from_module(module, layer["prev_op"]),
                layers=linear_layers,
                inp=input_feat[layer["inp"]],
            )

            if "module2inspect" in layer and layer["module2inspect"] is not None:
                if layer["module2inspect"] == "":
                    layer_dict["module2inspect"] = module
                else:
                    layer_dict["module2inspect"] = get_nested_attr_from_module(module, layer["module2inspect"])
            if has_kwargs:
                layer_dict["kwargs"] = module_kwargs
                has_kwargs = False

            layers.append(layer_dict)

        return has_kwargs

    def get_moe_down_proj_layers(
        module: nn.Module,
        input_feat: dict[str, Any],
        module_kwargs: dict[str, Any],
        matched_layers: list[str],
        layers: list[dict[str, Any]],
    ) -> None:
        for i in range(len(matched_layers)):
            prefix = ".".join(matched_layers[i].split(".")[:-1])  # feed_forward.experts.0.up_proj
            linear_layer = get_nested_attr_from_module(module, matched_layers[i])

            # pre_layer
            prev_op = get_nested_attr_from_module(module, prefix + "." + layer["prev_op"])

            inp = input_feat[prefix + "." + layer["inp"]]

            layer_dict = dict(
                prev_op=prev_op,
                layers=[linear_layer],
                inp=inp,
            )

            layers.append(layer_dict)

    layers: list[dict[str, Any]] = []
    has_kwargs = True  # For first layer from module, input kwargs.

    for layer in scaling_layers:
        try:  # dense
            _ = get_nested_attr_from_module(module, layer["layers"][0])  # check is_moe
            has_kwargs = get_dense_layers(module, input_feat, module_kwargs, layer, layers, has_kwargs)

        except (AttributeError, KeyError):  # moe
            if fnmatch.filter(input_feat.keys(), "*" + layer["inp"]):  # moe: gate|up|down_proj
                # match layers
                matched_layers = []
                for layer_name in layer["layers"]:
                    matched_layers += fnmatch.filter(input_feat.keys(), "*" + layer_name)

                try:
                    get_moe_down_proj_layers(module, input_feat, module_kwargs, matched_layers, layers)

                except (AttributeError, KeyError):  # no matched patten
                    logger.warning(f"Skip smoothing this layer as no matched pattern is found for {layer}.")

    return layers


def get_model_layers(model: nn.Module, layers_name: str) -> nn.ModuleList:
    model_layer = get_nested_attr_from_module(model, layers_name)
    return cast(nn.ModuleList, model_layer)


def init_device_map(model: nn.Module) -> dict[str, torch.device]:
    from collections import defaultdict

    k_name_v_device: dict[Any, torch.device] = {}
    if hasattr(model, "hf_device_map"):
        if len(model.hf_device_map) == 1:
            device = [v for _, v in model.hf_device_map.items()][0]
            k_name_v_device = defaultdict(lambda: device)
        else:
            k_name_v_device = {
                layer_name: (
                    torch.device(layer_device)
                    if isinstance(layer_device, str)
                    else torch.device(f"cuda:{layer_device}")
                )
                for layer_name, layer_device in model.hf_device_map.items()
            }
    else:
        # `device` is an attribute for transformers.PretrainedModel models, but not nn.Module in general.
        device = model.device if hasattr(model, "device") else next(model.parameters()).device
        k_name_v_device = defaultdict(lambda: device)
    return k_name_v_device


def reset_model_kv_cache(model: nn.Module, use_cache: bool = False) -> bool:
    forward_pass_use_cache = True
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        forward_pass_use_cache = model.config.use_cache
        model.config.use_cache = use_cache
    elif (
        hasattr(model, "config")
        and hasattr(model.config, "text_config")
        and hasattr(model.config.text_config, "use_cache")
    ):
        forward_pass_use_cache = model.config.text_config.use_cache
        model.config.text_config.use_cache = use_cache
    return forward_pass_use_cache


def init_blockwise_algo(
    model: nn.Module, model_decoder_layers: str | None, data_loader: DataLoader[torch.Tensor]
) -> tuple[nn.ModuleList, dict[str, Any], list[torch.Tensor]]:
    assert model_decoder_layers is not None
    modules = get_model_layers(model, model_decoder_layers)
    forward_pass_use_cache = reset_model_kv_cache(model, use_cache=False)
    modules, layer_kwargs, inputs = cache_model_inps(model, modules, data_loader)
    reset_model_kv_cache(model, use_cache=forward_pass_use_cache)
    return modules, layer_kwargs, inputs
