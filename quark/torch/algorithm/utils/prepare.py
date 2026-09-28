#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import contextlib
import fnmatch
import inspect
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from quark.common.utils.log import ScreenLogger
from quark.torch.algorithm.utils.module import get_device, get_layer_idx, get_nested_attr_from_module
from quark.torch.algorithm.utils.utils import clear_memory

logger = ScreenLogger(__name__)


class _StopForward(Exception):
    """Raised by the calibration pre-hook to abort the forward once layer 0's inputs are captured."""


def _has_mixed_layer_types(model: nn.Module) -> bool:
    """Check if the model has mixed layer types.
    Args:
        model: The PyTorch neural network module to check.
    Returns:
        True if the model has more than one unique layer type, False otherwise.
    """
    model_config = getattr(model, "config", None)
    config = getattr(model_config, "text_config", model_config)
    layer_types = getattr(config, "layer_types", None)
    return layer_types is not None and len(set(layer_types)) > 1


def cache_model_inps(
    model: nn.Module, modules: nn.ModuleList, samples: DataLoader[torch.Tensor]
) -> tuple[nn.ModuleList, dict[str, Any], list[torch.Tensor]]:
    """Capture calibration input embeddings and forward kwargs.

    For uniform-attention models we hook layer 0 and early-exit the forward as soon as its inputs are
    captured, replaying that one shared kwargs dict through every layer.

    Models like Gemma2/Gemma3 interleave ``sliding_attention`` and ``full_attention`` layers, which
    receive a different ``attention_mask`` (sliding vs full causal) and -- for Gemma3 -- a different
    ``position_embeddings`` (local vs global RoPE table). Replaying layer 0's kwargs through every layer
    would apply the wrong mask/rotary to layers of the other type. For these we hook *every* layer and run
    a real full forward *once* to record each layer's own kwargs keyed by layer index under the
    ``_per_layer_kwargs`` key for replay to resolve.

    Only the *first* sample runs the full forward: per-layer kwargs depend on the input shape/positions,
    not its values, so same-shaped calibration samples produce identical kwargs (and replay already uses
    one representative set against every captured input). Subsequent samples early-exit at layer 0 like the
    uniform path, so a mixed model costs one full forward plus N-1 layer-0-only forwards rather than N.
    """
    mixed = _has_mixed_layer_types(model)
    inps: list[torch.Tensor] = []
    layer_kwargs: dict[str, Any] = {}
    per_layer_kwargs: dict[int, dict[str, Any]] = {}
    captured_per_layer = False  # set once the first full forward has recorded every layer's kwargs
    offloaded_devices: dict[int, torch.device] = {}  # layers pulled on-device for the full forward

    def catch_hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Forward pre-hook to capture layer inputs and kwargs for calibration.
        Captures the input tensors and forward kwargs for each layer during model calibration.
        Handles device management for offloaded layers and controls forward propagation flow.
        Args:
            module: The layer module being hooked.
            args: Positional arguments passed to the module's forward method.
            kwargs: Keyword arguments passed to the module's forward method.
        Side Effects:
            - Moves offloaded layers to the input device and tracks original device in offloaded_devices.
            - Captures per-layer kwargs in per_layer_kwargs for mixed-layer models.
            - Appends layer 0 inputs to inps list.
            - Updates layer_kwargs with layer 0's forward kwargs.
            - Raises _StopForward to abort forward pass after layer 0 input capture (conditional).
        Raises:
            _StopForward: When layer 0 input is captured and per-layer kwargs are not needed.
        """
        # Pull an offloaded layer onto the input's device just before it runs (no-op when already there).
        # Accelerate-managed layers stay excluded: their hook owns the execution device, so moving
        # them here just swaps a cuda/cpu mismatch for cuda:N/cuda:M. Callers must therefore not
        # strand them off-device (see ``AutoSmoothQuantProcessor.init_quant``).
        hidden = args[0] if args else kwargs["hidden_states"]
        module_device = get_device(module)
        if not hasattr(module, "_hf_hook") and module_device != hidden.device:
            offloaded_devices[id(module)] = module_device
            module.to(hidden.device)
        own_kwargs = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        if len(args) > 1:
            params = list(inspect.signature(module.forward).parameters)
            for name, value in zip(params[1 : len(args)], args[1:], strict=True):
                own_kwargs.setdefault(name, value)
        if mixed and not captured_per_layer:
            per_layer_kwargs[get_layer_idx(module)] = own_kwargs
        if module is modules[0]:
            inps.append(args[0].detach() if args else kwargs["hidden_states"].detach())
            layer_kwargs.clear()
            layer_kwargs.update(own_kwargs)
            # Per-layer kwargs need one full forward; afterwards (and always for uniform models) only
            # layer 0's input is needed, so abort the rest of the forward.
            if not mixed or captured_per_layer:
                raise _StopForward
        if mixed and not captured_per_layer and module is modules[-1]:
            orig_device = offloaded_devices.pop(id(module), None)
            if orig_device is not None:
                module.to(orig_device)
            raise _StopForward

    def restore_hook(module: nn.Module, args: Any, output: Any) -> None:
        """Restore offloaded layers to their original device after forward pass.
        This hook is called after a module's forward pass completes. If the module was
        temporarily moved to a different device for execution, it is moved back to its
        original device.
        Args:
            module: The module that just completed its forward pass.
            args: Forward pass arguments (unused).
            output: Forward pass output (unused).
        """
        # Send a just-in-time-loaded layer back to where it came from once its forward is done.
        orig_device = offloaded_devices.pop(id(module), None)
        if orig_device is not None:
            module.to(orig_device)

    cur_layer_device = (
        get_device(modules[0])
        if get_device(modules[0]) != torch.device("meta")
        else modules[0]._hf_hook.execution_device
    )

    # Mixed models need every layer's kwargs, so hook all layers and run a full forward; uniform models
    # only need layer 0 and bail out via _StopForward.
    hooked = modules if mixed else modules[:1]
    handles = [m.register_forward_pre_hook(catch_hook, with_kwargs=True) for m in hooked]
    handles += [m.register_forward_hook(restore_hook) for m in hooked]

    logger.info("Caching model inputs for quantization algorithm...")
    with torch.no_grad():
        for sample in tqdm(samples, desc="Caching layer inputs"):
            with contextlib.suppress(_StopForward):  # full forward only on the first mixed-model sample
                if isinstance(sample, torch.Tensor):
                    model(sample.to(cur_layer_device), use_cache=False)
                else:
                    model(**{key: val.to(cur_layer_device) for key, val in sample.items()})
            # After the first mixed-model sample, every layer's kwargs are recorded; later samples
            # early-exit at layer 0 like the uniform path.
            captured_per_layer = True

    for h in handles:
        h.remove()
    del samples
    clear_memory()

    if mixed:
        layer_kwargs["_per_layer_kwargs"] = per_layer_kwargs
    return modules, layer_kwargs, inps


def move_embed(
    model: nn.Module, embedding_layer_name_list: list[str], device: dict[str, torch.device] | torch.device
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

            layer_dict = {
                "prev_op": get_nested_attr_from_module(module, layer["prev_op"]),
                "layers": linear_layers,
                "inp": input_feat[layer["inp"]],
            }

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

            layer_dict = {
                "prev_op": prev_op,
                "layers": [linear_layer],
                "inp": inp,
            }

            layers.append(layer_dict)

    layers: list[dict[str, Any]] = []
    has_kwargs = True  # For first layer from module, input kwargs.

    for layer in scaling_layers:
        try:  # dense
            _ = get_nested_attr_from_module(module, layer["layers"][0])  # OK for dense, and exception for moe
            has_kwargs = get_dense_layers(module, input_feat, module_kwargs, layer, layers, has_kwargs)

        except (AttributeError, KeyError):  # moe
            if fnmatch.filter(input_feat.keys(), "*" + layer["inp"]):  # moe: gate|up|down_proj
                # match layers
                matched_layers = []
                for layer_name in layer["layers"]:
                    matched_layers += fnmatch.filter(input_feat.keys(), "*" + layer_name)

                # moe gate/up_proj cannot use AWQ/SQ/ASQ, because moe has the pattern: post_attention_layernorm + (router, gate_proj, up_proj)
                try:  # moe down_proj
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
