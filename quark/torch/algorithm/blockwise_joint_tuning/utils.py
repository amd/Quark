#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import inspect
import math
import os
import time
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from quark.common.utils.log import ScreenLogger
from quark.torch.algorithm.blockwise_joint_tuning.optim.amp import NativeScalerWithGradNormCount
from quark.torch.algorithm.utils.module import get_device, move_to_device
from quark.torch.algorithm.utils.utils import clear_memory

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = ScreenLogger(__name__)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def enable_weight_requires_grad(model: nn.Module, trainable_modules: list[str]) -> tuple[list[nn.Parameter], list[str]]:
    params: list[nn.Parameter] = []
    names: list[str] = []
    for module_name, module in model.named_modules():
        if len(trainable_modules) > 0 and not any(keyword in module_name for keyword in trainable_modules):
            continue
        for param_name, p in module.named_parameters(recurse=False):
            if param_name == "weight":
                p.requires_grad = True
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                params.append(p)
                names.append(full_name)
    return params, names


def enable_learnable_qparams_requires_grad(
    model: nn.Module, trainable_modules: list[str], quant_trainable_modules: list[str]
) -> tuple[list[nn.Parameter], list[str]]:
    params: list[nn.Parameter] = []
    names: list[str] = []
    for param_name, p in model.named_parameters():
        if "scale" not in param_name and "zero_point" not in param_name:
            continue
        if len(trainable_modules) > 0 and not any(keyword in param_name for keyword in trainable_modules):
            continue
        if len(quant_trainable_modules) > 0 and not any(keyword in param_name for keyword in quant_trainable_modules):
            continue
        p.requires_grad = True
        params.append(p)
        names.append(param_name)
    return params, names


def _move_nested_to_device(v: Any, device: torch.device) -> Any:
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return move_to_device(v, device)
    if isinstance(v, dict):
        return {kk: _move_nested_to_device(vv, device) for kk, vv in v.items()}
    if isinstance(v, tuple):
        return tuple(_move_nested_to_device(x, device) for x in v)
    if isinstance(v, list):
        return [_move_nested_to_device(x, device) for x in v]
    return v


def _align_attention_mask_for_input(attention_mask: Any, input_tensor: torch.Tensor) -> Any:
    if attention_mask is None or not isinstance(attention_mask, torch.Tensor):
        return attention_mask
    mask = attention_mask
    batch = input_tensor.shape[0]
    if mask.dim() >= 1:
        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, *mask.shape[1:])
        elif mask.shape[0] != batch:
            mask = mask[:1].expand(batch, *mask.shape[1:])
    mask = move_to_device(mask, input_tensor.device)
    if torch.is_floating_point(mask):
        mask = mask.to(input_tensor.dtype)
    return mask


def _build_layer_forward_kwargs(
    layer: nn.Module, module_kwargs: dict[str, Any], input_tensor: torch.Tensor, device: torch.device
) -> dict[str, Any]:
    params = inspect.signature(layer.forward).parameters
    accepts_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    filtered_kwargs: dict[str, Any] = {}
    for k, v in module_kwargs.items():
        if accepts_var_kwargs or k in params:
            filtered_kwargs[k] = _move_nested_to_device(v, device)
    if "attention_mask" in filtered_kwargs:
        filtered_kwargs["attention_mask"] = _align_attention_mask_for_input(
            filtered_kwargs["attention_mask"], input_tensor
        )
    if "past_key_value" in filtered_kwargs:
        filtered_kwargs["past_key_value"] = None
    if "past_key_values" in filtered_kwargs:
        filtered_kwargs["past_key_values"] = None
    return filtered_kwargs


def block_batch_forward(
    layer: nn.Module,
    module_kwargs: dict[str, Any],
    input_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    additional_layer_inputs = _build_layer_forward_kwargs(layer, module_kwargs, input_tensor, device)
    output = layer(input_tensor, **additional_layer_inputs)
    if isinstance(output, tuple):
        if not isinstance(output[0], torch.Tensor):
            raise ValueError(f"Unexpected layer output[0] type: {type(output[0])}")
        output = output[0]
    elif not isinstance(output, torch.Tensor):
        raise ValueError(f"Unexpected layer output type: {type(output)}")
    return output


@torch.no_grad()
def block_forward(
    layer: nn.Module,
    module_kwargs: dict[str, Any],
    num_batches: int,
    device: torch.device,
    layer_inputs: list[torch.Tensor],
    outputs_acc: list[torch.Tensor],
    cache_examples_on_gpu: bool,
) -> list[torch.Tensor]:
    if get_device(layer) != torch.device("meta"):
        layer = move_to_device(layer, device)
    assert not isinstance(layer_inputs, torch.Tensor)
    for j in range(num_batches):
        layer_input = move_to_device(layer_inputs[j], device)
        layer_output = block_batch_forward(layer, module_kwargs, layer_input, device)
        layer_output = move_to_device(layer_output, device if cache_examples_on_gpu else torch.device("cpu"))
        outputs_acc.append(layer_output)
    return outputs_acc


def blockwise_joint_training(
    layer: nn.Module,
    module_kwargs: dict[str, Any],
    trainable_modules: list[str],
    quant_trainable_modules: list[str],
    layer_inputs: list[torch.Tensor],
    fp_layer_outputs: list[torch.Tensor],
    quant_val_inps: list[torch.Tensor],
    fp_val_outputs: list[torch.Tensor],
    device: torch.device,
    epochs: int,
    quant_lr: float,
    weight_lr: float,
    min_lr_factor: float,
    weight_decay: float,
    qparam_weight_decay: float,
    max_grad_norm: float,
    layer_index: int,
    loss_func: nn.Module,
) -> None:
    del quant_val_inps, fp_val_outputs, max_grad_norm
    set_requires_grad(layer, False)

    num_update_steps_per_epoch = max(len(layer_inputs), 1)
    max_steps = int(epochs * num_update_steps_per_epoch)

    param_groups: list[dict[str, object]] = []
    quant_scheduler: CosineAnnealingLR | None = None
    weight_scheduler: CosineAnnealingLR | None = None
    quant_index: int | None = None
    weight_index: int | None = None

    qparam_params, _ = enable_learnable_qparams_requires_grad(layer, trainable_modules, quant_trainable_modules)
    if quant_lr > 0 and len(qparam_params) > 0:
        param_groups.append({"params": qparam_params, "lr": quant_lr, "weight_decay": qparam_weight_decay})
        empty_optimizer = torch.optim.AdamW([torch.tensor(0.0)], lr=quant_lr)
        quant_scheduler = CosineAnnealingLR(empty_optimizer, T_max=max_steps, eta_min=quant_lr / min_lr_factor)
        quant_index = len(param_groups) - 1

    weight_params, _ = enable_weight_requires_grad(layer, trainable_modules)
    if weight_lr > 0 and len(weight_params) > 0:
        param_groups.append({"params": weight_params, "lr": weight_lr, "weight_decay": weight_decay})
        empty_optimizer = torch.optim.AdamW([torch.tensor(0.0)], lr=weight_lr)
        weight_scheduler = CosineAnnealingLR(empty_optimizer, T_max=max_steps, eta_min=weight_lr / min_lr_factor)
        weight_index = len(param_groups) - 1

    if len(param_groups) == 0:
        logger.info("Skip blockwise joint tuning for this layer (no trainable params matched).")
        return

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
    loss_scaler = NativeScalerWithGradNormCount()

    for epoch in range(epochs):
        start_time = time.time()
        loss_list: list[torch.Tensor] = []
        for quant_inputs, fp_targets in zip(layer_inputs, fp_layer_outputs, strict=False):
            with torch.amp.autocast("cuda"):
                quant_inputs = quant_inputs.to(device)
                fp_targets = fp_targets.to(device)
                outputs = block_batch_forward(layer, module_kwargs, quant_inputs, device)
                loss = loss_func(fp_targets, outputs)
            if not math.isfinite(loss.item()):
                raise RuntimeError("Loss is NaN/Inf during blockwise joint tuning.")
            loss_list.append(loss.detach().cpu())
            optimizer.zero_grad()
            loss_scaler(loss, optimizer, parameters=(p for p in layer.parameters() if p.requires_grad))

            if quant_index is not None and quant_scheduler is not None:
                quant_scheduler.step()
                optimizer.param_groups[quant_index]["lr"] = quant_scheduler.get_last_lr()[0]
            if weight_index is not None and weight_scheduler is not None:
                weight_scheduler.step()
                optimizer.param_groups[weight_index]["lr"] = weight_scheduler.get_last_lr()[0]

        loss_mean = torch.stack(loss_list).mean()
        quant_lr_cur = quant_scheduler.get_last_lr()[0] if quant_scheduler is not None else 0.0
        weight_lr_cur = weight_scheduler.get_last_lr()[0] if weight_scheduler is not None else 0.0
        logger.info(
            f"Block {layer_index}, Epoch {epoch}, loss={loss_mean:.6f}, "
            f"quant_lr={quant_lr_cur:.6g}, weight_lr={weight_lr_cur:.6g}, "
            f"time={(time.time() - start_time):.3f}s"
        )

    optimizer.zero_grad()
    del optimizer
    clear_memory()
