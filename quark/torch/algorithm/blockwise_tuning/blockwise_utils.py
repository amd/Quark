#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations
import time
from typing import Any, Dict, List, Union, cast, Tuple
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from quark.torch.algorithm.utils.module import move_to_device, get_dtype
from quark.torch.algorithm.utils.utils import TensorData, clear_memory

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def block_batch_forward(layer: nn.Module, module_kwargs: Dict[str, Any], input: torch.Tensor,
                        device: torch.device) -> torch.Tensor:

    additional_layer_inputs: Dict[str, Union[None, torch.Tensor, nn.Module]] = {}
    for k, v in module_kwargs.items():
        if isinstance(v, torch.Tensor):
            additional_layer_inputs[k] = move_to_device(v, device)
        else:
            additional_layer_inputs[k] = v
    if "past_key_value" in additional_layer_inputs:
        additional_layer_inputs['past_key_value'] = None
    output = layer(input, **additional_layer_inputs)[0]
    return cast(torch.Tensor, output)


@torch.no_grad()
def block_forward(layer: nn.Module, module_kwargs: Dict[str, Any], num_batches: int, device: torch.device,
                  layer_inputs: List[torch.Tensor], fp_layer_outputs: List[torch.Tensor],
                  cache_examples_on_gpu: bool) -> List[torch.Tensor]:

    layer = move_to_device(layer, device)
    for j in range(num_batches):
        layer_input = move_to_device(layer_inputs[j], device)
        layer_output = block_batch_forward(layer, module_kwargs, layer_input, device)
        layer_output = move_to_device(layer_output, device if cache_examples_on_gpu else torch.device("cpu"))

        fp_layer_outputs.append(layer_output)

    return fp_layer_outputs


def blockwise_training(layer: nn.Module, module_kwargs: Dict[str, Any], trainable_modules: List[str],
                       layer_inputs: List[torch.Tensor], fp_layer_outputs: List[torch.Tensor], device: torch.device,
                       epochs: int, weight_lr: float, min_lr_factor: float, weight_decay: float, max_grad_norm: float,
                       layer_index: int) -> None:

    criterion = nn.MSELoss()

    num_update_steps_per_epoch = max(len(layer_inputs), 1)
    max_steps = int(epochs * num_update_steps_per_epoch)

    tensordata = TensorData(layer_inputs, fp_layer_outputs, device)

    tensordata_loader = DataLoader(tensordata, batch_size=None, shuffle=True)

    scaler = GradScaler()

    # module forward and calculates the loss
    before_tuning_loss = blockwise_eval(layer, module_kwargs, tensordata_loader, criterion, device)

    train_parameters = set_trainable_parameters(layer, trainable_modules, layer_index)

    optimizer, weight_scheduler = prepare_optimizer_and_scheduler(train_parameters, weight_decay, weight_lr,
                                                                  min_lr_factor, max_steps)

    layer_dtype = get_dtype(layer)

    for epoch in range(epochs):
        start_time = time.time()
        layer.train()

        for input, fp_outputs in tensordata_loader:
            if layer_dtype == torch.float16:
                layer.float()
                with autocast(device_type="cuda"):
                    outputs = block_batch_forward(layer, module_kwargs, input, device)
                    loss = criterion(outputs, fp_outputs)
                scaler.scale(loss).backward()
            else:
                outputs = block_batch_forward(layer, module_kwargs, input, device)
                loss = criterion(outputs, fp_outputs)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(train_parameters, max_grad_norm)

            if layer_dtype == torch.float16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            weight_scheduler.step()
            optimizer.zero_grad()
            layer.zero_grad()

        after_tuning_loss = blockwise_eval(layer, module_kwargs, tensordata_loader, criterion, device)

        lr = weight_scheduler.get_lr()[0]  # type: ignore

        logger.info(
            f"Blocks: {layer_index}, Epoch: {epoch}, Before Tuning Loss:{before_tuning_loss:.8f}, After Tuning Loss:{after_tuning_loss:.8f}, Weight LR:{lr:.6f}, Max Allocated Memory: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB, Traing Time: {(time.time()-start_time):.4f} seconds."
        )

    if layer_dtype == torch.float16:
        layer.half()

    clear_memory()


def blockwise_eval(layer: nn.Module, module_kwargs: Dict[str, Any], tensordata_loader: DataLoader[tuple[torch.Tensor,
                                                                                                        torch.Tensor]],
                   criterion: nn.Module, device: torch.device) -> float:

    ret_loss = 0.0
    with torch.no_grad():
        layer.eval()
        for input, fp_outputs in tensordata_loader:
            with autocast(device_type="cuda"):
                outputs = block_batch_forward(layer, module_kwargs, input, device)
                loss = criterion(outputs, fp_outputs)
                ret_loss += (loss.detach().cpu().item()) * len(input)

    return ret_loss / len(tensordata_loader)


def set_trainable_parameters(model: nn.Module, trainable_modules: List[str], layer_index: int) -> List[nn.Parameter]:
    params = []
    names = []

    trainable_keywords = set(trainable_modules + ['layernorm'])
    for n, p in model.named_parameters():
        if not any(keyword in n for keyword in trainable_keywords):
            p.requires_grad = False
        else:
            p.requires_grad = True
            params.append(p)
            names.append(n)

    logger.info(
        f"Trainable parameter number: {sum(p.nelement() for p in params)/1e6}M. Trainable modules: {', '.join(map(str, names))}"
    )
    return params


def prepare_optimizer_and_scheduler(train_parameters: List[torch.nn.Parameter], weight_decay: float, weight_lr: float,
                                    min_lr_factor: float, max_steps: int) -> Tuple[AdamW, CosineAnnealingLR]:

    optimizer = AdamW(train_parameters, weight_decay=weight_decay, lr=weight_lr)

    weight_scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=weight_lr / min_lr_factor)

    return optimizer, weight_scheduler
