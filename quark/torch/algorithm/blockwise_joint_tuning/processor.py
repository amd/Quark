#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from quark.torch.algorithm.config import BlockwiseJointTuningConfig

from quark.common.utils.log import ScreenLogger
from quark.experimental.torch.algorithm.blockwise_joint_tuning.quantize.learnable_linear import (
    ExperimentalLearnableQuantizedLinear,
)
from quark.experimental.torch.algorithm.blockwise_joint_tuning.quantize.utils import set_op_by_name, set_quant_state
from quark.torch.algorithm.blockwise_joint_tuning.utils import block_forward, blockwise_joint_training
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import get_device, get_dtype, move_to_device
from quark.torch.algorithm.utils.prepare import get_model_layers, init_blockwise_algo, init_device_map
from quark.torch.algorithm.utils.utils import clear_memory

logger = ScreenLogger(__name__)

__all__ = ["BlockwiseJointTuningProcessor"]

CPU = torch.device("cpu")


def _infer_weight_quant_params_from_layer(layer: nn.Module) -> tuple[int, int]:
    """Infer (num_bits, group_size) from the first QuantLinear in a layer."""
    for module in layer.modules():
        if not isinstance(module, nn.Linear):
            continue

        weight_quantizer = getattr(module, "_weight_quantizer", None)
        if weight_quantizer is None:
            continue

        dtype = getattr(weight_quantizer, "dtype", None)
        group_size = getattr(weight_quantizer, "group_size", None)
        if dtype is None or group_size is None:
            continue

        num_bits = dtype.to_bitwidth()
        return int(num_bits), int(group_size)

    raise ValueError(
        "Failed to infer quantization params from layer: no QuantLinear with valid dtype/group_size was found."
    )


def _replace_linear_with_learnable_quant_linear(
    layer: nn.Module, num_bits: int | None = None, group_size: int | None = None
) -> None:
    # TODO: ExperimentalLearnableQuantizedLinear is a temporary implementation used for QAT weight quantization.
    # Replace with the official Quark QuantLinear once it supports learnable scale/zero_point for QAT.
    if num_bits is None or group_size is None:
        inferred_num_bits, inferred_group_size = _infer_weight_quant_params_from_layer(layer)
        num_bits = inferred_num_bits if num_bits is None else num_bits
        group_size = inferred_group_size if group_size is None else group_size

    assert num_bits is not None and group_size is not None

    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.Linear):
            quant_linear = ExperimentalLearnableQuantizedLinear(module, num_bits, group_size)
            set_op_by_name(layer, name, quant_linear)
            del module
    set_quant_state(layer, weight_quant=True)


class BlockwiseJointTuningProcessor(BaseAlgoProcessor):
    """Blockwise joint tuning processor.

    Difference from `BlockwiseTuningProcessor`:
    - `BlockwiseTuningProcessor` uses module-only trainable selection.
    - `BlockwiseJointTuningProcessor` is for joint optimization with an extra
      quant-parameter target group (e.g., scale/zero_point).
    """

    def __init__(
        self,
        fp_model: nn.Module,
        model: nn.Module,
        algo_config: BlockwiseJointTuningConfig,
        data_loader: DataLoader[torch.Tensor],
    ) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.flags(enabled=True, allow_tf32=False)

        self.fp_model = fp_model
        self.model = model
        self.epochs = algo_config.epochs
        self.weight_lr = algo_config.weight_lr
        # Backward compatible with possible `quant_lr` naming in other branches.
        self.quant_lr = getattr(algo_config, "quant_lr", algo_config.qparam_lr)
        self.min_lr_factor = algo_config.min_lr_factor
        self.weight_decay = algo_config.weight_decay
        self.qparam_weight_decay = algo_config.qparam_weight_decay
        self.max_grad_norm = algo_config.max_grad_norm
        self.model_decoder_layers = algo_config.model_decoder_layers
        self.trainable_modules = algo_config.trainable_modules
        self.quant_trainable_modules = algo_config.quant_trainable_modules

        if isinstance(data_loader, tuple | list) and len(data_loader) >= 2:
            self.traindata_loader = data_loader[0]
            self.valdata_loader = data_loader[1]
        else:
            self.traindata_loader = data_loader
            self.valdata_loader = data_loader

        self.device_map = init_device_map(self.model)
        self.modules, self.module_kwargs, self.train_inputs = init_blockwise_algo(
            self.model, self.model_decoder_layers, self.traindata_loader
        )
        self.modules_fp = get_model_layers(self.fp_model, self.model_decoder_layers)
        self.modules_fp_val, _, self.val_inputs = init_blockwise_algo(
            self.fp_model, self.model_decoder_layers, self.valdata_loader
        )

        # Tuning on GPU, keep inactive blocks on CPU.
        for i in range(len(self.modules)):
            self.modules[i] = self.modules[i].to("cpu")
        for i in range(len(self.modules_fp)):
            self.modules_fp[i] = self.modules_fp[i].to("cpu")
            self.modules_fp_val[i] = self.modules_fp_val[i].to("cpu")
        clear_memory()

    def apply(self) -> None:
        cache_examples_on_gpu = False

        num_train_batches = len(self.train_inputs)
        num_val_batches = len(self.val_inputs)

        layer_train_inputs = [inp.detach().requires_grad_(False) for inp in self.train_inputs]
        layer_train_outputs: list[torch.Tensor] = []
        fp_layer_train_inputs = list(layer_train_inputs)
        fp_layer_train_outputs: list[torch.Tensor] = []

        layer_val_inputs = [inp.detach().requires_grad_(False) for inp in self.val_inputs]
        fp_layer_val_inputs = list(layer_val_inputs)
        fp_layer_val_outputs: list[torch.Tensor] = []

        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        self.fp_model.config.use_cache = False

        for i in tqdm(range(len(self.modules)), desc="BlockWise_Joint_Tuning"):
            logger.info(f"Start joint tuning layer {i + 1}/{len(self.modules)}")
            layer = self.modules[i]
            fp_layer = self.modules_fp[i]
            fp_layer_val = self.modules_fp_val[i]
            layer_dtype = get_dtype(layer)

            _replace_linear_with_learnable_quant_linear(layer)

            force_layer_back_to_cpu = False
            if get_device(layer) == CPU:
                move_to_device(layer, self.device_map[f"{self.model_decoder_layers}.{i}"])
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer)

            fp_layer_train_outputs = block_forward(
                fp_layer,
                self.module_kwargs,
                num_train_batches,
                cur_layer_device,
                fp_layer_train_inputs,
                fp_layer_train_outputs,
                cache_examples_on_gpu,
            )
            fp_layer_val_outputs = block_forward(
                fp_layer_val,
                self.module_kwargs,
                num_val_batches,
                cur_layer_device,
                fp_layer_val_inputs,
                fp_layer_val_outputs,
                cache_examples_on_gpu,
            )

            fp_layer = move_to_device(fp_layer, CPU if force_layer_back_to_cpu else cur_layer_device)
            fp_layer_val = move_to_device(fp_layer_val, CPU if force_layer_back_to_cpu else cur_layer_device)

            # Align with joint-QAT behavior: cast layer to fp32 before AMP training.
            with torch.no_grad():
                layer.float()

            blockwise_joint_training(
                layer=layer,
                module_kwargs=self.module_kwargs,
                trainable_modules=self.trainable_modules,
                quant_trainable_modules=self.quant_trainable_modules,
                layer_inputs=layer_train_inputs,
                fp_layer_outputs=fp_layer_train_outputs,
                quant_val_inps=layer_val_inputs,
                fp_val_outputs=fp_layer_val_outputs,
                device=cur_layer_device,
                epochs=self.epochs,
                quant_lr=self.quant_lr,
                weight_lr=self.weight_lr,
                min_lr_factor=self.min_lr_factor,
                weight_decay=self.weight_decay,
                qparam_weight_decay=self.qparam_weight_decay,
                max_grad_norm=self.max_grad_norm,
                layer_index=i,
                loss_func=torch.nn.MSELoss(),
            )

            layer_train_outputs = block_forward(
                layer,
                self.module_kwargs,
                num_train_batches,
                cur_layer_device,
                layer_train_inputs,
                layer_train_outputs,
                cache_examples_on_gpu,
            )

            # Restore original block dtype to keep model-wide dtype consistency
            # for downstream eval (e.g. lm_head matmul expects fp16 inputs).
            with torch.no_grad():
                layer.to(dtype=layer_dtype)
            layer = move_to_device(layer, CPU if force_layer_back_to_cpu else cur_layer_device)

            del layer
            del fp_layer
            del fp_layer_val
            del layer_train_inputs
            del fp_layer_train_inputs
            layer_train_inputs, layer_train_outputs = layer_train_outputs, []  # noqa: F841
            fp_layer_train_inputs, fp_layer_train_outputs = fp_layer_train_outputs, []  # noqa: F841
            fp_layer_val_inputs, fp_layer_val_outputs = fp_layer_val_outputs, []
            clear_memory()

        self.model.config.use_cache = forward_pass_use_cache
        self.fp_model.config.use_cache = forward_pass_use_cache

        del self.fp_model
        clear_memory()
