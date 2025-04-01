#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
from typing import Optional, Tuple
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.algorithm.utils.utils import get_device_map, set_device_map
from quark.torch.pruning.config import Config
from quark.torch.quantization.utils import set_op_by_name, get_op_by_name

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def process_model_pruning(
    model: nn.Module,
    config: Config,
    is_accelerate: Optional[bool],
) -> nn.Module:

    logger.info("Pruning model start.")
    before_pruning_parameters = sum(p.numel() for p in model.parameters())

    device_map = get_device_map(model, is_accelerate)

    model = model.cpu()
    clear_memory()

    pruned_model, pruned_intermediate_size = model_pruning_on_cpu(model, config)

    if config.algo_config is not None and hasattr(config.algo_config, 'mlp_intermediate_size_name'):
        pruned_model.config.__setattr__(config.algo_config.mlp_intermediate_size_name, pruned_intermediate_size)

    del model
    clear_memory()

    pruned_model = set_device_map(pruned_model, device_map)

    after_pruning_parameters = sum(p.numel() for p in pruned_model.parameters())
    logger.info("#Param before pruning: {}, #Param after pruning: {}, Pruning Ratio = {:.4f}%".format(
        before_pruning_parameters, after_pruning_parameters,
        100.0 * after_pruning_parameters / before_pruning_parameters))
    logger.info("Pruning model end.")

    return pruned_model


def prune_weights_tool(weights: torch.Tensor, bias: torch.Tensor,
                       mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    indices_to_keep = torch.nonzero(~mask).squeeze()  # Find the indices to keep

    if weights.shape[0] == len(mask):
        pruned_weights = weights[indices_to_keep, :]
    elif weights.shape[0] % len(mask) == 0:
        indices_to_keep_doubled = indices_to_keep.clone()
        for n in range(1, weights.shape[0] // len(mask)):
            indices_to_keep_doubled_add = indices_to_keep + len(mask) * n
            indices_to_keep_doubled = torch.cat((indices_to_keep_doubled, indices_to_keep_doubled_add))
        pruned_weights = weights[indices_to_keep_doubled, :]
    elif weights.shape[1] == len(mask):
        pruned_weights = weights[:, indices_to_keep]
    else:
        raise ValueError(
            f"The shape of the weight tensor {weights.shape} does not match the shape of the mask {mask.shape}")

    if bias is not None and bias.shape[0] == len(mask):
        pruned_bias = bias[indices_to_keep]
    else:
        pruned_bias = bias

    return pruned_weights, pruned_bias


def prune_layer(layer: nn.Module, zero_input_channels: torch.Tensor) -> nn.Module:
    if isinstance(layer, nn.Linear):
        with torch.no_grad():
            weight = layer.weight.data.clone()

            bias = layer.bias.data.clone() if layer.bias is not None else None

            pruned_weights, pruned_bias = prune_weights_tool(weight, bias, zero_input_channels)

            pruned_layer = nn.Linear(pruned_weights.shape[1], pruned_weights.shape[0], bias=(bias is not None))
            pruned_layer.weight.data = pruned_weights.contiguous()
            if bias is not None:
                pruned_layer.bias.data = pruned_bias
            return pruned_layer

    return layer


def model_pruning_on_cpu(model: nn.Module, config: Config) -> Tuple[nn.Module, int]:

    if config.algo_config is not None and hasattr(config.algo_config, 'mlp_pruning_modules') and hasattr(
            config.algo_config, 'mlp_pruning_ratio') and hasattr(
                config.algo_config, 'mlp_intermediate_size_name') and hasattr(config.algo_config, 'mlp_scaling_layers'):

        pruned_size = int((1 - config.algo_config.mlp_pruning_ratio) *
                          model.config.__getattribute__(config.algo_config.mlp_intermediate_size_name))

        if pruned_size % 128 != 0:
            pruned_size = int(round(pruned_size / 128) * 128)

        for name, module in model.named_modules():
            for mlp_pruning_module in config.algo_config.mlp_pruning_modules:
                if mlp_pruning_module in name:

                    weights = module.weight.data.cpu()
                    zero_input_channels = torch.all(weights == 0, dim=0)

                    original_input_channel = weights.shape[1]
                    remove_input_channel = sum(zero_input_channels)

                    current_pruned_size = int(original_input_channel - remove_input_channel)

                    # adjust for opt since opt's activation is relu
                    if current_pruned_size <= pruned_size:
                        diff = pruned_size - current_pruned_size
                        for i in range(len(zero_input_channels)):
                            if zero_input_channels[i] is True:
                                zero_input_channels[i] = False
                                diff -= 1
                            if diff == 0:
                                break
                    else:
                        raise ValueError("Pruning not in effect.")

                    logger.info(
                        f"Start pruning layer {name}, original input channel: {original_input_channel} -> pruned input channel {pruned_size}"
                    )

                    pruned_layer = prune_layer(module, zero_input_channels)

                    set_op_by_name(model, name, pruned_layer)

                    for prev_name in config.algo_config.mlp_scaling_layers[mlp_pruning_module]:

                        prev_module_name = name.replace(mlp_pruning_module, prev_name)

                        prev_pruned_module = prune_layer(get_op_by_name(model, prev_module_name), zero_input_channels)

                        logger.info(
                            f"Start pruning prev_layer {prev_module_name}, original out channel: {original_input_channel} -> pruned out channel {pruned_size}"
                        )

                        set_op_by_name(model, prev_module_name, prev_pruned_module)

        return model, pruned_size
    else:
        raise ValueError("Algorithm configuration is not set.")
