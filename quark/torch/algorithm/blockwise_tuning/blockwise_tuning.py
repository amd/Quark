#
# Copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations
from typing import List, TYPE_CHECKING
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
if TYPE_CHECKING:
    from quark.torch.pruning.config import BlockwiseTuningConfig
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import get_device, move_to_device
from quark.torch.algorithm.utils.prepare import init_blockwise_algo, init_device_map, get_model_layers
from quark.torch.algorithm.blockwise_tuning.blockwise_utils import block_forward, blockwise_training
from quark.torch.algorithm.utils.utils import clear_memory

from tqdm import tqdm

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ["BlockwiseTuningProcessor"]

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


class BlockwiseTuningProcessor(BaseAlgoProcessor):

    def __init__(self, fp_model: nn.Module, model: nn.Module, algo_config: BlockwiseTuningConfig,
                 data_loader: DataLoader[torch.Tensor]) -> None:
        self.fp_model = fp_model
        self.model = model
        self.epochs = algo_config.epochs
        self.weight_lr = algo_config.weight_lr
        self.min_lr_factor = algo_config.min_lr_factor
        self.weight_decay = algo_config.weight_decay
        self.max_grad_norm = algo_config.max_grad_norm
        self.model_decoder_layers = algo_config.model_decoder_layers
        self.trainable_modules = algo_config.trainable_modules
        self.data_loader = data_loader
        self.device_map = init_device_map(self.model)
        self.modules, self.module_kwargs, self.inps = init_blockwise_algo(self.model, self.model_decoder_layers,
                                                                          self.data_loader)
        self.modules_fp = get_model_layers(self.fp_model, self.model_decoder_layers)

    def apply(self) -> None:
        # set output on cpu
        cache_examples_on_gpu = False

        num_batches = len(self.inps)

        layer_inputs = [inp.detach().requires_grad_(False) for inp in self.inps]
        layer_outputs: List[torch.Tensor] = []

        fp_layer_inputs = [inputs for inputs in layer_inputs]
        fp_layer_outputs: List[torch.Tensor] = []

        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        self.fp_model.config.use_cache = False

        # tuning on gpu, other blocks in cpu
        for i in range(len(self.modules)):
            self.modules[i] = self.modules[i].to("cpu")
        clear_memory()

        for i in tqdm(range(len(self.modules)), desc="BlockWise_Tuning"):
            logger.info(f"Start tuning layer {i + 1}/{len(self.modules)}")
            layer = self.modules[i]
            layer_fp = self.modules_fp[i]

            force_layer_back_to_cpu = False
            if get_device(layer) == CPU:
                move_to_device(layer, self.device_map[f"{self.model_decoder_layers}.{i}"])
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer)

            # layer_fp.forward
            fp_layer_outputs = block_forward(layer_fp, self.module_kwargs, num_batches, cur_layer_device,
                                             fp_layer_inputs, fp_layer_outputs, cache_examples_on_gpu)

            layer_fp = move_to_device(layer_fp, CPU if force_layer_back_to_cpu else cur_layer_device)

            # train
            blockwise_training(layer, self.module_kwargs, self.trainable_modules, layer_inputs, fp_layer_outputs,
                               cur_layer_device, self.epochs, self.weight_lr, self.min_lr_factor, self.weight_decay,
                               self.max_grad_norm, i)

            # layer.forward
            layer_outputs = block_forward(layer, self.module_kwargs, num_batches, cur_layer_device, layer_inputs,
                                          layer_outputs, cache_examples_on_gpu)

            layer = move_to_device(layer, CPU if force_layer_back_to_cpu else cur_layer_device)

            del layer
            del layer_fp
            del layer_inputs
            del fp_layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []
            fp_layer_inputs, fp_layer_outputs = fp_layer_outputs, []
            clear_memory()
        self.model.config.use_cache = forward_pass_use_cache
        self.fp_model.config.use_cache = forward_pass_use_cache

        del self.fp_model
        clear_memory()
