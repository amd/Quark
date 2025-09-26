#
# Copyright(c) 2025 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import move_to_device
from quark.torch.algorithm.utils.prepare import get_model_layers, init_blockwise_algo, init_device_map
from quark.torch.pruning.config import LayerImportancePruneConfig
from quark.torch.utils import setattr_recursive

logger = ScreenLogger(__name__)

__all__ = ["LayerImportancePrunerProcessor"]

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


class LayerImportancePrunerProcessor(BaseAlgoProcessor):
    def __init__(
        self,
        model: nn.Module,
        pruning_algo_config: LayerImportancePruneConfig,
        data_loader: DataLoader[torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        # -------init prune config ---------
        self.model_decoder_layers: str = pruning_algo_config.model_decoder_layers
        self.layer_norm_field: str = pruning_algo_config.layer_norm_field
        self.delete_layers_index: list[int] = pruning_algo_config.delete_layers_index
        self.delete_layer_num: int = pruning_algo_config.delete_layer_num
        self.save_memory = pruning_algo_config.save_gpu_memory
        self.layer_num_field: str = pruning_algo_config.layer_num_field

        self.device_map = init_device_map(self.model)
        self.decode_layers = get_model_layers(self.model, self.model_decoder_layers)
        self.num_hidden_layers: int = len(self.decode_layers)
        assert len(self.layer_num_field) != 0, "lack of layer_num_field"
        assert getattr(self.model.config, self.layer_num_field) == self.num_hidden_layers

        # ----------valid data set & config ---------------
        self.test_dataset: list[torch.Tensor] = data_loader  # type: ignore
        assert len(set([x.numel() for x in self.test_dataset])) == 1
        self.seqlen_for_eval: int = [x.numel() for x in self.test_dataset][0]
        # ---------evaluation---------------
        self.min_ppl = torch.tensor(float("inf"))
        self.best_del_idx: list[int] = []
        self.ppl_list: list[torch.Tensor] = []
        self.delete_list: list[list[int]] = []

        # ----------slow mode evaluation--------
        # if GPU memory is limited, we will using layer-by-layer forward to save momory
        if self.save_memory:
            self.eval_func = self._slow_eval_model
            _, self.module_kwargs, self.layer_inputs = init_blockwise_algo(
                self.model, self.model_decoder_layers, self.test_dataset
            )  # type: ignore
            for i in range(len(self.decode_layers)):  # to save memory
                self.decode_layers[i] = self.decode_layers[i].to("cpu")
            torch.cuda.empty_cache()
        else:
            self.eval_func = self._fast_eval_model  # type: ignore

    @torch.no_grad()  # using fully in GPU memory, speed fast but using much memory
    def _fast_eval_model(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        nlls = []
        loss_fct = torch.nn.CrossEntropyLoss()
        for i in tqdm(range(len(self.test_dataset))):
            batch = self.test_dataset[i]
            lm_logits = model(batch)["logits"]
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = self.test_dataset[i][:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * self.seqlen_for_eval
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (len(self.test_dataset) * self.seqlen_for_eval))
        return ppl

    @torch.no_grad()  # using partly in GPU memory, speed slow but saving memory
    def _slow_eval_model(self, model: nn.Module, remain_layer_idx: list[int] = []) -> torch.Tensor:
        torch.cuda.empty_cache()
        layers_list = get_model_layers(self.model, self.model_decoder_layers)
        num_batches = len(self.layer_inputs)
        layer_outputs: list[torch.Tensor] = []
        layer_inputs = [inp for inp in self.layer_inputs]
        for i in tqdm(range(len(layers_list)), desc="PPL influence"):
            layer = layers_list[i]
            # NOTE at least using one gpu
            layer_device = (
                self.device_map[f"{self.model_decoder_layers}.{remain_layer_idx[i]}"]
                if f"{self.model_decoder_layers}.{remain_layer_idx[i]}" in self.device_map
                else CUDA
            )
            move_to_device(layer, layer_device)
            additional_layer_inputs = {}
            for k, v in self.module_kwargs.items():
                if isinstance(v, torch.Tensor):
                    additional_layer_inputs[k] = move_to_device(v, layer_device)
                elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
                    additional_layer_inputs[k] = tuple([move_to_device(i, layer_device) for i in v])  # type: ignore
                else:
                    additional_layer_inputs[k] = v
            if "past_key_value" in additional_layer_inputs:
                additional_layer_inputs["past_key_value"] = None  # type: ignore

            layer_outputs = []
            for j in range(num_batches):
                layer_input = move_to_device(layer_inputs[j], layer_device)
                layer_output = layer(layer_input, **additional_layer_inputs)[0]
                layer_outputs.append(layer_output)

            layer = move_to_device(layer, CPU)
            layer_inputs, layer_outputs = layer_outputs, []
            torch.cuda.empty_cache()

        # NOTE assume deepseek/llama/opt model all have norm layer
        if self.layer_norm_field is not None or len(self.layer_norm_field) > 0:
            norm_layer_device = (
                self.device_map[self.layer_norm_field] if self.layer_norm_field in self.device_map else CUDA
            )
            layer_norm = get_model_layers(self.model, self.layer_norm_field)
            move_to_device(layer_norm, norm_layer_device)

        # TODO lm_head my change name in different model
        lm_head_device = self.device_map["lm_head"] if "lm_head" in self.device_map else CUDA
        move_to_device(model.lm_head, lm_head_device)
        self.test_dataset = [move_to_device(x, lm_head_device) for x in self.test_dataset]
        loss_fct = nn.CrossEntropyLoss()
        nlls = []
        for i in range(len(layer_inputs)):
            hidden_states = layer_inputs[i].unsqueeze(0)
            if self.layer_norm_field is not None or len(self.layer_norm_field) > 0:
                hidden_states = move_to_device(hidden_states, norm_layer_device)
                hidden_states = layer_norm(hidden_states)

            hidden_states = move_to_device(hidden_states, lm_head_device)
            lm_logits = model.lm_head(hidden_states)[0]
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = self.test_dataset[i][:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * self.seqlen_for_eval
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (len(self.layer_inputs) * self.seqlen_for_eval))
        return ppl

    def _trim_layers(self, layers_to_trim: list[int] | None = None) -> None:
        logger.info(f"Removing from {layers_to_trim} of {len(self.decode_layers)} layers")
        demand_layers = []
        for idx in range(len(self.decode_layers)):
            if idx not in layers_to_trim:  # type: ignore
                demand_layers.append(self.decode_layers[idx])
        pruned_decoders = nn.ModuleList(demand_layers)
        setattr_recursive(self.model, self.model_decoder_layers, pruned_decoders)
        return

    def apply(self) -> None:
        forward_pass_use_cache = self.model.config.use_cache
        original_model_ppl = self.eval_func(self.model, remain_layer_idx=[i for i in range(self.num_hidden_layers)])
        bf_prune_param = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Original PPL: {original_model_ppl.item()} param: {bf_prune_param}")
        # assume you skip the quant process
        if len(self.delete_layers_index) > 0:
            self._trim_layers(self.delete_layers_index)
            remained_layer = [i for i in range(self.num_hidden_layers) if i not in self.delete_layers_index]
            ppl = self.eval_func(self.model, remain_layer_idx=remained_layer)
            logger.info(f"User assigned to delete {self.delete_layers_index}, final PPL: {ppl.item()}")
            return

        total_layer_num = len(self.decode_layers)
        for i in range(total_layer_num + 1 - self.delete_layer_num):
            layers_to_trim = [num for num in range(i, i + self.delete_layer_num)]
            self._trim_layers(layers_to_trim)
            remained_layer = [i for i in range(self.num_hidden_layers) if i not in layers_to_trim]
            ppl = self.eval_func(self.model, remain_layer_idx=remained_layer)
            logger.info(f"After delete {layers_to_trim} layers, the PPL: {ppl.item()}")
            #  results recording
            if ppl < self.min_ppl:
                self.min_ppl = ppl
                self.best_del_idx = layers_to_trim
            self.ppl_list.append(ppl)
            self.delete_list.append(layers_to_trim)

        # generate pruned model TODO
        self._trim_layers(self.best_del_idx)
        for need_delete_layer in self.best_del_idx[::-1]:
            self.decode_layers[need_delete_layer].to(CPU)
            del self.decode_layers[need_delete_layer]

        # TODO may modify config after pruning
        self.model.config.num_hidden_layers = self.num_hidden_layers - len(self.best_del_idx)

        logger.info(
            f"PPL influence pruning finished, finally delete {self.best_del_idx} as has minmal PPL: {self.min_ppl}"
        )
        aft_prune_param = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Before pruning param: {bf_prune_param} PPL: {original_model_ppl.item()} \n \
                    after pruning param: {aft_prune_param} PPL: {self.min_ppl.item()}"
        )
        torch.cuda.empty_cache()
        self.model.config.use_cache = forward_pass_use_cache
