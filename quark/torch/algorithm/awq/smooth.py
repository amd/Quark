#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023 MIT HAN Lab
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import functools
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.awq.scale import apply_scale
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import get_device, get_named_linears, move_to_device
from quark.torch.algorithm.utils.prepare import (
    cache_model_inps,
    get_layers_for_scaling,
    get_model_layers,
    init_device_map,
    reset_model_kv_cache,
)
from quark.torch.algorithm.utils.utils import get_num_attn_heads_from_model
from quark.torch.quantization.debug import QUARK_ALGO_DEBUG
from quark.torch.quantization.utils import clear_memory
from quark.torch.utils.torch_utils import get_op_name

logger = ScreenLogger(__name__)

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import SmoothQuantConfig

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


class SmoothQuantProcessor(BaseAlgoProcessor):
    def __init__(
        self, model: nn.Module, quant_algo_config: SmoothQuantConfig, data_loader: DataLoader[torch.Tensor]
    ) -> None:
        self.model = model

        # `device` is an attribute for transformers.PretrainedModel models, but not nn.Module in general.
        self.device = model.device if hasattr(model, "device") else next(model.parameters()).device

        self.data_loader = data_loader
        self.alpha = quant_algo_config.alpha
        self.clamp_min = quant_algo_config.scale_clamp_min
        self.model_decoder_layers = quant_algo_config.model_decoder_layers
        self.scaling_layers = quant_algo_config.scaling_layers
        self.device_map = init_device_map(self.model)
        self.modules, self.module_kwargs, self.inps = self.init_quant()
        self.num_attention_heads, self.num_key_value_heads = get_num_attn_heads_from_model(model)

    def apply(self) -> None:
        forward_pass_use_cache = reset_model_kv_cache(self.model, use_cache=False)

        for i in tqdm(range(len(self.modules)), desc="SmoothQuant"):
            # [STEP 1]: Get layer, extract linear modules, extract act scales
            named_linears = get_named_linears(self.modules[i])
            input_feat, act_scales = self._get_act_scale_and_input_feat(self.modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply act scales
            assert self.scaling_layers is not None
            module_config: list[dict[str, Any]] = get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs, self.scaling_layers
            )
            scales_list = []
            for layer in module_config:
                scales = self._search_best_scale(
                    self.modules[i], act_scales, **layer
                )  # scales: (pre_layer, layer, best_scales)
                if scales is not None:
                    if QUARK_ALGO_DEBUG:
                        logger.info(
                            f"SmoothQuant for layer {i}: {scales[1]}, scales_max={scales[2].max().item()}, scales_min={scales[2].min().item()}"
                        )
                    scales_list.append(scales)
                else:
                    logger.warning(f"SmoothQuant is skipped for layer {i}: {scales[1]}!")
            apply_scale(
                self.modules[i],
                scales_list,
                input_feat_dict=None,
                device=self.device,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
            )
            clear_memory()

        reset_model_kv_cache(self.model, use_cache=forward_pass_use_cache)

    @torch.no_grad()
    def _search_best_scale(
        self,
        module: nn.Module,
        act_scales: dict[str, torch.Tensor],
        prev_op: nn.Module,
        layers: list[nn.Linear],
        inp: torch.Tensor,
        module2inspect: nn.Module | None = None,
        kwargs: dict[str, Any] = {},
    ) -> tuple[str, tuple[str, ...], torch.Tensor]:
        for fc in layers:
            assert isinstance(fc, nn.Linear)
            assert fc.in_features == act_scales[get_op_name(module, fc)].numel()

        # [STEP 1]: Compute maximum of weight
        device, dtype = layers[0].weight.device, layers[0].weight.dtype
        layer_act_scales = act_scales[get_op_name(module, layers[0])].to(device).to(dtype)
        weight_scales = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in layers], dim=0)
        weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
        # [STEP 2]: Balance quant error between weight and act
        best_scales = (layer_act_scales.pow(self.alpha) / weight_scales.pow(1 - self.alpha)).clamp(min=self.clamp_min)
        assert torch.isnan(best_scales).sum() == 0, best_scales
        return (get_op_name(module, prev_op), tuple([get_op_name(module, m) for m in layers]), best_scales)

    @torch.no_grad()
    def _get_act_scale_and_input_feat(
        self, layer: nn.ModuleList, named_linears: dict[str, nn.Linear]
    ) -> tuple[dict[str, list[torch.Tensor]], dict[str, torch.Tensor]]:
        act_scales: dict[str, torch.Tensor] = {}
        num_batches = len(self.inps)
        layer_inputs = [inp for inp in self.inps]
        cur_layer_device = get_device(layer)
        layer_outputs = []
        cache_examples_on_gpu = True

        def stat_tensor(name: str, tensor: torch.Tensor) -> None:
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float()
            if name in act_scales:
                act_scales[name] = torch.max(act_scales[name], comming_max)
            else:
                act_scales[name] = comming_max

        # collect act scale
        def cache_input_hook(
            m: nn.Module,
            x: tuple[torch.Tensor, ...],
            y: torch.Tensor,
            name: str,
            feat_dict: dict[str, list[torch.Tensor]],
        ) -> None:
            x = x[0]
            x = x.detach()
            feat_dict[name] = []
            stat_tensor(name, x)

        input_feat: dict[str, list[torch.Tensor]] = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )

        for j in range(num_batches):
            layer_input = move_to_device(layer_inputs[j], cur_layer_device)
            output = layer(layer_input, **self.module_kwargs)
            if isinstance(output, tuple):
                layer_output = output[0]
            elif isinstance(output, torch.Tensor):
                layer_output = output
            else:
                raise ValueError(f"Unexpected output type: {type(output)}")
            layer_output = cast(
                torch.Tensor, move_to_device(layer_output, cur_layer_device if cache_examples_on_gpu else CPU)
            )
            layer_outputs.append(layer_output)

        # get output as next layer's input
        self.inps = layer_outputs
        for h in handles:
            h.remove()

        return input_feat, act_scales

    def init_quant(self) -> tuple[nn.ModuleList, dict[str, Any], list[torch.Tensor]]:
        assert self.model_decoder_layers is not None
        modules = get_model_layers(self.model, self.model_decoder_layers)

        forward_pass_use_cache = reset_model_kv_cache(self.model, use_cache=False)
        modules, layer_kwargs, inputs = cache_model_inps(self.model, modules, self.data_loader)

        reset_model_kv_cache(self.model, use_cache=forward_pass_use_cache)

        return modules, layer_kwargs, inputs
