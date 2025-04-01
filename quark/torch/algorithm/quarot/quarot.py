#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from tqdm import tqdm
from typing import Any, TYPE_CHECKING
import torch
import torch.nn as nn
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.algorithm.rotation.rotation_utils import get_rotation_matrix
from quark.torch.algorithm.rotation.hadamard import matmul_hadU
from quark.torch.algorithm.rotation.rotation import RotationProcessor
from quark.torch.algorithm.utils.prepare import get_model_layers
from quark.torch.algorithm.quarot.utils import add_qk_rotation_after_function_call_in_forward, R4Wrapper, rotate_in_channels2, rotate_out_channels2

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import QuaRotConfig

__all__ = ["QuaRotProcessor"]


class QuaRotProcessor(RotationProcessor):

    def __init__(self, model: nn.Module, pre_quant_opt_config: QuaRotConfig, _data_loader: Any) -> None:
        self.quarot_opt_config = pre_quant_opt_config
        self.scaling_layers = self.quarot_opt_config.scaling_layers
        assert self.scaling_layers is not None
        self.model = model
        self.optimized_rotation_path = self.quarot_opt_config.optimized_rotation_path
        self.backbone = get_model_layers(self.model, self.quarot_opt_config.backbone)
        self.layers = get_model_layers(self.model, self.quarot_opt_config.model_decoder_layers)

    def apply(self) -> None:
        if self.optimized_rotation_path is None:
            rotation1 = get_rotation_matrix(self.model.config.hidden_size, random=self.quarot_opt_config.random)
        else:
            rotation1 = torch.load(self.optimized_rotation_path)["R1"]
        self.rotate(rotation1)
        clear_memory()
        self.r2()
        if self.quarot_opt_config.online_had:
            if self.quarot_opt_config.kv_cache_quant:
                self.r3()
            if self.quarot_opt_config.act_quant:
                self.r4()

    def r2(self) -> None:
        rotation2 = get_rotation_matrix(self.backbone.config.hidden_size // self.backbone.config.num_attention_heads,
                                        random=self.quarot_opt_config.random2)
        for idx, layer in tqdm(enumerate(self.layers), desc="R2 Rotation: ", total=self.model.config.num_hidden_layers):
            layer_proj_v = get_model_layers(layer, self.quarot_opt_config.v_proj)
            layer_proj_o = get_model_layers(layer, self.quarot_opt_config.o_proj)
            if self.optimized_rotation_path is not None:
                rotation2 = torch.load(self.optimized_rotation_path)[f"model.layers.{idx}.self_attn.R2"].to(
                    layer_proj_v.weight.device)
            rotation2 = rotation2.to(layer_proj_v.weight.device)
            rotate_out_channels2(layer_proj_v, rotation=rotation2)
            rotation2 = rotation2.to(layer_proj_o.weight.device)
            rotate_in_channels2(layer_proj_o, rotation=rotation2)
            clear_memory()

    def r3(self) -> None:
        for layer in tqdm(self.layers, desc="R3 Rotation: "):
            add_qk_rotation_after_function_call_in_forward(
                get_model_layers(layer, self.quarot_opt_config.self_attn),
                "apply_rotary_pos_emb"  # this is the name of the function called in Llama that actually does RoPE
            )
            clear_memory()

    def r4(self) -> None:
        for layer in tqdm(self.layers, desc="R4 Rotation: "):
            mlp = get_model_layers(layer, self.quarot_opt_config.mlp)
            dtype = mlp.down_proj.weight.dtype
            mlp.down_proj.weight.data = matmul_hadU(mlp.down_proj.weight.data).to(dtype)
            mlp.down_proj = R4Wrapper(mlp.down_proj)
            clear_memory()
