#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from quark.torch.algorithm.quarot.utils import InputRotationWrapper, add_qk_rotation_after_function_call_in_forward
from quark.torch.algorithm.rotation.hadamard import matmul_hadU
from quark.torch.algorithm.rotation.rotation import RotationProcessor
from quark.torch.algorithm.rotation.rotation_utils import get_rotation_matrix, rotate_in_channels_, rotate_out_channels_
from quark.torch.algorithm.utils.prepare import get_model_layers
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.utils.accelerate_helper import untie_parameters

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import QuaRotConfig

__all__ = ["QuaRotProcessor"]


class QuaRotProcessor(RotationProcessor):
    def __init__(self, model: nn.Module, quarot_config: QuaRotConfig, _data_loader: Any) -> None:
        self.quarot_config = quarot_config
        self.scaling_modules = quarot_config.scaling_layers
        self.modules = get_model_layers(model, quarot_config.model_decoder_layers)
        self.scaling_layers = self.get_scaling_layers()
        assert self.scaling_layers is not None
        self.model = model
        self.optimized_rotation_path = quarot_config.optimized_rotation_path
        self.backbone = get_model_layers(self.model, quarot_config.backbone)
        self.layers = get_model_layers(self.model, quarot_config.model_decoder_layers)

        self.rotation_size: int | None = quarot_config.rotation_size
        self.random_r1 = quarot_config.random_r1
        self.random_r2 = quarot_config.random_r2

    def apply(self) -> None:
        # R1 needs to be applied on embed_tokens as:
        # W_e' = W_e @ R1
        # R1^(-1) needs to be applied on lm_head as:
        # W_lm' = R1^(-1) @ W_lm.
        # With tied weights we have W_e = W_lm in memory,
        # and end up with `R1^(-1) @ W_e @ R1` which is wrong.
        self.model = untie_parameters(self.model)

        # R1 can be disabled. In Quarot/SpinQuant, it is always offline (can be fully fused into other layers).
        if self.quarot_config.r1:
            self.r1()

        # R2 can be disabled. In Quarot/SpinQuant, it is always offline (can be fully fused into other layers).
        if self.quarot_config.r2:
            self.r2()

        # R3 is useful only in case KV cache is quantized. It is online (can not be fully fused into other layers).
        if self.quarot_config.r3:
            self.r3()

        # R4 is online, can not be fully fused into other layers.
        if self.quarot_config.r4:
            self.r4()

    def r1(self) -> None:
        if self.rotation_size is not None:
            r1_rotation_size = self.rotation_size
        else:
            r1_rotation_size = self.model.config.hidden_size

        if self.optimized_rotation_path is None:
            rotation1 = get_rotation_matrix(r1_rotation_size, random=self.random_r1)
        else:
            rotation1 = torch.load(self.optimized_rotation_path)["R1"]

        self.rotate(rotation1)

        clear_memory()

    def r2(self) -> None:
        if self.rotation_size is not None:
            r2_rotation_size = self.rotation_size
        else:
            r2_rotation_size = self.backbone.config.hidden_size // self.backbone.config.num_attention_heads

        if self.optimized_rotation_path is None:
            rotation2 = get_rotation_matrix(r2_rotation_size, random=self.random_r2)
        else:
            rotations2 = torch.load(self.optimized_rotation_path)

        for idx, layer in tqdm(enumerate(self.layers), desc="R2 Rotation", total=self.model.config.num_hidden_layers):
            layer_proj_v = get_model_layers(layer, self.quarot_config.v_proj)
            layer_proj_o = get_model_layers(layer, self.quarot_config.o_proj)

            if self.optimized_rotation_path is not None:
                rotation2 = rotations2[f"model.layers.{idx}.self_attn.R2"]

            rotation2 = rotation2.to(layer_proj_v.weight.device)
            rotate_out_channels_(layer_proj_v, rotation=rotation2)

            rotation2 = rotation2.to(layer_proj_o.weight.device)
            rotate_in_channels_(layer_proj_o, rotation=rotation2)

            clear_memory()

    def r3(self) -> None:
        if self.rotation_size is not None:
            raise NotImplementedError("R3 does not support custom rotation size at the moment. Please open an issue.")

        for layer in tqdm(self.layers, desc="R3 Rotation"):
            add_qk_rotation_after_function_call_in_forward(
                get_model_layers(layer, self.quarot_config.self_attn),
                "apply_rotary_pos_emb",  # this is the name of the function called in Llama that actually does RoPE
            )
            clear_memory()

    def r4(self) -> None:
        if self.rotation_size is not None:
            custom_rotation_size = True
        else:
            custom_rotation_size = False

        for layer in tqdm(self.layers, desc="R4 Rotation"):
            mlp = get_model_layers(layer, self.quarot_config.mlp)
            dtype = mlp.down_proj.weight.dtype

            if custom_rotation_size:
                rotation4 = get_rotation_matrix(self.rotation_size, random=False)  # type: ignore[arg-type]

                rotate_in_channels_(mlp.down_proj, rotation=rotation4)
            else:
                mlp.down_proj.weight.data = matmul_hadU(mlp.down_proj.weight.data).to(dtype)

            mlp.down_proj = InputRotationWrapper(mlp.down_proj, self.rotation_size)
            clear_memory()
