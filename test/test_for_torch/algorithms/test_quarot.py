#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from typing import Optional

import pytest
from testing_algorithms_utils import assert_non_destructive_transform

from quark.torch.quantization.config.config import QuaRotConfig


@pytest.mark.parametrize("rotation_size", [None, 32])
@pytest.mark.parametrize("r1", [True, False])
@pytest.mark.parametrize("r2", [True, False])
@pytest.mark.parametrize("r3", [True, False])
@pytest.mark.parametrize("r4", [True, False])
def test_non_destructive_transform(rotation_size: int | None, r1: bool, r2: bool, r3: bool, r4: bool):
    if rotation_size == 32 and r3:
        pytest.skip("Custom rotation_size + r3 is not implemented")

    # scaling_layers taken from quarot_config.json in quark examples.
    scaling_layers = {
        "first_layer": [
            {
                "prev_modules": ["model.embed_tokens"],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": [
                    "model.layers.layer_id.self_attn.q_proj",
                    "model.layers.layer_id.self_attn.k_proj",
                    "model.layers.layer_id.self_attn.v_proj",
                ],
            },
            {
                "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"],
            },
        ],
        "middle_layers": [
            {
                "prev_modules": ["model.layers.pre_layer_id.mlp.down_proj"],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": [
                    "model.layers.layer_id.self_attn.q_proj",
                    "model.layers.layer_id.self_attn.k_proj",
                    "model.layers.layer_id.self_attn.v_proj",
                ],
            },
            {
                "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"],
            },
        ],
        "last_layer": [
            {
                "prev_modules": ["model.layers.layer_id.mlp.down_proj"],
                "norm_module": "model.norm",
                "next_modules": ["lm_head"],
            }
        ],
    }

    quarot_config = QuaRotConfig(scaling_layers=scaling_layers, rotation_size=rotation_size, r1=r1, r2=r2, r3=r3, r4=r4)

    assert_non_destructive_transform(quarot_config)
