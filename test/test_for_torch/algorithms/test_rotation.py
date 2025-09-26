#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
import torch
import torch.nn as nn
from testing_algorithms_utils import assert_non_destructive_transform

from quark.torch.algorithm.rotation.hadamard import matmul_hadU
from quark.torch.algorithm.rotation.rotation_utils import get_rotation_matrix, rotate_out_channels_
from quark.torch.quantization.config.config import RotationConfig


def is_power_of_two(n: int):
    return (n != 0) and (n & (n - 1) == 0)


# TODO: e.g. 2 * 12 or 4 * 40 is broken = not equivalent.
# There should be a way to get the inverse of `get_rotation_matrix` for
# these dimensions as well.
@pytest.mark.parametrize("n", [12, 1024, 40])
def test_hadamard(n: int):
    inp = torch.rand(n, device="cuda")

    res_custom = matmul_hadU(inp.clone())

    rotation_matrix = get_rotation_matrix(n, random=False)
    rotation_matrix = rotation_matrix.to(inp.dtype)
    rotation_matrix = rotation_matrix.to(inp.device)

    if not is_power_of_two(n):
        rotation_matrix = rotation_matrix.T.contiguous()

    res_matmul = inp @ rotation_matrix

    absdiff = (res_matmul - res_custom).abs()

    assert torch.allclose(res_matmul, res_custom, atol=1e-2, rtol=1e-2)


def test_rotate_out_channels():
    # Create a Linear layer with bias
    in_features = 3
    out_features = 3
    module = nn.Linear(in_features, out_features, bias=True)

    # Initialize rotation matrix as an identity matrix (no rotation)
    rotation = torch.eye(out_features, dtype=torch.float64)

    # Clone original weights and biases for comparison
    original_weight = module.weight.data.clone()
    original_bias = module.bias.data.clone()

    # Apply the rotation
    rotate_out_channels_(module, rotation)

    # Check if weights and biases remain the same (since rotation is identity)
    assert torch.allclose(module.weight.data, original_weight, atol=1e-6), "Weights were incorrectly modified."
    assert torch.allclose(module.bias.data, original_bias, atol=1e-6), "Bias was incorrectly modified."

    # Apply a non-identity rotation matrix
    rotation = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)

    # Manually compute expected rotated weights and biases
    expected_weight = torch.matmul(rotation.T, original_weight.to(dtype=torch.float64))
    expected_bias = torch.matmul(rotation.T, original_bias.to(dtype=torch.float64))

    # Apply the rotation
    rotate_out_channels_(module, rotation)

    # Check if weights and biases match the expected values
    assert torch.allclose(module.weight.data, expected_weight.to(dtype=module.weight.dtype), atol=1e-6), (
        "Weights were not rotated correctly."
    )
    assert torch.allclose(module.bias.data, expected_bias.to(dtype=module.bias.dtype), atol=1e-6), (
        "Bias was not rotated correctly."
    )


def test_non_destructive_transform():
    # Taken from example's rotation_config.json.
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

    rotation_config = RotationConfig(model_decoder_layers="model.layers", scaling_layers=scaling_layers)

    assert_non_destructive_transform(rotation_config)
