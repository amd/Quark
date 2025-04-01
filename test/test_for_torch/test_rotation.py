#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
from quark.torch.algorithm.rotation.rotation_utils import rotate_out_channels

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
    rotate_out_channels(module, rotation)

    # Check if weights and biases remain the same (since rotation is identity)
    assert torch.allclose(module.weight.data, original_weight, atol=1e-6), "Weights were incorrectly modified."
    assert torch.allclose(module.bias.data, original_bias, atol=1e-6), "Bias was incorrectly modified."

    # Apply a non-identity rotation matrix
    rotation = torch.tensor([[0.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float64)

    # Manually compute expected rotated weights and biases
    expected_weight = torch.matmul(rotation.T, original_weight.to(dtype=torch.float64))
    expected_bias = torch.matmul(rotation.T, original_bias.to(dtype=torch.float64))

    # Apply the rotation
    rotate_out_channels(module, rotation)

    # Check if weights and biases match the expected values
    assert torch.allclose(module.weight.data, expected_weight.to(dtype=module.weight.dtype), atol=1e-6), "Weights were not rotated correctly."
    assert torch.allclose(module.bias.data, expected_bias.to(dtype=module.bias.dtype), atol=1e-6), "Bias was not rotated correctly."
