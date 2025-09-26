#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright [2024] Yujun Lin, Haotian Tang, Shang Yang, Song Han

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from scipy.linalg import hadamard

from quark.torch.algorithm.rotation.hadamard import get_hadamard_matrices, random_hadamard_matrix


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm normalization to hidden states."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def multiply_with_reshape(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if A.ndim > 2:
        raise ValueError(f"multiply_with_reshape only support A.ndim = 1 or 2, got {A.ndim}.")

    if A.shape[-1] % B.shape[0] != 0:
        raise ValueError(
            f"Expected A.shape[-1] % B.shape[0] == 0 in multiply_with_reshape, but got: incompatible A.shape[-1]={A.shape[-1]}, B.shape[0]={B.shape[0]}."
        )

    needs_reshape = False
    if A.shape[-1] != B.shape[0]:
        needs_reshape = True

    dtype = A.dtype
    device = A.device

    # The `needs_reshape` case comes down to applying the smaller B as a block diagonal matrix.
    if needs_reshape:
        A = A.reshape(*A.shape[:-1], -1, B.shape[0])

    A = A.to(torch.float64) @ B.to(device=device)

    if needs_reshape:
        A = A.reshape(*A.shape[:-2], -1)

    return A.to(device=device, dtype=dtype)


def rotate_with_size(x: torch.Tensor, rotation_size: int, rotation_matrix: torch.Tensor | None = None) -> torch.Tensor:
    """
    Rotates the input tensor `x` on its last dimension, per group of `rotation_size`.

    Denoting `k = rotation_size` and R_k the rotation of shape (k, k), and applying this rotation on x of shape (..., num_groups * k), the inverse transform is the block diagonal:

    [  R_k 0_k  ...  0_k ]
    [  0_k R_k           ]
    [     .    .         ]
    [     .      .       ]
    [     .              ]
    [     0_k   ...  R_k ]

    of shape (num_groups * k, num_groups * k).
    """
    if rotation_matrix is None:
        rotation_matrix = torch.tensor(
            hadamard(rotation_size, dtype=float), dtype=x.dtype, device=x.device
        ) / math.sqrt(rotation_size)
    elif rotation_matrix is not None and rotation_matrix.shape[0] != rotation_size:
        raise ValueError(
            f"The function rotate_with_size got the input rotation_size={rotation_size} and rotation_matrix of shape {rotation_matrix.shape}, which are incompatible."
        )

    x = x.reshape(*x.shape[:-1], -1, rotation_size) @ rotation_matrix

    return x.reshape(*x.shape[:-2], -1).to(x.dtype)


def rotate_in_channels_(module: nn.Module, rotation: torch.Tensor) -> None:
    """Rotate the input channels of a linear layer.
    If weight and rotation's sizes don't match, it reshapes weight in order to multiply them."""
    module.weight.data = multiply_with_reshape(module.weight.data, rotation)


def rotate_out_channels_(module: nn.Module, rotation: torch.Tensor) -> None:
    """Rotate the output channels of a linear layer.
    If weight/bias and rotation's sizes don't match
    it reshapes weight/bias in order to multiply them."""
    module.weight.data = multiply_with_reshape(module.weight.data.T, rotation)
    module.weight.data = module.weight.data.T

    if module.bias is not None:
        module.bias.data = multiply_with_reshape(module.bias.data, rotation)


def get_rotation_matrix(num_channels: int, random: bool = True) -> torch.Tensor:
    """Get a random rotation matrix for the given number of channels."""
    if random:
        return random_hadamard_matrix(num_channels)
    else:
        hadamard_1, hadamard_K, K = get_hadamard_matrices(num_channels)
        hadamard_1 = hadamard_1.to(dtype=torch.float64)
        if K == 1:
            rotation = hadamard_1
        else:
            assert hadamard_K is not None
            hadamard_K = hadamard_K.to(dtype=torch.float64)
            rotation = torch.kron(hadamard_1, hadamard_K)
        return rotation.mul_(1.0 / torch.tensor(num_channels, dtype=torch.float64).sqrt())


def transform_norm_and_linear(
    prev_modules: Iterable[nn.Module],
    norm_module: nn.Module,
    next_modules: Iterable[nn.Module],
    prev_out_channels_dims: list[int],
) -> None:
    transform_rms_norm_and_linear(norm_module, next_modules)
    if isinstance(norm_module, nn.LayerNorm):
        assert prev_modules is not None
        prev_modules_linear = [mod for mod in prev_modules if isinstance(mod, nn.Linear)]
        transform_layer_norm_to_rms_norm(norm_module, prev_modules_linear, prev_out_channels_dims)


def transform_rms_norm_and_linear(norm: nn.Module, next_modules: Iterable[nn.Module]) -> None:
    next_modules_linear = [mod for mod in next_modules if isinstance(mod, nn.Linear)]
    ln_w = norm.weight.data.to(dtype=torch.float64)
    norm.weight.data = torch.ones_like(norm.weight.data)
    if hasattr(norm, "bias") and norm.bias is not None:
        ln_b = norm.bias.data.to(dtype=torch.float64)
        norm.bias = None  # type: ignore
    else:
        ln_b = None
    for linear in next_modules_linear:
        dtype = linear.weight.dtype
        fc_w = linear.weight.data.to(dtype=torch.float64)
        linear.weight.data = (fc_w * ln_w).to(dtype=dtype)
        if ln_b is not None:
            if linear.bias is None:
                linear.bias = nn.Parameter(torch.zeros(linear.out_features, dtype=dtype, device=linear.weight.device))
            linear.bias.data = (linear.bias.data.to(dtype=torch.float64) + torch.matmul(fc_w, ln_b)).to(dtype=dtype)


def transform_layer_norm_to_rms_norm(
    norm: nn.Module,
    prev_modules: Iterable[nn.Linear],
    prev_out_channels_dims: list[int],
) -> None:
    assert isinstance(norm, nn.LayerNorm)
    assert len(norm.normalized_shape) == 1, f"LayerNorm's #dims must be 1, got {len(norm.normalized_shape)}"
    assert norm.bias is None, "LayerNorm's bias must be None"
    # region move substract mean to the previous linear modules
    assert len(prev_modules) > 0, "No previous modules found"
    if isinstance(prev_out_channels_dims, int):
        prev_out_channels_dims = [prev_out_channels_dims] * len(prev_modules)
    for module, dim in zip(prev_modules, prev_out_channels_dims, strict=False):
        if isinstance(module, nn.LayerNorm):
            module.bias = None
        else:
            if isinstance(module, nn.Linear):
                assert dim == 0, "Linear module's output channels dimension is 0"
            elif isinstance(module, nn.Embedding):
                assert dim == 1, "Embedding module's output channels dimension is 1"
            dtype = module.weight.dtype
            W = module.weight.data.to(dtype=torch.float64)
            module.weight.data = W.sub_(W.mean(dim=dim, keepdim=True)).to(dtype=dtype)
            if hasattr(module, "bias") and module.bias is not None:
                B = module.bias.data.to(dtype=torch.float64)
                module.bias.data = B.sub_(B.mean()).to(dtype=dtype)
    # region replace LayerNorm with RMSNorm
    rms = RMSNorm(hidden_size=norm.normalized_shape[0], eps=norm.eps)
    rms.weight.data = norm.weight.data
