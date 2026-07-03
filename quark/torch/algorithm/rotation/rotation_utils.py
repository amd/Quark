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
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn
from scipy.linalg import hadamard

from quark.common.utils.log import ScreenLogger
from quark.torch.algorithm.rotation.hadamard import (
    _get_hadamard_K,
    get_hadamard_matrices,
    matmul_hadU,
    random_hadamard_matrix,
)
from quark.torch.algorithm.rotation.monkeypatch import add_wrapper_after_function_call_in_method

logger = ScreenLogger(__name__)


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


def rotate_with_size(
    x: torch.Tensor, rotation_size: int | None = None, rotation_matrix: torch.Tensor | None = None
) -> torch.Tensor:
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
        assert rotation_size is not None

        rotation_matrix = torch.tensor(
            hadamard(rotation_size, dtype=float), dtype=x.dtype, device=x.device
        ) / math.sqrt(rotation_size)
    else:
        assert rotation_size is None
        rotation_size = rotation_matrix.shape[0]

        if x.shape[-1] % rotation_size != 0:
            raise ValueError(
                f"The function rotate_with_size got the input x with x.shape[-1]={x.shape[-1]} and rotation_matrix of shape {rotation_size}, which are incompatible."
            )

    dtype = x.dtype

    needs_reshape = False
    if x.shape[-1] != rotation_size:
        needs_reshape = True
        x = x.reshape(*x.shape[:-1], -1, rotation_size)

    if rotation_matrix.device != x.device:
        logger.warning(
            f"Device mismatch! rotation_matrix device: {rotation_matrix.device}, x device: {x.device}. This is likely causing unnecessary slowness. please open an issue."
        )
        rotation_matrix = rotation_matrix.to(x.device)

    # TODO: fix type mismatch?
    x = x.to(torch.float64) @ rotation_matrix.to(dtype=torch.float64)
    x = x.to(dtype)
    if needs_reshape:
        x = x.reshape(*x.shape[:-2], -1)

    return x


def rotate_in_channels_(module: nn.Module, rotation: torch.Tensor) -> None:
    """Rotate the input channels of a linear layer.
    If weight and rotation's sizes don't match, it reshapes weight in order to multiply them."""
    module.weight.data = rotate_with_size(module.weight.data, rotation_matrix=rotation)


def rotate_out_channels_(module: nn.Module, rotation: torch.Tensor) -> None:
    """Rotate the output channels of a linear layer.
    If weight/bias and rotation's sizes don't match
    it reshapes weight/bias in order to multiply them."""
    module.weight.data = rotate_with_size(module.weight.data.T, rotation_matrix=rotation)
    module.weight.data = module.weight.data.T.contiguous()

    if module.bias is not None:
        module.bias.data = rotate_with_size(module.bias.data, rotation_matrix=rotation)


def get_rotation_matrix(num_channels: int, device: torch.device | str, random: bool = True) -> torch.Tensor:
    """Get a random rotation matrix for the given number of channels."""
    if random:
        rotation = random_hadamard_matrix(num_channels)
    else:
        hadamard_1, hadamard_K, K = get_hadamard_matrices(num_channels)
        hadamard_1 = hadamard_1.to(dtype=torch.float64)
        if K == 1:
            rotation = hadamard_1
        else:
            assert hadamard_K is not None
            hadamard_K = hadamard_K.to(dtype=torch.float64)
            rotation = torch.kron(hadamard_K, hadamard_1)
        rotation = rotation.mul_(1.0 / torch.tensor(num_channels, dtype=torch.float64).sqrt())

    return rotation.to(device)


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


class QKRotation(nn.Module):
    """Performs R3 rotation after RoPE of both Q and K, but does not do K quantization"""

    def __init__(self, func: Callable[..., Any]):
        super().__init__()
        self.func = func

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        q, k = self.func(*args, **kwargs)

        q = matmul_hadU(q)

        # k is transposed later on in the attention Q @ K.T, so no need to use `inverse=True` here.
        k = matmul_hadU(k)

        return q, k


def add_qk_rotation_after_function_call_in_forward(module: nn.Module, function_name: str) -> None:
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.

    This function used to insert the R3 rotation after the output of the call of the RoPE operation.
    Implementating it like this is not ideal, since we need to modify the forward function's globals. However, this is the
    trick used by both QuaRot and SpinQuant to insert a rotation after the RoPE operation. Ultimately it would better to
    find a way to implement this feature without touching globals.
    """

    attr_name = f"{function_name}_qk_rotation"
    assert not hasattr(module, attr_name)
    wrapper = add_wrapper_after_function_call_in_method(module, "forward", function_name, QKRotation)
    setattr(module, attr_name, wrapper)


class InputRotationWrapper(nn.Module):
    """
    Wrapper around a nn.Module that applies a Hadamard rotation before the module.
    If the module is an nn.Linear or nn.Conv, then Quark will replace it by a quantized linear layer
    If there is activation quantization, it is applied in between, i.e. after the rotation
    but before the forward pass of the module
    """

    def __getattr__(self, name: str) -> Any:
        # TODO: try to do a check on `self.original_module` attributes here
        if name in {"weight", "bias", "in_features", "out_features"}:
            return getattr(self.original_module, name)
        else:
            return super().__getattr__(name)

    def __setattr__(self, key: str, value: Any) -> None:
        # TODO: try to do a check on `self.original_module` attributes here
        if key in {"weight", "bias", "in_features", "out_features"}:
            setattr(self.original_module, key, value)
        else:
            super().__setattr__(key, value)

    def state_dict(
        self, *args: tuple[Any], destination: dict[str, Any] | None = None, prefix: str = "", keep_vars: bool = False
    ) -> dict[str, Any]:
        destination_local = super().state_dict(*args, prefix=prefix, keep_vars=keep_vars)

        for param_name in list(destination_local):
            if ".original_module." in param_name:
                new_name = param_name.replace(".original_module.", ".")
                destination_local[new_name] = destination_local.pop(param_name)

        if destination is not None:
            destination.update(destination_local)
        else:
            destination = destination_local

        return destination

    def forward(self, x: torch.Tensor) -> Any:
        x = self.transform(x)

        # quantization will happen here, in between (since it happens before a nn.Linear layer)
        x = self.original_module(x)
        return x


class InputRotationWrapperHadamard(InputRotationWrapper):
    def __init__(
        self,
        original_module: nn.Linear,
        rotation_size: int | None = None,
        hadamard_K: torch.Tensor | None = None,
        K: int | None = None,
    ):
        super().__init__()

        if not isinstance(original_module, nn.Linear):
            raise ValueError(
                f"InputRotationWrapper only supports module instance of torch.nn.Linear, got {original_module.__class__.__name__}"
            )

        self.original_module = original_module

        in_features = original_module.in_features

        if rotation_size is not None and in_features != rotation_size:
            if in_features % rotation_size == 0:
                self.rotation_size = rotation_size
                self.use_matmul_hadU = False
            else:
                raise ValueError(f"rotation_size={rotation_size} is not compatible with in_features={in_features}.")
        else:
            self.rotation_size = in_features
            self.use_matmul_hadU = True

        if hadamard_K is None or K is None:
            rotation_matrix, K = _get_hadamard_K(self.rotation_size)
        else:
            rotation_matrix = hadamard_K

        self.K = K

        rotation_matrix = rotation_matrix.to(self.original_module.weight.dtype)
        rotation_matrix = rotation_matrix.to(self.original_module.weight.device)

        input_rotation = rotation_matrix.clone()

        if input_rotation.shape[0] != self.rotation_size:
            assert self.K is not None
            hadamard_1, _ = _get_hadamard_K(self.rotation_size // self.K)

            hadamard_1 = hadamard_1.to(input_rotation.device)

            input_rotation = input_rotation.to(dtype=torch.float64)
            input_rotation = torch.kron(input_rotation, hadamard_1)

        assert input_rotation.shape[0] == self.rotation_size

        assert (
            input_rotation[input_rotation == 1].numel() + input_rotation[input_rotation == -1].numel()
            == input_rotation.numel()
        )

        input_rotation = input_rotation.to(torch.int8)

        self.register_buffer("input_rotation", input_rotation)

        self.transform = HadamardTransform(
            use_matmul_hadU=self.use_matmul_hadU, rotation_size=self.rotation_size, hadamard_K=rotation_matrix, K=K
        )


class InputRotationWrapperOrthogonal(InputRotationWrapper):
    def __init__(
        self,
        original_module: nn.Linear,
        rotation_matrix: torch.Tensor,
    ):
        super().__init__()

        if not isinstance(original_module, nn.Linear):
            raise ValueError(
                f"InputRotationWrapper only supports module instance of torch.nn.Linear, got {original_module.__class__.__name__}"
            )

        self.original_module = original_module

        assert rotation_matrix is not None
        assert rotation_matrix.dtype == torch.float64

        rotation_matrix = rotation_matrix.to(self.original_module.weight.device)

        self.transform = OrthogonalTransform(rotation_matrix)

        input_rotation = rotation_matrix.clone()
        self.register_buffer("input_rotation", input_rotation)


class OrthogonalTransform(nn.Module):
    def __init__(
        self,
        rotation_matrix: torch.Tensor,
    ):
        super().__init__()

        assert rotation_matrix is not None

        self.rotation_matrix = rotation_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rotate_with_size(x, rotation_matrix=self.rotation_matrix)
        return x


class HadamardTransform(nn.Module):
    def __init__(
        self,
        rotation_size: int,
        use_matmul_hadU: bool,
        hadamard_K: torch.Tensor | None = None,
        K: int | None = None,
    ):
        super().__init__()

        self.use_matmul_hadU = use_matmul_hadU
        self.rotation_size = rotation_size

        if not use_matmul_hadU:
            if hadamard_K is None:
                rotation_matrix, _ = _get_hadamard_K(self.rotation_size)
            else:
                rotation_matrix = hadamard_K

            # Ensure rotation matrix stays in float32 for numerical match.
            rotation_matrix = rotation_matrix.to(torch.float32)
            rotation_matrix = rotation_matrix / math.sqrt(self.rotation_size)
        else:
            assert K is not None
            assert hadamard_K is not None
            rotation_matrix = hadamard_K

        self.K = K
        self.rotation_matrix = rotation_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_matmul_hadU:
            # TODO: ideally, rotate_with_size should handle this case well.
            x = matmul_hadU(x, hadamard_K=self.rotation_matrix, K=self.K)
        else:
            x = rotate_with_size(x, rotation_matrix=self.rotation_matrix)

        return x
