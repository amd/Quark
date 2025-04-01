# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.
#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Any, Tuple

from quark.torch.algorithm.rotation.hadamard import matmul_hadU
from quark.torch.algorithm.quarot.monkeypatch import add_wrapper_after_function_call_in_method


def multiply_with_reshape(A: torch.Tensor, B: torch.Tensor, transpose: bool) -> torch.Tensor:
    dtype = A.dtype
    device = A.device
    if transpose:
        A = A.t()
    shape = A.shape
    A = A.reshape(-1, shape[-1] // B.shape[0], B.shape[0])
    C = A.to(torch.float64) @ B.to(device=device)
    C = C.reshape(shape)
    if transpose:
        C = C.t()
    return C.to(device=device, dtype=dtype)


def rotate_in_channels2(module: nn.Module, rotation: torch.Tensor) -> None:
    """ Rotate the input channels of a linear layer.
        If weight and rotation's sizes don't match, it reshapes weight in order to multiply them."""
    module.weight.data = multiply_with_reshape(module.weight.data, rotation, transpose=False)


def rotate_out_channels2(module: nn.Module, rotation: torch.Tensor) -> None:
    """Rotate the output channels of a linear layer.
        If weight/bias and rotation's sizes don't match
        it reshapes weight/bias in order to multiply them."""
    module.weight.data = multiply_with_reshape(module.weight.data, rotation, transpose=True)
    if module.bias is not None:
        module.bias.data = multiply_with_reshape(module.bias.data, rotation, transpose=True)


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """ Applies Hadamard transform to x (without dividing by sqrt n). Ideally should be replaced by a hardware
    optimized kernel, since Hadamard transforms can in theory be done much faster than general matrix multiplications.

    Code from: https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/fast_hadamard_transform/fast_hadamard_transform_interface.py
    """
    from scipy.linalg import hadamard
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    return out[..., :dim].reshape(*x_shape)


def hadamard_multiply(x: torch.Tensor) -> torch.Tensor:
    """ Applies hadamard transform to x with dividing by sqrt n """
    dtype = x.dtype
    return (hadamard_transform(x.float()) / math.sqrt(x.shape[-1])).to(dtype)


class QKRotation(nn.Module):
    """ Performs R3 rotation after RoPE of both Q and K, but does not do K quantization"""

    def __init__(self, func: Callable[..., Any]):
        super().__init__()
        self.func = func

    def forward(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        q, k = self.func(*args, **kwargs)
        q = hadamard_multiply(q)
        k = hadamard_multiply(k)
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


class R4Wrapper(nn.Module):
    """
    Wrapper around a nn.Module that applies a Hadamard rotation before the module.
    If the module is an nn.Linear or nn.Conv, then Quark will replace it by a quantized linear layer
    If there is activation quantization, it is applied in between, i.e. after the rotation
    but before the forward pass of the module
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> Any:
        x = matmul_hadU(x)
        # quantization will happen here, in between (since it happens before a nn.Linear layer)
        x = self.module(x)
        return x
