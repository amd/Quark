#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Tests for dequantize_fp8_per_block and dequantize_int_per_group_affine kernels."""

from __future__ import annotations

import pytest
import torch

from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
    dequantize_fp8,
    dequantize_fp8_per_block_impl,
    dequantize_int_per_group_affine,
)
from quark.torch.quantization.config.type import QSchemeType


# =============================================================================
# dequantize_fp8_per_block
# =============================================================================
def test_fp8_per_block_basic_and_different_scales() -> None:
    """Per-block dequantize: correct shape/dtype, uniform scale, and per-block scale."""
    M, N, BM, BN = 4, 8, 4, 4
    inputs = torch.ones(M, N).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 3.0]])  # 1 row-block, 2 col-blocks

    result = dequantize_fp8_per_block_impl(inputs, scale, (BM, BN))
    assert result.shape == (M, N) and result.dtype == torch.float32
    assert torch.allclose(result[:, :4], torch.ones(M, 4))
    assert torch.allclose(result[:, 4:], torch.full((M, 4), 3.0))


def test_fp8_per_block_non_divisible_fallback() -> None:
    """Non-divisible dimensions use the padding fallback path."""
    M, N, BM, BN = 10, 8, 4, 4
    inputs = torch.ones(M, N).to(torch.float8_e4m3fn)
    scale = torch.full((3, 2), 2.0)  # ceil(10/4)=3

    result = dequantize_fp8_per_block_impl(inputs, scale, (BM, BN))
    assert result.shape == (M, N)
    assert torch.allclose(result, torch.full((M, N), 2.0))


# =============================================================================
# dequantize_int_per_group_affine
# =============================================================================
def test_int_per_group_scale_and_zero_point() -> None:
    """Per-group dequantize: scale and zero_point applied correctly per group."""
    inputs = torch.full((1, 8), 10, dtype=torch.int8)
    scale = torch.tensor([[2.0, 3.0]])
    zp = torch.tensor([[3.0, 5.0]])

    result = dequantize_int_per_group_affine(inputs, scale, zp, axis=1, group_size=4)
    assert result.shape == (1, 8)
    # (10 - 3) * 2 = 14 for group 0, (10 - 5) * 3 = 15 for group 1
    assert torch.allclose(result[0, :4], torch.full((4,), 14.0))
    assert torch.allclose(result[0, 4:], torch.full((4,), 15.0))


def test_int_per_group_identity() -> None:
    """Scale=1, zero_point=0 should return the input as float."""
    inputs = torch.randint(-128, 127, (4, 16), dtype=torch.int8)
    scale = torch.ones(4, 2)
    zp = torch.zeros(4, 2)

    result = dequantize_int_per_group_affine(inputs, scale, zp, axis=1, group_size=8)
    assert result.shape == (4, 16)
    assert torch.allclose(result, inputs.float())


# =============================================================================
# dequantize_fp8 dispatcher: per_block branch
# =============================================================================
def test_dequantize_fp8_per_block_dispatch() -> None:
    """The per_block branch derives block_size from scale.shape and dequantizes ``input * scale``."""
    M, N = 4, 8
    # Distinct, fp8-exact input values per block so the test catches an impl that
    # ignores the input and returns the scale alone.
    inputs = torch.empty(M, N)
    inputs[:, :4] = 1.0
    inputs[:, 4:] = 2.0
    inputs = inputs.to(torch.float8_e4m3fn)
    scale = torch.tensor([[2.0, 4.0]])  # 1 row-block × 2 col-blocks → block_size (4, 4)
    result = dequantize_fp8(inputs, scale=scale, qscheme=QSchemeType.per_block.value, quant_dtype="fp8_e4m3")
    assert result.shape == (M, N)
    assert torch.allclose(result[:, :4], torch.full((M, 4), 2.0))  # 1.0 * 2.0
    assert torch.allclose(result[:, 4:], torch.full((M, 4), 8.0))  # 2.0 * 4.0


def test_dequantize_fp8_per_block_requires_scale() -> None:
    """per_block without scale → ValueError."""
    inputs = torch.ones(4, 8).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="requires a scale tensor"):
        dequantize_fp8(inputs, scale=None, qscheme=QSchemeType.per_block.value, quant_dtype="fp8_e4m3")


def test_dequantize_fp8_per_block_rejects_non_2d() -> None:
    """per_block requires both inputs and scale to be 2D."""
    inputs = torch.ones(2, 4, 8).to(torch.float8_e4m3fn)
    scale = torch.ones(1, 2)
    with pytest.raises(ValueError, match="expects 2D inputs and scale"):
        dequantize_fp8(inputs, scale=scale, qscheme=QSchemeType.per_block.value, quant_dtype="fp8_e4m3")
