#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from quark.common.data_type import BaseFP8_E5M3
from quark.torch.kernel.float8_e5m3.lookup import E5M3_LOOKUP_TABLE


@triton.jit  # type: ignore[untyped-decorator]
def _fp8_e5m3_qdq_kernel(
    input_ptr: tl.tensor,
    output_ptr: tl.tensor,
    lookup_table_ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Fused Triton kernel to quantize FP32 to E5M3 and immediately dequantize back to FP32."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load FP32 values
    fp32_vals = tl.load(input_ptr + offsets, mask=mask)

    # Quantize using shared device function
    e5m3_bits = _quantize_fp32_to_e5m3_bits(fp32_vals)

    # Dequantize: E5M3 bits -> FP32 using lookup table
    dequantized = tl.load(lookup_table_ptr + e5m3_bits.to(tl.int64), mask=mask)

    # Store dequantized FP32 result
    tl.store(output_ptr + offsets, dequantized, mask=mask)


def fp8_e5m3_qdq(inputs_f32: torch.Tensor) -> torch.Tensor:
    """
    Fused quantize-dequantize for E5M3 using a single Triton kernel.
    Takes FP32 input, quantizes to E5M3, and immediately dequantizes back to FP32.

    This is faster than separate quantize + dequantize operations because:
    - Single kernel launch instead of two
    - No intermediate uint8 tensor allocation
    - Better data locality
    """
    if inputs_f32.dtype != torch.float32:
        raise ValueError(f"Input must be float32, got {inputs_f32.dtype}.")

    inputs_f32 = torch.clamp(inputs_f32, BaseFP8_E5M3.min_value, BaseFP8_E5M3.max_value)

    output = torch.empty_like(inputs_f32, dtype=torch.float32)
    n_elements = inputs_f32.numel()

    # Launch fused QDQ kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # NOTE: `with torch.cuda.device` is necessary, otherwise the Triton kernel may be dispatched to a wrong device.
    with torch.cuda.device(inputs_f32.device):
        _fp8_e5m3_qdq_kernel[grid](
            inputs_f32,
            output,
            E5M3_LOOKUP_TABLE[inputs_f32.device],
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output


@triton.jit  # type: ignore[untyped-decorator]
def _quantize_fp32_to_e5m3_bits(fp32_vals: tl.tensor) -> tl.tensor:
    """
    Triton device function to convert FP32 values to E5M3 bit representation.
    This is a helper function that can be called from multiple kernels.

    Args:
        fp32_vals: FP32 tensor values

    Returns:
        uint8 tensor with E5M3 bit patterns
    """
    # Reinterpret FP32 as int32 for bit manipulation
    inputs_bits = fp32_vals.to(tl.int32, bitcast=True)

    # Extract FP32 components
    exponent_f32 = (inputs_bits >> 23) & 0xFF
    mantissa_f32 = inputs_bits & 0x7FFFFF

    # Compute E5M3 exponent (adjust bias from 127 to 15)
    exponent_e5m3_raw = exponent_f32 - 112

    # Determine if value should be E5M3 subnormal
    is_subnormal = exponent_e5m3_raw <= 0

    # ========================================================================
    # PATH 1: FP32 normal → E5M3 subnormal
    # ========================================================================
    mantissa_with_implicit = (1 << 23) | mantissa_f32
    shift_amount = 21 - exponent_e5m3_raw
    shift_amount = tl.minimum(tl.maximum(shift_amount, 0), 31)

    mantissa_shifted = mantissa_with_implicit >> shift_amount
    lsb_sub = mantissa_shifted & 1

    # Round-to-nearest-ties-to-even
    round_bit_pos = shift_amount - 1
    round_bit_sub = (mantissa_with_implicit >> round_bit_pos) & 1
    sticky_mask = (1 << round_bit_pos) - 1
    sticky_bits_sub = mantissa_with_implicit & sticky_mask

    round_up_sub = round_bit_sub & ((sticky_bits_sub != 0) | lsb_sub)
    mantissa_rounded = mantissa_shifted + round_up_sub

    mantissa_e5m3_sub = mantissa_rounded & 0x7
    exp_e5m3_sub = (mantissa_rounded >> 3) & 1

    result_subnormal = ((exp_e5m3_sub << 3) | mantissa_e5m3_sub).to(tl.uint8)

    # ========================================================================
    # PATH 2: FP32 normal → E5M3 normal
    # ========================================================================
    round_bit_norm = (mantissa_f32 >> 19) & 1
    sticky_bits_norm = mantissa_f32 & 0x7FFFF
    lsb_norm = (mantissa_f32 >> 20) & 1

    mantissa_e5m3_norm = (mantissa_f32 >> 20) & 0x7
    round_up_norm = round_bit_norm & ((sticky_bits_norm != 0) | lsb_norm)
    mantissa_e5m3_norm = mantissa_e5m3_norm + round_up_norm

    exp_increment = (mantissa_e5m3_norm >> 3) & 1
    mantissa_e5m3_norm = mantissa_e5m3_norm & 0x7

    exp_e5m3_norm = exponent_e5m3_raw + exp_increment
    exp_e5m3_norm = tl.minimum(tl.maximum(exp_e5m3_norm, 0), 31)

    result_normal = ((exp_e5m3_norm << 3) | mantissa_e5m3_norm).to(tl.uint8)

    # Combine results
    return tl.where(is_subnormal, result_subnormal, result_normal)


@triton.jit  # type: ignore[untyped-decorator]
def _fp32_to_e5m3_kernel(
    input_ptr: tl.tensor,
    output_ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Triton kernel to convert FP32 to E5M3 bit representation."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load FP32 values
    fp32_vals = tl.load(input_ptr + offsets, mask=mask)

    # Quantize using device function
    result = _quantize_fp32_to_e5m3_bits(fp32_vals)

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


def triton_convert_fp32_to_e5m3_bits(inputs_f32: torch.Tensor) -> torch.Tensor:
    """
    Convert float32 tensor to E5M3 uint8 bit representation using Triton kernel.
    Uses round-to-nearest-ties-to-even rounding.

    Handles conversion of FP32 normals to E5M3 subnormals when values fall
    in range [2^(-17), 2^(-14)).
    """
    if inputs_f32.dtype != torch.float32:
        raise ValueError(f"Input must be float32, got {inputs_f32.dtype}.")

    output = torch.empty_like(inputs_f32, dtype=torch.uint8)
    n_elements = inputs_f32.numel()

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # NOTE: `with torch.cuda.device` is necessary, otherwise the Triton kernel may be dispatched to a wrong device.
    with torch.cuda.device(inputs_f32.device):
        _fp32_to_e5m3_kernel[grid](
            inputs_f32,
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output
