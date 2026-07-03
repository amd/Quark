#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


import torch

from quark.common.data_type import BaseFP8_E5M3
from quark.torch.kernel.float8_e5m3.lookup import E5M3_LOOKUP_TABLE


def reference_convert_fp32_to_e5m3_bits(inputs_f32: torch.Tensor) -> torch.Tensor:
    """
    Convert float32 tensor to E5M3 uint8 bit representation.
    Uses round-to-nearest-ties-to-even rounding.

    Handles conversion of FP32 normals to E5M3 subnormals when values fall
    in range [2^(-17), 2^(-14)).
    """
    if inputs_f32.dtype != torch.float32:
        raise ValueError(f"Input must be float32, got {inputs_f32.dtype}.")

    inputs_bits = inputs_f32.view(torch.int32)

    # Extract FP32 components: S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM
    exponent_f32 = (inputs_bits >> 23) & 0xFF  # 8 exponent bits
    mantissa_f32 = inputs_bits & 0x7FFFFF  # 23 mantissa bits

    # Adjust exponent bias from 127 (float32) to 15 (E5M3): exp_e5m3 = exponent_f32 - 112

    # exponent_f32_unbiased = exponent_f32 - 127
    # exponent_e5m3_raw = exponent_f32_unbiased + 15
    exponent_e5m3_raw = exponent_f32.to(torch.int32) - 112

    # Determine if value should be E5M3 subnormal (exponent_e5m3_raw <= 0)
    is_subnormal = exponent_e5m3_raw <= 0

    # ========================================================================
    # PATH 1: FP32 normal → E5M3 subnormal (denormalization required)
    # ========================================================================
    # For subnormals, we need to denormalize: shift the mantissa to represent
    # smaller values with exp=0.
    # E5M3 subnormals represent: 2^(-14) * (mantissa/8) for exp=0, mantissa in [1,7]
    # E5M3 subnormal: b2 * 2**(-1) + b1 * 2**(-2) + b0 * 2**(-3)

    # FP32 normal = E5M3 subnormal
    # 2^exponent_f32_unbiased × (mantissa_with_implicit / 2^23) = 2^(-14) × M × 2^(-3) = 2^(-17) × M
    # Solving for M (3-bits binary):

    # M = 2^(exponent_f32_unbiased + 17) × (mantissa_with_implicit / 2^23)
    # M = mantissa_with_implicit × 2^(exponent_f32_unbiased + 17 - 23)
    # M = mantissa_with_implicit × 2^(exponent_f32_unbiased - 6)
    # M = mantissa_with_implicit × 2^(exponent_e5m3_raw - 21)
    #
    # hence the `shift_amount` below.

    # 1 = 2^23 / 2^23, hence 1.mantissa = mantissa_with_implicit / 2^23.
    mantissa_with_implicit = (1 << 23) | mantissa_f32

    # Denormalization shift: shift by (21 - exponent_e5m3_raw) to place the value
    # in the subnormal range. The extra +1 accounts for removing implicit leading 1.
    shift_amount = 21 - exponent_e5m3_raw
    shift_amount = torch.clamp(shift_amount, 0, 31)

    mantissa_shifted = mantissa_with_implicit >> shift_amount
    lsb = mantissa_shifted & 1  # Least significant of the 3-bit result, used for ties-to-even.

    # Round-to-nearest-ties-to-even
    round_bit_pos = shift_amount - 1  # Bit index deciding to round up/down.
    round_bit = (mantissa_with_implicit >> round_bit_pos) & 1  # 1 = round up, 0 = round down.
    sticky_mask = (1 << round_bit_pos) - 1  # Mask for bits below the round bit.
    sticky_bits = mantissa_with_implicit & sticky_mask  # Used for ties to even, check if exact halfway (is a tie).

    # Round-to-nearest-ties-to-even: round up if round_bit AND (sticky_bits OR lsb)
    round_up = round_bit & ((sticky_bits != 0) | lsb)

    mantissa_rounded = mantissa_shifted + round_up

    # For subnormals: keep all 3 bits (no implicit leading 1 to remove)
    mantissa_e5m3 = mantissa_rounded & 0x7

    # Handle mantissa overflow (8 -> exponent becomes 1, mantissa becomes 0)
    # If mantissa rounds to 8, we've rounded up to the smallest normal number.
    exp_e5m3 = (mantissa_rounded >> 3) & 1  # 1 if mantissa_rounded >= 8, 0 otherwise

    result_subnormal = ((exp_e5m3 << 3) | mantissa_e5m3).to(torch.uint8)

    # ========================================================================
    # PATH 2: FP32 normal → E5M3 normal (standard quantization)
    # ========================================================================
    # Round mantissa from 23 bits to 3 bits using round to nearest, ties to even.
    # Mantissa:
    #  b   b   b   b  ...  b   b
    # 22  21  20  19  ...  1   0

    round_bit = (mantissa_f32 >> 19) & 1  # Bit at position 19, determines rounding direction.
    sticky_bits = mantissa_f32 & 0x7FFFF  # Lower 19 bits, check if exact halfway (is a tie).
    lsb = (mantissa_f32 >> 20) & 1  # Least significant of the 3-bit result, used for ties-to-even.

    # Truncated mantissa (top 3 bits of 23-bit mantissa)
    mantissa_e5m3 = (mantissa_f32 >> 20) & 0x7  # Extract bits [22:20]

    # Round-to-nearest-ties-to-even: round up if round_bit AND (sticky_bits OR lsb)
    round_up = round_bit & ((sticky_bits != 0) | lsb)
    mantissa_e5m3 = mantissa_e5m3 + round_up

    # Handle mantissa overflow (8 -> increment exponent, mantissa becomes 0)
    # If mantissa rounds to 8, that means we need to carry to the next exponent.
    exp_increment = (mantissa_e5m3 >> 3) & 1  # 1 if mantissa >= 8
    mantissa_e5m3 = mantissa_e5m3 & 0x7  # Keep only lower 3 bits

    exp_e5m3 = exponent_e5m3_raw + exp_increment
    exp_e5m3 = torch.clamp(exp_e5m3, 0, 31)

    result_normal = ((exp_e5m3 << 3) | mantissa_e5m3).to(torch.uint8)

    # Combine subnormal and normal result.
    result = torch.where(is_subnormal, result_subnormal, result_normal)

    return result


def reference_dequantize_float8_e5m3(inputs: torch.Tensor) -> torch.Tensor:
    """
    Dequantize E5M3 format (stored as uint8) back to float32 using pure lookup table.
    E5M3 format: 5 exponent bits, 3 mantissa bits, no sign bit.

    This version uses a pre-computed lookup table for all 256 possible E5M3 values,
    making it a simple index operation.
    """
    if inputs.dtype != torch.uint8:
        raise ValueError(f"dequantize_float8_e5m3_v2 expected uint8 input, got inputs.dtype={inputs.dtype}.")

    # Move lookup table to the same device as inputs
    lookup_table = E5M3_LOOKUP_TABLE[inputs.device]

    # Simple lookup operation - convert uint8 to long for indexing
    result = lookup_table[inputs.long()]

    return result


def reference_float32_qdq_to_float8_e5m3(inputs_f32: torch.Tensor) -> torch.Tensor:
    """Reference implementation using separate quantize + dequantize."""
    inputs_f32 = torch.clamp(inputs_f32, BaseFP8_E5M3.min_value, BaseFP8_E5M3.max_value)
    e5m3_bits = reference_convert_fp32_to_e5m3_bits(inputs_f32)
    return reference_dequantize_float8_e5m3(e5m3_bits)
