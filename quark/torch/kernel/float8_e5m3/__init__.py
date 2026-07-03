#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

import torch

from quark.common.utils.import_utils import is_triton_available
from quark.torch.kernel.float8_e5m3.reference import (
    reference_convert_fp32_to_e5m3_bits,
    reference_dequantize_float8_e5m3,
    reference_float32_qdq_to_float8_e5m3,
)

if is_triton_available():
    from quark.torch.kernel.float8_e5m3.triton import (
        fp8_e5m3_qdq as fp8_e5m3_qdq_triton,
    )
    from quark.torch.kernel.float8_e5m3.triton import (
        triton_convert_fp32_to_e5m3_bits,
    )
else:

    def _raise_import_error_when_used(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ImportError(
            "Failed to import amdfp4 quantization kernels for Triton: Please ensure triton is installed correctly."
        )

    fp8_e5m3_qdq_triton = _raise_import_error_when_used
    triton_convert_fp32_to_e5m3_bits = _raise_import_error_when_used

__all__ = [
    "convert_fp32_to_e5m3_bits_func",
    "float32_qdq_to_float8_e5m3_func",
    "dequantize_float8_e5m3_func",
]


def convert_fp32_to_e5m3_bits_func(inputs_f32: torch.Tensor) -> torch.Tensor:
    """
    Convert float32 tensor to E5M3 uint8 bit representation.
    """
    if inputs_f32.device.type == "cpu":
        return reference_convert_fp32_to_e5m3_bits(inputs_f32)
    else:
        return triton_convert_fp32_to_e5m3_bits(inputs_f32)


def float32_qdq_to_float8_e5m3_func(inputs_f32: torch.Tensor) -> torch.Tensor:
    """
    Fused quantize-dequantize for E5M3 format.
    """
    if inputs_f32.device.type == "cpu":
        return reference_float32_qdq_to_float8_e5m3(inputs_f32)
    else:
        return fp8_e5m3_qdq_triton(inputs_f32)


def dequantize_float8_e5m3_func(inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """
    Dequantize E5M3 format (stored as uint8) back to float32.
    Uses reference implementation for both CPU and GPU.
    """
    return reference_dequantize_float8_e5m3(inputs)
