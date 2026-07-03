#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
AMD Aiter kernel wrappers for native FP8 and MXFP4 inference.

This module provides wrapper functions for AMD Aiter's optimized GEMM kernels
used for native quantized inference on AMD GPUs.
"""

from quark.torch.kernel.aiter.gemm import (
    gemm_fp8,
    gemm_fp8_blockscale,
    is_aiter_available,
)
from quark.torch.kernel.aiter.quant import (
    dynamic_per_group_quant_fp4,
    dynamic_per_tensor_quant_fp8,
    dynamic_per_token_quant_fp8,
)

__all__ = [
    "is_aiter_available",
    "gemm_fp8",
    "gemm_fp8_blockscale",
    "dynamic_per_token_quant_fp8",
    "dynamic_per_group_quant_fp4",
    "dynamic_per_tensor_quant_fp8",
]
