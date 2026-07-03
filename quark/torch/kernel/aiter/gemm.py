#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
GEMM kernel wrappers for AMD Aiter.

This module provides wrapper functions for Aiter's optimized GEMM kernels:
- gemm_a8w8: FP8/INT8 GEMM with per-tensor or per-token scaling
- gemm_a8w8_blockscale: FP8 GEMM with per-group (block) scaling
"""

import torch
from torch import Tensor

from quark.common.utils.import_utils import is_aiter_available as _check_aiter_package
from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

# Aiter availability flag (for kernel-specific imports)
_aiter_kernels_available: bool = False
_aiter_import_error: str | None = None

# Only try to import kernels if the package is available
if _check_aiter_package():
    try:
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8 as _aiter_gemm_a8w8  # type: ignore[import-not-found]
        from aiter.ops.gemm_op_a8w8 import (  # type: ignore[import-not-found]
            gemm_a8w8_blockscale as _aiter_gemm_a8w8_blockscale,
        )
        from aiter.ops.gemm_op_a8w8 import (  # type: ignore[import-not-found]
            gemm_a8w8_bpreshuffle as _aiter_gemm_a8w8_bpreshuffle,
        )
        from aiter.utility import dtypes as aiter_dtypes  # type: ignore[import-not-found]

        _aiter_kernels_available = True
        logger.debug("AMD Aiter kernels are available for native inference.")
    except ImportError as e:
        _aiter_import_error = str(e)
        logger.debug(f"AMD Aiter kernel import failed: {e}")
else:
    _aiter_import_error = "aiter package not installed"
    logger.debug("AMD Aiter package not available")

# Define placeholder dtypes for type hints when aiter is not available
if not _aiter_kernels_available:

    class _PlaceholderDtypes:
        bf16 = torch.bfloat16
        fp16 = torch.float16
        fp32 = torch.float32

    aiter_dtypes = _PlaceholderDtypes()


def is_aiter_available() -> bool:
    """Check if AMD Aiter kernels are available for native inference."""
    return _aiter_kernels_available


def _check_aiter_available() -> None:
    """Raise ImportError if Aiter is not available."""
    if not is_aiter_available():
        raise ImportError(
            f"Native inference requires AMD Aiter. "
            f"Please install Aiter from https://github.com/ROCm/aiter. "
            f"Import error: {_aiter_import_error}"
        )


def gemm_fp8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """
    FP8 GEMM with per-tensor or per-channel scaling.

    Computes: Y = (XQ * x_scale) @ (WQ * w_scale).T + bias

    Args:
        XQ: Quantized input tensor [M, K] in FP8 format
        WQ: Quantized weight tensor [N, K] in FP8 format
        x_scale: Input scale tensor [M, 1] or [1] for per-tensor
        w_scale: Weight scale tensor [N] or [1] for per-tensor
        bias: Optional bias tensor [N]
        output_dtype: Output data type (default: bfloat16)

    Returns:
        Output tensor [M, N] in output_dtype
    """

    # Ensure scales are in the right format
    if x_scale.dim() == 0:
        x_scale = x_scale.view(1, 1)
    elif x_scale.dim() == 1:
        x_scale = x_scale.view(-1, 1)

    if w_scale.dim() == 0:
        w_scale = w_scale.view(1)

    return _aiter_gemm_a8w8(
        XQ,
        WQ,
        x_scale.to(torch.bfloat16),
        w_scale.to(torch.bfloat16),
        bias=bias,
        dtype=output_dtype,
    )


def gemm_fp8_blockscale(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """
    FP8 GEMM with per-group (block) scaling.

    Computes GEMM where scales are applied per block (typically 128 elements).

    Args:
        XQ: Quantized input tensor [M, K] in FP8 format
        WQ: Quantized weight tensor [N, K] in FP8 format
        x_scale: Input block scales [M, K/block_size]
        w_scale: Weight block scales [N, K/block_size]
        output_dtype: Output data type (default: bfloat16)

    Returns:
        Output tensor [M, N] in output_dtype
    """

    return _aiter_gemm_a8w8_blockscale(
        XQ,
        WQ,
        x_scale,
        w_scale,
        dtype=output_dtype,
    )


def gemm_fp8_bpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """
    FP8 GEMM with pre-shuffled weights for better performance.

    This variant expects weights to be pre-shuffled for optimal memory access
    patterns.  Unlike ``gemm_fp8`` which takes ``w_scale`` as 1-D ``[N]``,
    the bpreshuffle CK kernel expects:

    * ``x_scale``: ``[M, 1]``  (per-token or per-tensor broadcast)
    * ``w_scale``: ``[1, N]``  (per-channel or per-tensor broadcast)

    The wrapper automatically reshapes and broadcasts scalar / 1-D inputs.

    Args:
        XQ: Quantized input tensor [M, K] in FP8 format
        WQ: Pre-shuffled quantized weight tensor [N, K] in FP8 format
        x_scale: Input scale — scalar, [1], [M], or [M, 1]
        w_scale: Weight scale — scalar, [1], [N], or [1, N]
        bias: Optional bias tensor [N]
        output_dtype: Output data type (default: bfloat16)

    Returns:
        Output tensor [M, N] in output_dtype
    """

    M = XQ.shape[0]
    N = WQ.shape[0]

    # x_scale -> [M, 1]
    if x_scale.numel() == 1:
        x_scale = x_scale.reshape(1, 1).expand(M, 1).contiguous()
    elif x_scale.dim() == 1:
        x_scale = x_scale.reshape(-1, 1)

    # w_scale -> [1, N]  (bpreshuffle expects 2-D, unlike regular gemm_a8w8)
    if w_scale.numel() == 1:
        w_scale = w_scale.reshape(1, 1).expand(1, N).contiguous()
    elif w_scale.dim() == 1:
        w_scale = w_scale.reshape(1, -1)

    # Unlike gemm_a8w8 which expects bf16 scales, the bpreshuffle CK kernel
    # requires float32 scales.
    return _aiter_gemm_a8w8_bpreshuffle(
        XQ,
        WQ,
        x_scale.to(torch.float32),
        w_scale.to(torch.float32),
        bias=bias,
        dtype=output_dtype,
    )
