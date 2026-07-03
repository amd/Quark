#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Dynamic quantization operations using AMD Aiter.

This module provides wrapper functions for Aiter's dynamic quantization kernels
used for input quantization during native inference.
"""

import torch
from torch import Tensor

from quark.common.utils.import_utils import is_aiter_available as _check_aiter_package
from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

# Import Aiter quantization ops
_aiter_quant_available: bool = False
_aiter_quant_import_error: str | None = None

# Only try to import if the package is available
if _check_aiter_package():
    try:
        from aiter.ops.quant import (  # type: ignore[import-not-found]
            per_1x32_f4_quant_hip as _aiter_per_1x32_f4_quant,
        )
        from aiter.ops.quant import (  # type: ignore[import-not-found]
            per_group_quant_hip as _aiter_per_group_quant,
        )
        from aiter.ops.quant import (  # type: ignore[import-not-found]
            per_tensor_quant_hip as _aiter_per_tensor_quant,
        )
        from aiter.ops.quant import (  # type: ignore[import-not-found]
            per_token_quant_hip as _aiter_per_token_quant,
        )
        from aiter.utility import dtypes as _aiter_dtypes  # type: ignore[import-not-found]

        _aiter_quant_available = True
    except ImportError as e:
        _aiter_quant_import_error = str(e)
        logger.debug(f"AMD Aiter quantization ops import failed: {e}")
else:
    _aiter_quant_import_error = "aiter package not installed"
    logger.debug("AMD Aiter package not available for quantization ops")
    _aiter_dtypes = None  # type: ignore[misc, assignment]


def _check_aiter_quant_available() -> None:
    """Raise ImportError if Aiter quantization ops are not available."""
    if not _aiter_quant_available:
        raise ImportError(
            f"Native inference quantization requires AMD Aiter. "
            f"Please install Aiter from https://github.com/ROCm/aiter. "
            f"Import error: {_aiter_quant_import_error}"
        )


def _aiter_fp8_quant_dtype() -> torch.dtype:
    """FP8 storage dtype for Aiter quant kernels (GFX-aware via ``aiter.utility.dtypes``)."""
    if _aiter_dtypes is None:
        raise ImportError("Aiter dtypes are unavailable")
    return _aiter_dtypes.fp8


def dynamic_per_token_quant_fp8(
    x: Tensor,
    scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Dynamic per-token quantization to FP8.

    Quantizes input tensor dynamically using per-token (per-row) scaling.

    Args:
        x: Input tensor [M, K] in float16/bfloat16/float32
        scale: Optional pre-computed scale. If None, computed dynamically.

    Returns:
        Tuple of:
            - Quantized tensor [M, K] in FP8 format
            - Scale tensor [M, 1] in float32
    """
    _check_aiter_quant_available()

    return _aiter_per_token_quant(
        x,
        scale=scale,
        quant_dtype=_aiter_fp8_quant_dtype(),
    )


def dynamic_per_group_quant_fp8(
    x: Tensor,
    group_size: int = 128,
    scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Dynamic per-group quantization to FP8.

    Quantizes input tensor dynamically using per-group scaling.

    Args:
        x: Input tensor [M, K] in float16/bfloat16/float32
        group_size: Size of quantization groups (default: 128)
        scale: Optional pre-computed scale. If None, computed dynamically.

    Returns:
        Tuple of:
            - Quantized tensor [M, K] in FP8 format
            - Scale tensor [M, K/group_size] in float32
    """
    _check_aiter_quant_available()

    return _aiter_per_group_quant(
        x,
        scale=scale,
        quant_dtype=_aiter_fp8_quant_dtype(),
        group_size=group_size,
    )


def dynamic_per_tensor_quant_fp8(
    x: Tensor,
    scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Dynamic per-tensor quantization to FP8.

    Quantizes input tensor dynamically using per-tensor scaling.

    Args:
        x: Input tensor [M, K] in float16/bfloat16/float32
        scale: Optional pre-computed scale. If None, computed dynamically.

    Returns:
        Tuple of:
            - Quantized tensor [M, K] in FP8 format
            - Scale tensor [1] in float32
    """
    _check_aiter_quant_available()

    return _aiter_per_tensor_quant(
        x,
        scale=scale,
        quant_dtype=_aiter_fp8_quant_dtype(),
    )


def dynamic_per_group_quant_fp4(
    x: Tensor,
    group_size: int = 32,
    shuffle_scale: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Dynamic per-group quantization to MXFP4.

    Quantizes input tensor dynamically using per-32-element block scaling
    to MXFP4 format with e8m0 scales.

    Args:
        x: Input tensor [M, K] in float16/bfloat16/float32
        group_size: Size of quantization groups (default: 32 for MXFP4)
        shuffle_scale: Whether to shuffle scales for kernel compatibility

    Returns:
        Tuple of:
            - Quantized tensor [M, K/2] in fp4x2 packed format
            - Scale tensor in e8m0 format
    """
    _check_aiter_quant_available()

    # Ensure 2D input
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])

    # MXFP4 uses 32-element blocks by default
    y, scale = _aiter_per_1x32_f4_quant(
        x,
        shuffle=shuffle_scale,
    )

    return y, scale
