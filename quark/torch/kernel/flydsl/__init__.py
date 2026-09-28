#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""FlyDSL kernels for the A8W4 / SVDQuant native inference backends (gfx950).

The GEMM is the vendored snapshot in ``.kernels.preshuffle_gemm``. It carries the
fused SVD epilogue (ROCm/FlyDSL#957), which has not landed upstream, so it is the
only implementation that can serve ``flydsl_svdquant``; its MFMA epilogue and
pipeline helpers are unmodified and are imported from ``aiter`` instead. It therefore
needs the ``flydsl`` compiler/runtime wheel plus ``aiter``, and nothing else;
``docs/source/pytorch/pytorch_troubleshooting.rst`` explains why the flydsl pin is
exact. The MXFP8 activation-quant kernel in ``.mxfp8_quant`` has no upstream
equivalent.
"""

from functools import cache
from typing import Any

_INSTALL_HINT = (
    "FlyDSL native inference requires the flydsl compiler/runtime and aiter; see the "
    "FlyDSL section of docs/source/pytorch/pytorch_troubleshooting.rst for the pinned "
    "flydsl version and the aiter install command."
)


@cache
def _preshuffle_gemm() -> Any:
    """Import the vendored A8W4 GEMM module (needs the ``flydsl`` wheel and ``aiter``)."""
    try:
        from .kernels import preshuffle_gemm
    except ImportError as e:
        raise ImportError(f"{_INSTALL_HINT} Import error: {e}") from e
    return preshuffle_gemm


def get_a8w4_compile() -> Any:
    """Return the ``compile_preshuffle_gemm_a8w4`` factory (FP8 act x MXFP4 weight)."""
    return _preshuffle_gemm().compile_preshuffle_gemm_a8w4


@cache
def get_mxfp8_quant() -> Any:
    """Return the ``quantize_mxfp8_1x32_flydsl`` activation-quant callable."""
    try:
        from .mxfp8_quant import quantize_mxfp8_1x32_flydsl
    except ImportError as e:
        raise ImportError(f"{_INSTALL_HINT} Import error: {e}") from e
    return quantize_mxfp8_1x32_flydsl


def is_flydsl_quant_available() -> bool:
    """True if the native FlyDSL MXFP8 quant kernel is importable."""
    try:
        get_mxfp8_quant()
        return True
    except ImportError:
        return False
