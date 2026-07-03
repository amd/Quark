#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Defensive numerics helpers used across observers, packers, and exporters.

The functions here cover two recurring failure modes in the codebase:

1. ``log2`` of a non-positive value produces ``-inf`` / ``NaN`` and silently
   poisons downstream casts (``.to(torch.int8)``, ``.to(torch.uint8)``,
   bit-reinterprets) — the result is garbage scale bytes in packed tensors and
   exported checkpoints. See prior fix commit a92e3acbb95 for the same family
   in the MXFP4 triton kernel.

2. ``Tensor.max()`` / ``Tensor.min()`` raise ``RuntimeError`` on empty tensors,
   which is reachable from observer code paths that filter inputs (e.g.
   ``_skip_zeros`` on an all-zero calibration batch).
"""

from __future__ import annotations

import torch

# Smallest positive normal float32. Matches the epsilon used in the prior MXFP4
# triton fix; gives a finite, well-defined result for ``log2`` of zero without
# perturbing any positive input above the subnormal range.
_FP32_MIN_NORMAL = 2.0**-126


def safe_log2(x: torch.Tensor) -> torch.Tensor:
    """``torch.log2`` clamped to keep the output finite.

    Non-positive inputs (zero, negative, or any value smaller than the smallest
    positive normal float32) are clamped to ``2**-126`` before the log. Inputs
    above that threshold are unchanged, so this is a drop-in replacement for
    ``torch.log2`` at scale-computation sites where degenerate blocks would
    otherwise emit ``-inf`` / ``NaN`` and corrupt downstream integer casts.
    """
    return torch.log2(torch.clamp(x, min=_FP32_MIN_NORMAL))


def safe_max(x: torch.Tensor, default: float = 0.0) -> float:
    """``x.max().item()`` that returns ``default`` when ``x`` is empty.

    Reaches into the codebase wherever an observer or calibrator might filter
    its input down to nothing (e.g. ``_skip_zeros=True`` on an all-zero batch).
    """
    if x.numel() == 0:
        return default
    return x.max().item()


def safe_min(x: torch.Tensor, default: float = 0.0) -> float:
    """``x.min().item()`` that returns ``default`` when ``x`` is empty."""
    if x.numel() == 0:
        return default
    return x.min().item()


def to_e8m0_uint8(scale: torch.Tensor) -> torch.Tensor:
    """Convert a positive float scale tensor to its UE8M0 (biased uint8) byte
    representation used by MXFP4 / e8m0 export paths.

    Uses :func:`safe_log2` so degenerate (zero) scales stay finite — without
    it, ``log2(0) -> -inf`` would bit-cast through ``int16`` to garbage and
    produce a corrupted scale byte in the exported tensor.
    """
    return (safe_log2(scale).round().to(torch.int16).clamp(-127, 127) + 127).to(torch.uint8)
