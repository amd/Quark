# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""SRHT (Structured Random Hadamard Transform) helpers for factored exports.

These are the exact rotation primitives used by TwoBitScalar PTQ to build the
per-layer rotation `R` (random permutation -> random sign flip -> block
Walsh-Hadamard). The factored/shared-rotation converters reuse them to
materialize the dense rotation matrices from the per-layer sidecar
(`srht_perm`, `srht_signs`). Keeping them here makes the factored-export tools
self-contained.

Conventions (match the project-wide SRHT):
- `rotate_last_dim(x, perm, signs, blk)` rotates the last dim of `x`.
- `srht_block(in_dim)` = largest power-of-2 factor of in_dim, capped at 1024.
- Orthogonal: applying the transpose recovers the input exactly.
"""

import torch


def _largest_pow2_factor(n: int) -> int:
    return n & (-n) if n else 0


def _hadamard(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    orig_shape = x.shape
    y = x.reshape(-1, n)
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).view(-1, n)
        h *= 2
    return y.view(*orig_shape)


def srht_block(in_dim: int) -> int:
    return min(_largest_pow2_factor(in_dim), 1024)


def rotate_last_dim(x: torch.Tensor, perm: torch.Tensor, signs: torch.Tensor, blk: int) -> torch.Tensor:
    n = x.shape[-1]
    leading = x.shape[:-1]
    perm_d = perm.to(x.device)
    s = signs.to(device=x.device, dtype=x.dtype)
    x = x[..., perm_d] * s
    if blk >= 2:
        nblk = n // blk
        x = _hadamard(x.reshape(*leading, nblk, blk)) / (blk**0.5)
        x = x.reshape(*leading, n)
    return x
