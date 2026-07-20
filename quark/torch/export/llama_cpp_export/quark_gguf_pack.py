#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Pack Quark real-quantized weights into llama.cpp GGUF Q4_0 / Q4_1 blocks without re-quantizing."""

from __future__ import annotations

import torch

from quark.torch.export.llama_cpp_export.quark_awq_unpack import (
    unpack_signed_int4,
    unpack_unsigned_int4,
)

_BLOCK_BYTES_Q4_1 = 20
_BLOCK_BYTES_Q4_0 = 18


def _f16_bytes(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.to(torch.float16).contiguous()
    return t.view(torch.uint8).reshape(*t.shape[:-1], 2)


def _expand_scale_and_zero(
    q: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match ``dequantize_uint4_weight`` / ``dequantize_weight`` broadcasting."""
    if scales.shape[0] * group_size == q.shape[0] and scales.shape[1] == q.shape[1]:
        s = scales.repeat_interleave(group_size, dim=0)
        z = zeros.repeat_interleave(group_size, dim=0)
    elif scales.shape[0] == q.shape[0] and scales.shape[1] * group_size == q.shape[1]:
        s = scales.repeat_interleave(group_size, dim=1)
        z = zeros.repeat_interleave(group_size, dim=1)
    else:
        raise ValueError(
            f"Unsupported Quark scale/zero shape {scales.shape} / {zeros.shape} for q {q.shape}"
        )
    return s, z


def _pack_q4_1_rows(q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    """Pack uint4 values with per-group scale/zero into Q4_1 blocks along the last dim."""
    rows, cols = q.shape
    group_size = 32
    if cols % group_size != 0:
        raise ValueError(f"Last dimension {cols} is not divisible by {group_size}")
    num_groups = cols // group_size

    q = q.reshape(rows, num_groups, group_size)
    s = scales.reshape(rows, num_groups, group_size)[:, :, 0:1].to(torch.float32)
    z = zeros.reshape(rows, num_groups, group_size)[:, :, 0:1].to(torch.float32)

    d = s.to(torch.float16)
    m = (-z * s).to(torch.float16)

    left = q[:, :, : group_size // 2]
    right = q[:, :, group_size // 2 :]
    nibbles = left | (right << 4)

    blocks = torch.cat([_f16_bytes(d), _f16_bytes(m), nibbles], dim=-1)
    return blocks.reshape(rows, num_groups * _BLOCK_BYTES_Q4_1).to(torch.uint8)


def pack_affine_uint4_to_q4_1(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    *,
    group_size: int,
    pack_reorder: bool,
    trim_output_dim: int | None = None,
) -> torch.Tensor:
    """Pack Quark uint4 WO tensors into Q4_1 rows in llama.cpp layout (post-transpose)."""
    q = unpack_unsigned_int4(qweight, pack_reorder=pack_reorder)
    z = unpack_unsigned_int4(qzeros, pack_reorder=pack_reorder).to(torch.float32)
    s = scales.to(torch.float32)

    if trim_output_dim is not None:
        q = q[:, :trim_output_dim]
        z = z[:, :trim_output_dim]
        s = s[:, :trim_output_dim]

    s, z = _expand_scale_and_zero(q, s, z, group_size)
    q = q.T.contiguous()
    s = s.T.contiguous()
    z = z.T.contiguous()
    return _pack_q4_1_rows(q.to(torch.uint8), s, z)


def pack_symmetric_int4_to_q4_0(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    pack_reorder: bool,
    trim_output_dim: int | None = None,
) -> torch.Tensor:
    """Pack Quark int4 WO tensors into Q4_0 rows in llama.cpp layout (post-transpose)."""
    q = unpack_signed_int4(qweight, pack_reorder=pack_reorder)
    s = scales.to(torch.float32)

    if trim_output_dim is not None:
        q = q[:, :trim_output_dim]
        s = s[:, :trim_output_dim]

    qs = (q + 8).clamp(0, 15).to(torch.uint8)
    if s.shape[0] * group_size == qs.shape[0] and s.shape[1] == qs.shape[1]:
        s = s.repeat_interleave(group_size, dim=0)
    elif s.shape[0] == qs.shape[0] and s.shape[1] * group_size == qs.shape[1]:
        s = s.repeat_interleave(group_size, dim=1)
    else:
        raise ValueError(f"Unsupported Quark scale shape {s.shape} for q {qs.shape}")

    qs = qs.T.contiguous()
    s = s.T.contiguous()

    rows, cols = qs.shape
    num_groups = cols // group_size
    qs = qs.reshape(rows, num_groups, group_size)
    s = s.reshape(rows, num_groups, group_size)[:, :, 0:1].to(torch.float16)

    left = qs[:, :, : group_size // 2]
    right = qs[:, :, group_size // 2 :]
    nibbles = left | (right << 4)

    blocks = torch.cat([_f16_bytes(s), nibbles], dim=-1)
    return blocks.reshape(rows, num_groups * _BLOCK_BYTES_Q4_0).to(torch.uint8)
