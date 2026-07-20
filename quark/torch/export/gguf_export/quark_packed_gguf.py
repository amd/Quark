#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Pack Quark real-quantized safetensor payloads into GGUF Q4_0 / Q4_1 blocks.

Internal helpers for ``gguf_export``; does not change public export APIs.
Maps:

* ``uint4_wo_32`` (asymmetric, group size 32) -> Q4_1
* ``int4_wo_32`` (symmetric, group size 32) -> Q4_0

Packing preserves Quark quant indices (no float re-quantization).
"""

from __future__ import annotations

import torch

_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
_BLOCK_BYTES_Q4_1 = 20
_BLOCK_BYTES_Q4_0 = 18


def sign_extend_int4(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.int8)
    return torch.where(values >= 8, values - 16, values)


def unpack_unsigned_int4(packed: torch.Tensor, *, pack_reorder: bool) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    unpacked = (packed.to(torch.int32)[:, :, None] >> shifts) & 0xF
    if pack_reorder:
        order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.long)
    else:
        order = torch.arange(8, dtype=torch.long)
    return unpacked[:, :, order].reshape(packed.shape[0], -1).to(torch.uint8)


def unpack_signed_int4(packed: torch.Tensor, *, pack_reorder: bool) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    unpacked = (packed.to(torch.int32)[:, :, None] >> shifts) & 0xF
    if pack_reorder:
        order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.long)
    else:
        order = torch.arange(8, dtype=torch.long)
    unpacked = unpacked[:, :, order].reshape(packed.shape[0], -1)
    return sign_extend_int4(unpacked)


def _f16_bytes(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.to(torch.float16).contiguous()
    return t.view(torch.uint8).reshape(*t.shape[:-1], 2)


def _expand_scale_and_zero(
    q: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def pack_uint4_wo32_to_q4_1(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    *,
    group_size: int = 32,
    pack_reorder: bool = True,
) -> torch.Tensor:
    """Pack Quark ``uint4_wo_32`` tensors into GGUF Q4_1 byte rows (llama.cpp layout)."""
    if group_size != 32:
        raise ValueError(f"Q4_1 native pack requires group_size=32, got {group_size}")
    q = unpack_unsigned_int4(qweight, pack_reorder=pack_reorder)
    z = unpack_unsigned_int4(qzeros, pack_reorder=pack_reorder).to(torch.float32)
    s = scales.to(torch.float32)
    s, z = _expand_scale_and_zero(q, s, z, group_size)
    q = q.T.contiguous()
    s = s.T.contiguous()
    z = z.T.contiguous()
    return _pack_q4_1_rows(q, s, z)


def pack_int4_wo32_to_q4_0(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = 32,
    pack_reorder: bool = True,
) -> torch.Tensor:
    """Pack Quark ``int4_wo_32`` tensors into GGUF Q4_0 byte rows (llama.cpp layout)."""
    if group_size != 32:
        raise ValueError(f"Q4_0 native pack requires group_size=32, got {group_size}")
    q = unpack_signed_int4(qweight, pack_reorder=pack_reorder)
    s = scales.to(torch.float32)
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
