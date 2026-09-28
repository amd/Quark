#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Cover Pack_uint2.unpack truncation via origin_packed_axis_size (2-D and 1-D)."""

import torch

from quark.torch.utils.pack import Pack_uint2


def test_unpack_truncates_to_origin_axis_size_2d():
    p = Pack_uint2(qscheme="per_tensor", dtype="uint2")
    orig = torch.randint(0, 4, (8, 4))
    packed = p.pack(orig, reorder=True)
    full = p.unpack(packed, reorder=True)
    assert tuple(full.shape) == (8, 4)
    trunc = p.unpack(packed, reorder=True, origin_packed_axis_size=6)
    assert tuple(trunc.shape) == (6, 4)


def test_unpack_truncates_to_origin_axis_size_1d():
    p = Pack_uint2(qscheme="per_tensor", dtype="uint2")
    orig = torch.randint(0, 4, (8,))
    packed = p.pack(orig, reorder=True)
    full = p.unpack(packed, reorder=True)
    assert full.dim() == 1 and full.shape[0] == 8
    trunc = p.unpack(packed, reorder=True, origin_packed_axis_size=6)
    assert tuple(trunc.shape) == (6,)
