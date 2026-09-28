#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for QAD 2-bit fake-quant + packed export round-trip (CPU, tiny)."""

import os
import sys

import pytest
import torch

# The QAD example utilities live under examples/.../llm_qad/.
_QAD = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../examples/torch/language_modeling/llm_qad"))
if _QAD not in sys.path:
    sys.path.insert(0, _QAD)

from export_qad_2bit_packed import DEFAULT_LEVELS, pack_qad_2bit, unpack_qad_2bit  # noqa: E402
from qad_weight_fakequant import fake_quant_weight_group_gptq_style  # noqa: E402

LEVELS = torch.tensor(DEFAULT_LEVELS)


def _exact_4level_weight(o=8, i=128, g=64, seed=0):
    """Construct a weight that is exactly 4-level x per-group scale."""
    torch.manual_seed(seed)
    idx = torch.randint(0, 4, (o, i))
    scale = torch.rand(o, i // g) * 2 + 0.5
    return (LEVELS[idx] * scale.repeat_interleave(g, dim=1)).float()


def test_pack_unpack_roundtrip_exact_4level():
    g = 64
    w = _exact_4level_weight(g=g)
    packed, scale = pack_qad_2bit(w, g, LEVELS)
    # packed is 4 int2 values per uint8 byte
    assert packed.dtype == torch.uint8
    assert packed.shape == (w.shape[0], w.shape[1] // 4)
    assert scale.dtype == torch.float16 and scale.shape == (w.shape[0], w.shape[1] // g)

    w_rec = unpack_qad_2bit(packed, scale, w.shape[1], g, LEVELS)
    # Exact 4-level input -> only fp16 scale-storage error remains.
    assert (w - w_rec).abs().max().item() < 1e-2


def test_packed_indices_in_range():
    g = 64
    w = _exact_4level_weight(g=g)
    packed, _ = pack_qad_2bit(w, g, LEVELS)
    # every unpacked index must be in {0,1,2,3}
    for k in range(4):
        idx = (packed >> (2 * k)) & 0x3
        assert int(idx.max()) <= 3 and int(idx.min()) >= 0


def test_fakequant_is_4_level_per_group():
    """fake_quant_weight_group_gptq_style snaps each group to 4 levels x scale."""
    torch.manual_seed(0)
    g = 64
    w = torch.randn(8, 128)
    wq = fake_quant_weight_group_gptq_style(w, group_size=g)
    assert wq.shape == w.shape
    # Within each group, dividing by max-abs should give values near {-1,-1/3,1/3,1}.
    o, i = w.shape
    wq_g = wq.view(o, i // g, g)
    scale = wq_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    u = wq_g / scale
    nearest = (u.unsqueeze(-1) - LEVELS.view(1, 1, 1, 4)).abs().min(dim=-1).values
    assert nearest.max().item() < 1e-2, "fake-quant output not on the 4-level grid"


def test_pack_requires_divisible_in_features():
    w = torch.randn(8, 130)  # 130 not divisible by 64
    with pytest.raises(ValueError):
        pack_qad_2bit(w, 64, LEVELS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
