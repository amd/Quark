#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""E8M0 activation-scale rounding must be ceil.

Regression test for a bug that shipped because the implementations were written to be
bit-exact to each other and were all wrong the same way, so every assertion here
compares against ``ceil(log2(amax / dtype_max))`` computed from first principles rather
than against another backend. Rounding down makes ``amax / scale`` exceed the dtype max
and the clamp then truncates the block max -- a magnitude loss that cosine similarity is
blind to.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

FP8_MAX = 448.0


def _expected_e8m0(amax: float, dtype_max: float = FP8_MAX) -> int:
    """ceil(log2(amax/dtype_max)) as an E8M0 byte, derived independently of any backend."""
    return int(math.ceil(math.log2(amax / dtype_max))) + 127


@pytest.mark.parametrize("amax", [0.5, 1.0, 1.5, 1.9, 3.0, 7.3, 0.017, 1024.0])
def test_e8m0_exponent_is_ceil(amax: float) -> None:
    """The eager reference must pick the ceil exponent for a single 32-block."""
    from quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear import (
        _quantize_mxfp8_1x32_eager,
    )

    x = torch.full((1, 32), amax, device="cuda", dtype=torch.bfloat16)
    _, scale = _quantize_mxfp8_1x32_eager(x)
    got = int(scale.view(torch.uint8).flatten()[0].item())
    assert got == _expected_e8m0(amax), (
        f"amax={amax}: e8m0 {got}, expected ceil -> {_expected_e8m0(amax)}. "
        f"A smaller exponent makes amax/scale exceed {FP8_MAX} and the clamp truncates."
    )


@pytest.mark.parametrize("amax", [0.5, 1.0, 1.9, 7.3])
def test_block_max_never_saturates(amax: float) -> None:
    """amax/scale must stay <= FP8_MAX, i.e. the block max must not hit the clamp."""
    from quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear import (
        _quantize_mxfp8_1x32_eager,
    )

    x = torch.full((1, 32), amax, device="cuda", dtype=torch.bfloat16)
    _, scale = _quantize_mxfp8_1x32_eager(x)
    s = 2.0 ** (int(scale.view(torch.uint8).flatten()[0].item()) - 127)
    assert amax / s <= FP8_MAX + 1e-6, f"amax/scale = {amax / s:.1f} > {FP8_MAX}: clamp truncates"


def test_all_ones_roundtrip_is_exact() -> None:
    """The canonical failure: round-to-nearest gave exactly 7/8 here."""
    from quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear import (
        _e8m0_to_fp32,
        _quantize_mxfp8_1x32_eager,
    )

    x = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
    q, scale = _quantize_mxfp8_1x32_eager(x)
    s = _e8m0_to_fp32(scale.view(torch.uint8).flatten()[0]).item()
    deq = q.view(torch.float8_e4m3fn).float().flatten()[0].item() * s
    assert deq == pytest.approx(1.0, abs=1e-6), f"all-ones dequantized to {deq} (7/8 = the old bug)"


def test_gain_is_unbiased_on_random_input() -> None:
    """Aggregate magnitude must not drift low -- the property cosine cannot see."""
    from quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear import (
        _e8m0_to_fp32,
        _quantize_mxfp8_1x32_eager,
    )

    torch.manual_seed(0)
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    q, scale = _quantize_mxfp8_1x32_eager(x)
    deq = (
        q.view(torch.float8_e4m3fn).float().view(-1, 32) * _e8m0_to_fp32(scale.view(torch.uint8)).float().view(-1, 1)
    ).view(x.shape)
    gain = float(deq.norm() / x.float().norm())
    # Round-to-nearest measured ~0.963 here; ceil sits within a few 1e-3 of 1.0.
    assert 0.99 <= gain <= 1.01, f"quantization gain {gain:.4f} is biased (expect ~1.0)"


@pytest.mark.flydsl_gfx950
def test_kernel_matches_the_eager_reference() -> None:
    """Only now compare the kernel: the eager side is already pinned to the ceil above."""
    from quark.torch.quantization.nn.modules import flydsl_a8w4_inference_linear as M

    if not M._flydsl_quant_available():
        pytest.skip("flydsl MXFP8 quant kernel unavailable; run tools/ci/setup_flydsl.sh")

    torch.manual_seed(0)
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    eq, es = M._quantize_mxfp8_1x32_eager(x)
    es = M._shuffle_e8m0_scale(es.view(torch.uint8)).view(torch.uint8).contiguous()
    fq, fs = M._quantize_mxfp8_1x32_flydsl_shuffled(x)
    assert torch.equal(eq.view(torch.uint8), fq.view(torch.uint8)), "flydsl quant != eager"
    assert torch.equal(es, fs), "flydsl e8m0 scale != eager"


@pytest.mark.flydsl_gfx950
def test_the_eager_reference_is_bit_exact_to_the_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """_quantize_activation_mxfp8 must return identical bytes on both of its paths.

    The GEMM cannot tell which path produced its operands, so "bit-exact to the eager
    reference" is a hard requirement rather than a tolerance: a kernel that rounded
    differently would make results depend on whether the flydsl wheel is installed.
    """
    from quark.torch.quantization.nn.modules import flydsl_a8w4_inference_linear as M

    if not M._flydsl_quant_available():
        pytest.skip("flydsl MXFP8 quant kernel unavailable; run tools/ci/setup_flydsl.sh")

    torch.manual_seed(0)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)

    q_kernel, s_kernel = M._quantize_activation_mxfp8(x)
    # Same input, kernel reported as unavailable -> the pure-torch reference runs.
    monkeypatch.setattr(M, "_flydsl_quant_available", lambda: False)
    q_eager, s_eager = M._quantize_activation_mxfp8(x)

    assert torch.equal(q_kernel, q_eager), "quantized activations differ between the two paths"
    # _shuffle_e8m0_scale pads M up to a multiple of 256 and leaves the pad rows
    # uninitialized, so only the defined region is meaningful.
    rows, cols = x.shape[0], x.shape[-1] // 32
    k_2d = s_kernel.view(-1, cols)[:rows]
    e_2d = s_eager.view(-1, cols)[:rows]
    assert torch.equal(k_2d, e_2d), "e8m0 scales differ between the two paths"
