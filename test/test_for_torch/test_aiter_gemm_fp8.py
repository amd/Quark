#
# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Tests for Aiter FP8 GEMM kernels: regular gemm_fp8, gemm_fp8_bpreshuffle,
and gemm_fp8_blockscale.

Verifies correctness against a Torch reference GEMM in the dequantized domain.

Run:
    pytest test/test_for_torch/test_aiter_gemm_fp8.py -v
"""

import pytest
import torch


def _skip_if_no_aiter():
    """Skip test if Aiter kernels are unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")

    import quark.torch.kernel.aiter.gemm as aiter_gemm

    if not aiter_gemm.is_aiter_available():
        pytest.skip("Aiter GEMM kernels not available")


def _skip_if_no_shuffle():
    """Skip test if Aiter shuffle op is unavailable."""
    _skip_if_no_aiter()
    try:
        from aiter.ops.shuffle import shuffle_weight  # noqa: F401
    except ImportError:
        pytest.skip("aiter.ops.shuffle not available")


def _get_fp8_dtype():
    """Return the hardware-native FP8 dtype.

    MI300X (gfx942) uses float8_e4m3fnuz; MI325/gfx950 uses float8_e4m3fn.
    Detect by running dynamic_per_tensor_quant_fp8 on a tiny tensor.
    """
    from quark.torch.kernel.aiter.quant import dynamic_per_tensor_quant_fp8

    probe = torch.ones(1, 64, device="cuda", dtype=torch.bfloat16)
    qp, _ = dynamic_per_tensor_quant_fp8(probe)
    return qp.dtype


MATRIX_SHAPES = [
    # (M, K, N) — cover small, medium, and production-like sizes.
    # K must be > 192 and K % 64 == 0 for Aiter heuristic dispatch.
    (16, 256, 256),
    (32, 512, 512),
    (64, 256, 1024),
    (128, 512, 2048),
    (256, 256, 5120),
    (2048, 5120, 5120),
]

BLOCK_SCALE_SHAPES = [
    # (M, K, N, block_n, block_k) — 2D blocking on both N and K.
    (16, 256, 256, 128, 128),
    (32, 512, 512, 128, 128),
    (64, 256, 1024, 128, 128),
    (128, 512, 2048, 128, 128),
    (256, 256, 5120, 128, 128),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_per_tensor_test_data(m, k, n, seed=42):
    """Build quantized input/weight with per-tensor scales and a reference output."""
    torch.manual_seed(seed)
    from quark.torch.kernel.aiter.quant import dynamic_per_tensor_quant_fp8

    fp8_dtype = _get_fp8_dtype()

    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    xq, x_scale = dynamic_per_tensor_quant_fp8(x)
    assert xq.dtype == fp8_dtype

    w_float = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.1
    w_scale = torch.tensor(0.25, device="cuda", dtype=torch.float32)
    wq = (w_float / w_scale).to(fp8_dtype)

    # Torch reference in dequantized domain.
    x_deq = xq.float() * x_scale.float().item()
    w_deq = wq.float() * w_scale.float().item()
    out_ref = torch.matmul(x_deq, w_deq.t()).to(torch.bfloat16)

    return xq, x_scale, wq, w_scale, out_ref


def _build_per_token_test_data(m, k, n, seed=42):
    """Build quantized input with per-token scale and weight with per-channel scale."""
    torch.manual_seed(seed)
    from quark.torch.kernel.aiter.quant import dynamic_per_token_quant_fp8

    fp8_dtype = _get_fp8_dtype()

    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    xq, x_scale = dynamic_per_token_quant_fp8(x)
    assert xq.dtype == fp8_dtype

    w_float = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.1
    w_max = w_float.abs().amax(dim=1)
    fp8_max = 448.0
    w_scale = (w_max / fp8_max).clamp(min=1e-12)
    wq = (w_float / w_scale.unsqueeze(1)).to(fp8_dtype)

    # Torch reference
    x_deq = xq.float() * x_scale.float()  # [M, 1] broadcast
    w_deq = wq.float() * w_scale.float().unsqueeze(1)  # [N, 1] broadcast
    out_ref = torch.matmul(x_deq, w_deq.t()).to(torch.bfloat16)

    return xq, x_scale, wq, w_scale, out_ref


def _build_blockscale_test_data(m, k, n, block_n=128, block_k=128, seed=42):
    """Build quantized input/weight with 2D block scales and a reference output.

    The blockscale kernel expects:
        x_scale: [M, scale_k]          where scale_k = ceil(K / block_k)
        w_scale: [scale_n, scale_k]    where scale_n = ceil(N / block_n)
    """
    torch.manual_seed(seed)
    import math

    fp8_dtype = _get_fp8_dtype()
    scale_k = math.ceil(k / block_k)
    scale_n = math.ceil(n / block_n)

    # Quantize x and w into FP8 with random block scales
    x_float = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
    w_float = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.1

    x_scale = torch.rand(m, scale_k, dtype=torch.float32, device="cuda").clamp(min=0.01)
    w_scale = torch.rand(scale_n, scale_k, dtype=torch.float32, device="cuda").clamp(min=0.01)

    # Quantize x per [row, k-block]
    xq = torch.zeros(m, k, device="cuda", dtype=fp8_dtype)
    for bk in range(scale_k):
        ks = bk * block_k
        ke = min(ks + block_k, k)
        xq[:, ks:ke] = (x_float[:, ks:ke].float() / x_scale[:, bk : bk + 1]).to(fp8_dtype)

    # Quantize w per [n-block, k-block]
    wq = torch.zeros(n, k, device="cuda", dtype=fp8_dtype)
    for bn in range(scale_n):
        ns = bn * block_n
        ne = min(ns + block_n, n)
        for bk in range(scale_k):
            ks = bk * block_k
            ke = min(ks + block_k, k)
            wq[ns:ne, ks:ke] = (w_float[ns:ne, ks:ke].float() / w_scale[bn, bk]).to(fp8_dtype)

    # Torch reference: dequantize blocks and do matmul
    x_deq = torch.zeros(m, k, device="cuda", dtype=torch.float32)
    w_deq = torch.zeros(n, k, device="cuda", dtype=torch.float32)
    for bk in range(scale_k):
        ks = bk * block_k
        ke = min(ks + block_k, k)
        x_deq[:, ks:ke] = xq[:, ks:ke].float() * x_scale[:, bk : bk + 1].float()
    for bn in range(scale_n):
        ns = bn * block_n
        ne = min(ns + block_n, n)
        for bk in range(scale_k):
            ks = bk * block_k
            ke = min(ks + block_k, k)
            w_deq[ns:ne, ks:ke] = wq[ns:ne, ks:ke].float() * w_scale[bn, bk].float()
    out_ref = torch.matmul(x_deq, w_deq.t()).to(torch.bfloat16)

    return xq, x_scale, wq, w_scale, out_ref


def _make_scales(x_scale, w_scale, m, n, strategy):
    """Format scales according to the strategy being tested."""
    if strategy == "new_2d":
        xs = x_scale.float().reshape(1, 1).expand(m, 1).contiguous()
        ws = w_scale.float().reshape(1, 1).expand(1, n).contiguous()
    elif strategy == "old_1d":
        xs = x_scale.float().reshape(-1).repeat(m)  # [M]
        ws = w_scale.float().reshape(-1).repeat(n)  # [N]
    elif strategy == "scalar":
        xs = x_scale.float().reshape(1)  # [1]
        ws = w_scale.float().reshape(1)  # [1]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return xs, ws


def _get_tolerances(m, n):
    """Return atol/rtol depending on output size (larger accumulations -> more error)."""
    total = m * n
    atol = 0.02 if total > 1_000_000 else 5e-3
    rtol = 0.01 if total > 1_000_000 else 5e-3
    return atol, rtol


# ---------------------------------------------------------------------------
# Regular gemm_fp8 tests
# ---------------------------------------------------------------------------


class TestRegularGemmFP8:
    """Test gemm_fp8 (standard FP8 GEMM without pre-shuffled weights)."""

    @pytest.mark.parametrize("m,k,n", MATRIX_SHAPES)
    def test_per_tensor_vs_torch_ref(self, m, k, n):
        """Per-tensor scaled gemm_fp8 should produce output close to Torch reference."""
        _skip_if_no_aiter()
        import quark.torch.kernel.aiter.gemm as aiter_gemm

        xq, x_scale, wq, w_scale, out_ref = _build_per_tensor_test_data(m, k, n)

        # gemm_fp8 expects x_scale [M,1] or [1], w_scale [N] or [1]
        xs = x_scale.float().reshape(-1).repeat(m)
        ws = w_scale.float().reshape(-1).repeat(n)

        try:
            out = aiter_gemm.gemm_fp8(
                XQ=xq,
                WQ=wq,
                x_scale=xs,
                w_scale=ws,
                bias=None,
                output_dtype=torch.bfloat16,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "not supported" in msg or "unsupported" in msg:
                pytest.skip(f"gemm_fp8 unsupported for ({m},{k},{n}): {exc}")
            raise

        assert out.shape == (m, n), f"Expected ({m},{n}), got {out.shape}"
        assert int(torch.isnan(out).sum()) == 0, "NaN in gemm_fp8 output"
        assert int(torch.isinf(out).sum()) == 0, "Inf in gemm_fp8 output"

        atol, rtol = _get_tolerances(m, n)
        torch.testing.assert_close(
            out,
            out_ref,
            atol=atol,
            rtol=rtol,
            msg=lambda s: f"gemm_fp8 per-tensor shape=({m},{k},{n}): {s}",
        )

    @pytest.mark.parametrize("m,k,n", MATRIX_SHAPES)
    def test_per_token_vs_torch_ref(self, m, k, n):
        """Per-token/per-channel scaled gemm_fp8 should match Torch reference."""
        _skip_if_no_aiter()
        import quark.torch.kernel.aiter.gemm as aiter_gemm

        xq, x_scale, wq, w_scale, out_ref = _build_per_token_test_data(m, k, n)

        try:
            out = aiter_gemm.gemm_fp8(
                XQ=xq,
                WQ=wq,
                x_scale=x_scale,
                w_scale=w_scale,
                bias=None,
                output_dtype=torch.bfloat16,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "not supported" in msg or "unsupported" in msg:
                pytest.skip(f"gemm_fp8 unsupported for ({m},{k},{n}): {exc}")
            raise

        assert out.shape == (m, n), f"Expected ({m},{n}), got {out.shape}"
        assert int(torch.isnan(out).sum()) == 0, "NaN in gemm_fp8 output"
        assert int(torch.isinf(out).sum()) == 0, "Inf in gemm_fp8 output"

        atol, rtol = _get_tolerances(m, n)
        torch.testing.assert_close(
            out,
            out_ref,
            atol=atol,
            rtol=rtol,
            msg=lambda s: f"gemm_fp8 per-token shape=({m},{k},{n}): {s}",
        )

    @pytest.mark.parametrize("m,k,n", MATRIX_SHAPES)
    def test_output_dtype(self, m, k, n):
        """gemm_fp8 output dtype should match the requested output_dtype."""
        _skip_if_no_aiter()
        import quark.torch.kernel.aiter.gemm as aiter_gemm

        xq, x_scale, wq, w_scale, _ = _build_per_tensor_test_data(m, k, n)
        xs = x_scale.float().reshape(1)
        ws = w_scale.float().reshape(1).expand(n)

        for dtype in [torch.bfloat16, torch.float16]:
            try:
                out = aiter_gemm.gemm_fp8(
                    XQ=xq,
                    WQ=wq,
                    x_scale=xs,
                    w_scale=ws,
                    bias=None,
                    output_dtype=dtype,
                )
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "not supported" in msg or "unsupported" in msg:
                    continue
                raise
            assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


# ---------------------------------------------------------------------------
# Bpreshuffle gemm_fp8 tests
# ---------------------------------------------------------------------------


class TestBpreshuffleDirectAiter:
    """Sanity check: call Aiter bpreshuffle kernel directly (bypass wrapper)."""

    def test_direct_aiter_call(self):
        """Verify the raw Aiter kernel works with known-good arguments."""
        _skip_if_no_shuffle()
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle as _raw_bpreshuffle
        from aiter.ops.shuffle import shuffle_weight

        fp8_dtype = _get_fp8_dtype()
        m, k, n = 16, 256, 256

        torch.manual_seed(0)
        x = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.1).to(fp8_dtype)
        w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.1).to(fp8_dtype)
        w_shuffled = shuffle_weight(w)

        x_scale = torch.rand([m, 1], dtype=torch.float32, device="cuda")
        w_scale = torch.rand([1, n], dtype=torch.float32, device="cuda")

        try:
            out = _raw_bpreshuffle(x, w_shuffled, x_scale, w_scale, dtype=torch.bfloat16)
        except RuntimeError as exc:
            pytest.fail(
                f"Direct Aiter bpreshuffle call failed: {exc}\n"
                f"  x.dtype={x.dtype}, w.dtype={w.dtype}\n"
                f"  x_scale.dtype={x_scale.dtype}, shape={x_scale.shape}\n"
                f"  w_scale.dtype={w_scale.dtype}, shape={w_scale.shape}\n"
                f"  output_dtype=bfloat16"
            )

        assert out.shape == (m, n)
        assert out.dtype == torch.bfloat16
        assert int(torch.isnan(out).sum()) == 0, "Direct Aiter call produced NaN"


class TestBpreshuffleScaleShapes:
    """Compare gemm_fp8_bpreshuffle output across scale shape strategies."""

    @pytest.mark.parametrize("m,k,n", MATRIX_SHAPES)
    @pytest.mark.parametrize("strategy", ["new_2d", "old_1d"])
    def test_bpreshuffle_vs_torch_ref(self, m, k, n, strategy):
        """Each strategy should produce output close to the Torch reference."""
        _skip_if_no_shuffle()
        from aiter.ops.shuffle import shuffle_weight

        import quark.torch.kernel.aiter.gemm as aiter_gemm

        xq, x_scale, wq, w_scale, out_ref = _build_per_tensor_test_data(m, k, n)
        wq_shuffled = shuffle_weight(wq)
        xs, ws = _make_scales(x_scale, w_scale, m, n, strategy)

        try:
            out = aiter_gemm.gemm_fp8_bpreshuffle(
                XQ=xq,
                WQ=wq_shuffled,
                x_scale=xs,
                w_scale=ws,
                bias=None,
                output_dtype=torch.bfloat16,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "not supported" in msg or "unsupported" in msg:
                pytest.skip(f"Aiter bpreshuffle unsupported for ({m},{k},{n}): {exc}")
            raise

        assert out.shape == (m, n), f"Expected ({m},{n}), got {out.shape}"
        nan_count = int(torch.isnan(out).sum())
        inf_count = int(torch.isinf(out).sum())
        assert nan_count == 0, f"strategy={strategy}, shape=({m},{k},{n}): {nan_count} NaN in output"
        assert inf_count == 0, f"strategy={strategy}, shape=({m},{k},{n}): {inf_count} Inf in output"

        atol, rtol = _get_tolerances(m, n)
        torch.testing.assert_close(
            out,
            out_ref,
            atol=atol,
            rtol=rtol,
            msg=lambda s: f"strategy={strategy}, shape=({m},{k},{n}): {s}",
        )

    @pytest.mark.parametrize("m,k,n", MATRIX_SHAPES)
    def test_all_strategies_match_each_other(self, m, k, n):
        """All scale strategies should produce identical kernel output."""
        _skip_if_no_shuffle()
        from aiter.ops.shuffle import shuffle_weight

        import quark.torch.kernel.aiter.gemm as aiter_gemm

        xq, x_scale, wq, w_scale, _ = _build_per_tensor_test_data(m, k, n)
        wq_shuffled = shuffle_weight(wq)

        outputs = {}
        for strategy in ["new_2d", "old_1d"]:
            xs, ws = _make_scales(x_scale, w_scale, m, n, strategy)
            try:
                out = aiter_gemm.gemm_fp8_bpreshuffle(
                    XQ=xq,
                    WQ=wq_shuffled,
                    x_scale=xs,
                    w_scale=ws,
                    bias=None,
                    output_dtype=torch.bfloat16,
                )
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "not supported" in msg or "unsupported" in msg:
                    pytest.skip(f"Aiter bpreshuffle unsupported for ({m},{k},{n}): {exc}")
                raise
            outputs[strategy] = out

        for s in ["old_1d"]:
            nan_a = int(torch.isnan(outputs["new_2d"]).sum())
            nan_b = int(torch.isnan(outputs[s]).sum())
            if nan_a > 0 or nan_b > 0:
                pytest.fail(f"shape=({m},{k},{n}): NaN detected — new_2d has {nan_a} NaN, {s} has {nan_b} NaN")
            max_diff = float((outputs["new_2d"].float() - outputs[s].float()).abs().max())
            assert max_diff == 0, (
                f"shape=({m},{k},{n}): new_2d vs {s} max_diff={max_diff} "
                f"(expected bit-identical after wrapper normalization)"
            )


# ---------------------------------------------------------------------------
# Block-scale gemm_fp8 tests
# ---------------------------------------------------------------------------


class TestBlockScaleGemmFP8:
    """Test gemm_fp8_blockscale (FP8 GEMM with 2D block scaling)."""

    @pytest.mark.parametrize("m,k,n,block_n,block_k", BLOCK_SCALE_SHAPES)
    def test_blockscale_vs_torch_ref(self, m, k, n, block_n, block_k):
        """Block-scaled gemm should produce output close to Torch reference."""
        _skip_if_no_aiter()
        import quark.torch.kernel.aiter.gemm as aiter_gemm

        xq, x_scale, wq, w_scale, out_ref = _build_blockscale_test_data(
            m,
            k,
            n,
            block_n=block_n,
            block_k=block_k,
        )

        try:
            out = aiter_gemm.gemm_fp8_blockscale(
                XQ=xq,
                WQ=wq,
                x_scale=x_scale,
                w_scale=w_scale,
                output_dtype=torch.bfloat16,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "not supported" in msg or "unsupported" in msg:
                pytest.skip(f"gemm_fp8_blockscale unsupported for ({m},{k},{n}), block=({block_n},{block_k}): {exc}")
            raise

        assert out.shape == (m, n), f"Expected ({m},{n}), got {out.shape}"
        assert int(torch.isnan(out).sum()) == 0, "NaN in blockscale output"
        assert int(torch.isinf(out).sum()) == 0, "Inf in blockscale output"

        # Block-scale quantization introduces more error than per-tensor.
        atol = 0.05 if m * n > 1_000_000 else 0.02
        rtol = 0.02
        torch.testing.assert_close(
            out,
            out_ref,
            atol=atol,
            rtol=rtol,
            msg=lambda s: (f"gemm_fp8_blockscale shape=({m},{k},{n}), block=({block_n},{block_k}): {s}"),
        )

    @pytest.mark.parametrize("m,k,n,block_n,block_k", BLOCK_SCALE_SHAPES)
    def test_blockscale_no_nan_inf(self, m, k, n, block_n, block_k):
        """Block-scale output must not contain NaN or Inf."""
        _skip_if_no_aiter()
        import quark.torch.kernel.aiter.gemm as aiter_gemm

        xq, x_scale, wq, w_scale, _ = _build_blockscale_test_data(
            m,
            k,
            n,
            block_n=block_n,
            block_k=block_k,
        )

        try:
            out = aiter_gemm.gemm_fp8_blockscale(
                XQ=xq,
                WQ=wq,
                x_scale=x_scale,
                w_scale=w_scale,
                output_dtype=torch.bfloat16,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "not supported" in msg or "unsupported" in msg:
                pytest.skip(f"gemm_fp8_blockscale unsupported for ({m},{k},{n}), block=({block_n},{block_k}): {exc}")
            raise

        assert int(torch.isnan(out).sum()) == 0
        assert int(torch.isinf(out).sum()) == 0


# ---------------------------------------------------------------------------
# Cross-kernel comparison
# ---------------------------------------------------------------------------


class TestCrossKernelComparison:
    """Compare outputs across different GEMM kernel variants."""

    @pytest.mark.parametrize("m,k,n", MATRIX_SHAPES)
    def test_regular_vs_bpreshuffle(self, m, k, n):
        """Regular gemm_fp8 and bpreshuffle should agree on the same inputs."""
        _skip_if_no_shuffle()
        from aiter.ops.shuffle import shuffle_weight

        import quark.torch.kernel.aiter.gemm as aiter_gemm

        xq, x_scale, wq, w_scale, _ = _build_per_tensor_test_data(m, k, n)
        wq_shuffled = shuffle_weight(wq)

        xs_reg = x_scale.float().reshape(-1).repeat(m)
        ws_reg = w_scale.float().reshape(-1).repeat(n)

        xs_bp = x_scale.float().reshape(1, 1).expand(m, 1).contiguous()
        ws_bp = w_scale.float().reshape(1, 1).expand(1, n).contiguous()

        try:
            out_reg = aiter_gemm.gemm_fp8(
                XQ=xq,
                WQ=wq,
                x_scale=xs_reg,
                w_scale=ws_reg,
                bias=None,
                output_dtype=torch.bfloat16,
            )
            out_bp = aiter_gemm.gemm_fp8_bpreshuffle(
                XQ=xq,
                WQ=wq_shuffled,
                x_scale=xs_bp,
                w_scale=ws_bp,
                bias=None,
                output_dtype=torch.bfloat16,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "not supported" in msg or "unsupported" in msg:
                pytest.skip(f"GEMM unsupported for ({m},{k},{n}): {exc}")
            raise

        assert int(torch.isnan(out_reg).sum()) == 0
        assert int(torch.isnan(out_bp).sum()) == 0

        torch.testing.assert_close(
            out_bp,
            out_reg,
            atol=0.05,
            rtol=0.05,
            msg=lambda s: f"bpreshuffle vs regular shape=({m},{k},{n}): {s}",
        )
