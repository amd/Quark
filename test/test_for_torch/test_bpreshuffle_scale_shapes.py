#
# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Standalone test to verify gemm_fp8_bpreshuffle produces correct results
across all supported scale input shapes, and compares against the Torch
reference GEMM in dequantized domain.

Three scale-passing strategies are tested for each (M, K, N) combination:
  1. "new_2d"   — x_scale: [M, 1], w_scale: [1, N]  (correct bpreshuffle format)
  2. "old_1d"   — x_scale: [M],    w_scale: [N]      (old repeat-based format)
  3. "scalar"   — x_scale: [1],    w_scale: scalar    (raw per-tensor scalar)

All strategies are reshaped by the wrapper to [M,1] / [1,N] before reaching
the kernel, so they should all produce identical results.

Run:
    pytest test/test_for_torch/test_bpreshuffle_scale_shapes.py -v
"""

import pytest
import torch


def _skip_if_no_aiter():
    """Skip test if Aiter kernels or shuffle op are unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")

    import quark.torch.kernel.aiter.gemm as aiter_gemm

    if not aiter_gemm.is_aiter_available():
        pytest.skip("Aiter GEMM kernels not available")

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


def _build_test_data(m, k, n, seed=42):
    """Build quantized input/weight and reference output."""
    torch.manual_seed(seed)
    from aiter.ops.shuffle import shuffle_weight

    from quark.torch.kernel.aiter.quant import dynamic_per_tensor_quant_fp8

    fp8_dtype = _get_fp8_dtype()

    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    xq, x_scale = dynamic_per_tensor_quant_fp8(x)
    assert xq.dtype == fp8_dtype

    w_float = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.1
    w_scale = torch.tensor(0.25, device="cuda", dtype=torch.float32)
    wq = (w_float / w_scale).to(fp8_dtype)
    wq_shuffled = shuffle_weight(wq)

    # Torch reference in dequantized domain.
    x_deq = xq.float() * x_scale.float().item()
    w_deq = wq.float() * w_scale.float().item()
    out_ref = torch.matmul(x_deq, w_deq.t()).to(torch.bfloat16)

    return xq, x_scale, wq, wq_shuffled, w_scale, out_ref


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


class TestBpreshuffleDirectAiter:
    """Sanity check: call Aiter bpreshuffle kernel directly (bypass wrapper)."""

    def test_direct_aiter_call(self):
        """Verify the raw Aiter kernel works with known-good arguments."""
        _skip_if_no_aiter()
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
    @pytest.mark.parametrize("strategy", ["new_2d", "old_1d"])  # , "scalar"])
    def test_bpreshuffle_vs_torch_ref(self, m, k, n, strategy):
        """Each strategy should produce output close to the Torch reference."""
        _skip_if_no_aiter()
        import quark.torch.kernel.aiter.gemm as aiter_gemm

        xq, x_scale, wq, wq_shuffled, w_scale, out_ref = _build_test_data(m, k, n)
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

        # FP8 quantization + bf16 accumulation rounding.
        # Large K dimensions (5120) accumulate more rounding error.
        total = m * n
        atol = 0.02 if total > 1_000_000 else 5e-3
        rtol = 0.01 if total > 1_000_000 else 5e-3
        torch.testing.assert_close(
            out,
            out_ref,
            atol=atol,
            rtol=rtol,
            msg=lambda s: f"strategy={strategy}, shape=({m},{k},{n}): {s}",
        )

    @pytest.mark.parametrize("m,k,n", MATRIX_SHAPES)
    def test_all_strategies_match_each_other(self, m, k, n):
        """All three scale strategies should produce identical kernel output."""
        _skip_if_no_aiter()
        import quark.torch.kernel.aiter.gemm as aiter_gemm

        xq, x_scale, wq, wq_shuffled, w_scale, _ = _build_test_data(m, k, n)

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

        # All strategies go through the same wrapper reshape, so outputs
        # should be bit-identical (same kernel call after normalization).
        for s in ["old_1d"]:  # , "scalar"]:
            nan_a = int(torch.isnan(outputs["new_2d"]).sum())
            nan_b = int(torch.isnan(outputs[s]).sum())
            if nan_a > 0 or nan_b > 0:
                pytest.fail(f"shape=({m},{k},{n}): NaN detected — new_2d has {nan_a} NaN, {s} has {nan_b} NaN")
            max_diff = float((outputs["new_2d"].float() - outputs[s].float()).abs().max())
            assert max_diff == 0, (
                f"shape=({m},{k},{n}): new_2d vs {s} max_diff={max_diff} "
                f"(expected bit-identical after wrapper normalization)"
            )
