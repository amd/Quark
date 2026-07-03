#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Test case to demonstrate Bug #3: Log2 of zero in MXFP4 triton kernel.

The bug is in quark/torch/kernel/mx/triton.py lines 82-83:

    # In EVEN rounding mode:
    # eps =  tl.where(max_val == 0.0, 2**(-126), 0.0)
    max_val = max_val.to(tl.int32, bitcast=True)
    max_val = (max_val + 0x200000).to(tl.uint32, bitcast=True) & 0x7F800000
    max_val = max_val.to(tl.float32, bitcast=True)
    # scale_e8m0_unbiased = tl.log2(max_val + eps).floor() - _get_max_quant_exp(mx_tensor_dtype)
    scale_e8m0_unbiased = tl.log2(max_val).floor() - _get_max_quant_exp(mx_tensor_dtype)  # no eps

The issue is that when the input block contains all zeros, max_val becomes 0.
Then tl.log2(0) produces -inf, which propagates through the computation and
produces incorrect scale values. The commented-out eps handling would fix this.

This bug affects:
1. All-zero input blocks produce incorrect/undefined scale values
2. The dequantized output may contain NaN or incorrect values
3. Numerical instability in edge cases
"""

import pytest
import torch

from quark.common.utils.import_utils import is_triton_available
from quark.torch.quantization.utils import even_round


class TestBug3Log2Zero:
    """Test cases demonstrating the log2(0) bug in MXFP4 triton kernel"""

    def test_all_zeros_block_produces_finite_scale(self):
        """
        When an input block is all zeros, the reference even_round implementation
        must produce a finite scale (the kernel should match this behavior).
        """
        max_val = torch.zeros(1, dtype=torch.float32)

        scale = even_round(max_val, "fp4")

        assert torch.isfinite(scale).all(), "Scale should be finite even for zero input"
        assert not torch.isnan(scale).any(), "Scale should not be NaN for zero input"

    def test_log2_zero_produces_negative_inf(self):
        """
        Demonstrate that log2(0) produces -inf, which is the root cause of the bug.
        """
        zero = torch.tensor(0.0)
        log2_zero = torch.log2(zero)

        assert torch.isinf(log2_zero), "log2(0) should produce inf"
        assert log2_zero < 0, "log2(0) should produce negative inf"

    def test_mixed_zeros_and_nonzeros(self):
        """
        Test blocks where some are all zeros and some have values.
        The bug would cause only the zero blocks to have incorrect scales.
        """
        max_val = torch.tensor([0.0, 1.0], dtype=torch.float32)

        scale = even_round(max_val, "fp4")

        assert torch.isfinite(scale[0]), "Zero block should have finite scale"
        assert torch.isfinite(scale[1]), "Non-zero block should have finite scale"
        assert scale[1] > 0, "Non-zero block should have positive scale"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not is_triton_available(), reason="Triton not available")
    def test_triton_kernel_with_zero_block(self):
        """
        Test the actual triton kernel with all-zero input.
        This should trigger the bug if it exists.
        """
        try:
            from quark.torch.kernel.mx.triton import qdq_mxfp4_triton
        except ImportError:
            pytest.skip("Could not import qdq_mxfp4_triton")

        # Create all-zero input on GPU
        x = torch.zeros(1, 32, dtype=torch.float16, device="cuda")

        try:
            result = qdq_mxfp4_triton(x, scale_calculation_mode="even")

            # Check for NaN in output (indicates the bug)
            if torch.isnan(result).any():
                pytest.fail(
                    "Bug #3 confirmed: qdq_mxfp4_triton produces NaN for all-zero input. "
                    "This is caused by log2(0) = -inf in scale computation."
                )

            # Check for inf in output
            if torch.isinf(result).any():
                pytest.fail(
                    "Bug #3 confirmed: qdq_mxfp4_triton produces inf for all-zero input. "
                    "This is caused by log2(0) = -inf in scale computation."
                )

            # For all-zero input, output should also be all zeros
            assert torch.allclose(result, torch.zeros_like(result)), "All-zero input should produce all-zero output"

        except Exception as e:
            if "log" in str(e).lower() or "inf" in str(e).lower() or "nan" in str(e).lower():
                pytest.fail(f"Bug #3 triggered: {e}")
            raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not is_triton_available(), reason="Triton not available")
    def test_triton_kernel_with_mixed_zeros(self):
        """
        Test the triton kernel with mixed zero and non-zero blocks.
        """
        try:
            from quark.torch.kernel.mx.triton import qdq_mxfp4_triton
        except ImportError:
            pytest.skip("Could not import qdq_mxfp4_triton")

        # Create input: alternating zero and non-zero blocks
        x = torch.zeros(4, 32, dtype=torch.float16, device="cuda")
        x[1, :] = torch.randn(32, dtype=torch.float16, device="cuda")
        x[3, :] = torch.randn(32, dtype=torch.float16, device="cuda")

        try:
            result = qdq_mxfp4_triton(x, scale_calculation_mode="even")

            # Check for NaN in output
            nan_mask = torch.isnan(result)
            if nan_mask.any():
                nan_rows = nan_mask.any(dim=1)
                pytest.fail(
                    f"Bug #3 confirmed: NaN produced in rows {nan_rows.nonzero().squeeze().tolist()}. "
                    "Zero blocks produce NaN due to log2(0) = -inf."
                )

            # Zero input blocks should produce zero output
            assert torch.allclose(result[0], torch.zeros_like(result[0])), "Zero input block should produce zero output"
            assert torch.allclose(result[2], torch.zeros_like(result[2])), "Zero input block should produce zero output"

        except Exception as e:
            if "log" in str(e).lower() or "inf" in str(e).lower() or "nan" in str(e).lower():
                pytest.fail(f"Bug #3 triggered: {e}")
            raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not is_triton_available(), reason="Triton not available")
    def test_triton_downcast_upcast_with_zeros(self):
        """
        Test the downcast/upcast functions directly with zero input.
        """
        try:
            from quark.torch.kernel.mx.triton import downcast_to_mxfp, upcast_from_mxfp
        except ImportError:
            pytest.skip("Could not import downcast/upcast functions")

        # All-zero input
        x = torch.zeros(1, 64, dtype=torch.float16, device="cuda")

        try:
            # Downcast to MXFP4
            quantized, scale, _ = downcast_to_mxfp(
                x,
                out_quant_type=torch.uint8,
                axis=-1,
            )

            # Check scale is not NaN or inf
            if torch.isnan(scale).any():
                pytest.fail("Bug #3 confirmed: downcast produces NaN scale for zero input")
            if torch.isinf(scale).any():
                pytest.fail("Bug #3 confirmed: downcast produces inf scale for zero input")

            # Upcast back
            dequantized = upcast_from_mxfp(
                quantized,
                scale,
                dtype=torch.float16,
                axis=-1,
            )

            # Check output is not NaN
            if torch.isnan(dequantized).any():
                pytest.fail("Bug #3 confirmed: upcast produces NaN for zero input")

        except Exception as e:
            if "log" in str(e).lower() or "inf" in str(e).lower() or "nan" in str(e).lower():
                pytest.fail(f"Bug #3 triggered: {e}")
            raise


class TestBug3EdgeCases:
    """Additional edge cases related to log2(0) bug"""

    def test_very_small_values_near_zero(self):
        """
        Test with very small max values that might underflow.
        """
        max_val = torch.tensor([1e-40, 1e-38, 1e-35], dtype=torch.float32)

        scale = even_round(max_val, "fp4")

        assert torch.isfinite(scale).all(), "Scale should be finite for very small values"

    def test_single_nonzero_max(self):
        """
        Test single non-zero max value (edge case).
        """
        max_val = torch.tensor([1.0], dtype=torch.float32)

        scale = even_round(max_val, "fp4")

        assert torch.isfinite(scale).all(), "Scale should be finite"
        assert scale.item() > 0, "Scale should be positive"


class TestBug3ProposedFix:
    """Test the proposed fix for the log2(0) bug"""

    def test_eps_protection_prevents_nan(self):
        """
        Demonstrate that adding epsilon protection prevents NaN/inf.

        The fix is to change:
            scale_e8m0_unbiased = tl.log2(max_val).floor() - _get_max_quant_exp(mx_tensor_dtype)
        To:
            eps = tl.where(max_val == 0.0, 2**(-126), 0.0)
            scale_e8m0_unbiased = tl.log2(max_val + eps).floor() - _get_max_quant_exp(mx_tensor_dtype)
        """
        # Simulate the buggy behavior
        max_val_zero = torch.tensor(0.0)
        buggy_result = torch.log2(max_val_zero)  # -inf

        assert torch.isinf(buggy_result), "Buggy version produces -inf"

        # Simulate the fixed behavior
        eps = torch.tensor(2 ** (-126))
        fixed_result = torch.log2(max_val_zero + eps)  # finite value

        assert torch.isfinite(fixed_result), "Fixed version produces finite value"
        assert fixed_result == -126.0, "log2(2^-126) should be -126"
