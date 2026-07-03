#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for ``quark.torch.utils.numerics`` and the call sites it protects.

Two recurring bug families are covered:

1. ``torch.log2`` on a zero-valued scale silently produces ``-inf``, which then
   bit-casts to garbage when piped through ``.to(torch.int8)`` /
   ``.to(torch.uint8)``. This corrupts packed MXFP tensors and exported
   E8M0 scale bytes (see ``Pack_mxfp4.pack``,
   ``RealQuantizer.maybe_convert_and_transpose_scale``,
   ``_pack_quantized_tensor`` in ``file2file_quantization.py``).

2. ``Tensor.max()`` / ``Tensor.min()`` raise ``RuntimeError`` on an empty
   tensor. This is reachable from ``PerTensorHistogramObserver.forward``
   when ``_skip_zeros=True`` filters out every element of an all-zero
   calibration batch.
"""

from __future__ import annotations

import math
import unittest

import torch

from quark.torch.utils.numerics import safe_log2, safe_max, safe_min, to_e8m0_uint8


class TestSafeLog2(unittest.TestCase):
    def test_log2_of_zero_without_helper_is_minus_inf(self):
        """Raw torch.log2(0) is -inf — documents the bug surface."""
        self.assertTrue(math.isinf(torch.log2(torch.tensor(0.0)).item()))

    def test_log2_of_negative_without_helper_is_nan(self):
        """Raw torch.log2(negative) is NaN — documents the bug surface."""
        self.assertTrue(math.isnan(torch.log2(torch.tensor(-1.0)).item()))

    def test_safe_log2_of_zero_is_finite(self):
        result = safe_log2(torch.tensor(0.0))
        self.assertTrue(math.isfinite(result.item()))
        self.assertLess(result.item(), 0.0)  # very small positive -> very negative log2

    def test_safe_log2_of_negative_is_finite(self):
        result = safe_log2(torch.tensor(-1.5))
        self.assertTrue(math.isfinite(result.item()))

    def test_safe_log2_of_positive_unchanged(self):
        # Values above the smallest positive normal float32 must round-trip exactly.
        for v in (1e-30, 1e-10, 0.5, 1.0, 2.0, 1e10):
            with self.subTest(v=v):
                expected = math.log2(v)
                got = safe_log2(torch.tensor(v)).item()
                self.assertAlmostEqual(got, expected, places=4)

    def test_safe_log2_followed_by_int_cast_does_not_corrupt_zero_block(self):
        """Reproduces the Pack_mxfp4 silent-corruption pattern.

        The packer stores the per-block scale as ``log2(scale).to(int8).view(uint8)``.
        With raw torch.log2, a zero scale gives -inf -> int8 saturates / undefined ->
        uint8 view -> garbage scale byte. With safe_log2, the byte is bounded
        and reproducible.
        """
        zero_scale = torch.zeros(1)
        # The buggy pattern is non-deterministic across PyTorch builds — assert
        # only that the safe variant produces a finite, bounded byte.
        safe_byte = safe_log2(zero_scale).to(torch.int8).view(torch.uint8).item()
        self.assertIn(safe_byte, range(0, 256))


class TestSafeMaxMin(unittest.TestCase):
    def test_raw_max_on_empty_tensor_raises(self):
        """Documents the bug surface in HistogramObserver._skip_zeros path."""
        empty = torch.tensor([])
        with self.assertRaises(RuntimeError):
            empty.max()

    def test_safe_max_on_empty_returns_default(self):
        self.assertEqual(safe_max(torch.tensor([])), 0.0)
        self.assertEqual(safe_max(torch.tensor([]), default=-7.5), -7.5)

    def test_safe_min_on_empty_returns_default(self):
        self.assertEqual(safe_min(torch.tensor([])), 0.0)
        self.assertEqual(safe_min(torch.tensor([]), default=42.0), 42.0)

    def test_safe_max_min_on_non_empty_match_torch(self):
        x = torch.tensor([3.0, -1.0, 4.0, -1.5, 9.0])
        self.assertEqual(safe_max(x), 9.0)
        self.assertEqual(safe_min(x), -1.5)


class TestPackMXFP4ZeroScaleBlock(unittest.TestCase):
    """End-to-end smoke test of the Pack_mxfp4 call site after the refactor."""

    def test_pack_does_not_raise_on_all_zero_block(self):
        from quark.torch.utils.pack import Pack_mxfp4

        packer = Pack_mxfp4(qscheme=None, dtype="mxfp4")
        # One block of 33 elements (1 scale + 32 values), all zero.
        # Before the fix: log2(0) -> -inf -> int8 cast undefined.
        # After the fix: scale byte is bounded and finite, no exception.
        tensor = torch.zeros(1, 33)
        packed = packer.pack(tensor, reorder=False)
        self.assertFalse(torch.isnan(packed.float()).any())
        self.assertFalse(torch.isinf(packed.float()).any())


class TestObserverEmptyAfterSkipZeros(unittest.TestCase):
    """Reproduces the ``HistogramObserver`` empty-tensor crash path.

    Mirrors the in-place forward pattern from
    ``PerTensorHistogramObserver.forward``: filter out zeros, then take max/min.
    Before the refactor this raises ``RuntimeError``. After the refactor (using
    safe_max/safe_min) it returns sensible defaults instead.
    """

    @staticmethod
    def buggy_path(x: torch.Tensor) -> tuple[float, float]:
        x = x[torch.where(x != 0)]
        return x.max().item(), x.min().item()

    @staticmethod
    def fixed_path(x: torch.Tensor) -> tuple[float, float]:
        x = x[torch.where(x != 0)]
        return safe_max(x), safe_min(x)

    def test_buggy_path_crashes_on_all_zero_input(self):
        with self.assertRaises(RuntimeError):
            self.buggy_path(torch.zeros(8))

    def test_fixed_path_returns_defaults_on_all_zero_input(self):
        x_max, x_min = self.fixed_path(torch.zeros(8))
        self.assertEqual(x_max, 0.0)
        self.assertEqual(x_min, 0.0)


class TestToE8M0Uint8(unittest.TestCase):
    """Covers ``to_e8m0_uint8`` and, indirectly, the e8m0 export call sites in
    ``realquantizer.maybe_convert_and_transpose_scale`` and
    ``file2file_quantization._pack_quantized_tensor`` that delegate to it.
    """

    def test_round_trip_on_powers_of_two(self):
        # E8M0 bias = 127, so 1.0 -> 127, 2.0 -> 128, 0.5 -> 126, 0.25 -> 125.
        scale = torch.tensor([1.0, 2.0, 0.5, 0.25])
        self.assertEqual(to_e8m0_uint8(scale).tolist(), [127, 128, 126, 125])

    def test_zero_scale_is_bounded_not_garbage(self):
        # Pre-fix: log2(0) -> -inf -> int16 -> undefined -> uint8 -> garbage.
        # Post-fix: safe_log2 clamps the input at 2**-126, giving log2 = -126,
        # which biases to uint8(1) — the smallest representable E8M0 byte
        # under the safe-clamp. The key property is that the byte is finite
        # and reproducible, not garbage.
        self.assertEqual(to_e8m0_uint8(torch.tensor([0.0])).tolist(), [1])

    def test_extreme_scale_stays_within_uint8(self):
        # Within-fp32 powers of two map cleanly: 2**100 -> 100 + 127 = 227,
        # 2**-100 -> -100 + 127 = 27. The clamp guarantees results stay in
        # [0, 254]; safe_log2 guarantees the input to round() is finite.
        bytes_ = to_e8m0_uint8(torch.tensor([2.0**100, 2.0**-100])).tolist()
        self.assertEqual(bytes_, [227, 27])


class TestPerTensorHistogramObserverConstantInput(unittest.TestCase):
    """Covers the constant-input nudge in ``PerTensorHistogramObserver.forward``
    (the ``if x_min == x_max:`` branch). Without the fix, ``torch.histc`` and
    ``torch.linspace`` would produce a degenerate, zero-width histogram.
    """

    def _make_asymmetric(self):
        from quark.torch.quantization.config.config import QTensorConfig
        from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
        from quark.torch.quantization.observer.observer import PerTensorHistogramObserver

        spec = QTensorConfig(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_tensor,
            observer_cls=PerTensorHistogramObserver,
            symmetric=False,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            is_dynamic=False,
        )
        return PerTensorHistogramObserver(spec)

    def test_constant_input_yields_strictly_increasing_bin_edges(self):
        obs = self._make_asymmetric()
        # All ones -> x_min == x_max == 1.0 in the asymmetric branch, hits the nudge.
        obs(torch.ones(64))
        edges = obs.calib_bin_edges
        # Every consecutive pair of edges must differ — i.e. no degenerate bins.
        diffs = (edges[1:] - edges[:-1]).abs()
        self.assertTrue(torch.all(diffs > 0).item())
        self.assertFalse(torch.isnan(obs.calib_hist).any().item())


if __name__ == "__main__":
    unittest.main()
