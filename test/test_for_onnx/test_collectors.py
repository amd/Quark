#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
from unittest.mock import patch

import numpy as np

from quark.onnx.calibration.calibrators import LayerWisePercentileCalibrater
from quark.onnx.calibration.collectors import (
    OverridedHistogramCollector,
    compute_lwp_metric_from_histogram,
    compute_lwp_scale_zp,
)
from quark.onnx.quantization.quant_utils import ExtendedQuantType, compute_minmse


class TestOverridedHistogramCollector(unittest.TestCase):
    def test_collect_absolute_value_uses_last_sample_for_layerwise_percentile(self) -> None:
        collector = OverridedHistogramCollector(
            method="percentile",
            symmetric=True,
            num_bins=4,
            num_quantized_bins=128,
            percentile=99.9,
            layer_wise=True,
        )
        last_sample = np.array([-1.0, 2.0], dtype=np.float32)

        collector.collect_absolute_value(
            {
                "tensor": [
                    np.array([-100.0, 100.0], dtype=np.float32),
                    last_sample,
                ]
            }
        )

        hist, hist_edges, min_value, max_value = collector.histogram_dict["tensor"]
        expected_hist, expected_edges = np.histogram(np.absolute(last_sample.flatten()), bins=4)

        np.testing.assert_array_equal(hist, expected_hist)
        np.testing.assert_allclose(hist_edges, expected_edges.astype(np.float32))
        self.assertEqual(min_value, np.float32(-1.0))
        self.assertEqual(max_value, np.float32(2.0))

    def test_collect_value_uses_last_sample_for_layerwise_percentile(self) -> None:
        collector = OverridedHistogramCollector(
            method="percentile",
            symmetric=False,
            num_bins=4,
            num_quantized_bins=128,
            percentile=99.9,
            layer_wise=True,
        )
        last_sample = np.array([-1.0, 3.0], dtype=np.float32)

        collector.collect_value(
            {
                "tensor": [
                    np.array([-100.0, 100.0], dtype=np.float32),
                    last_sample,
                ]
            }
        )

        hist, hist_edges, min_value, max_value, threshold = collector.histogram_dict["tensor"]
        expected_threshold = np.array(3.0, dtype=np.float32)
        expected_hist, expected_edges = np.histogram(last_sample.flatten(), bins=4, range=(-3.0, 3.0))

        np.testing.assert_array_equal(hist, expected_hist)
        np.testing.assert_allclose(hist_edges, expected_edges)
        self.assertEqual(min_value, np.float32(-1.0))
        self.assertEqual(max_value, np.float32(3.0))
        np.testing.assert_array_equal(threshold, expected_threshold)

    def test_collect_value_converts_float16_for_histogram(self) -> None:
        collector = OverridedHistogramCollector(
            method="percentile",
            symmetric=False,
            num_bins=4,
            num_quantized_bins=128,
            percentile=99.9,
        )
        data = np.array([-1.0, 0.5, 2.0], dtype=np.float16)
        original_histogram = np.histogram
        observed: dict[str, np.dtype] = {}

        def capture_histogram(
            histogram_data: np.ndarray, *args: object, **kwargs: object
        ) -> tuple[np.ndarray, np.ndarray]:
            observed["dtype"] = histogram_data.dtype
            return original_histogram(histogram_data, *args, **kwargs)

        with patch("quark.onnx.calibration.collectors.np.histogram", side_effect=capture_histogram):
            collector.collect_value({"tensor": [data]})

        hist, hist_edges, min_value, max_value, threshold = collector.histogram_dict["tensor"]
        expected_threshold = np.array(2.0, dtype=np.float16)
        expected_hist, expected_edges = np.histogram(
            data.astype(np.float32).flatten(),
            bins=4,
            range=(-expected_threshold, expected_threshold),
        )

        self.assertEqual(observed["dtype"], np.dtype(np.float32))
        np.testing.assert_array_equal(hist, expected_hist)
        np.testing.assert_allclose(hist_edges, expected_edges)
        self.assertEqual(min_value, np.float16(-1.0))
        self.assertEqual(max_value, np.float16(2.0))
        np.testing.assert_array_equal(threshold, expected_threshold)

    def test_collect_value_uses_zero_range_for_empty_tensor(self) -> None:
        collector = OverridedHistogramCollector(
            method="percentile",
            symmetric=False,
            num_bins=4,
            num_quantized_bins=128,
            percentile=99.9,
        )
        data = np.array([], dtype=np.float32)

        collector.collect_value({"tensor": [data]})

        hist, hist_edges, min_value, max_value, threshold = collector.histogram_dict["tensor"]
        expected_zero = np.array(0, dtype=np.float32)
        expected_hist, expected_edges = np.histogram(
            data.flatten(),
            bins=4,
            range=(-expected_zero, expected_zero),
        )

        np.testing.assert_array_equal(hist, expected_hist)
        np.testing.assert_allclose(hist_edges, expected_edges)
        np.testing.assert_array_equal(min_value, expected_zero)
        np.testing.assert_array_equal(max_value, expected_zero)
        np.testing.assert_array_equal(threshold, expected_zero)

    def test_collect_value_empty_tensor_uses_original_dtype_for_zero_min_max(self) -> None:
        collector = OverridedHistogramCollector(
            method="percentile",
            symmetric=False,
            num_bins=4,
            num_quantized_bins=128,
            percentile=99.9,
        )
        data = np.array([], dtype=np.float16)

        collector.collect_value({"tensor": [data]})

        _, _, min_value, max_value, threshold = collector.histogram_dict["tensor"]
        expected_zero = np.array(0, dtype=np.float16)

        np.testing.assert_array_equal(min_value, expected_zero)
        np.testing.assert_array_equal(max_value, expected_zero)
        np.testing.assert_array_equal(threshold, expected_zero)
        self.assertEqual(min_value.dtype, np.dtype(np.float16))
        self.assertEqual(max_value.dtype, np.dtype(np.float16))

    def test_compute_minmse_caps_histogram_bins_at_2048(self) -> None:
        data = np.arange(2049, dtype=np.float32)
        observed: dict[str, int] = {}

        class HistogramCalled(Exception):
            pass

        def capture_histogram(histogram_data: np.ndarray, bins: int, *args: object, **kwargs: object) -> None:
            observed["bins"] = bins
            raise HistogramCalled

        with (
            patch("quark.onnx.quantization.quant_utils.np.histogram", side_effect=capture_histogram),
            self.assertRaises(HistogramCalled),
        ):
            compute_minmse(data, ExtendedQuantType.QInt8.tensor_type, minmse_mode="HistCenter")

        self.assertEqual(observed["bins"], 2048)

    def test_compute_lwp_scale_zp_range_spanning_zero(self) -> None:
        # Well-conditioned candidate (rmin < 0 < rmax): the 0-clamp inside
        # compute_scale_zp is a no-op, so scale/zp equal the plain affine formula.
        scale, zp = compute_lwp_scale_zp(-2.0, 2.0, -128, 127)
        self.assertAlmostEqual(scale, 4.0 / 255, places=6)
        self.assertEqual(zp, 0)
        scale, zp = compute_lwp_scale_zp(-1.0, 3.0, -128, 127)
        self.assertAlmostEqual(scale, 4.0 / 255, places=6)
        # LWP convention q = round(x/s) - zp, so zp = -round(qmin - rmin/scale) = 64.
        self.assertEqual(zp, 64)

    def test_compute_lwp_scale_zp_all_positive_range_clamps_to_zero(self) -> None:
        # All-positive candidate: compute_scale_zp folds 0 into the range so the
        # zero-point stays inside [q_min, q_max]. The plain formula would give a
        # zp far outside the quantized range, mispredicting the real quant error.
        scale, zp = compute_lwp_scale_zp(0.5, 3.0, -128, 127)
        self.assertAlmostEqual(scale, 3.0 / 255, places=6)  # range clamped to [0, 3]
        self.assertEqual(zp, 128)

    def test_compute_lwp_scale_zp_degenerate_range(self) -> None:
        # Near-constant tensor (rmax == rmin): after clamping to [0, 1] the scale
        # is well-defined; it must not blow up to a huge zp like the old 1e-6 floor.
        scale, zp = compute_lwp_scale_zp(1.0, 1.0, -128, 127)
        self.assertAlmostEqual(scale, 1.0 / 255, places=6)  # range clamped to [0, 1]
        self.assertEqual(zp, 128)

    @staticmethod
    def _reference_metric(hist, hist_edges, rmin, rmax, q_min, q_max, metric):
        """Independent re-implementation used to validate the function under test.

        Deliberately does NOT call ``compute_lwp_scale_zp`` (the helper the function
        under test uses); it re-derives scale/zp inline from the ``r = s(q - z)``
        convention so the assertion checks real correctness, not self-consistency.
        """
        hist_f64 = np.asarray(hist, dtype=np.float64)
        total = hist_f64.sum()
        bin_centres = (np.asarray(hist_edges[:-1], dtype=np.float64) + np.asarray(hist_edges[1:], dtype=np.float64)) / 2
        # Same affine derivation compute_scale_zp performs: clamp range to include 0,
        # then scale = (rmax - rmin) / (qmax - qmin), z = round(qmin - rmin/scale).
        # Mirror compute_scale_zp exactly: clamp to include 0, compute dr/dq in
        # float64, then cast scale to float32 (the dtype of rmin/rmax) before deriving
        # the zero-point. Matching this precision path keeps the comparison exact.
        rmn = np.minimum(np.float32(rmin), np.float32(0))
        rmx = np.maximum(np.float32(rmax), np.float32(0))
        dr = np.array(rmx - rmn, dtype=np.float64)
        dq = np.float64(q_max) - np.float64(q_min)
        scale = np.array(dr / dq)
        if scale < np.finfo(np.float32).tiny:
            scale = 1.0
            z = 0
        else:
            z = int(np.round(q_min - rmn / scale))
            scale = float(scale.astype(np.float32))
        zp = -z  # LWP quant/dequant uses q = round(x/s) - zp, i.e. zp = -z
        q = np.clip(np.round(bin_centres / scale - zp), q_min, q_max)
        dq = (q + zp) * scale
        diff = bin_centres - dq
        if metric == "mse":
            return float(np.sum(hist_f64 * diff * diff) / total)
        return float(np.sum(hist_f64 * np.abs(diff)) / total)

    def test_compute_lwp_metric_from_histogram_empty_returns_zeros(self) -> None:
        # total == 0 edge case: no samples fell into the histogram, so every
        # candidate must score 0.0 and argmin picks index 0 (the first candidate).
        hist = np.zeros(4, dtype=np.int64)
        hist_edges = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        candidate_ranges = [(-2.0, 2.0), (-1.5, 1.5), (-1.0, 1.0)]
        metrics = compute_lwp_metric_from_histogram(
            (hist, hist_edges, np.float32(-2.0), np.float32(2.0)), candidate_ranges, -128, 127, "mae"
        )
        self.assertEqual(len(metrics), len(candidate_ranges))
        self.assertTrue(np.all(metrics == 0.0))
        self.assertEqual(int(np.argmin(metrics)), 0)

    def test_compute_lwp_metric_symmetric_mae_matches_reference(self) -> None:
        # Symmetric (absolute-value) histogram, mae: validates bin-centre
        # computation and count-weighting against an independent reference.
        hist = np.array([10, 40, 30, 20], dtype=np.int64)
        hist_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        candidate_ranges = [(-2.0, 2.0), (-1.5, 1.5), (-1.0, 1.0)]
        metrics = compute_lwp_metric_from_histogram(
            (hist, hist_edges, np.float32(0.0), np.float32(2.0)), candidate_ranges, -128, 127, "mae"
        )
        for i, (rmin, rmax) in enumerate(candidate_ranges):
            expected = self._reference_metric(hist, hist_edges, rmin, rmax, -128, 127, "mae")
            self.assertAlmostEqual(metrics[i], expected, places=10)

    def test_compute_lwp_metric_mse_differs_from_mae(self) -> None:
        # mse vs mae branch: both must match their own reference and, for a
        # non-degenerate histogram, produce different numbers.
        hist = np.array([5, 15, 25, 35], dtype=np.int64)
        hist_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        candidate_ranges = [(-2.0, 2.0), (-1.0, 1.0)]
        mae = compute_lwp_metric_from_histogram(
            (hist, hist_edges, np.float32(0.0), np.float32(2.0)), candidate_ranges, -128, 127, "mae"
        )
        mse = compute_lwp_metric_from_histogram(
            (hist, hist_edges, np.float32(0.0), np.float32(2.0)), candidate_ranges, -128, 127, "mse"
        )
        for i, (rmin, rmax) in enumerate(candidate_ranges):
            self.assertAlmostEqual(
                mae[i], self._reference_metric(hist, hist_edges, rmin, rmax, -128, 127, "mae"), places=10
            )
            self.assertAlmostEqual(
                mse[i], self._reference_metric(hist, hist_edges, rmin, rmax, -128, 127, "mse"), places=10
            )
        self.assertFalse(np.allclose(mae, mse))

    def test_compute_lwp_metric_asymmetric_signed_histogram(self) -> None:
        # Asymmetric case: a signed histogram spanning negative and positive
        # bins is handled by the same bin-centre path.
        hist = np.array([20, 10, 10, 20], dtype=np.int64)
        hist_edges = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        candidate_ranges = [(-2.0, 2.0), (-1.0, 1.5)]
        metrics = compute_lwp_metric_from_histogram(
            (hist, hist_edges, np.float32(-2.0), np.float32(2.0)), candidate_ranges, -128, 127, "mae"
        )
        for i, (rmin, rmax) in enumerate(candidate_ranges):
            expected = self._reference_metric(hist, hist_edges, rmin, rmax, -128, 127, "mae")
            self.assertAlmostEqual(metrics[i], expected, places=10)

    def test_compute_lwp_metric_unknown_metric_raises(self) -> None:
        hist = np.array([1, 2, 3, 4], dtype=np.int64)
        hist_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_lwp_metric_from_histogram(
                (hist, hist_edges, np.float32(0.0), np.float32(2.0)), [(-2.0, 2.0)], -128, 127, "MAE"
            )


class TestCalOneLayerMetric(unittest.TestCase):
    """Direct tests for the raw-data scoring path LayerWisePercentileCalibrater.cal_one_layer_metric.

    Mirrors the histogram-path metric tests above so both selection paths have the
    same unknown-metric guard covered.
    """

    @staticmethod
    def _bare_calibrater(lwp_metric: str) -> LayerWisePercentileCalibrater:
        # Build a bare instance without the heavy __init__ (model / data reader):
        # cal_one_layer_metric only reads lwp_metric and q_min/q_max.
        cal = object.__new__(LayerWisePercentileCalibrater)
        cal.lwp_metric = lwp_metric
        cal.q_min, cal.q_max = -128, 127
        return cal

    def test_cal_one_layer_metric_mse_differs_from_mae(self) -> None:
        # Coarse scale so quantization error is large enough for mse and mae to
        # be clearly distinct (small scales drive both toward ~0 and the check
        # becomes flaky).
        x = np.array([0.1, 0.9, -0.7, 1.3], dtype=np.float32)
        mse = self._bare_calibrater("mse").cal_one_layer_metric(x, 1.0, 0)
        mae = self._bare_calibrater("mae").cal_one_layer_metric(x, 1.0, 0)
        self.assertGreater(mse, 0.0)
        self.assertGreater(mae, 0.0)
        self.assertNotAlmostEqual(mse, mae)

    def test_cal_one_layer_metric_unknown_metric_raises(self) -> None:
        x = np.array([0.1, 0.5, -0.3, 0.8], dtype=np.float32)
        with self.assertRaises(ValueError):
            self._bare_calibrater("MAE").cal_one_layer_metric(x, 0.01, 0)


if __name__ == "__main__":
    unittest.main()
