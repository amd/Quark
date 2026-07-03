#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
from unittest.mock import patch

import numpy as np

from quark.onnx.calibration.collectors import OverridedHistogramCollector
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


if __name__ == "__main__":
    unittest.main()
