#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for ``check_quantization_preference_arguments`` in input_check.py.

Each test class targets one preference value (or the None / invalid path) and
drives every conditional branch inside that preference block.
"""

import multiprocessing
import unittest

from onnxruntime.quantization.calibrate import CalibrationMethod

from quark.onnx.calibration import LayerWiseMethod, PowerOfTwoMethod
from quark.onnx.quantization.input_check import check_quantization_preference_arguments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opts(**kwargs):
    """Return an extra_options dict with QuantizationPreference set."""
    return dict(kwargs)


# ---------------------------------------------------------------------------
# No preference (None) — early return, nothing mutated
# ---------------------------------------------------------------------------


class TestPreferenceNone(unittest.TestCase):
    def test_returns_without_mutation(self):
        opts = {}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)
        self.assertEqual(opts, {})

    def test_returns_without_mutation_with_fast_ft(self):
        opts = {}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        # FastFinetune dict should NOT be created when preference is None
        self.assertNotIn("FastFinetune", opts)


# ---------------------------------------------------------------------------
# Invalid preference value
# ---------------------------------------------------------------------------


class TestPreferenceInvalid(unittest.TestCase):
    def test_raises_for_unknown_value(self):
        opts = {"QuantizationPreference": "turbo"}
        with self.assertRaises(ValueError):
            check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)

    def test_raises_for_empty_string(self):
        opts = {"QuantizationPreference": ""}
        with self.assertRaises(ValueError):
            check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)


# ---------------------------------------------------------------------------
# "accuracy" preference
# ---------------------------------------------------------------------------


class TestPreferenceAccuracy(unittest.TestCase):
    # --- MinMSEModePof2Scale ---

    def test_pof2_minmse_sets_minmse_mode_when_not_all(self):
        opts = {"QuantizationPreference": "accuracy", "MinMSEModePof2Scale": "MostCommon"}
        check_quantization_preference_arguments(PowerOfTwoMethod.MinMSE, False, opts)
        self.assertEqual(opts["MinMSEModePof2Scale"], "All")

    def test_pof2_minmse_leaves_minmse_mode_when_already_all(self):
        opts = {"QuantizationPreference": "accuracy", "MinMSEModePof2Scale": "All"}
        check_quantization_preference_arguments(PowerOfTwoMethod.MinMSE, False, opts)
        self.assertEqual(opts["MinMSEModePof2Scale"], "All")

    def test_non_pof2_method_does_not_set_minmse_mode(self):
        opts = {"QuantizationPreference": "accuracy"}
        check_quantization_preference_arguments(CalibrationMethod.Percentile, False, opts)
        self.assertNotIn("MinMSEModePof2Scale", opts)

    # --- FastFinetune not enabled ---

    def test_no_fast_ft_dict_created_when_include_fast_ft_false(self):
        opts = {"QuantizationPreference": "accuracy"}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)
        self.assertNotIn("FastFinetune", opts)

    # --- FastFinetune adjustments ---

    def test_fast_ft_earlystop_disabled_when_true(self):
        opts = {"QuantizationPreference": "accuracy", "FastFinetune": {"EarlyStop": True}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertFalse(opts["FastFinetune"]["EarlyStop"])

    def test_fast_ft_earlystop_not_touched_when_already_false(self):
        opts = {"QuantizationPreference": "accuracy", "FastFinetune": {"EarlyStop": False}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertFalse(opts["FastFinetune"]["EarlyStop"])

    def test_fast_ft_update_bias_enabled_when_false(self):
        opts = {"QuantizationPreference": "accuracy", "FastFinetune": {"UpdateBias": False}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertTrue(opts["FastFinetune"]["UpdateBias"])

    def test_fast_ft_update_bias_not_touched_when_already_true(self):
        opts = {"QuantizationPreference": "accuracy", "FastFinetune": {"UpdateBias": True}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertTrue(opts["FastFinetune"]["UpdateBias"])

    def test_fast_ft_output_qdq_enabled_when_false(self):
        opts = {"QuantizationPreference": "accuracy", "FastFinetune": {"OutputQDQ": False}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertTrue(opts["FastFinetune"]["OutputQDQ"])

    def test_fast_ft_output_qdq_not_touched_when_already_true(self):
        opts = {"QuantizationPreference": "accuracy", "FastFinetune": {"OutputQDQ": True}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertTrue(opts["FastFinetune"]["OutputQDQ"])

    def test_fast_ft_dict_created_when_absent_and_include_fast_ft_true(self):
        opts = {"QuantizationPreference": "accuracy"}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertIn("FastFinetune", opts)
        self.assertTrue(opts["FastFinetune"]["UpdateBias"])
        self.assertTrue(opts["FastFinetune"]["OutputQDQ"])


# ---------------------------------------------------------------------------
# "speed" preference
# ---------------------------------------------------------------------------


class TestPreferenceSpeed(unittest.TestCase):
    # --- NumBins for PowerOfTwoMethod.MinMSE ---

    def test_pof2_minmse_sets_num_bins_when_absent(self):
        opts = {"QuantizationPreference": "speed"}
        check_quantization_preference_arguments(PowerOfTwoMethod.MinMSE, False, opts)
        self.assertEqual(opts["NumBins"], 2048)

    def test_pof2_minmse_sets_num_bins_when_zero(self):
        opts = {"QuantizationPreference": "speed", "NumBins": 0}
        check_quantization_preference_arguments(PowerOfTwoMethod.MinMSE, False, opts)
        self.assertEqual(opts["NumBins"], 2048)

    def test_pof2_minmse_does_not_overwrite_existing_num_bins(self):
        opts = {"QuantizationPreference": "speed", "NumBins": 512}
        check_quantization_preference_arguments(PowerOfTwoMethod.MinMSE, False, opts)
        self.assertEqual(opts["NumBins"], 512)

    def test_pof2_minmse_skips_num_bins_when_mode_not_all(self):
        opts = {"QuantizationPreference": "speed", "MinMSEModePof2Scale": "MostCommon"}
        check_quantization_preference_arguments(PowerOfTwoMethod.MinMSE, False, opts)
        self.assertNotIn("NumBins", opts)

    # --- LWPUseHistogram for LayerWisePercentile ---

    def test_lwp_sets_use_histogram_when_false(self):
        opts = {"QuantizationPreference": "speed", "LWPUseHistogram": False}
        check_quantization_preference_arguments(LayerWiseMethod.LayerWisePercentile, False, opts)
        self.assertTrue(opts["LWPUseHistogram"])

    def test_lwp_does_not_touch_use_histogram_when_already_true(self):
        opts = {"QuantizationPreference": "speed", "LWPUseHistogram": True}
        check_quantization_preference_arguments(LayerWiseMethod.LayerWisePercentile, False, opts)
        self.assertTrue(opts["LWPUseHistogram"])

    # --- CalibOptimizeMem ---

    def test_calib_optimize_mem_set_false_when_absent(self):
        opts = {"QuantizationPreference": "speed"}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)
        self.assertFalse(opts["CalibOptimizeMem"])

    def test_calib_optimize_mem_set_false_when_true(self):
        opts = {"QuantizationPreference": "speed", "CalibOptimizeMem": True}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)
        self.assertFalse(opts["CalibOptimizeMem"])

    # --- CalibWorkerNum ---

    def test_calib_worker_num_set_when_absent(self):
        opts = {"QuantizationPreference": "speed"}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)
        self.assertEqual(opts["CalibWorkerNum"], min(multiprocessing.cpu_count(), 8))

    def test_calib_worker_num_not_overwritten_when_already_one(self):
        # NOTE: the current guard is `"key" not in opts OR value > 1`, so an
        # explicit CalibWorkerNum=1 is NOT overwritten (1 is not > 1).
        # This is a known limitation documented in the code review.
        opts = {"QuantizationPreference": "speed", "CalibWorkerNum": 1}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)
        self.assertEqual(opts["CalibWorkerNum"], 1)

    # --- FastFinetune adjustments ---

    def test_fast_ft_earlystop_enabled_when_false(self):
        opts = {"QuantizationPreference": "speed", "FastFinetune": {"EarlyStop": False}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertTrue(opts["FastFinetune"]["EarlyStop"])

    def test_fast_ft_earlystop_not_touched_when_already_true(self):
        opts = {"QuantizationPreference": "speed", "FastFinetune": {"EarlyStop": True}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertTrue(opts["FastFinetune"]["EarlyStop"])

    def test_fast_ft_mem_opt_level_set_zero_when_nonzero(self):
        opts = {"QuantizationPreference": "speed", "FastFinetune": {"MemOptLevel": 1}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertEqual(opts["FastFinetune"]["MemOptLevel"], 0)

    def test_fast_ft_mem_opt_level_not_touched_when_already_zero(self):
        opts = {"QuantizationPreference": "speed", "FastFinetune": {"MemOptLevel": 0}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertEqual(opts["FastFinetune"]["MemOptLevel"], 0)

    def test_fast_ft_num_workers_set_when_one(self):
        opts = {"QuantizationPreference": "speed", "FastFinetune": {"NumWorkers": 1}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertEqual(opts["FastFinetune"]["NumWorkers"], min(multiprocessing.cpu_count(), 8))

    def test_fast_ft_num_workers_not_touched_when_already_greater_than_one(self):
        opts = {"QuantizationPreference": "speed", "FastFinetune": {"NumWorkers": 4}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertEqual(opts["FastFinetune"]["NumWorkers"], 4)

    def test_no_fast_ft_dict_created_when_include_fast_ft_false(self):
        opts = {"QuantizationPreference": "speed"}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)
        self.assertNotIn("FastFinetune", opts)

    def test_fast_ft_dict_created_when_absent_and_include_fast_ft_true(self):
        opts = {"QuantizationPreference": "speed"}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertIn("FastFinetune", opts)
        self.assertTrue(opts["FastFinetune"]["EarlyStop"])
        self.assertEqual(opts["FastFinetune"]["MemOptLevel"], 0)


# ---------------------------------------------------------------------------
# "resource_efficiency" preference
# ---------------------------------------------------------------------------


class TestPreferenceResourceEfficiency(unittest.TestCase):
    # --- NumBins for PowerOfTwoMethod.MinMSE ---

    def test_pof2_minmse_sets_num_bins_when_absent(self):
        opts = {"QuantizationPreference": "resource_efficiency"}
        check_quantization_preference_arguments(PowerOfTwoMethod.MinMSE, False, opts)
        self.assertEqual(opts["NumBins"], 2048)

    def test_pof2_minmse_sets_num_bins_when_zero(self):
        opts = {"QuantizationPreference": "resource_efficiency", "NumBins": 0}
        check_quantization_preference_arguments(PowerOfTwoMethod.MinMSE, False, opts)
        self.assertEqual(opts["NumBins"], 2048)

    def test_pof2_minmse_does_not_overwrite_existing_num_bins(self):
        opts = {"QuantizationPreference": "resource_efficiency", "NumBins": 1024}
        check_quantization_preference_arguments(PowerOfTwoMethod.MinMSE, False, opts)
        self.assertEqual(opts["NumBins"], 1024)

    def test_pof2_minmse_skips_num_bins_when_mode_not_all(self):
        opts = {"QuantizationPreference": "resource_efficiency", "MinMSEModePof2Scale": "MostCommon"}
        check_quantization_preference_arguments(PowerOfTwoMethod.MinMSE, False, opts)
        self.assertNotIn("NumBins", opts)

    # --- LWPUseHistogram for LayerWisePercentile ---

    def test_lwp_sets_use_histogram_when_false(self):
        opts = {"QuantizationPreference": "resource_efficiency", "LWPUseHistogram": False}
        check_quantization_preference_arguments(LayerWiseMethod.LayerWisePercentile, False, opts)
        self.assertTrue(opts["LWPUseHistogram"])

    def test_lwp_does_not_touch_use_histogram_when_already_true(self):
        opts = {"QuantizationPreference": "resource_efficiency", "LWPUseHistogram": True}
        check_quantization_preference_arguments(LayerWiseMethod.LayerWisePercentile, False, opts)
        self.assertTrue(opts["LWPUseHistogram"])

    # --- CalibOptimizeMem ---

    def test_calib_optimize_mem_set_true_when_absent(self):
        opts = {"QuantizationPreference": "resource_efficiency"}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)
        self.assertTrue(opts["CalibOptimizeMem"])

    def test_calib_optimize_mem_set_true_when_false(self):
        opts = {"QuantizationPreference": "resource_efficiency", "CalibOptimizeMem": False}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)
        self.assertTrue(opts["CalibOptimizeMem"])

    # --- CalibWorkerNum ---

    def test_calib_worker_num_set_one_when_absent(self):
        opts = {"QuantizationPreference": "resource_efficiency"}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)
        self.assertEqual(opts["CalibWorkerNum"], 1)

    def test_calib_worker_num_set_one_when_greater_than_one(self):
        opts = {"QuantizationPreference": "resource_efficiency", "CalibWorkerNum": 4}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)
        self.assertEqual(opts["CalibWorkerNum"], 1)

    # --- FastFinetune adjustments ---

    def test_fast_ft_mem_opt_level_set_two_when_not_two(self):
        opts = {"QuantizationPreference": "resource_efficiency", "FastFinetune": {"MemOptLevel": 1}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertEqual(opts["FastFinetune"]["MemOptLevel"], 2)

    def test_fast_ft_mem_opt_level_not_touched_when_already_two(self):
        opts = {"QuantizationPreference": "resource_efficiency", "FastFinetune": {"MemOptLevel": 2}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertEqual(opts["FastFinetune"]["MemOptLevel"], 2)

    def test_fast_ft_num_workers_set_one_when_greater_than_one(self):
        opts = {"QuantizationPreference": "resource_efficiency", "FastFinetune": {"NumWorkers": 4}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertEqual(opts["FastFinetune"]["NumWorkers"], 1)

    def test_fast_ft_num_workers_not_touched_when_already_one(self):
        opts = {"QuantizationPreference": "resource_efficiency", "FastFinetune": {"NumWorkers": 1}}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertEqual(opts["FastFinetune"]["NumWorkers"], 1)

    def test_no_fast_ft_dict_created_when_include_fast_ft_false(self):
        opts = {"QuantizationPreference": "resource_efficiency"}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, False, opts)
        self.assertNotIn("FastFinetune", opts)

    def test_fast_ft_dict_created_when_absent_and_include_fast_ft_true(self):
        opts = {"QuantizationPreference": "resource_efficiency"}
        check_quantization_preference_arguments(CalibrationMethod.MinMax, True, opts)
        self.assertIn("FastFinetune", opts)
        self.assertEqual(opts["FastFinetune"]["MemOptLevel"], 2)


if __name__ == "__main__":
    unittest.main()
