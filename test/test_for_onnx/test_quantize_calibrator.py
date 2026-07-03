#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest

import numpy as np
import onnx
import onnxruntime
from onnx_testing_utils import prepare_model
from onnxruntime.quantization import CalibrationDataReader, QuantType

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import CalibrationMethod, Config, ModelQuantizer, PowerOfTwoMethod
from quark.onnx.quantization.config import get_default_config

input_tensor = np.array(
    [
        [
            [
                [0.26921557, 0.79500909, 0.6102178, 0.04375664],
                [0.06221361, 0.98258356, 0.38635129, 0.06492238],
                [0.49631707, 0.35442799, 0.51719146, 0.52100111],
                [0.04145599, 0.88960236, 0.50627326, 0.57204613],
            ],
            [
                [0.99185097, 0.93582153, 0.13174529, 0.42896287],
                [0.14552133, 0.02538564, 0.0732355, 0.25725371],
                [0.09856916, 0.43015628, 0.55679755, 0.66560074],
                [0.9439425, 0.45701841, 0.86791293, 0.64728276],
            ],
            [
                [0.29159685, 0.79021383, 0.3117182, 0.11342342],
                [0.16660495, 0.46426165, 0.31348552, 0.143383],
                [0.96454802, 0.63258874, 0.30295267, 0.96720039],
                [0.29879457, 0.79916527, 0.02905061, 0.20115725],
            ],
        ]
    ]
).astype(np.float32)

output_tensor_minmse_all = np.array(
    [
        [
            [
                [0.25390625, 0.16015625, 0.0234375, -0.046875],
                [0.1484375, -0.015625, 0.13671875, 0.0390625],
                [0.125, 0.2734375, 0.421875, 0.11328125],
                [0.1484375, 0.13671875, 0.234375, 0.28515625],
            ]
        ]
    ]
).astype(np.float32)

output_tensor_minmse_mostcommon = np.array(
    [
        [
            [
                [0.25390625, 0.16015625, 0.0234375, -0.046875],
                [0.1484375, -0.015625, 0.13671875, 0.0390625],
                [0.125, 0.2734375, 0.421875, 0.11328125],
                [0.1484375, 0.13671875, 0.234375, 0.28515625],
            ]
        ]
    ]
).astype(np.float32)

output_tensor_minmse_percentile = np.array(
    [
        [
            [
                [0.25390625, 0.16015625, 0.0234375, -0.046875],
                [0.1484375, -0.015625, 0.13671875, 0.0390625],
                [0.125, 0.2734375, 0.421875, 0.11328125],
                [0.1484375, 0.13671875, 0.234375, 0.28515625],
            ]
        ]
    ]
).astype(np.float32)

output_tensor_percentile_sym = np.array(
    [
        [
            [
                [0.24136764, 0.1602976, 0.02395251, -0.04790503],
                [0.14371508, -0.03500752, 0.13818759, 0.03685002],
                [0.12344757, 0.26900515, 0.41456276, 0.11423507],
                [0.1510851, 0.13818759, 0.23584014, 0.28558767],
            ]
        ]
    ]
).astype(np.float32)

output_tensor_percentile_asym = np.array(
    [
        [
            [
                [0.2412573, 0.1602243, 0.02394156, -0.04788313],
                [0.14364938, -0.03499152, 0.1381244, 0.03683317],
                [0.12339114, 0.2688822, 0.41437322, 0.11418284],
                [0.15101601, 0.1381244, 0.23573232, 0.2854571],
            ]
        ]
    ]
).astype(np.float32)

output_tensor_entropy = np.array(
    [
        [
            [
                [0.24146867, 0.16036469, 0.02396254, -0.04792508],
                [0.14377524, -0.03502217, 0.13824542, 0.03686544],
                [0.12349924, 0.26911774, 0.41473624, 0.11428288],
                [0.14930505, 0.13824542, 0.23593885, 0.2857072],
            ]
        ]
    ]
).astype(np.float32)

output_tensor_distribution = np.array(
    [
        [
            [
                [0.1883611, 0.16192444, 0.02313206, -0.04956871],
                [0.10905115, -0.07270077, 0.1354878, 0.03965497],
                [0.11566032, 0.26767102, 0.3734176, 0.11566032],
                [0.14209697, 0.13879238, 0.23792979, 0.26767102],
            ]
        ]
    ]
).astype(np.float32)


class DataReader(CalibrationDataReader):
    def __init__(self, input_tensor):
        self.data = [input_tensor]
        self.input_name = "input"
        self.index = 0

    def get_next(self):
        if self.index < len(self.data):
            input_dict = {self.input_name: self.data[self.index]}
            self.index += 1
            return input_dict
        return None

    def __len__(self):
        return len(self.data)

    def rewind(self):
        self.index = 0


def prepare_minmse_all_multiple_workers_config():
    quant_config = get_default_config("XINT8")
    quant_config.calibrate_method = PowerOfTwoMethod.MinMSE
    quant_config.extra_options["CalibOptimizeMem"] = True
    quant_config.extra_options["CalibWorkerNum"] = 1000
    return Config(global_quant_config=quant_config)


def prepare_minmse_all_config():
    quant_config = get_default_config("XINT8")
    quant_config.calibrate_method = PowerOfTwoMethod.MinMSE
    quant_config.extra_options["MinMSEModePof2Scale"] = "All"
    quant_config.extra_options["CalibOptimizeMem"] = True
    return Config(global_quant_config=quant_config)


def prepare_minmse_mostcommon_config():
    quant_config = get_default_config("XINT8")
    quant_config.calibrate_method = PowerOfTwoMethod.MinMSE
    quant_config.extra_options["MinMSEModePof2Scale"] = "MostCommon"
    quant_config.extra_options["CalibTensorRangeSymmetric"] = True
    return Config(global_quant_config=quant_config)


def prepare_minmse_percentile_config():
    quant_config = get_default_config("XINT8")
    quant_config.calibrate_method = PowerOfTwoMethod.MinMSE
    quant_config.extra_options["MinMSEModePof2Scale"] = "Percentile"
    quant_config.extra_options["CalibOptimizeMem"] = False
    return Config(global_quant_config=quant_config)


def prepare_percentile_multiple_workers_config():
    quant_config = get_default_config("S8S8_AAWS")
    quant_config.calibrate_method = CalibrationMethod.Percentile
    quant_config.extra_options["CalibWorkerNum"] = 1000
    return Config(global_quant_config=quant_config)


def prepare_percentile_sym_config():
    quant_config = get_default_config("S8S8_AAWS")
    quant_config.calibrate_method = CalibrationMethod.Percentile
    quant_config.extra_options["CalibTensorRangeSymmetric"] = True
    return Config(global_quant_config=quant_config)


def prepare_percentile_asym_config():
    quant_config = get_default_config("S8S8_AAWS")
    quant_config.calibrate_method = CalibrationMethod.Percentile
    quant_config.extra_options["CalibTensorRangeSymmetric"] = False
    return Config(global_quant_config=quant_config)


def prepare_entropy_config():
    quant_config = get_default_config("S8S8_AAWS")
    quant_config.calibrate_method = CalibrationMethod.Entropy
    return Config(global_quant_config=quant_config)


def prepare_distribution_config():
    quant_config = get_default_config("S8S8_AAWS")
    quant_config.calibrate_method = CalibrationMethod.Distribution
    quant_config.extra_options["CalibWorkerNum"] = 2
    return Config(global_quant_config=quant_config)


def prepare_data():
    data_reader = DataReader(input_tensor)
    return data_reader


def prepare_quantizer(quant_config):
    quantizer = ModelQuantizer(quant_config)
    return quantizer


def quantize_static(quantizer, input_model_path, output_model_path, data_reader):
    input_model = onnx.load(input_model_path)
    output_model = quantizer.quantize_model(input_model, calibration_data_reader=data_reader)
    onnx.save(output_model, output_model_path)
    print("Static quantized the ONNX model and saved it at:", output_model_path)
    return output_model_path


def infer_quantized_model(quantized_model_path):
    sess = onnxruntime.InferenceSession(quantized_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(output_dir, quant_config):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestComputeMinmseFromHistogram(unittest.TestCase):
    def test_matches_raw_minmse_on_known_data(self):
        """Histogram MSE worker must select same scale as raw-data worker on uniform data."""
        from quark.onnx import PowerOfTwoMethod
        from quark.onnx.calibration.collectors import compute_minmse_from_histogram, compute_minmse_worker
        from quark.onnx.quantization.quant_utils import get_tensor_type_from_qType

        rng = np.random.default_rng(42)
        data = rng.uniform(-1.0, 1.0, size=10000).astype(np.float32)

        num_bins = 2048
        hist, edges = np.histogram(data, bins=num_bins)
        rmin = np.array(data.min(), dtype=np.float32)
        rmax = np.array(data.max(), dtype=np.float32)
        histogram = (hist, edges.astype(np.float32), rmin, rmax)

        act_type = get_tensor_type_from_qType(QuantType.QInt8)
        name, (thresh_min, thresh_max) = compute_minmse_from_histogram(
            tensor_name="test_tensor",
            histogram=histogram,
            quantized_tensor_type={},
            activation_qType=act_type,
            symmetric=True,
        )

        _, (ref_thresh_min, ref_thresh_max) = compute_minmse_worker(
            "test_tensor", [data], {}, "All", act_type, True, PowerOfTwoMethod.MinMSE, 99.999
        )

        self.assertEqual(name, "test_tensor")
        self.assertAlmostEqual(float(thresh_min), -float(thresh_max), places=4)
        self.assertTrue(
            np.isclose(float(thresh_max), float(ref_thresh_max)),
            f"Histogram result {float(thresh_max)} does not match reference worker {float(ref_thresh_max)}",
        )

    def test_quantized_tensor_type_override(self):
        """quantized_tensor_type override must change act_type used for candidate search."""
        from quark.onnx.calibration.collectors import compute_minmse_from_histogram
        from quark.onnx.quantization.quant_utils import get_tensor_type_from_qType

        rng = np.random.default_rng(7)
        data = rng.uniform(-0.5, 0.5, size=5000).astype(np.float32)
        hist, edges = np.histogram(data, bins=512)
        rmin = np.array(data.min(), dtype=np.float32)
        rmax = np.array(data.max(), dtype=np.float32)
        histogram = (hist, edges.astype(np.float32), rmin, rmax)

        act_type_int8 = get_tensor_type_from_qType(QuantType.QInt8)

        _, (min8, max8) = compute_minmse_from_histogram("t", histogram, {}, act_type_int8, symmetric=True)
        _, (min16, max16) = compute_minmse_from_histogram(
            "t", histogram, {"t": QuantType.QInt16}, act_type_int8, symmetric=True
        )
        self.assertNotEqual(float(max8), float(max16), "INT16 override should produce a different threshold than INT8")


class TestPowOfTwoCollectorHistogram(unittest.TestCase):
    def _make_collector(self, num_bins=64):
        from quark.onnx.calibration.collectors import PowOfTwoCollector
        from quark.onnx.calibration.methods import PowerOfTwoMethod

        return PowOfTwoCollector(
            activation_type=QuantType.QInt8,
            method=PowerOfTwoMethod.MinMSE,
            symmetric=True,
            minmse_mode="All",
            num_bins=num_bins,
        )

    def test_single_batch_creates_histogram(self):
        """A single call to collect_histogram_value should create a histogram entry with the correct shape and counts."""
        collector = self._make_collector()
        data = np.linspace(-1.0, 1.0, 200, dtype=np.float32)
        collector.collect_histogram_value({"t": [data]})
        self.assertIn("t", collector.histogram_dict)
        hist, edges, rmin, rmax = collector.histogram_dict["t"]
        self.assertEqual(len(hist), 64)
        self.assertEqual(len(edges), 65)
        self.assertAlmostEqual(float(rmin), -1.0, places=4)
        self.assertAlmostEqual(float(rmax), 1.0, places=4)
        self.assertEqual(int(hist.sum()), 200)

    def test_two_batches_same_range_counts_merge(self):
        """Two batches with identical range should accumulate counts without expanding the histogram."""
        collector = self._make_collector()
        data = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
        collector.collect_histogram_value({"t": [data]})
        collector.collect_histogram_value({"t": [data]})
        hist, _, _, _ = collector.histogram_dict["t"]
        self.assertEqual(int(hist.sum()), 200)

    def test_expanding_range_preserves_all_counts(self):
        """A batch that exceeds the current range should expand the histogram while preserving all sample counts."""
        collector = self._make_collector()
        batch1 = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
        batch2 = np.linspace(-2.0, 2.0, 100, dtype=np.float32)
        collector.collect_histogram_value({"t": [batch1]})
        collector.collect_histogram_value({"t": [batch2]})
        hist, edges, rmin, rmax = collector.histogram_dict["t"]
        self.assertEqual(int(hist.sum()), 200)
        self.assertLessEqual(float(edges[0]), -2.0)
        self.assertGreaterEqual(float(edges[-1]), 2.0)
        self.assertAlmostEqual(float(rmin), -2.0, places=4)
        self.assertAlmostEqual(float(rmax), 2.0, places=4)


class TestPowOfTwoCollectorComputeMinmse(unittest.TestCase):
    def test_compute_collection_result_uses_histogram_when_available(self):
        """compute_collection_result should return symmetric thresholds derived from the accumulated histogram."""
        from quark.onnx.calibration.collectors import PowOfTwoCollector
        from quark.onnx.calibration.methods import PowerOfTwoMethod

        collector = PowOfTwoCollector(
            activation_type=QuantType.QInt8,
            method=PowerOfTwoMethod.MinMSE,
            symmetric=True,
            minmse_mode="All",
            num_bins=256,
        )
        rng = np.random.default_rng(0)
        data = rng.uniform(-0.8, 0.8, 2000).astype(np.float32)
        collector.collect_histogram_value({"act": [data[:1000]]})
        collector.collect_histogram_value({"act": [data[1000:]]})

        result = collector.compute_collection_result()
        self.assertIn("act", result)
        lo, hi = result["act"]
        self.assertLess(float(lo), 0.0)
        self.assertGreater(float(hi), 0.0)
        self.assertAlmostEqual(float(lo), -float(hi), places=3)

    def test_compute_collection_result_raises_when_no_data(self):
        """compute_collection_result should raise ValueError when neither histogram nor raw data has been collected."""
        from quark.onnx.calibration.collectors import PowOfTwoCollector
        from quark.onnx.calibration.methods import PowerOfTwoMethod

        collector = PowOfTwoCollector(
            activation_type=QuantType.QInt8,
            method=PowerOfTwoMethod.MinMSE,
            symmetric=True,
        )
        with self.assertRaises(ValueError):
            collector.compute_collection_result()


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_tensor_quantize_minmse_all_multiple_workers(self, tmpdir: str):
        quant_config = prepare_minmse_all_multiple_workers_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, output_tensor_minmse_all, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_tensor_quantize_minmse_all(self, tmpdir: str):
        quant_config = prepare_minmse_all_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, output_tensor_minmse_all, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_tensor_quantize_minmse_mostcommon(self, tmpdir: str):
        quant_config = prepare_minmse_mostcommon_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, output_tensor_minmse_mostcommon, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_tensor_quantize_minmse_percentile(self, tmpdir: str):
        quant_config = prepare_minmse_percentile_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, output_tensor_minmse_percentile, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_tensor_quantize_percentile_multiple_workers(self, tmpdir: str):
        quant_config = prepare_percentile_multiple_workers_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, output_tensor_percentile_sym, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_tensor_quantize_percentile_sym(self, tmpdir: str):
        quant_config = prepare_percentile_sym_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, output_tensor_percentile_sym, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_tensor_quantize_percentile_asym(self, tmpdir: str):
        quant_config = prepare_percentile_asym_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, output_tensor_percentile_asym, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_tensor_quantize_entropy(self, tmpdir: str):
        quant_config = prepare_entropy_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, output_tensor_entropy, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_tensor_quantize_distribution(self, tmpdir: str):
        quant_config = prepare_distribution_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, output_tensor_distribution, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)


class TestMinMSEHistogramMode(unittest.TestCase):
    @use_temporary_directory
    def test_histogram_mode_no_disk_writes(self, tmpdir: str):
        """Histogram mode must not write .npystream files."""
        import glob
        import os

        from quark.onnx.calibration.calibrators import PowOfTwoCalibrater
        from quark.onnx.calibration.methods import PowerOfTwoMethod as Pof2

        # Build the tiny test model
        input_model_path, _ = prepare_model(tmpdir)
        augmented_path = os.path.join(tmpdir, "augmented.onnx")

        calibrator = PowOfTwoCalibrater(
            model_input=input_model_path,
            augmented_model_path=augmented_path,
            method=Pof2.MinMSE,
            minmse_mode="All",
            symmetric=True,
            num_bins=2048,
        )
        calibrator.augment_graph()
        calibrator.execution_providers = ["CPUExecutionProvider"]
        calibrator.create_inference_session()
        calibrator.collect_data(DataReader(input_tensor))

        npystream_files = glob.glob(os.path.join(tmpdir, "*.npystream"))
        self.assertEqual(len(npystream_files), 0, f"Unexpected disk files: {npystream_files}")
        result = calibrator.compute_data()
        self.assertIsNotNone(result)


class TestMinMSENumBinsOption(unittest.TestCase):
    @use_temporary_directory
    def test_custom_num_bins_accepted(self, tmpdir: str):
        """MinMSENumBins extra_option is forwarded to PowOfTwoCalibrater via create_calibrator_power_of_two."""
        import os

        from quark.onnx.calibration.calibrators import create_calibrator_power_of_two
        from quark.onnx.calibration.methods import PowerOfTwoMethod as Pof2

        input_model_path, _ = prepare_model(tmpdir)
        augmented_path = os.path.join(tmpdir, "augmented.onnx")

        calibrator = create_calibrator_power_of_two(
            model_input=input_model_path,
            augmented_model_path=augmented_path,
            calibrate_method=Pof2.MinMSE,
            extra_options={"minmse_mode": "All", "num_bins": 512},
        )
        self.assertEqual(calibrator.num_bins, 512)


if __name__ == "__main__":
    unittest.main()
