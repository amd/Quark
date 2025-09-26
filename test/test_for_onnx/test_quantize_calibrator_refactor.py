#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest

import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader
from testing_utils import prepare_model

from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config.config import QConfig
from quark.onnx.quantization.config.spec import CalibMethod, Int8Spec, QLayerConfig, XInt8Spec
from quark.shares.utils.testing_utils import use_temporary_directory

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

    def rewind(self):
        self.index = 0


def prepare_minmse_all_multiple_workers_config():
    quant_config = QConfig(
        QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()), CalibOptimizeMem=True, CalibWorkerNum=1000
    )
    return quant_config


def prepare_minmse_all_config():
    quant_config = QConfig(
        QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()), MinMSEMode="All", CalibOptimizeMem=True
    )
    return quant_config


def prepare_minmse_mostcommon_config():
    quant_config = QConfig(
        QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()),
        MinMSEMode="MostCommon",
        CalibTensorRangeSymmetric=True,
    )
    return quant_config


def prepare_minmse_percentile_config():
    quant_config = QConfig(
        QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()), MinMSEMode="Percentile", CalibOptimizeMem=False
    )
    return quant_config


def prepare_percentile_multiple_workers_config():
    quant_config = QConfig(QLayerConfig(activation=Int8Spec(), weight=Int8Spec()), CalibWorkerNum=1000)
    return quant_config


def prepare_percentile_sym_config():
    quant_config = QConfig(QLayerConfig(activation=Int8Spec(), weight=Int8Spec()), CalibTensorRangeSymmetric=True)
    return quant_config


def prepare_percentile_asym_config():
    quant_config = QConfig(QLayerConfig(activation=Int8Spec(), weight=Int8Spec()), CalibTensorRangeSymmetric=False)
    return quant_config


def prepare_entropy_config():
    act_spec = Int8Spec()
    act_spec.set_calibration_method(CalibMethod.Entropy)
    quant_config = QConfig(QLayerConfig(activation=act_spec, weight=Int8Spec()), CalibWorkerNum=1000)
    return quant_config


def prepare_distribution_config():
    act_spec = Int8Spec()
    act_spec.set_calibration_method(CalibMethod.Distribution)
    quant_config = QConfig(QLayerConfig(act_spec, weight=Int8Spec()), CalibWorkerNum=2)
    return quant_config


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


if __name__ == "__main__":
    unittest.main()
