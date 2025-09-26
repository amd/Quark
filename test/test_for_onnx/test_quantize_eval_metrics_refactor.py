#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest

import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader
from testing_utils import prepare_model

from quark.onnx import ModelQuantizer, get_library_path
from quark.onnx.quantization.config.config import QConfig
from quark.onnx.quantization.config.spec import QLayerConfig, XInt8Spec
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

XINT8_golden_output = np.array(
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


class CalibDataReader(CalibrationDataReader):
    def __init__(self, input_tensor):
        self.data = [input_tensor]
        self.input_name = "input"
        self.index = 0

    def get_next(self):
        if self.index < len(self.data):
            input_dict = {self.input_name: self.data[self.index]}
            self.index += 1
            return input_dict
        else:
            return None

    def reset_iter(self):
        self.index = 0


class EvalDataReader(CalibrationDataReader):
    def __init__(self, input_tensor):
        self.data = [input_tensor]
        self.input_name = "input"
        self.index = 0

    def get_next(self):
        if self.index < len(self.data):
            input_dict = {self.input_name: self.data[self.index]}
            self.index += 1
            return input_dict
        else:
            return None

    def reset_iter(self):
        self.index = 0


def infer_quantized_model(quantized_model_path):
    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path())
    sess = onnxruntime.InferenceSession(quantized_model_path, sess_options)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize_eval_metrics(output_dir):
    input_model_path, quantized_model_path = prepare_model(output_dir)
    data_reader = CalibDataReader(input_tensor)
    quant_config = QConfig(QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()), EvalMetrics=True)
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(input_model_path, quantized_model_path, data_reader)
    print("Quantized the ONNX model and saved it at:", quantized_model_path)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_eval_datareader(output_dir):
    input_model_path, quantized_model_path = prepare_model(output_dir)
    data_reader = CalibDataReader(input_tensor)
    quant_config = QConfig(
        QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()),
        EvalMetrics=True,
        EvalDataReader=EvalDataReader(input_tensor),
    )
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(input_model_path, quantized_model_path, data_reader)
    print("Quantized the ONNX model and saved it at:", quantized_model_path)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_EvalMetrics(self, tmpdir: str):
        output = tensor_quantize_eval_metrics(tmpdir)
        comp_equal = np.allclose(output, XINT8_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_EvalDataReader(self, tmpdir: str):
        output = tensor_quantize_eval_datareader(tmpdir)
        comp_equal = np.allclose(output, XINT8_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)


if __name__ == "__main__":
    unittest.main()
