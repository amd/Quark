#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import unittest

import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader
from testing_utils import prepare_model_vit

from quark.onnx import ModelQuantizer
from quark.onnx.quant_utils import is_version_below
from quark.onnx.quantization.config.config import Config
from quark.onnx.quantization.config.custom_config import MATMUL_NBITS_CONFIG
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

output_tensor = np.array(
    [
        [
            0.13596348,
            0.7069947,
            -0.745268,
            -0.8128995,
            0.25301066,
            0.43890983,
            0.10404871,
            0.88007313,
            -0.11417447,
            -0.7310455,
        ]
    ]
).astype(np.float32)
output_tensor_gptq = np.array(
    [
        [
            0.14073174,
            0.72223854,
            -0.7341187,
            -0.8268701,
            0.22473133,
            0.44346523,
            0.1087392,
            0.8950456,
            -0.13907433,
            -0.74229425,
        ]
    ]
).astype(np.float32)
output_tensor_hqq = np.array(
    [
        [
            0.13278924,
            0.7027687,
            -0.7434849,
            -0.8260411,
            0.26645112,
            0.4434718,
            0.10804553,
            0.89408153,
            -0.1228957,
            -0.7243466,
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
        else:
            return None

    def rewind(self):
        self.index = 0


def prepare_config():
    config_copy = copy.deepcopy(MATMUL_NBITS_CONFIG)
    config_copy.extra_options["MatMulNBitsParams"]["Symmetric"] = False
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_config_gptq():
    config_copy = copy.deepcopy(MATMUL_NBITS_CONFIG)
    config_copy.extra_options["MatMulNBitsParams"]["Symmetric"] = False
    config_copy.extra_options["MatMulNBitsParams"]["Algorithm"] = "GPTQ"
    config_copy.extra_options["GPTQParams"] = {"MSE": False, "GroupSize": 32, "ActOrder": True, "PerChannel": True}
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_config_hqq():
    config_copy = copy.deepcopy(MATMUL_NBITS_CONFIG)
    config_copy.extra_options["MatMulNBitsParams"]["Symmetric"] = False
    config_copy.extra_options["MatMulNBitsParams"]["Algorithm"] = "HQQ"
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_data():
    data_reader = DataReader(input_tensor)
    return data_reader


def prepare_quantizer(quant_config):
    quantizer = ModelQuantizer(quant_config)
    return quantizer


def quantize_static(quantizer, input_model_path, output_model_path, data_reader):
    quantizer.quantize_model(input_model_path, output_model_path, data_reader)
    print("Quantized the ONNX model and saved it at:", output_model_path)
    return output_model_path


def infer_quantized_model(quantized_model_path):
    sess = onnxruntime.InferenceSession(quantized_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize_matmul_4bits(output_dir):
    input_model_path, output_model_path = prepare_model_vit(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_matmul_4bits_none_calibration_data_reader(output_dir):
    input_model_path, output_model_path = prepare_model_vit(output_dir)

    quant_config = prepare_config()
    quant_config.global_quant_config.extra_options["UseRandomData"] = True

    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, None)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_matmul_4bits_gptq(output_dir):
    input_model_path, output_model_path = prepare_model_vit(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config_gptq()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_matmul_4bits_hqq(output_dir):
    input_model_path, output_model_path = prepare_model_vit(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config_hqq()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize(self, tmpdir: str):
        output = tensor_quantize_matmul_4bits(tmpdir)
        comp_equal = np.allclose(output, output_tensor, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_none_calibration_data_reader(self, tmpdir: str):
        output = tensor_quantize_matmul_4bits_none_calibration_data_reader(tmpdir)
        comp_equal = np.allclose(output, output_tensor, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_gptq(self, tmpdir: str):
        output_gptq = tensor_quantize_matmul_4bits_gptq(tmpdir)
        comp_equal = np.allclose(output_gptq, output_tensor_gptq, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_hqq(self, tmpdir: str):
        if not is_version_below(onnxruntime, "1.18.0"):
            output_hqq = tensor_quantize_matmul_4bits_hqq(tmpdir)
            comp_equal = np.allclose(output_hqq, output_tensor_hqq, atol=1e-1)
            self.assertEqual(comp_equal, True)


if __name__ == "__main__":
    unittest.main()
