#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
import torch
import torch.nn as nn
import numpy as np
import onnxruntime
import copy
from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx import get_library_path
from quark.onnx.quantization.config.config import Config
from quark.onnx.quantization.config.custom_config import XINT8_CONFIG, INT8_TRANSFORMER_DEFAULT_CONFIG, S8S8_AAWS_CONFIG
from quark.shares.utils.testing_utils import use_temporary_directory
from pathlib import Path

input_tensor = np.array([[[[0.26921557, 0.79500909, 0.6102178, 0.04375664],
                           [0.06221361, 0.98258356, 0.38635129, 0.06492238],
                           [0.49631707, 0.35442799, 0.51719146, 0.52100111],
                           [0.04145599, 0.88960236, 0.50627326, 0.57204613]],
                          [[0.99185097, 0.93582153, 0.13174529, 0.42896287],
                           [0.14552133, 0.02538564, 0.0732355, 0.25725371],
                           [0.09856916, 0.43015628, 0.55679755, 0.66560074],
                           [0.9439425, 0.45701841, 0.86791293, 0.64728276]],
                          [[0.29159685, 0.79021383, 0.3117182, 0.11342342],
                           [0.16660495, 0.46426165, 0.31348552, 0.143383],
                           [0.96454802, 0.63258874, 0.30295267, 0.96720039],
                           [0.29879457, 0.79916527, 0.02905061, 0.20115725]]]]).astype(np.float32)

golden_output = np.array([[0.14453125, 0.04492188, -0.20898438, -0.16210938, 0.05078125, -0.23046875]]).astype(np.float32)

XINT8_golden_output = np.array([[0.14453125, 0.04492188, -0.20898438, -0.16210938, 0.04882812, -0.23242188]]).astype(np.float32)

S8S8_AAWS_golden_output = np.array([[0.13598002, 0.04434131, -0.20840417, -0.17736524, 0.05025349, -0.20692612]]).astype(np.float32)

INT8_TRANSFORMER_golden_output = np.array([[0.13601275, 0.04435198, -0.20845431, -0.17740792, 0.05026558, -0.20549752]]).astype(np.float32)

class DataReader(CalibrationDataReader):
    def __init__(self, input_tensor):
        self.data = [input_tensor]
        self.input_name = 'input'
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


class Extra_Types_To_Quantize_Model(nn.Module):
    def __init__(self):
        super(Extra_Types_To_Quantize_Model, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv_transpose = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1, padding=0)
        self.gemm = nn.Linear(27, 6)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm1d(6)
        self.avgpool = nn.AvgPool2d(kernel_size=3)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.conv_transpose(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = x.reshape(x.size(0), -1)
        x = self.gemm(x)
        x = self.bn2(x)
        return x


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = Extra_Types_To_Quantize_Model()
    dummy_input = torch.randn(1, 3, 4, 4)

    onnx_model_path = Path(output_dir, 'extra_op_types_to_quantize_model.onnx').as_posix()
    quant_onnx_model_path = Path(output_dir, 'extra_op_types_to_quantize_model_quantized.onnx').as_posix()
    torch.onnx.export(model,
                      dummy_input,
                      onnx_model_path,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=17)

    print(f'Model has been saved to {onnx_model_path}')
    return onnx_model_path, quant_onnx_model_path


def prepare_config(config, op_types_to_quantize, extra_op_types_to_quantize):
    config_copy = copy.deepcopy(config)
    config_copy.op_types_to_quantize = op_types_to_quantize
    config_copy.extra_op_types_to_quantize = extra_op_types_to_quantize
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
    print('Quantized the ONNX model and saved it at:', output_model_path)
    return output_model_path


def infer_quantized_model(quantized_model_path):
    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path())
    sess = onnxruntime.InferenceSession(quantized_model_path, sess_options)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f'Model output: {output}')
    return output


def tensor_quantize(config, op_types_to_quantize, extra_op_types_to_quantize, output_dir):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config(config, op_types_to_quantize, extra_op_types_to_quantize)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_Raise(self, tmpdir: str):
        with self.assertLogs('quark.onnx.quant_utils_screen', level='WARNING') as cm:
            output = tensor_quantize(XINT8_CONFIG, [], ['BatchNormalization', 'xxx'], tmpdir)
        self.assertTrue(any("The model does not contain the following op types: " in message for message in cm.output))

        comp_equal = np.allclose(output, XINT8_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize(self, tmpdir: str):
        output = tensor_quantize(XINT8_CONFIG, ['Conv'], ['Gemm'], tmpdir)
        comp_equal = np.allclose(output, golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_XINT8(self, tmpdir: str):
        output = tensor_quantize(XINT8_CONFIG, [], ['BatchNormalization'], tmpdir)
        comp_equal = np.allclose(output, XINT8_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_S8S8_AAWS(self, tmpdir: str):
        output = tensor_quantize(S8S8_AAWS_CONFIG, [], ['PRelu'], tmpdir)
        comp_equal = np.allclose(output, S8S8_AAWS_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_INT8_TRANSFORMER(self, tmpdir: str):
        output = tensor_quantize(INT8_TRANSFORMER_DEFAULT_CONFIG, [], ['ConvTranspose', 'PRelu'], tmpdir)
        comp_equal = np.allclose(output, INT8_TRANSFORMER_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)


if __name__ == '__main__':
    unittest.main()
