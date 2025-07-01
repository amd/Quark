#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import unittest
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

from pathlib import Path
from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config.custom_config import XINT8_CONFIG
from quark.onnx.quantization.config.config import Config
from quark.shares.utils.testing_utils import use_temporary_directory

input_data = np.array([[0.36239759, 0.55816052, 0.28596501, 0.2115006]]).astype(np.float32)
golden_output = np.array([[27.0, 15.5, 15.0, 19.0]]).astype(np.float32)


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


class MultiMulAddModel(nn.Module):
    def __init__(self):
        super(MultiMulAddModel, self).__init__()
        self.mul1 = nn.Linear(4, 4, bias=False)
        self.mul2 = nn.Linear(4, 4, bias=False)
        self.add = nn.Linear(4, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.mul1(x) * 2
        x2 = self.mul2(x) * 3
        x = x1 + x2
        x = self.add(x)
        x = x * 2 + 1
        x = x * 5
        x1 = self.relu(x)
        x2 = x + 6
        x = x1 + x2
        return x


def prepare_model(output_dir: str):
    torch.manual_seed(42)

    model = MultiMulAddModel()
    onnx_model_path = Path(output_dir, 'multi_mul_add.onnx').as_posix()
    quant_onnx_model_path = Path(output_dir, 'multi_mul_add_quantized.onnx').as_posix()

    dummy_input = torch.randn([1, 4])
    torch.onnx.export(model,
                      dummy_input,
                      onnx_model_path,
                      input_names=['input'],
                      output_names=['output'],
                      keep_initializers_as_inputs=False,
                      do_constant_folding=False,
                      opset_version=17)

    print(f'Model has been saved to {onnx_model_path}')
    return onnx_model_path, quant_onnx_model_path


def prepare_config(config):
    config_copy = copy.deepcopy(config)
    config_copy.extra_options['RemoveQDQMulAdd'] = True
    config_copy.debug_mode = True
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_data(input_tensor):
    data_reader = DataReader(input_tensor)
    return data_reader


def prepare_quantizer(quant_config):
    quantizer = ModelQuantizer(quant_config)
    return quantizer


def quantize_static(quantizer, input_model_path, output_model_path, data_reader):
    quantizer.quantize_model(input_model_path, output_model_path, data_reader)
    print('Quantized the ONNX model and saved it at:', output_model_path)
    return output_model_path


def infer_quantized_model(input_data, quantized_model_path):
    # Disabling ORT Graph Optimization to achieve reproducible golden numbers across different servers
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(quantized_model_path, sess_options=so)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {input_name: input_data})
    print(f'Model output: {output}')
    return output


def tensor_quantize(input_data, output_dir: str):
    data_reader = prepare_data(input_data)
    input_model_path, output_model_path = prepare_model(output_dir)
    quant_config = prepare_config(XINT8_CONFIG)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(input_data, quantized_model_path)
    return output, quantized_model_path


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_MultiMulAddModel(self, tmpdir: str):
        with self.assertLogs('quark.onnx.tools.remove_qdq_mul_add_screen', level='DEBUG') as cm:
            output, quantized_model_path = tensor_quantize(input_data, output_dir=tmpdir)
        self.assertTrue(any("Skip pattern match: output of DequantizeLinear" in message for message in cm.output))
        self.assertTrue(any("Removed QuantizeLinear & DequantizeLinear operations: mul-add." in message for message in cm.output))
        comp_equal = np.allclose(output, golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)


if __name__ == "__main__":
    unittest.main()
