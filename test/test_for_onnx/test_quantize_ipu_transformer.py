#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config.custom_config import INT8_TRANSFORMER_DEFAULT_CONFIG, INT8_TRANSFORMER_ACCURATE_CONFIG, INT16_TRANSFORMER_DEFAULT_CONFIG, INT16_TRANSFORMER_ACCURATE_CONFIG
from quark.onnx.quantization.config.config import Config
from testing_utils import prepare_model_vit
from quark.shares.utils.testing_utils import use_temporary_directory

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

INT8_TRANSFORMER_golden_output = np.array([[0.4097347, -0.09403747, 0.06716962, -0.71199805, 0.44331953,
                                            0.14777318, 0.27539545, 0.47018737, 0.6649793, 0.03358481]]).astype(np.float32)

INT8_TRANSFORMER_ACCURATE_golden_output = np.array([[0.36263505, -0.28204948, 0.04700825, -0.75213194, 0.57081443,
                                                     0.08058557, 0.26861855, 0.38278145, 0.7655629, 0.17460206]]).astype(np.float32)

INT16_TRANSFORMER_golden_output = np.array([[0.13535856, 0.6938596, -0.7329068, -0.8139547, 0.2314869,
                                             0.44130704, 0.11938943, 0.8988707, -0.11625311, -0.7321228]]).astype(np.float32)

INT16_TRANSFORMER_ACCURATE_golden_output = np.array([[0.13543288, 0.6938355, -0.73287404, -0.8139299, 0.23148753,
                                                      0.4413654, 0.1193628, 0.89848727, -0.11622717, -0.7321162]]).astype(np.float32)


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

def prepare_config(config):
    quant_config = Config(global_quant_config=config)
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
    sess = onnxruntime.InferenceSession(quantized_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f'Model output: {output}')
    return output


def tensor_quantize_ipu_transformer(config, output_dir):
    input_model_path, output_model_path = prepare_model_vit(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config(config)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_INT8_TRANSFORMER(self, tmpdir: str):
        config = INT8_TRANSFORMER_DEFAULT_CONFIG
        output = tensor_quantize_ipu_transformer(config, tmpdir)
        comp_equal = np.allclose(output, INT8_TRANSFORMER_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_INT8_TRANSFORMER_ACCURATE(self, tmpdir: str):
        config = INT8_TRANSFORMER_ACCURATE_CONFIG
        output = tensor_quantize_ipu_transformer(config, tmpdir)
        comp_equal = np.allclose(output, INT8_TRANSFORMER_ACCURATE_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_INT16_TRANSFORMER(self, tmpdir: str):
        config = INT16_TRANSFORMER_DEFAULT_CONFIG
        output = tensor_quantize_ipu_transformer(config, tmpdir)
        comp_equal = np.allclose(output, INT16_TRANSFORMER_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_INT16_TRANSFORMER_ACCURATE(self, tmpdir: str):
        config = INT16_TRANSFORMER_ACCURATE_CONFIG
        output = tensor_quantize_ipu_transformer(config, tmpdir)
        comp_equal = np.allclose(output, INT16_TRANSFORMER_ACCURATE_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)

if __name__ == '__main__':
    unittest.main()
