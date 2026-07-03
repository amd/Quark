#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import unittest

import numpy as np
import onnxruntime
from onnx_testing_utils import prepare_model
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization.quant_utils import QuantFormat

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import Config, ModelQuantizer
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

QuantFormat_QOperator_with_CPU_golden_output = np.array(
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

QuantFormat_QOperator_with_NPU_golden_output = np.array(
    [
        [
            [
                [0.25390625, 0.16015625, 0.0234375, -0.046875],
                [0.1484375, -0.01953125, 0.13671875, 0.0390625],
                [0.125, 0.26953125, 0.41796875, 0.11328125],
                [0.1484375, 0.13671875, 0.234375, 0.28515625],
            ]
        ]
    ]
).astype(np.float32)

Int16Scale_golden_output = np.array(
    [
        [
            [
                [0.24002075, 0.16062927, 0.02400208, -0.04800415],
                [0.14216614, -0.03692627, 0.1366272, 0.03692627],
                [0.123703, 0.26771545, 0.41357422, 0.11262512],
                [0.14955139, 0.13847351, 0.23448181, 0.28433228],
            ]
        ]
    ]
).astype(np.float32)

QuantFormat_QDQ_golden_output = np.array(
    [
        [
            [
                [0.24002075, 0.16062927, 0.02400208, -0.04800415],
                [0.14216614, -0.03692627, 0.1366272, 0.03692627],
                [0.123703, 0.26771545, 0.41357422, 0.11262512],
                [0.14955139, 0.13847351, 0.23448181, 0.28433228],
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
        else:
            return None

    def rewind(self):
        self.index = 0


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


def tensor_quantize(quant_config, output_dir):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_QuantFormat_QOperator_with_CPU(self, tmpdir: str):
        quant_config = get_default_config("U8S8_AAWS")
        config_copy = copy.deepcopy(quant_config)
        config_copy.quant_format = QuantFormat.QOperator
        config = Config(global_quant_config=config_copy)
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, QuantFormat_QOperator_with_CPU_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_QuantFormat_QOperator_with_NPU(self, tmpdir: str):
        quant_config = get_default_config("XINT8")
        config_copy = copy.deepcopy(quant_config)
        config_copy.quant_format = QuantFormat.QOperator
        config = Config(global_quant_config=config_copy)
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, QuantFormat_QOperator_with_NPU_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_Int16Scale(self, tmpdir: str):
        quant_config = get_default_config("U8S8_AAWS")
        config_copy = copy.deepcopy(quant_config)
        config_copy.quant_format = QuantFormat.QOperator
        config_copy.extra_options["Int16Scale"] = True
        config = Config(global_quant_config=config_copy)
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, Int16Scale_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_QuantFormat_QDQ(self, tmpdir: str):
        quant_config = get_default_config("U8S8_AAWS")
        config_copy = copy.deepcopy(quant_config)
        config_copy.quant_format = QuantFormat.QDQ
        config_copy.extra_options["Int16Scale"] = True
        config = Config(global_quant_config=config_copy)
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, QuantFormat_QDQ_golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)


if __name__ == "__main__":
    unittest.main()
