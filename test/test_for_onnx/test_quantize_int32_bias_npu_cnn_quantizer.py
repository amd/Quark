#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader

from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantFormat, QuantType
from quark.onnx.quantization.config.config import Config, QuantizationConfig
from quark.shares.utils.testing_utils import use_temporary_directory

input_data = np.array(
    [
        [
            [
                [0.26619988, 0.73333566, 0.32430612, 0.56555123, 0.78568403],
                [0.50381943, 0.62112556, 0.78376413, 0.2894883, 0.46732242],
                [0.28120838, 0.53861799, 0.83088573, 0.0888585, 0.30219859],
                [0.80025317, 0.88537935, 0.42602682, 0.78531207, 0.76150828],
                [0.88925415, 0.18487376, 0.71942776, 0.04007276, 0.84051725],
            ],
            [
                [0.83338162, 0.5661508, 0.59231535, 0.28232884, 0.11760868],
                [0.75736037, 0.12840651, 0.18621735, 0.85781309, 0.73346954],
                [0.3070585, 0.03626074, 0.22557921, 0.2237572, 0.78784106],
                [0.68366023, 0.25022015, 0.29810134, 0.60772729, 0.34931635],
                [0.84850974, 0.55294383, 0.31268, 0.61667239, 0.28753261],
            ],
            [
                [0.20067241, 0.95934905, 0.86314381, 0.01692715, 0.34158923],
                [0.24051579, 0.57178108, 0.57631192, 0.75122361, 0.00370697],
                [0.35564212, 0.58467473, 0.58606206, 0.27266265, 0.05458511],
                [0.7195592, 0.20194915, 0.90723205, 0.96791405, 0.39916769],
                [0.27560292, 0.40176254, 0.25091583, 0.39977971, 0.78865324],
            ],
        ]
    ]
).astype(np.float32)

int32_bias_true_golden_output = np.array(
    [
        [
            -0.125,
            -0.07519531,
            0.04882812,
            -0.04296875,
            -0.04980469,
            0.02636719,
            -0.03222656,
            -0.08398438,
            -0.09960938,
            -0.05566406,
        ]
    ]
).astype(np.float32)
int32_bias_false_golden_output = np.array(
    [
        [
            -0.09375,
            -0.08105469,
            0.03320312,
            -0.03320312,
            -0.03613281,
            0.01757812,
            -0.05566406,
            -0.08789062,
            -0.09960938,
            -0.06542969,
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


class Int32BiasNpuCnnQuantizerModel(nn.Module):
    def __init__(self):
        super(Int32BiasNpuCnnQuantizerModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def prepare_model(output_dir):
    torch.manual_seed(42)

    model = Int32BiasNpuCnnQuantizerModel()

    dummy_input = torch.randn([1, 3, 5, 5])
    onnx_model_path = Path(output_dir, "int32_bias_npu_cnn_quantizer.onnx").as_posix()
    onnx_quantized_model_path = Path(output_dir, "int32_bias_npu_cnn_quantizer_quantized.onnx").as_posix()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        keep_initializers_as_inputs=False,
        do_constant_folding=False,
        opset_version=17,
    )

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_quantized_model_path


def prepare_config(int32_bias=False):
    config = QuantizationConfig(
        calibrate_method=PowerOfTwoMethod.NonOverflow,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        enable_npu_cnn=True,
        extra_options={"Int32Bias": int32_bias},
    )
    quant_config = Config(global_quant_config=config)
    return quant_config


def prepare_data(input_tensor):
    data_reader = DataReader(input_tensor)
    return data_reader


def prepare_quantizer(quant_config):
    quantizer = ModelQuantizer(quant_config)
    return quantizer


def quantize_static(quantizer, input_model_path, output_model_path, data_reader):
    quantizer.quantize_model(input_model_path, output_model_path, data_reader)
    print("Quantized the ONNX model and saved it at:", output_model_path)
    return output_model_path


def infer_quantized_model(input_data, quantized_model_path):
    sess = onnxruntime.InferenceSession(quantized_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(input_data, int32_bias, output_dir):
    data_reader = prepare_data(input_data)
    input_model_path, output_model_path = prepare_model(output_dir)
    quant_config = prepare_config(int32_bias)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(input_data, quantized_model_path)
    return output, quantized_model_path


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_npu_cnn_quantizer_int32_bias_true(self, tmpdir: str):
        output, quantized_model_path = tensor_quantize(input_data, True, tmpdir)
        comp_equal = np.allclose(output, int32_bias_true_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_npu_cnn_quantizer_int32_bias_false(self, tmpdir: str):
        output, quantized_model_path = tensor_quantize(input_data, False, tmpdir)
        comp_equal = np.allclose(output, int32_bias_false_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)


if __name__ == "__main__":
    unittest.main()
