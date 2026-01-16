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

from quark.onnx import BFloat16Spec, ModelQuantizer, QConfig, QLayerConfig
from quark.onnx.operators.custom_ops import get_library_path
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
golden_output = np.array(
    [
        [
            -0.3125,
            -0.09667969,
            -0.359375,
            -0.02172852,
            0.01782227,
            -0.828125,
            0.8046875,
            -0.25195312,
            -0.06494141,
            -0.66015625,
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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_relu = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.conv_leaky_relu = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.01)
        )
        self.conv_prelu = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), nn.PReLU())
        self.mul = nn.Linear(8 * 5 * 5, 10)
        self.add = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv_relu(x)
        x = self.conv_leaky_relu(x)
        x = self.conv_prelu(x)
        x = x.view(x.size(0), -1)
        x = self.mul(x) * 0.5
        x = self.add(x + 1.0)
        return x


def prepare_model(output_dir):
    torch.manual_seed(42)

    model = Model()

    dummy_input = torch.randn([1, 3, 5, 5])
    onnx_model_path = Path(output_dir, "onnx_input.onnx").as_posix()
    onnx_quantized_model_path_qdq = Path(output_dir, "onnx_quantized_qdq.onnx").as_posix()
    onnx_quantized_model_path_clip = Path(output_dir, "onnx_quantized_clip.onnx").as_posix()

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        keep_initializers_as_inputs=False,
        do_constant_folding=False,
        opset_version=17,
        dynamo=False,
    )

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_quantized_model_path_qdq, onnx_quantized_model_path_clip


def prepare_config(is_insert_clip_bfloat16_qdq):
    if is_insert_clip_bfloat16_qdq:
        quant_config = QConfig(QLayerConfig(activation=BFloat16Spec(), weight=BFloat16Spec()), BF16WithClip=True)
    else:
        quant_config = QConfig(QLayerConfig(activation=BFloat16Spec(), weight=BFloat16Spec()), BF16WithClip=False)
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
    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path("CPU"))
    sess = onnxruntime.InferenceSession(quantized_model_path, so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(input_data, is_replace_bfloat16_qdq_clip, output_dir):
    data_reader = prepare_data(input_data)
    input_model_path, output_model_path_qdq, output_model_path_clip = prepare_model(output_dir)
    quant_config = prepare_config(is_replace_bfloat16_qdq_clip)
    quantizer = prepare_quantizer(quant_config)
    if is_replace_bfloat16_qdq_clip:
        quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path_clip, data_reader)
    else:
        quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path_qdq, data_reader)
    output = infer_quantized_model(input_data, quantized_model_path)
    return output, quantized_model_path


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_MultiMulAddModel(self, tmpdir: str):
        output_qdq, quantized_model_path_qdq = tensor_quantize(input_data, False, tmpdir)
        output_clip, quantized_model_path_clip = tensor_quantize(input_data, True, tmpdir)
        comp_equal = np.allclose(output_qdq, golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)
        comp_equal = np.allclose(output_clip, output_qdq, atol=1e-1)
        self.assertEqual(comp_equal, True)


if __name__ == "__main__":
    unittest.main()
