#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import unittest
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import Config, ModelQuantizer
from quark.onnx.quantization.config.custom_config import S16S8_ASWS_CONFIG, XINT8_CONFIG

input_tensor = np.array([3.0]).astype(np.float32)


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


class SimpleMulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0]))

    def forward(self, x):
        return x * self.weight


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = SimpleMulModel()

    dummy_input = torch.randn(1)

    onnx_model_path = Path(output_dir, "simple_mul_model.onnx").as_posix()
    onnx_quantized_model_path = Path(output_dir, "simple_mul_model_quantized.onnx").as_posix()

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_quantized_model_path


def prepare_config(tmp_config):
    config_copy = copy.deepcopy(tmp_config)
    config_copy.extra_options["AlignEltwiseQuantType"] = True
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


def tensor_quantize(tmp_config, output_dir):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config(tmp_config)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_eltwise(self, tmpdir: str):
        output = tensor_quantize(S16S8_ASWS_CONFIG, tmpdir)
        golden = np.array([[6.0]], dtype=np.float32)
        self.assertEqual(output, golden)

    @use_temporary_directory
    def test_quantize_eltwise_warning(self, tmpdir: str):
        output = tensor_quantize(XINT8_CONFIG, tmpdir)
        golden = np.array([[6.0]], dtype=np.float32)
        self.assertEqual(output, golden)


if __name__ == "__main__":
    unittest.main()
