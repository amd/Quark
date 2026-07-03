#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import unittest
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import yaml
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import CLEConfig, Int16Spec, ModelQuantizer, QConfig, QLayerConfig, XInt8Spec


def make_input_tensor():
    return np.array(
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


output_golden = np.array([[-0.46404064]], dtype=np.float32)


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


class DoubleConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1, 1)

        with torch.no_grad():
            self.conv2.weight *= 100.0

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.clip(x, 0, 6)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = DoubleConvModel()
    dummy_input = torch.randn(1, 3, 4, 4)
    onnx_model_path = Path(output_dir, f"double_conv_model_{np.random.randint(0, 10000)}.onnx").as_posix()
    onnx_quantized_model_path = Path(
        output_dir, f"double_conv_model_quantized_{np.random.randint(0, 10000)}.onnx"
    ).as_posix()
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


def prepare_yaml(output_dir):
    yaml_path = Path(output_dir, "preprocess.yaml").as_posix()
    config = {
        "passes": {"onnx_convert_opset_version": {"target_opset_version": 21}},
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def prepare_config(yaml_path):
    cle_algo = CLEConfig(cle_steps=2)
    quant_config = QConfig(
        global_config=QLayerConfig(input_tensors=XInt8Spec(), weight=XInt8Spec()),
        specific_layer_config={
            QLayerConfig(input_tensors=Int16Spec(), weight=Int16Spec()): ["/conv1/Conv", "/conv2/Conv"]
        },
        layer_type_config={None: ["Gemm"]},
        algo_config=[cle_algo],
        extra_options={"SimplifyModel": False, "Int32Bias": False, "PreprocessYAML": yaml_path},
    )
    return quant_config


def prepare_data():
    data_reader = DataReader(make_input_tensor())
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
    input_data = make_input_tensor()
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(output_dir, yaml_path):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config(yaml_path)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_preprocess_yaml(self, tmpdir: str):
        yaml_path = prepare_yaml(tmpdir)
        output = tensor_quantize(tmpdir, yaml_path)
        comp_equal = np.allclose(output, output_golden, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_preprocess_xint8(self, tmpdir: str):
        output = tensor_quantize(tmpdir, "xint8")
        comp_equal = np.allclose(output, output_golden, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_preprocess_a8w8(self, tmpdir: str):
        output = tensor_quantize(tmpdir, "a8w8")
        comp_equal = np.allclose(output, output_golden, atol=1e-1)
        self.assertEqual(comp_equal, True)


if __name__ == "__main__":
    unittest.main()
