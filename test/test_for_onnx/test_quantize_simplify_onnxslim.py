#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader

from quark.onnx import Int8Spec, ModelQuantizer, QConfig, QLayerConfig
from quark.shares.utils.testing_utils import use_temporary_directory


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
        self.fc = nn.Linear(1, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        return self.gelu(x)


def prepare_model(input_shape, model_type, output_dir: str):
    torch.manual_seed(42)
    onnx_model_path = Path(output_dir, model_type + ".onnx").as_posix()
    quant_onnx_model_path = Path(output_dir, model_type + "_quantized.onnx").as_posix()

    opset_version = 17
    model = Model()
    onnx_model_path = Path(output_dir, "gelu.onnx").as_posix()
    quant_onnx_model_path = Path(output_dir, "gelu_quantized.onnx").as_posix()

    dummy_input = torch.randn(input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        opset_version=opset_version,
        dynamo=False,
    )

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, quant_onnx_model_path


def prepare_config():
    quant_config = QConfig(
        global_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec()),
        SimplifyModel=True,
        SimplifyModelOptions={"skip_fusion_patterns": ["EliminationSlice"]},
    )
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
    # Disabling ORT Graph Optimization to achieve reproducible golden numbers across different servers
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(quantized_model_path, sess_options=so)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(input_data, input_shape, model_type, output_dir: str):
    data_reader = prepare_data(input_data)
    input_model_path, output_model_path = prepare_model(input_shape, model_type=model_type, output_dir=output_dir)
    quant_config = prepare_config()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(input_data, quantized_model_path)
    return output, quantized_model_path


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_optimize_fuse_gelu(self, tmpdir: str):
        input_shape = [1]
        input_data = np.array([1.0], dtype=np.float32)
        output, quantized_model_path = tensor_quantize(input_data, input_shape, "fuse_gelu", tmpdir)
        fuse_gelu_golden_output = np.array([1.5061977], dtype=np.float32)
        comp_equal = np.allclose(output, fuse_gelu_golden_output, atol=1e-1)
        self.assertEqual(comp_equal, True)


if __name__ == "__main__":
    unittest.main()
