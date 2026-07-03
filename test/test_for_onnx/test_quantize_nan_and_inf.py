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

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import ModelQuantizer, QConfig, QLayerConfig, XInt8Spec

input_data_1 = np.array(
    [[np.nan, 0.96671784, 0.260476, 0.8972365, 0.37674972], [0.33622175, 0.45137647, 0.8402551, 0.12310214, 0.5430262]]
).astype(np.float32)

input_data_2 = np.array(
    [[np.nan, 0.96671784, 0.260476, 0.8972365, 0.37674972], [np.nan, np.inf, -np.inf, 0.12310214, 0.5430262]]
).astype(np.float32)


golden_output_1 = np.array(
    [
        [-0.1171875, -0.171875, -0.4296875, 0.6328125, 0.375, 0.0703125, -0.53125, -0.1328125, 0.6171875, 0.25],
        [-0.25, -0.1484375, 0.2265625, 0.59375, -0.2734375, -0.40625, 0.328125, 0.3515625, 0.6875, -0.3203125],
    ]
).astype(np.float32)

golden_output_2 = np.array(
    [
        [-256.0, -256.0, -256.0, 254.0, 254.0, 254.0, -256.0, -256.0, -256.0, 254.0],
        [254.0, -256.0, -256.0, 254.0, 254.0, 254.0, -256.0, -256.0, -256.0, 254.0],
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


class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mul = nn.Linear(5, 10)

    def forward(self, x):
        x = self.mul(x)
        return x


def prepare_model(output_dir):
    torch.manual_seed(42)

    model = CnnModel()
    onnx_model_path = Path(output_dir, "float_cnn.onnx").as_posix()

    dummy_input = torch.randn([2, 5])
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
    return onnx_model_path


def infer_model(input_data, model_path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model_path, sess_options=so)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(input_data, output_dir):
    quantized_model_path = Path(output_dir, "quantized_cnn.onnx").as_posix()
    data_reader = DataReader(input_data)
    input_model_path = prepare_model(output_dir)
    quant_config = QConfig(global_config=QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()))
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(input_model_path, quantized_model_path, data_reader)
    print("Quantized the ONNX model and saved it at:", quantized_model_path)
    output = infer_model(input_data, quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_Nan_and_Inf_Model_1(self, tmpdir: str):
        with self.assertLogs("quark.onnx.utils.model_utils_screen", level="WARNING") as cm:
            output = tensor_quantize(input_data_1, tmpdir)
        self.assertTrue(any("Non-finite values" in message for message in cm.output))
        output = tensor_quantize(input_data_1, tmpdir)
        comp_equal = np.allclose(output, golden_output_1, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_Nan_and_Inf_Model_2(self, tmpdir: str):
        with self.assertLogs("quark.onnx.utils.model_utils_screen", level="WARNING") as cm:
            output = tensor_quantize(input_data_2, tmpdir)
        self.assertTrue(any("Non-finite values" in message for message in cm.output))
        output = tensor_quantize(input_data_2, tmpdir)
        comp_equal = np.allclose(output, golden_output_2, atol=1e-1)
        self.assertEqual(comp_equal, True)


if __name__ == "__main__":
    unittest.main()
