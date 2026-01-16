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

from quark.onnx import Int8Spec, ModelQuantizer, QConfig, QLayerConfig, UInt8Spec
from quark.shares.utils.testing_utils import delete_directory_content, use_temporary_directory

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

int8_input_tensor = np.array(
    [
        [
            [[50, 55, -88, 9], [100, 41, -78, -15], [-40, -8, 90, -7], [-88, -65, -81, -108]],
            [[-76, -97, 13, 4], [-30, -65, -73, -6], [-65, -124, 1, 42], [-78, -4, -122, 50]],
            [[-72, 0, -66, 44], [-54, 44, -1, -6], [-90, -113, -121, 81], [118, 28, -27, -43]],
        ]
    ]
).astype(np.int8)

uint8_input_tensor = np.array(
    [
        [
            [[139, 251, 55, 193], [100, 95, 138, 34], [76, 44, 228, 168], [152, 87, 31, 113]],
            [[196, 191, 20, 152], [61, 21, 192, 115], [213, 60, 142, 177], [101, 17, 203, 62]],
            [[97, 135, 9, 59], [216, 195, 115, 91], [170, 175, 150, 33], [127, 71, 212, 244]],
        ]
    ]
).astype(np.uint8)


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


class SimpleConvModel(nn.Module):
    def __init__(self):
        super(SimpleConvModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = x.to(torch.float)
        x = self.conv(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def prepare_model(output_dir: str, input_dtype="float"):
    torch.manual_seed(42)
    model = SimpleConvModel()
    if input_dtype == "float":
        dummy_input = torch.randn(1, 3, 4, 4)
    elif input_dtype == "int8":
        dummy_input = torch.randint(low=0, high=100, size=(1, 3, 4, 4), dtype=torch.int8)
    elif input_dtype == "uint8":
        dummy_input = torch.randint(low=0, high=100, size=(1, 3, 4, 4), dtype=torch.uint8)
    else:
        raise NotImplementedError(f"Unspecified input dtype {input_dtype}!")
    onnx_model_path = Path(output_dir, "simple_conv_model.onnx").as_posix()
    quant_onnx_model_path = Path(output_dir, "simple_conv_model_quantized.onnx").as_posix()
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
    return onnx_model_path, quant_onnx_model_path


def prepare_config():
    quant_config = QConfig(global_config=QLayerConfig(activation=UInt8Spec(), weight=Int8Spec()))
    return quant_config


def prepare_random_input_config_with_specific(input_dtype="float"):
    if input_dtype == "float":
        quant_config = QConfig(
            global_config=QLayerConfig(activation=UInt8Spec(), weight=Int8Spec()),
            UseRandomData=True,
            RandomDataReaderInputDataRange={"input": [-1e-1, 1e-1]},
            RandomDataReaderInputShape={"input": [1, 3, 4, 4]},
        )
    elif input_dtype == "int8":
        quant_config = QConfig(
            global_config=QLayerConfig(activation=UInt8Spec(), weight=Int8Spec()),
            UseRandomData=True,
            RandomDataReaderInputDataRange={"input": [-128, 127]},
            RandomDataReaderInputShape={"input": [1, 3, 4, 4]},
        )
    elif input_dtype == "uint8":
        quant_config = QConfig(
            global_config=QLayerConfig(activation=UInt8Spec(), weight=Int8Spec()),
            UseRandomData=True,
            RandomDataReaderInputDataRange={"input": [0, 255]},
            RandomDataReaderInputShape={"input": [1, 3, 4, 4]},
        )
    else:
        raise NotImplementedError(f"Unspecified input dtype {input_dtype}!")
    return quant_config


def prepare_random_input_config_with_coarse():
    quant_config = QConfig(global_config=QLayerConfig(activation=UInt8Spec(), weight=Int8Spec()), UseRandomData=True)
    return quant_config


def prepare_abnormal_random_input_config(abnormal_type="shape_key"):
    if abnormal_type == "shape_key":
        quant_config = QConfig(
            global_config=QLayerConfig(activation=UInt8Spec(), weight=Int8Spec()),
            UseRandomData=True,
            RandomDataReaderInputDataRange={"input": [-1e-1, 1e-1]},
            RandomDataReaderInputShape={"inp": [1, 3, 4, 4]},
        )
    elif abnormal_type == "range_key":
        quant_config = QConfig(
            global_config=QLayerConfig(activation=UInt8Spec(), weight=Int8Spec()),
            UseRandomData=True,
            RandomDataReaderInputDataRange={"inp": [-1e-1, 1e-1]},
            RandomDataReaderInputShape={"input": [1, 3, 4, 4]},
        )
    elif abnormal_type == "dtype":
        quant_config = QConfig(
            global_config=QLayerConfig(activation=UInt8Spec(), weight=Int8Spec()),
            UseRandomData=True,
            RandomDataReaderInputDataRange={"input": [-1e-1, 1e-1]},
            RandomDataReaderInputShape=[1, 3, 4, 4],
        )
    else:
        raise NotImplementedError(f"Unspecified abnormal type {abnormal_type}!")
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


def infer_quantized_model(quantized_model_path, input_dtype="float"):
    sess = onnxruntime.InferenceSession(quantized_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    if input_dtype == "float":
        input_data = input_tensor
    elif input_dtype == "int8":
        input_data = int8_input_tensor
    elif input_dtype == "uint8":
        input_data = uint8_input_tensor
    else:
        raise NotImplementedError(f"Unspecified input dtype {input_dtype}!")
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(output_dir: str):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_random_input_with_specific(output_dir: str, input_dtype="float"):
    input_model_path, output_model_path = prepare_model(output_dir=output_dir, input_dtype=input_dtype)
    quant_config = prepare_random_input_config_with_specific(input_dtype=input_dtype)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, None)
    output = infer_quantized_model(quantized_model_path, input_dtype=input_dtype)
    return output


def tensor_quantize_random_input_with_coarse(output_dir: str):
    input_model_path, output_model_path = prepare_model(output_dir)
    quant_config = prepare_random_input_config_with_coarse()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, None)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_random_input_with_abnormal(output_dir: str, abnormal_type="shape_key"):
    input_model_path, output_model_path = prepare_model(output_dir)
    quant_config = prepare_abnormal_random_input_config(abnormal_type=abnormal_type)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, None)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize(self, tmpdir: str):
        self.assertEqual(tensor_quantize(tmpdir), np.array([[-0.9733977]], dtype=np.float32))

    @use_temporary_directory
    def test_quantize_random_input_with_specific(self, tmpdir: str):
        comp_equal = np.allclose(
            tensor_quantize_random_input_with_specific(output_dir=tmpdir, input_dtype="float"),
            np.array([[-0.9953757]]),
            atol=1e-1,
        )
        self.assertEqual(comp_equal, True)
        delete_directory_content(tmpdir)

        comp_equal = np.allclose(
            tensor_quantize_random_input_with_specific(output_dir=tmpdir, input_dtype="int8"),
            np.array([[-1.9877828]]),
            atol=1e-1,
        )
        self.assertEqual(comp_equal, True)
        delete_directory_content(tmpdir)

        comp_equal = np.allclose(
            tensor_quantize_random_input_with_specific(output_dir=tmpdir, input_dtype="uint8"),
            np.array([[4.504177]]),
            atol=1e-1,
        )
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_random_input_with_coarse(self, tmpdir: str):
        comp_equal = np.allclose(
            tensor_quantize_random_input_with_coarse(output_dir=tmpdir), np.array([[-0.9756757]]), atol=1e-1
        )
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_random_input_with_abnormal(self, tmpdir: str):
        with self.assertRaises(ValueError) as context:
            tensor_quantize_random_input_with_abnormal(output_dir=tmpdir, abnormal_type="shape_key")
        self.assertIn(
            ' Please check whether the parameter config.global_quant_config.extra_options["RandomDataReaderInputShape"] is correct.',
            str(context.exception),
        )
        delete_directory_content(tmpdir)

        with self.assertRaises(ValueError) as context:
            tensor_quantize_random_input_with_abnormal(output_dir=tmpdir, abnormal_type="range_key")
        self.assertIn(
            ' Please check whether the parameter config.global_quant_config.extra_options["RandomDataReaderInputDataRange"] is correct.',
            str(context.exception),
        )
        delete_directory_content(tmpdir)

        with self.assertRaises(TypeError) as context:
            tensor_quantize_random_input_with_abnormal(output_dir=tmpdir, abnormal_type="dtype")
        self.assertIn("The RandomDataReaderInputShape must be a Dict[str, List[int]]", str(context.exception))


if __name__ == "__main__":
    unittest.main()
