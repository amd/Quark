#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import unittest
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import yaml
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.experimental.cli.main import main as cli
from quark.onnx import BFloat16Spec, ModelQuantizer, QConfig, QLayerConfig

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


class DataReader(CalibrationDataReader):
    """
    A CalibrationDataReader implementation for ONNX model calibration.

    This class provides input data for calibrating ONNX models during quantization.
    It wraps input tensors and iterates through them for the calibration process.

    Attributes:
        data (list): List containing the input tensor for calibration.
        input_name (str): Name of the input tensor, defaults to 'input'.
        index (int): Current position in the data iteration.
    """

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
    """
    A simple CNN model for testing purposes.

    This model consists of a sequential block containing:
    - Conv2d layer: 3 input channels, 8 output channels, 3x3 kernel, stride=1, padding=1
    - ReLU activation function
    """

    def __init__(self):
        super().__init__()
        self.conv_relu = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), nn.ReLU())

    def forward(self, x):
        x = self.conv_relu(x)
        return x


def prepare_float_onnx_model(output_dir):
    """
    Prepare and export a float ONNX model.

    This function creates a simple convolutional neural network model, exports it to ONNX format,
    and saves it to the specified output directory.

    Args:
        output_dir: Directory path where the ONNX model will be saved.

    Returns:
        str: Path to the saved ONNX model file.
    """
    torch.manual_seed(42)

    model = Model()
    onnx_model_path = Path(output_dir, "onnx_input.onnx").as_posix()

    dummy_input = torch.randn([1, 3, 5, 5])
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


def quantize_onnx_model(output_dir: str, input_data):
    """
    Quantize an ONNX model using BFloat16 quantization.

    Args:
        output_dir (str): Directory path where the quantized models will be saved.
        input_data: Calibration input data for quantization.

    Returns:
        tuple: A tuple containing:
            - quantized_model_path (str): Path to the quantized ONNX model with QDQ nodes.
            - replace_bfloat16_qdq_with_cast_model_path (str): Path for the model with cast operations.
    """
    quantized_model_path = Path(output_dir, "onnx_quantized_qdq.onnx").as_posix()
    replace_bfloat16_qdq_with_cast_model_path = Path(output_dir, "replace_bfloat16_qdq_with_cast_model.onnx").as_posix()
    data_reader = DataReader(input_data)
    input_model_path = prepare_float_onnx_model(output_dir)
    quant_config = QConfig(
        global_config=QLayerConfig(activation=BFloat16Spec(), weight=BFloat16Spec()), BF16QDQToCast=False
    )
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(input_model_path, quantized_model_path, data_reader)
    return quantized_model_path, replace_bfloat16_qdq_with_cast_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    """
    Prepare a YAML configuration file for replacing BFloat16 QDQ nodes with Cast nodes.
    This function creates a YAML configuration file that specifies the input model path,
    optimization passes to apply, and the output model path for the Shapeshifter.
    Args:
        output_dir: Directory where the YAML file will be saved.
        onnx_model_path: Path to the input ONNX model.
        onnx_optimized_model_path: Path where the optimized ONNX model will be saved.
    Returns:
        str: Path to the created YAML configuration file.
    """
    yaml_path = Path(output_dir, "replace_bfloat16_qdq_with_cast.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_replace_bfloat16_qdq_with_cast": {
                "replace_bfloat16_qdq_with_cast": True,
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_model(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)

    cast_count = 0
    quant_count = 0
    dequant_count = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            cast_count += 1
        if node.op_type == "ExtendedQuantizeLinear":
            quant_count += 1
        if node.op_type == "ExtendedDequantizeLinear":
            dequant_count += 1

    return cast_count == 8 and quant_count == 0 and dequant_count == 0


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_replace_bfloat16_qdq_with_cast(self, tmpdir: str):
        quantized_model_path, replace_bfloat16_qdq_with_cast_model_path = quantize_onnx_model(tmpdir, input_data)
        yaml_path = prepare_yaml(tmpdir, quantized_model_path, replace_bfloat16_qdq_with_cast_model_path)
        cli(["shapeshifter", yaml_path])
        flag = check_model(replace_bfloat16_qdq_with_cast_model_path)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
