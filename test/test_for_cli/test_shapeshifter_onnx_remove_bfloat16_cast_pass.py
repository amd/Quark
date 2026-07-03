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
                [0.26619988, 0.73333566, 0.32430612, 0.56555123],
                [0.50381943, 0.62112556, 0.78376413, 0.2894883],
                [0.28120838, 0.53861799, 0.83088573, 0.0888585],
                [0.80025317, 0.88537935, 0.42602682, 0.78531207],
            ],
            [
                [0.83338162, 0.5661508, 0.59231535, 0.28232884],
                [0.75736037, 0.12840651, 0.18621735, 0.85781309],
                [0.3070585, 0.03626074, 0.22557921, 0.2237572],
                [0.68366023, 0.25022015, 0.29810134, 0.60772729],
            ],
            [
                [0.20067241, 0.95934905, 0.86314381, 0.01692715],
                [0.24051579, 0.57178108, 0.57631192, 0.75122361],
                [0.35564212, 0.58467473, 0.58606206, 0.27266265],
                [0.7195592, 0.20194915, 0.90723205, 0.96791405],
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
        """
        Get the next calibration data item.

        This method retrieves the next input tensor for model calibration during iteration.
        It returns a dictionary containing the input data with the input name as the key.
        When all data has been consumed, it returns None.

        Returns:
            dict or None: A dictionary mapping input_name to the current data tensor,
                         or None if iteration is complete.
        """
        if self.index < len(self.data):
            input_dict = {self.input_name: self.data[self.index]}
            self.index += 1
            return input_dict
        else:
            return None

    def rewind(self):
        """
        Reset the iteration index to the beginning.

        This method allows the DataReader to be reused for multiple calibration
        passes by resetting the internal index counter to 0.
        """
        self.index = 0


class Model(nn.Module):
    """
    A simple convolutional neural network model.
    This model consists of a single 2D convolutional layer that transforms
    input tensors with 3 channels to output tensors with 8 channels.
    Attributes:
        conv (nn.Conv2d): Convolutional layer with kernel size 3, stride 1, and padding 1.
    """

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
        """
        Perform the forward pass computation.

        Args:
            x: Input tensor to be processed through the convolutional layer.

        Returns:
            Processed tensor after applying the convolutional operation.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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

    dummy_input = torch.randn([1, 3, 4, 4])
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


def quantize_onnx_model(output_dir, input_data):
    """
    Quantize an ONNX model using BFloat16 quantization.

    Args:
        output_dir: Directory path where the quantized models will be saved.
        input_data: Calibration input data for quantization.

    Returns:
        tuple: A tuple containing:
            - quantized_model_path: Path to the quantized ONNX model.
    """
    quantized_model_path = Path(output_dir, "onnx_quantized_qdq.onnx").as_posix()
    data_reader = DataReader(input_data)
    input_model_path = prepare_float_onnx_model(output_dir)
    quant_config = QConfig(
        global_config=QLayerConfig(activation=BFloat16Spec(), weight=BFloat16Spec()),
        BF16QDQToCast=True,
        EnableVaimlBF16=False,
    )
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(input_model_path, quantized_model_path, data_reader)
    return quantized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    """
    Prepare a YAML configuration file for remove BFloat16 Cast nodes.
    This function creates a YAML configuration file that specifies the input model path,
    optimization passes to apply, and the output model path for the Shapeshifter.
    Args:
        output_dir: Directory where the YAML file will be saved.
        onnx_model_path: Path to the input ONNX model.
        onnx_optimized_model_path: Path where the optimized ONNX model will be saved.
    Returns:
        str: Path to the created YAML configuration file.
    """
    yaml_path = Path(output_dir, "remove_bfloat16_cast.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_remove_bfloat16_cast": {
                "remove_bfloat16_cast": True,
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_model(onnx_optimized_model_path, expected_cast_count):
    """
    Verify that the ONNX model contains exactly 2 Clip nodes.

    This function loads an ONNX model and counts the number of Clip operation nodes
    in the model graph to verify the expected count.

    Args:
        onnx_optimized_model_path (str): Path to the ONNX model file to be checked.
        expected_cast_count (int): Expected number of Cast nodes in the model.

    Returns:
        bool: True if the model contains exactly 2 Clip nodes, False otherwise.
    """
    model = onnx.load(onnx_optimized_model_path)
    cast_count = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            cast_count += 1
    return cast_count == expected_cast_count


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_remove_bfloat16_cast(self, tmpdir: str):
        quantized_model_path = quantize_onnx_model(tmpdir, input_data)
        flag = check_model(quantized_model_path, 24)
        self.assertTrue(flag)
        remove_bfloat16_cast_model_path = Path(tmpdir, "remove_bfloat16_cast_model.onnx").as_posix()
        yaml_path = prepare_yaml(tmpdir, quantized_model_path, remove_bfloat16_cast_model_path)
        cli(["shapeshifter", yaml_path])
        flag = check_model(remove_bfloat16_cast_model_path, 2)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
