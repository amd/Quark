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
from quark.onnx import ModelQuantizer, QConfig, QLayerConfig, XInt8Spec

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
        self.conv_leaky_relu = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.01)
        )
        self.conv_prelu = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), nn.PReLU())
        self.conv = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.ReLU()
        self.mul = nn.Linear(8 * 5 * 5, 10)

    def forward(self, x):
        x = self.conv_relu(x)
        x = self.conv_leaky_relu(x)
        x = self.conv_prelu(x)
        x = self.conv(x)
        x1 = self.relu1(x)
        x2 = self.relu2(x)
        x = x1 + x2
        x = x.view(x.size(0), -1)
        x = self.mul(x)
        return x


def prepare_float_onnx_model(onnx_model_path):
    """Prepare and export a float ONNX model for testing.

    This function creates a complex CNN model with multiple Conv layers and
    activation functions (ReLU, LeakyReLU, PReLU), followed by a linear layer.
    The model is designed to test Q/DQ node removal between specific operator pairs.

    Args:
        onnx_model_path (str): Path where the ONNX model will be saved.
    """
    torch.manual_seed(42)

    model = Model()

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


def quantize_onnx_model(input_data, input_model_path, quantized_model_path):
    """Quantize an ONNX model using PTQ (Post-Training Quantization).

    This function quantizes a float ONNX model to INT8 using the provided
    calibration data. The quantization configuration uses INT8 for both
    activations and weights. All Q/DQ removal options are disabled to ensure
    that Q/DQ nodes remain in the model for testing the removal pass.

    Args:
        input_data (np.ndarray): Calibration data for quantization.
        input_model_path (str): Path to the input float ONNX model.
        quantized_model_path (str): Path where the quantized model will be saved.
    """
    data_reader = DataReader(input_data)
    quant_config = QConfig(
        global_config=QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()),
        DebugMode=True,
        RemoveQDQConvRelu=False,
        RemoveQDQConvLeakyRelu=False,
        RemoveQDQConvPRelu=False,
        RemoveQDQMulAdd=False,
        RemoveQDQBetweenOps=[],
    )
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(input_model_path, quantized_model_path, data_reader)
    print("Quantized the ONNX model and saved it at:", quantized_model_path)


def prepare_yaml(yaml_path, onnx_model_path, onnx_optimized_model_path):
    """Prepare a YAML configuration file for the Shapeshifter CLI.

    This function creates a YAML configuration file that specifies the
    onnx_remove_qdq_between_op_types pass with operator type pairs between
    which Q/DQ nodes should be removed. The configuration is used by the CLI
    to apply the Q/DQ removal pass to the model.

    Args:
        yaml_path (str): Path where the YAML configuration file will be saved.
        onnx_model_path (str): Path to the input ONNX model.
        onnx_optimized_model_path (str): Path where the optimized model will be saved.
    """
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_remove_qdq_between_op_types": {
                "remove_qdq_between_op_types": [
                    ["Conv", "Relu"],
                    ["Conv", "LeakyRelu"],
                    ["Conv", "PRelu"],
                ],
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def check_model(onnx_optimized_model_path):
    """Count the number of QuantizeLinear and DequantizeLinear nodes in an ONNX model.

    This function loads an ONNX model and counts the number of QuantizeLinear
    and DequantizeLinear nodes. The counts can be used to verify that Q/DQ nodes
    have been removed correctly by the pass.

    Args:
        onnx_optimized_model_path (str): Path to the ONNX model file to be checked.

    Returns:
        tuple[int, int]: A tuple of (quantize_count, dequantize_count) representing
                         the number of QuantizeLinear and DequantizeLinear nodes
                         in the model, respectively.
    """
    model = onnx.load(onnx_optimized_model_path)
    quant_count = 0
    dequant_count = 0
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            quant_count += 1
        if node.op_type == "DequantizeLinear":
            dequant_count += 1
    return quant_count, dequant_count


class TestTensorQuantize(unittest.TestCase):
    """Test cases for Shapeshifter passes related to quantization.

    This test class contains unit tests for various Shapeshifter passes,
    including Q/DQ node removal between operator type pairs.
    """

    @use_temporary_directory
    def test_onnx_adapter_onnx_remove_qdq_between_op_types(self, tmpdir: str):
        """Test the onnx_remove_qdq_between_op_types pass via CLI.

        The test uses a model with multiple Conv layers followed by activation functions
        (ReLU, LeakyReLU, PReLU) to verify Q/DQ node removal.

        Test procedure:
        1. Creates a float ONNX model with multiple Conv layers and activation functions:
           - Conv+ReLU, Conv+LeakyReLU, Conv+PReLU sequences
        2. Quantizes the model to INT8 with all Q/DQ removal options disabled to ensure
           all Q/DQ nodes remain in the quantized model for testing
        3. Counts the initial Q/DQ nodes in the quantized model before applying the pass
        4. Applies the onnx_remove_qdq_between_op_types pass via CLI with configuration
           to remove Q/DQ nodes between the following operator pairs:
           - [Conv, Relu]
           - [Conv, LeakyRelu]
           - [Conv, PRelu]
        5. Counts the Q/DQ nodes after applying the pass and verifies that exactly
           3 QuantizeLinear nodes and 3 DequantizeLinear nodes have been removed

        The test validates that the pass correctly identifies and removes redundant
        Q/DQ operations between the specified operator pairs without affecting
        other Q/DQ nodes in the model.

        Args:
            tmpdir (str): Temporary directory path provided by the test fixture.
        """
        input_model_path = Path(tmpdir, "remove_qdq_between_ops_float.onnx").as_posix()
        quantized_model_path = Path(tmpdir, "remove_qdq_between_ops_quantized.onnx").as_posix()
        remove_qdq_model_path = Path(tmpdir, "remove_qdq_between_op_types_model.onnx").as_posix()
        yaml_path = Path(tmpdir, "remove_qdq_between_op_types.yaml").as_posix()

        # Step 1: Prepare float model with Conv layers and activation functions
        prepare_float_onnx_model(input_model_path)

        # Step 2: Quantize the model to INT8, keeping all Q/DQ nodes for testing
        quantize_onnx_model(input_data, input_model_path, quantized_model_path)

        # Step 3: Count initial Q/DQ nodes before applying the pass
        old_quant_count, old_dequant_count = check_model(quantized_model_path)

        # Step 4: Apply the onnx_remove_qdq_between_op_types pass via CLI
        # This will remove Q/DQ nodes between [Conv, Relu], [Conv, LeakyRelu],
        # and [Conv, PRelu] pairs
        prepare_yaml(yaml_path, quantized_model_path, remove_qdq_model_path)
        cli(["shapeshifter", yaml_path])

        # Step 5: Count Q/DQ nodes after applying the pass and verify removal
        new_quant_count, new_dequant_count = check_model(remove_qdq_model_path)

        # Verify that exactly 3 QuantizeLinear nodes were removed
        self.assertTrue(
            old_quant_count - new_quant_count == 3,
            f"Expected 3 QuantizeLinear nodes to be removed, but {old_quant_count - new_quant_count} were removed",
        )
        # Verify that exactly 3 DequantizeLinear nodes were removed
        self.assertTrue(
            old_dequant_count - new_dequant_count == 3,
            f"Expected 3 DequantizeLinear nodes to be removed, but {old_dequant_count - new_dequant_count} were removed",
        )


if __name__ == "__main__":
    unittest.main()
