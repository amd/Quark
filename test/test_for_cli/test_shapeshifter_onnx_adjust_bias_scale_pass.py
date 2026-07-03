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
from onnxruntime.quantization.onnx_quantizer import tensor_proto_to_array

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
        """Initialize the DataReader with calibration input data.

        Args:
            input_tensor: The input tensor to be used for model calibration.
        """
        self.data = [input_tensor]
        self.input_name = "input"
        self.index = 0

    def get_next(self):
        """Get the next calibration data sample.

        Returns:
            dict or None: A dictionary mapping input name to the next data sample
                if available, otherwise None when all data has been consumed.
        """
        if self.index < len(self.data):
            input_dict = {self.input_name: self.data[self.index]}
            self.index += 1
            return input_dict
        else:
            return None

    def rewind(self):
        """Reset the data reader index to the beginning.

        This method allows the calibration data to be read again from the start
        by resetting the iteration index to 0.
        """
        self.index = 0


class Model(nn.Module):
    """
    A simple CNN model for testing purposes.

    This model consists of a sequential block containing:
    - Conv2d layer: 3 input channels, 8 output channels, 3x3 kernel, stride=1, padding=1
    - ReLU activation function
    """

    def __init__(self):
        """Initialize the Model with a convolutional layer.

        Creates a Conv2d layer with 3 input channels, 8 output channels,
        a 3x3 kernel, stride of 1, and padding of 1.
        """
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, height, width].

        Returns:
            torch.Tensor: Output tensor after applying convolution.
        """
        x = self.conv(x)
        return x


def prepare_float_onnx_model(onnx_model_path):
    """Prepare and export a float ONNX model for testing.

    This function creates a simple CNN model, generates dummy input data,
    and exports it to ONNX format. The model consists of a single Conv2d layer
    with 3 input channels and 8 output channels.

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
    activations and weights, with INT32 bias. Bias scale adjustment is
    disabled to allow testing of the bias scale adjustment pass.

    Args:
        input_data (np.ndarray): Calibration data for quantization.
        input_model_path (str): Path to the input float ONNX model.
        quantized_model_path (str): Path where the quantized model will be saved.
    """
    data_reader = DataReader(input_data)
    quant_config = QConfig(
        global_config=QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()),
        DebugMode=True,
        Int32Bias=True,
        AdjustBiasScale=False,
    )
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(input_model_path, quantized_model_path, data_reader)
    print("Quantized the ONNX model and saved it at:", quantized_model_path)


def prepare_yaml(yaml_path, onnx_model_path, onnx_optimized_model_path):
    """Prepare a YAML configuration file for the Shapeshifter CLI.

    This function creates a YAML configuration file that specifies the
    onnx_adjust_bias_scale pass with adjust_bias_scale enabled. The configuration
    is used by the CLI to apply the bias scale adjustment pass to the model.

    Args:
        yaml_path (str): Path where the YAML configuration file will be saved.
        onnx_model_path (str): Path to the input ONNX model.
        onnx_optimized_model_path (str): Path where the optimized model will be saved.
    """
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_adjust_bias_scale": {
                "adjust_bias_scale": True,
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def check_model(model_path, init_name, expected_scale):
    """Check if a model's initializer has the expected scale value.

    This function loads an ONNX model, finds the specified initializer by name,
    and verifies that its scale value matches the expected value. This is used
    to verify that bias scale adjustment has been applied correctly.

    Args:
        model_path (str): Path to the ONNX model file.
        init_name (str): Name of the initializer to check (e.g., "conv.bias_quantized_scale").
        expected_scale (float): The expected scale value.

    Returns:
        bool: True if the initializer's scale matches the expected value, False otherwise.
    """
    model = onnx.load(model_path)
    init = [i for i in model.graph.initializer if i.name == init_name][0]
    val = tensor_proto_to_array(init).tolist()
    scale = val[0] if isinstance(val, list) else val
    return scale == expected_scale


def modify_scale(origin_model_path, new_model_path, init_name, expected_scale):
    """Modify a scale initializer in an ONNX model.

    This function loads an ONNX model, finds the specified initializer by name,
    and replaces its value with a new scale value. This is used to create a test
    scenario where the bias scale does not match the expected value (activation_scale * weights_scale),
    allowing verification that the bias scale adjustment pass corrects it.

    Args:
        origin_model_path (str): Path to the original ONNX model.
        new_model_path (str): Path where the modified model will be saved.
        init_name (str): Name of the initializer to modify (e.g., "conv.bias_quantized_scale").
        expected_scale (float): The new scale value to set.
    """
    model = onnx.load(origin_model_path)
    init = [i for i in model.graph.initializer if i.name == init_name][0]
    new_scale_val = onnx.numpy_helper.from_array(np.array(expected_scale).astype(np.float32), name=init_name)
    init.CopyFrom(new_scale_val)
    onnx.save(model, new_model_path)


class TestTensorQuantize(unittest.TestCase):
    """Test cases for Shapeshifter passes related to quantization.

    This test class contains unit tests for various Shapeshifter passes,
    including bias scale adjustment and Q/DQ node removal operations.
    """

    @use_temporary_directory
    def test_onnx_adapter_onnx_adjust_bias_scale(self, tmpdir: str):
        """Test the onnx_adjust_bias_scale pass via CLI.

        This test verifies that the onnx_adjust_bias_scale pass correctly adjusts
        bias scales in QDQ quantized models. The test procedure:
        1. Creates a float ONNX model with a Conv2d layer
        2. Quantizes the model to INT8 with INT32 bias
        3. Verifies the initial bias scale value
        4. Modifies the bias scale to an incorrect value (1.0)
        5. Applies the onnx_adjust_bias_scale pass via CLI
        6. Verifies that the bias scale has been corrected to match
           activation_scale * weights_scale

        Args:
            tmpdir (str): Temporary directory path provided by the test fixture.
        """
        input_model_path = Path(tmpdir, "adjust_bias_scale_float.onnx").as_posix()
        quantized_model_path = Path(tmpdir, "adjust_bias_scale_quantized.onnx").as_posix()
        modify_scale_quantized_model_path = Path(tmpdir, "modify_scale_adjust_bias_scale_quantized.onnx").as_posix()
        adjust_bias_scale_model_path = Path(tmpdir, "adjust_bias_scale_model.onnx").as_posix()
        yaml_path = Path(tmpdir, "adjust_bias_scale.yaml").as_posix()

        # Step 1: Prepare float model
        prepare_float_onnx_model(input_model_path)

        # Step 2: Quantize the model
        quantize_onnx_model(input_data, input_model_path, quantized_model_path)

        # Step 3: Verify initial bias scale (should be activation_scale * weights_scale)
        flag = check_model(quantized_model_path, "conv.bias_quantized_scale", 1.52587890625e-05)
        self.assertTrue(flag, "Initial bias scale should match activation_scale * weights_scale")

        # Step 4: Modify bias scale to incorrect value to test the pass
        modify_scale(quantized_model_path, modify_scale_quantized_model_path, "conv.bias_quantized_scale", 1.0)
        flag = check_model(modify_scale_quantized_model_path, "conv.bias_quantized_scale", 1.0)
        self.assertTrue(flag, "Modified bias scale should be 1.0")

        # Step 5: Apply the onnx_adjust_bias_scale pass via CLI
        prepare_yaml(yaml_path, modify_scale_quantized_model_path, adjust_bias_scale_model_path)
        cli(["shapeshifter", yaml_path])

        # Step 6: Verify that bias scale has been corrected
        flag = check_model(adjust_bias_scale_model_path, "conv.bias_quantized_scale", 1.52587890625e-05)
        self.assertTrue(flag, "Bias scale should be adjusted back to activation_scale * weights_scale")


if __name__ == "__main__":
    unittest.main()
