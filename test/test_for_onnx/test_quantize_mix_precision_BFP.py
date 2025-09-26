#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import unittest

import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod
from testing_utils import prepare_model

from quark.onnx import ExtendedQuantFormat, ExtendedQuantType, ModelQuantizer, VitisQuantFormat, VitisQuantType
from quark.onnx.quantization.config.config import Config, QuantizationConfig
from quark.onnx.quantization.config.custom_config import BF16_BFP16_CONFIG, BF16_MIXED_BFP16_CONFIG
from quark.shares.utils.testing_utils import use_temporary_directory

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

elementwise_mp_output_tensor = np.array(
    [
        [
            [
                [0.25195312, 0.16210938, 0.02490234, -0.04833984],
                [0.1484375, -0.01635742, 0.13867188, 0.03881836],
                [0.12255859, 0.27148438, 0.421875, 0.11474609],
                [0.15039062, 0.13867188, 0.23730469, 0.28710938],
            ]
        ]
    ],
).astype(np.float32)

layerwise_mp_output_tensor = np.array(
    [
        [
            [
                [0.25195312, 0.16113281, 0.02490234, -0.04785156],
                [0.1484375, -0.01867676, 0.13867188, 0.0378418],
                [0.12158203, 0.2734375, 0.421875, 0.11523438],
                [0.15039062, 0.13769531, 0.23730469, 0.28515625],
            ]
        ]
    ],
).astype(np.float32)

tensorwise_mp_output_tensor = np.array(
    [
        [
            [
                [0.25195312, 0.16210938, 0.02490234, -0.04833984],
                [0.1484375, -0.01647949, 0.13867188, 0.03881836],
                [0.12255859, 0.27148438, 0.421875, 0.11474609],
                [0.15039062, 0.13867188, 0.23730469, 0.28710938],
            ]
        ]
    ],
).astype(np.float32)

BFPandMX_mp_output_tensor = np.array(
    [
        [
            [
                [0.25390625, 0.16210938, 0.02612305, -0.04785156],
                [0.1484375, -0.0168457, 0.13867188, 0.0390625],
                [0.12304688, 0.2734375, 0.421875, 0.11425781],
                [0.15039062, 0.13867188, 0.23632812, 0.2890625],
            ]
        ]
    ],
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


def prepare_elementwise_config():
    config_copy = copy.deepcopy(BF16_BFP16_CONFIG)
    config_copy.extra_options["AddQDQPairToWeight"] = False
    return Config(global_quant_config=config_copy)


def prepare_layerwise_config():
    return Config(global_quant_config=BF16_MIXED_BFP16_CONFIG)


def prepare_tensorwise_config():
    quant_config = QuantizationConfig(
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=ExtendedQuantFormat.QDQ,
        activation_type=ExtendedQuantType.QBFloat16,
        weight_type=ExtendedQuantType.QBFloat16,
        specific_tensor_precision=True,
        extra_options={
            "MixedPrecisionTensor": {
                ExtendedQuantType.QBFP: ["conv.weight"]  # This is a specific name
            },
            "BFPAttributes": {
                "bfp_method": "to_bfp",
                "axis": 1,
                "bit_width": 16,
                "block_size": 8,
                "rounding_mode": 2,
            },
        },
    )

    return Config(global_quant_config=quant_config)


def prepare_BFPandMX_config():
    quant_config = QuantizationConfig(
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=ExtendedQuantFormat.QDQ,
        activation_type=ExtendedQuantType.QBFP,
        weight_type=ExtendedQuantType.QMX,
        extra_options={"AddQDQPairToWeight": False},
    )

    return Config(global_quant_config=quant_config)


def prepare_BFPandMX_compatibility_config():
    quant_config = QuantizationConfig(
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=ExtendedQuantFormat.QDQ,
        activation_type=VitisQuantType.QBFP,  # Test downward compatibility
        weight_type=VitisQuantType.QMX,  # Test downward compatibility
        extra_options={"AddQDQPairToWeight": False},
    )

    return Config(global_quant_config=quant_config)


def prepare_BFP_compatibility_config():
    quant_config = QuantizationConfig(
        calibrate_method=CalibrationMethod.MinMax, quant_format=VitisQuantFormat.BFPFixNeuron
    )  # Test downward compatibility

    return Config(global_quant_config=quant_config)


def prepare_MX_compatibility_config():
    quant_config = QuantizationConfig(
        calibrate_method=CalibrationMethod.MinMax, quant_format=VitisQuantFormat.MXFixNeuron
    )  # Test downward compatibility

    return Config(global_quant_config=quant_config)


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
    so = onnxruntime.SessionOptions()
    from quark.onnx import get_library_path as vai_lib_path

    so.register_custom_ops_library(vai_lib_path())
    sess = onnxruntime.InferenceSession(quantized_model_path, so, providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(output_dir, quant_config):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_elementwise_mix_precision(self, tmpdir: str):
        quant_config = prepare_elementwise_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, elementwise_mp_output_tensor, atol=1e-2)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_layerwise_mix_precision(self, tmpdir: str):
        quant_config = prepare_layerwise_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, layerwise_mp_output_tensor, atol=1e-2)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_tensorwise_mix_precision(self, tmpdir: str):
        quant_config = prepare_tensorwise_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, tensorwise_mp_output_tensor, atol=1e-2)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_BFPandMX_mix_precision(self, tmpdir: str):
        quant_config = prepare_BFPandMX_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, BFPandMX_mp_output_tensor, atol=1e-2)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_BFPandMX_compatibility_mix_precision(self, tmpdir: str):
        quant_config = prepare_BFPandMX_compatibility_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, BFPandMX_mp_output_tensor, atol=1e-2)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_BFP_compatibility_mix_precision(self, tmpdir: str):
        quant_config = prepare_BFP_compatibility_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, BFPandMX_mp_output_tensor, atol=1e-2)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MX_compatibility_mix_precision(self, tmpdir: str):
        quant_config = prepare_MX_compatibility_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, BFPandMX_mp_output_tensor, atol=1e-2)
        self.assertEqual(np.all(comp_equal), True)


if __name__ == "__main__":
    unittest.main()
