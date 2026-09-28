#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import os
import unittest

import numpy as np
import onnxruntime
from onnx_testing_utils import prepare_model
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import Config, ModelQuantizer, get_library_path
from quark.onnx.quantization.config.custom_config import S16S16_MIXED_S8S8_CONFIG
from quark.onnx.quantization.config.spec import BFloat16Spec, BFP16Spec, Float16Spec, MXInt8Spec, QLayerConfig

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

output_tensor = np.array(
    [
        [
            [
                [0.25217173, 0.16095717, 0.02391667, -0.04811294],
                [0.14599492, -0.01746433, 0.13771442, 0.0375311],
                [0.12354795, 0.26865387, 0.42113698, 0.11375473],
                [0.15047571, 0.13807288, 0.23625597, 0.28587446],
            ]
        ]
    ]
).astype(np.float32)


output_tensor_int16_mixed_fp16 = np.array(
    [
        [
            [
                [0.25048208, 0.16028333, 0.02336582, -0.04864665],
                [0.1463641, -0.01857184, 0.1368404, 0.03836467],
                [0.12164877, 0.26806426, 0.42113736, 0.11401439],
                [0.14929447, 0.1368404, 0.23535469, 0.28589067],
            ]
        ]
    ]
).astype(np.float32)


output_tensor_int16_mixed_bfp16 = np.array(
    [
        [
            [
                [0.25390625, 0.16210938, 0.02612305, -0.04785156],
                [0.1484375, -0.0168457, 0.13867188, 0.0390625],
                [0.12304688, 0.2734375, 0.421875, 0.11425781],
                [0.15039062, 0.13867188, 0.23632812, 0.2890625],
            ]
        ]
    ]
).astype(np.float32)


output_tensor_int16_mixed_fp16_and_bf16_and_bfp16 = np.array(
    [
        [
            [
                [0.25048208, 0.16028333, 0.02336582, -0.04864665],
                [0.1463641, -0.01857184, 0.1368404, 0.03836467],
                [0.12164877, 0.26806426, 0.42113736, 0.11401439],
                [0.14929447, 0.1368404, 0.23535469, 0.28589067],
            ]
        ]
    ]
).astype(np.float32)


output_tensor_int16_mixed_mxint8 = np.array(
    [
        [
            [
                [0.25172725, 0.16035496, 0.0248415, -0.04841405],
                [0.14599492, -0.01881215, 0.13760687, 0.03796843],
                [0.12227899, 0.26949984, 0.42113698, 0.11285857],
                [0.14995235, 0.13673222, 0.23390445, 0.28601784],
            ]
        ]
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


def prepare_config():
    config_copy = copy.deepcopy(S16S16_MIXED_S8S8_CONFIG)
    config_copy.extra_options["AutoMixprecision"]["MetricThreshold"] = 10000
    config_copy.extra_options["AutoMixprecision"]["SharedParamMode"] = "unshare"
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_config_int16_mixed_fp16():
    config_copy = copy.deepcopy(S16S16_MIXED_S8S8_CONFIG)
    config_copy.extra_options["AutoMixprecision"] = {
        "TargetLayerConfig": QLayerConfig(input_tensors=Float16Spec()),
        "DualQuantNodes": True,
    }
    config_copy.extra_options["ActivationSymmetric"] = True
    config_copy.extra_options["WeightSymmetric"] = True
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_config_int16_mixed_bfp16():
    config_copy = copy.deepcopy(S16S16_MIXED_S8S8_CONFIG)
    config_copy.extra_options["AutoMixprecision"] = {
        "TargetLayerConfig": {
            "input_tensors": {
                "data_type": "BFP16",
                "scale_type": "ScaleType.Float32",
                "calibration_method": "CalibMethod.MinMax",
                "quant_granularity": "QuantGranularity.Tensor",
                "symmetric": True,
            },
            "weight": {
                "data_type": "BFP16",
                "scale_type": "ScaleType.Float32",
                "calibration_method": "CalibMethod.MinMax",
                "quant_granularity": "QuantGranularity.Tensor",
                "symmetric": True,
            },
            "bias": {
                "data_type": "BFP16",
                "scale_type": "ScaleType.Float32",
                "calibration_method": "CalibMethod.MinMax",
                "quant_granularity": "QuantGranularity.Tensor",
                "symmetric": True,
            },
            "output_tensors": {
                "data_type": "BFP16",
                "scale_type": "ScaleType.Float32",
                "calibration_method": "CalibMethod.MinMax",
                "quant_granularity": "QuantGranularity.Tensor",
                "symmetric": True,
            },
        }
    }
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_config_int16_mixed_fp16_and_bf16_and_bfp16():
    config_copy = copy.deepcopy(S16S16_MIXED_S8S8_CONFIG)
    config_copy.extra_options["AutoMixprecision"] = {
        "TargetLayerConfig": [
            QLayerConfig(input_tensors=Float16Spec()),
            QLayerConfig(input_tensors=BFloat16Spec()),
            QLayerConfig(input_tensors=BFP16Spec()),
        ],
        "DualQuantNodes": True,
    }
    config_copy.extra_options["ActivationSymmetric"] = True
    config_copy.extra_options["WeightSymmetric"] = True
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_config_int16_mixed_mxint8_with_cache(output_dir: str):
    config_copy = copy.deepcopy(S16S16_MIXED_S8S8_CONFIG)
    config_copy.extra_options["AutoMixprecision"] = {
        "TargetLayerConfig": QLayerConfig(input_tensors=MXInt8Spec()),
        "SensitivityCacheFile": os.path.join(output_dir, "sensitivity_cache.json"),
    }
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
    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path())
    sess = onnxruntime.InferenceSession(quantized_model_path, so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(output_dir):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_int16_mixed_fp16(output_dir):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config_int16_mixed_fp16()
    ModelQuantizer(quant_config).quantize_model(input_model_path, output_model_path, data_reader)
    return infer_quantized_model(output_model_path)


def tensor_quantize_int16_mixed_bfp16(output_dir):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config_int16_mixed_bfp16()
    ModelQuantizer(quant_config).quantize_model(input_model_path, output_model_path, data_reader)
    return infer_quantized_model(output_model_path)


def tensor_quantize_int16_mixed_fp16_and_bf16_and_bfp16(output_dir):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config_int16_mixed_fp16_and_bf16_and_bfp16()
    ModelQuantizer(quant_config).quantize_model(input_model_path, output_model_path, data_reader)
    return infer_quantized_model(output_model_path)


def tensor_quantize_int16_mixed_mxint8_with_cache(output_dir: str):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config_int16_mixed_mxint8_with_cache(output_dir)
    ModelQuantizer(quant_config).quantize_model(input_model_path, output_model_path, data_reader)
    # Run the quantization again to test loading the cache file
    data_reader.rewind()
    quant_config.global_quant_config.extra_options["AutoMixprecision"]["MetricThreshold"] = 0.003
    ModelQuantizer(quant_config).quantize_model(input_model_path, output_model_path, data_reader)
    # Run the quantization again to test a different metric optimize object
    data_reader.rewind()
    quant_config.global_quant_config.extra_options["AutoMixprecision"]["MetricOptimizeObject"] = "quality"
    ModelQuantizer(quant_config).quantize_model(input_model_path, output_model_path, data_reader)
    # Run the quantization again with a extremely low metric threshold
    data_reader.rewind()
    quant_config.global_quant_config.extra_options["AutoMixprecision"]["MetricOptimizeObject"] = "speed"
    quant_config.global_quant_config.extra_options["AutoMixprecision"]["MetricThreshold"] = 0.0001
    ModelQuantizer(quant_config).quantize_model(input_model_path, output_model_path, data_reader)
    # Run the quantization again with a reasonable metric threshold
    data_reader.rewind()
    quant_config.global_quant_config.extra_options["AutoMixprecision"]["MetricThreshold"] = 0.004
    ModelQuantizer(quant_config).quantize_model(input_model_path, output_model_path, data_reader)
    return infer_quantized_model(output_model_path)


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_mix_precision(self, tmpdir: str):
        output = tensor_quantize(tmpdir)
        comp_equal = np.allclose(output, output_tensor, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_int16_mixed_fp16(self, tmpdir: str):
        output = tensor_quantize_int16_mixed_fp16(tmpdir)
        comp_equal = np.allclose(output, output_tensor_int16_mixed_fp16, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_int16_mixed_bfp16(self, tmpdir: str):
        output = tensor_quantize_int16_mixed_bfp16(tmpdir)
        comp_equal = np.allclose(output, output_tensor_int16_mixed_bfp16, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_int16_mixed_fp16_and_bf16_and_bfp16(self, tmpdir: str):
        output = tensor_quantize_int16_mixed_fp16_and_bf16_and_bfp16(tmpdir)
        comp_equal = np.allclose(output, output_tensor_int16_mixed_fp16_and_bf16_and_bfp16, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_int16_mixed_mxint8_with_cache(self, tmpdir: str):
        output = tensor_quantize_int16_mixed_mxint8_with_cache(tmpdir)
        comp_equal = np.allclose(output, output_tensor_int16_mixed_mxint8, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)


if __name__ == "__main__":
    unittest.main()
