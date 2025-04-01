#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
import numpy as np
import onnxruntime
import copy
import torch
from pathlib import Path
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod
from quark.onnx import ModelQuantizer, VitisQuantFormat, VitisQuantType
from quark.onnx.quantization.config.custom_config import BF16_MXINT8_CONFIG, BF16_MIXED_BFP16_ADAQUANT_CONFIG
from quark.onnx.quantization.config.config import Config, QuantizationConfig
from quark.shares.utils.testing_utils import use_temporary_directory

input_tensor = np.array([[[[0.26921557, 0.79500909, 0.6102178, 0.04375664],
                           [0.06221361, 0.98258356, 0.38635129, 0.06492238],
                           [0.49631707, 0.35442799, 0.51719146, 0.52100111],
                           [0.04145599, 0.88960236, 0.50627326, 0.57204613]],
                          [[0.99185097, 0.93582153, 0.13174529, 0.42896287],
                           [0.14552133, 0.02538564, 0.0732355, 0.25725371],
                           [0.09856916, 0.43015628, 0.55679755, 0.66560074],
                           [0.9439425, 0.45701841, 0.86791293, 0.64728276]],
                          [[0.29159685, 0.79021383, 0.3117182, 0.11342342],
                           [0.16660495, 0.46426165, 0.31348552, 0.143383],
                           [0.96454802, 0.63258874, 0.30295267, 0.96720039],
                           [0.29879457, 0.79916527, 0.02905061, 0.20115725]]]]).astype(np.float32)

elementwise_mp_output_tensor = np.array([[[[-0.00390625, -0.07128906, -0.05175781, -0.1328125],
                                           [-0.03662109, -0.06787109, -0.17871094, -0.1328125],
                                           [-0.00634766, -0.07177734, -0.05639648, -0.12255859],
                                           [-0.0625, -0.0859375, -0.17675781, -0.10009766]]]],).astype(np.float32)

layerwise_mp_output_tensor = np.array([[[[-0.00244141, -0.0703125, -0.05126953, -0.13183594],
                                         [-0.03613281, -0.06884766, -0.17871094, -0.1328125],
                                         [-0.00634766, -0.06982422, -0.05615234, -0.12207031],
                                         [-0.06347656, -0.08740234, -0.17578125, -0.10107422]]]],).astype(np.float32)

tensorwise_mp_output_tensor = np.array([[[[-0.00390625, -0.0703125, -0.05126953, -0.13183594],
                                          [-0.03613281, -0.06640625, -0.17871094, -0.13378906],
                                          [-0.00683594, -0.07128906, -0.05615234, -0.12158203],
                                          [-0.06298828, -0.08496094, -0.17578125, -0.09960938]]]],).astype(np.float32)

MXandBFP_standard_mp_output_tensor = np.array([[[[-0.00292969, -0.07226562, -0.05224609, -0.1328125],
                                                 [-0.03808594, -0.06835938, -0.1796875, -0.13476562],
                                                 [-0.00683594, -0.07128906, -0.05566406, -0.12207031],
                                                 [-0.0625, -0.08789062, -0.17578125, -0.09960938]]]],).astype(np.float32)

MXandBFP_dedicate_mp_output_tensor = np.array([[[[-0.00292969, -0.07128906, -0.05273438, -0.1328125],
                                                 [-0.03710938, -0.06738281, -0.1796875, -0.1328125],
                                                 [-0.00683594, -0.07324219, -0.05517578, -0.12402344],
                                                 [-0.06201172, -0.08789062, -0.17578125, -0.09960938]]]],).astype(np.float32)

MXandInt16_standard_mp_output_tensor = np.array([[[[-0.00390625, -0.0703125, -0.05175781, -0.13085938],
                                                   [-0.03808594, -0.06738281, -0.1796875, -0.1328125],
                                                   [-0.00683594, -0.07128906, -0.05664062, -0.12207031],
                                                   [-0.06347656, -0.08789062, -0.17578125, -0.1015625]]]],).astype(np.float32)

MXandInt16_dedicate_mp_output_tensor = np.array([[[[-0.00392081, -0.07062642, -0.05202574, -0.13329943],
                                                   [-0.03690144, -0.06802527, -0.17868596, -0.13241057],
                                                   [-0.00637473, -0.07137077, -0.05635554, -0.12234133],
                                                   [-0.06247669, -0.08598793, -0.17703912, -0.10038424]]]],).astype(np.float32)

MXandInt8_standard_mp_output_tensor = np.array([[[[-0.00350365, -0.07007293, -0.05115324, -0.13243783],
                                                  [-0.03503646, -0.06516782, -0.17728451, -0.13103637],
                                                  [-0.00280292, -0.0693722, -0.05395615, -0.12192689],
                                                  [-0.06166418, -0.08408751, -0.17518231, -0.09950355]]]],).astype(np.float32)

MXandInt8_dedicate_mp_output_tensor = np.array([[[[-0.00350365, -0.07007293, -0.05115324, -0.13243783],
                                                  [-0.03503646, -0.06516782, -0.17728451, -0.13103637],
                                                  [-0.00280292, -0.07007293, -0.05465688, -0.12192689],
                                                  [-0.06166418, -0.08408751, -0.17588304, -0.09950355]]]],).astype(np.float32)

MXandInt8_custom_mp_output_tensor = np.array([[[[-0.00350365, -0.07007293, -0.05115324, -0.13243783],
                                                [-0.03503646, -0.06516782, -0.17728451, -0.13103637],
                                                [-0.00280292, -0.0693722, -0.05395615, -0.12192689],
                                                [-0.06166418, -0.08408751, -0.17518231, -0.09950355]]]],).astype(np.float32)

MXQOperator_custom_mp_output_tensor = np.array([[[[-0.00350365, -0.07007293, -0.05115324, -0.13243783],
                                                  [-0.03503646, -0.06516782, -0.17728451, -0.13103637],
                                                  [-0.00280292, -0.0693722, -0.05395615, -0.12192689],
                                                  [-0.06166418, -0.08408751, -0.17518231, -0.09950355]]]],).astype(np.float32)


class ConvsModel(torch.nn.Module):

    def __init__(self):
        super(ConvsModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv21 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv22 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x1 = self.conv21(x)
        x2 = self.conv22(x)
        y = x1 + x2
        return y


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = ConvsModel()

    dummy_input = torch.randn(1, 3, 4, 4)

    onnx_model_path = Path(output_dir, 'simple_conv_model.onnx').as_posix()
    quant_onnx_model_path = Path(output_dir, 'simple_conv_model_quantized.onnx').as_posix()

    torch.onnx.export(model,
                      dummy_input,
                      onnx_model_path,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=17)

    print(f'Model has been saved to {onnx_model_path}')
    return onnx_model_path, quant_onnx_model_path


class DataReader(CalibrationDataReader):

    def __init__(self, input_tensor):
        self.data = [input_tensor]
        self.input_name = 'input'
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
    config_copy = copy.deepcopy(BF16_MXINT8_CONFIG)
    config_copy.extra_options['AddQDQPairToWeight'] = False
    return Config(global_quant_config=config_copy)


def prepare_layerwise_config():
    return Config(global_quant_config=BF16_MIXED_BFP16_ADAQUANT_CONFIG)


def prepare_tensorwise_config():
    quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                      quant_format=VitisQuantFormat.QDQ,
                                      activation_type=VitisQuantType.QBFloat16,
                                      weight_type=VitisQuantType.QBFloat16,
                                      specific_tensor_precision=True,
                                      extra_options={
                                          'MixedPrecisionTensor': {
                                              VitisQuantType.QMX: ['/conv1/Conv_output_0']  # This is a specific name
                                          },
                                          'MXAttributes': {
                                              'element_dtype': 'int8',
                                              'axis': 1,
                                              'block_size': 32,
                                              'rounding_mode': 2,
                                          },
                                      })

    return Config(global_quant_config=quant_config)


def prepare_MXandBFP_standard_config():
    quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                      quant_format=VitisQuantFormat.QDQ,
                                      activation_type=VitisQuantType.QMX,
                                      weight_type=VitisQuantType.QBFP,
                                      extra_options={
                                          'AddQDQPairToWeight': False
                                      })

    return Config(global_quant_config=quant_config)


def prepare_MXandBFP_dedicate_config():
    quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                      quant_format=VitisQuantFormat.QDQ,
                                      activation_type=VitisQuantType.QBFP,
                                      weight_type=VitisQuantType.QMX,
                                      extra_options={
                                          'AddQDQPairToWeight': False,
                                          'DedicatedQDQPair': True,
                                      })

    return Config(global_quant_config=quant_config)


def prepare_MXandInt16_standard_config():
    quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                      quant_format=VitisQuantFormat.QDQ,
                                      activation_type=VitisQuantType.QMX,
                                      weight_type=VitisQuantType.QInt16,
                                      per_channel=True,
                                      extra_options={
                                          'AddQDQPairToWeight': False
                                      })

    return Config(global_quant_config=quant_config)


def prepare_MXandInt16_dedicate_config():
    quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                      quant_format=VitisQuantFormat.QDQ,
                                      activation_type=VitisQuantType.QInt16,
                                      weight_type=VitisQuantType.QMX,
                                      # per_channel=True,
                                      extra_options={
                                          'AddQDQPairToWeight': False,
                                          'DedicatedQDQPair': True,
                                      })

    return Config(global_quant_config=quant_config)


def prepare_MXandInt8_standard_config():
    quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                      quant_format=VitisQuantFormat.QDQ,
                                      activation_type=VitisQuantType.QUInt8,
                                      weight_type=VitisQuantType.QInt8,
                                      specific_tensor_precision=True,
                                      extra_options={
                                          'MixedPrecisionTensor': {
                                              VitisQuantType.QMX: ['conv1.weight']  # This is a specific name
                                          },
                                      })

    return Config(global_quant_config=quant_config)


def prepare_MXandInt8_dedicate_config():
    quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                      quant_format=VitisQuantFormat.QDQ,
                                      activation_type=VitisQuantType.QUInt8,
                                      weight_type=VitisQuantType.QInt8,
                                      specific_tensor_precision=True,
                                      extra_options={
                                          'AddQDQPairToWeight': True,
                                          'DedicatedQDQPair': True,
                                          'MixedPrecisionTensor': {
                                              VitisQuantType.QMX: ['conv1.weight']  # This is a specific name
                                          },
                                      })

    return Config(global_quant_config=quant_config)


def prepare_MXandInt8_custom_config():
    quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                      quant_format=VitisQuantFormat.QDQ,
                                      activation_type=VitisQuantType.QUInt8,
                                      weight_type=VitisQuantType.QInt8,
                                      specific_tensor_precision=True,
                                      extra_options={
                                          'Int32Bias': False,
                                          'UseQDQVitisCustomOps': False,
                                          'MixedPrecisionTensor': {
                                              VitisQuantType.QMX: ['conv1.bias']  # This is a specific name
                                          },
                                      })

    return Config(global_quant_config=quant_config)


def prepare_MXQOperator_custom_config():
    from quark.onnx import QuantFormat, PowerOfTwoMethod

    quant_config = QuantizationConfig(calibrate_method=PowerOfTwoMethod.NonOverflow,
                                      quant_format=QuantFormat.QOperator,
                                      activation_type=VitisQuantType.QInt16,
                                      weight_type=VitisQuantType.QMX,
                                      extra_options={
                                          'AddQDQPairToWeight': False,
                                      })

    return Config(global_quant_config=quant_config)


def prepare_data():
    data_reader = DataReader(input_tensor)
    return data_reader


def prepare_quantizer(quant_config):
    quantizer = ModelQuantizer(quant_config)
    return quantizer


def quantize_static(quantizer, input_model_path, output_model_path, data_reader):
    quantizer.quantize_model(input_model_path, output_model_path, data_reader)
    print('Quantized the ONNX model and saved it at:', output_model_path)
    return output_model_path


def infer_quantized_model(quantized_model_path):
    so = onnxruntime.SessionOptions()
    from quark.onnx import get_library_path as vai_lib_path
    so.register_custom_ops_library(vai_lib_path())
    # from onnxruntime_extensions import get_library_path as ext_lib_path
    # so.register_custom_ops_library(ext_lib_path())
    sess = onnxruntime.InferenceSession(quantized_model_path, so, providers=['CPUExecutionProvider'])

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f'Model output: {output}')
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
    def test_quantize_MXandBFP_standard_mix_precision(self, tmpdir: str):
        quant_config = prepare_MXandBFP_standard_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, MXandBFP_standard_mp_output_tensor, atol=1e-2)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXandBFP_dedicate_mix_precision(self, tmpdir: str):
        quant_config = prepare_MXandBFP_dedicate_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, MXandBFP_dedicate_mp_output_tensor, atol=1e-2)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXandInt16_standard_mix_precision(self, tmpdir: str):
        quant_config = prepare_MXandInt16_standard_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, MXandInt16_standard_mp_output_tensor, atol=1e-2)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXandInt16_dedicate_mix_precision(self, tmpdir: str):
        quant_config = prepare_MXandInt16_dedicate_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, MXandInt16_dedicate_mp_output_tensor, atol=1e-2)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXandInt8_standard_mix_precision(self, tmpdir: str):
        quant_config = prepare_MXandInt8_standard_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, MXandInt8_standard_mp_output_tensor, atol=1e-2)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXandInt8_dedicate_mix_precision(self, tmpdir: str):
        quant_config = prepare_MXandInt8_dedicate_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, MXandInt8_dedicate_mp_output_tensor, atol=1e-2)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_MXandInt8_custom_mix_precision(self, tmpdir: str):
        quant_config = prepare_MXandInt8_custom_config()
        try:
            output = tensor_quantize(tmpdir, quant_config)
            comp_equal = np.allclose(output, MXandInt8_custom_mp_output_tensor, atol=1e-2)
            self.assertEqual(np.all(comp_equal), True)
        except Exception as e:
            # TODO: Support running ai.onnx.contrib Q or DQ
            print("This config will use ai.onnx.contrib Q or DQ, which has no implementation yet")

    @use_temporary_directory
    def test_quantize_MXQOperator_custom_mix_precision(self, tmpdir: str):
        quant_config = prepare_MXQOperator_custom_config()
        try:
            output = tensor_quantize(tmpdir, quant_config)
            comp_equal = np.allclose(output, MXQOperator_custom_mp_output_tensor, atol=1e-2)
            self.assertEqual(np.all(comp_equal), True)
        except Exception as e:
            # TODO: Support running the customized QOperator
            print("This config will generate customized QOperator, which has no implementation yet")


if __name__ == '__main__':
    unittest.main()
