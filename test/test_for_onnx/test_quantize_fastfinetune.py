#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
import numpy as np
import onnxruntime as ort
import copy
from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config.custom_config import U8U8_AAWA_CONFIG
from quark.onnx.quantization.config.config import Config
from quark.shares.utils.testing_utils import use_temporary_directory

import onnx
import torch
from pathlib import Path
from quark.onnx.finetuning.create_torch.base_qdq_quantizers import (default_round_func,
                                                                    int_quant_func, int_dequant_func, INTQuantDequantFunction,
                                                                    fp_quant_func, fp_dequant_func, fp_quant_dequant_func,
                                                                    INTQuantizer, FPQuantizer)
from quark.onnx.finetuning.create_torch.base_fn_quantizers import (BFPQuantizer, MXQuantizer)
from quark.onnx.finetuning.create_torch.quant_base_ops import (create_fn_quantizer, QuantizationModule)
from quark.onnx.quant_utils import COP_BFP_OP_NAME, COP_MX_OP_NAME

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


output_tensor = np.array([[[[-0.13814753, 0.34536883, -0.16577704, 1.2502352],
                            [-0.08288852, 0.26248032, 0.08288852, 0.91177374],
                            [-0.09670328, -0.09670328, -0.1519623, 1.5265303],
                            [-0.03453688, -0.16577704, -0.16577704, 1.5541598],
                            [-0.1174254, 0.10361066, 1.3331238, -0.1174254],
                            [-0.08288852, 0.23485081, 0.18649918, 0.8357926],
                            [1.4229196, -0.09670328, -0.02072213, -0.13124016],
                            [-0.16577704, 0.1519623, 1.4229196, -0.16577704],
                            [0.6147565, -0.1519623, -0.15886967, 1.0430139],
                            [0.9255885, -0.16577704, -0.13124016, 0.7045525],
                            [-0.1174254, 0.06216639, -0.0897959, 1.3331238],
                            [1.5817894, -0.16577704, -0.06216639, -0.16577704],
                            [0.40753523, 0.02072213, 0.84960735, -0.08288852],
                            [-0.1519623 , 0.3868131, 1.1604394, -0.1174254],
                            [0.69073766, 0.01381475, 0.5664049, -0.08288852],
                            [-0.16577704, 0.25557294, 1.3331238, -0.14505492]]]]).astype(np.float32)

# In order to cover all the op types we supported, we create a customized model here
class CustomModel(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=4, kernel_size=3, matmul_dim=4, layernorm_dim=4):
        super(CustomModel, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.matmul_weight = torch.nn.Parameter(torch.randn(matmul_dim, matmul_dim))
        self.layernorm = torch.nn.LayerNorm(layernorm_dim)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        x = self.conv(x)  # [batch_size, out_channels, height, width]
        # Flatten for MatMul (assuming batch x sequence x features)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(1))  # Reshape for MatMul
        x = torch.matmul(x, self.matmul_weight)
        x = self.layernorm(x)
        x = self.gelu(x)  # [batch_size, height * width, out_channels]
        return x


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = CustomModel()

    dummy_input = torch.randn(1, 3, 4, 4)

    onnx_model_path = Path(output_dir, 'simple_custom_model.onnx').as_posix()
    quant_onnx_model_path = Path(output_dir, 'simple_custom_model_quantized.onnx').as_posix()

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


def prepare_config():
    config_copy = copy.deepcopy(U8U8_AAWA_CONFIG)
    config_copy.per_channel = True
    config_copy.include_fast_ft = True
    config_copy.extra_options = {
        'WeightSymmetric': True,  # Per-channel supports symmetric weight only
        'FastFinetune': {
            'LearningRate': 0.1,
            'FixedSeed': 1705472343,
            'BatchSize': 1,
            'NumIterations': 100,
            'OptimAlgorithm': 'adaround',
            'TargetOpType': ['Conv', 'LayerNormalization'],  # Try skipping MatMul
            'OutputQDQ': True,
            'MemOptLevel': 0,
        }
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
    print('Quantized the ONNX model and saved it at:', output_model_path)
    return output_model_path


def infer_quantized_model(quantized_model_path):
    # Disabling ORT Graph Optimization to achieve reproducible golden numbers across different servers
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(quantized_model_path, sess_options=so)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f'Model output: {output}')
    return output


def tensor_quantize(output_dir):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_fastfinetune(self, tmpdir: str):
        output = tensor_quantize(tmpdir)
        # comp_equal = (output == output_tensor)
        comp_equal = np.allclose(output, output_tensor, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    def test_quantize_intfunc(self):
        int_quant_dequant_func = INTQuantDequantFunction.apply

        scale = torch.ones(size=(1,)) * 0.5
        zero_point = torch.ones(size=(1,)) * 0
        min_q = torch.ones(size=(1,)) * -128
        max_q = torch.ones(size=(1,)) * 127

        inp = torch.from_numpy(input_tensor)
        inp.requires_grad = True

        out0 = int_quant_func(inp, scale, zero_point, min_q, max_q, default_round_func)
        out1 = int_dequant_func(out0, scale, zero_point)
        out2 = int_quant_dequant_func(inp, scale, zero_point, min_q, max_q, default_round_func)
        out0.sum().backward()
        out1.sum().backward()
        out2.sum().backward()

        comp_equal = (out1.detach().numpy() == out2.detach().numpy())
        self.assertEqual(np.all(comp_equal), True)

    def test_quantize_fpfunc(self):
        scale = torch.ones(size=(1,)) * 0.5
        zero_point = torch.ones(size=(1,)) * 0
        min_q = torch.ones(size=(1,)) * -3.89e38
        max_q = torch.ones(size=(1,)) * 3.89e38

        inp = torch.from_numpy(input_tensor)
        inp.requires_grad = True

        out0 = fp_quant_func(inp, scale, zero_point, min_q, max_q, torch.bfloat16)
        out1 = fp_dequant_func(out0, scale, zero_point)

        out2 = fp_quant_dequant_func(inp, scale, zero_point, min_q, max_q, torch.bfloat16)
        comp_equal = (out1.detach().numpy() == out2.detach().numpy())
        self.assertEqual(np.all(comp_equal), True)

    def test_quantize_basic(self):
        scale = torch.ones(size=(1,)) * 0.5
        zero_point = torch.ones(size=(1,)) * 0
        min_q = torch.ones(size=(1,)) * -3.89e38
        max_q = torch.ones(size=(1,)) * 3.89e38

        inp = torch.from_numpy(input_tensor)
        inp.requires_grad = True

        itg = INTQuantizer(scale, zero_point, min_q, max_q)
        fp = FPQuantizer(scale, zero_point, min_q, max_q)
        bfp = BFPQuantizer({})
        mx = MXQuantizer({})

        create_fn_quantizer({'op_type': COP_MX_OP_NAME, 'op_attrs': {}})

        QuantizationModule({'op_type': COP_BFP_OP_NAME, 'op_attrs': {}})
        QuantizationModule((np.array([1]), np.array([0]), np.array([-128]), np.array([128]),
                            np.array([0]), False, onnx.onnx_pb.TensorProto.INT8))

        out0 = fp.quantize(inp)
        out1 = fp.dequantize(out0)
        out2 = fp.quantize_dequantize(inp)
        comp_equal = (out1.detach().numpy() == out2.detach().numpy())
        self.assertEqual(np.all(comp_equal), True)


if __name__ == '__main__':
    unittest.main()
