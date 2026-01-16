#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest

import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader
from testing_utils import prepare_model

from quark.onnx import AdaQuantConfig, BFP16Spec, ModelQuantizer, QConfig, QLayerConfig, get_library_path
from quark.shares.utils.testing_utils import require_torch_cuda, use_temporary_directory

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
                [0.2421875, 0.1484375, 0.015625, -0.05078125],
                [0.1328125, -0.03515625, 0.125, 0.03125],
                [0.109375, 0.25390625, 0.40625, 0.1015625],
                [0.13671875, 0.12109375, 0.21875, 0.2734375],
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


def prepare_config(op_device="CPU", in_device="CPU"):
    def device_config(device):
        if device == "CPU":
            true_device = "cpu"
        elif device == "CUDA":
            true_device = "cuda:0"
        else:
            true_device = "cpu"
        return true_device

    optim_device = device_config(op_device)
    infer_device = device_config(in_device)

    adaquant_algo = AdaQuantConfig(learning_rate=0.1, optim_device=optim_device, infer_device=infer_device)

    quant_config = QConfig(
        global_config=QLayerConfig(activation=BFP16Spec(), weight=BFP16Spec()), algo_config=[adaquant_algo]
    )
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


def infer_quantized_model(quantized_model_path, device="CPU"):
    if device != "CPU":
        if "ROCMExecutionProvider" in onnxruntime.get_available_providers():
            device = "ROCM"
            providers = ["ROCMExecutionProvider"]
        elif "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            device = "CUDA"
            providers = ["CUDAExecutionProvider"]
        else:
            device = "CPU"
            providers = ["CPUExecutionProvider"]
    else:
        device = "CPU"
        providers = ["CPUExecutionProvider"]

    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path(device))
    sess = onnxruntime.InferenceSession(quantized_model_path, so, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(output_dir: str, torch_device="CPU", ort_device="CPU"):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config(torch_device, ort_device)
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path, ort_device)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_BFP_cpu_cpu_fastfinetune(self, tmpdir: str):
        output = tensor_quantize(tmpdir, "CPU", "CPU")
        comp_equal = np.allclose(output, output_tensor, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    @require_torch_cuda
    def test_quantize_BFP_cuda_cpu_fastfinetune(self, tmpdir: str):
        output = tensor_quantize(tmpdir, "CUDA", "CPU")
        comp_equal = np.allclose(output, output_tensor, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    @require_torch_cuda
    def test_quantize_BFP_cpu_cuda_fastfinetune(self, tmpdir: str):
        output = tensor_quantize(tmpdir, "CPU", "CUDA")
        comp_equal = np.allclose(output, output_tensor, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    @require_torch_cuda
    def test_quantize_BFP_cuda_cuda_fastfinetune(self, tmpdir: str):
        output = tensor_quantize(tmpdir, "CUDA", "CUDA")
        comp_equal = np.allclose(output, output_tensor, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)


if __name__ == "__main__":
    unittest.main()
