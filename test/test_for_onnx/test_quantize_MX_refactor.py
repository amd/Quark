#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
from onnx import helper
from onnx.onnx_ml_pb2 import TensorProto
from onnxruntime.quantization import CalibrationDataReader

from quark.onnx import (
    ModelQuantizer,
    QConfig,
    QLayerConfig,
    get_library_path,
)
from quark.onnx.quantization.config.spec import MXInt8Spec
from quark.onnx.quantization.quant_utils import COP_DOMAIN, COP_MX_OP_NAME
from quark.shares.utils.testing_utils import use_temporary_directory
from quark.torch.kernel.hw_emulation.hw_emulation_interface import fake_quantize_mx
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.utils import get_dtype_params, reshape_to_blocks


def create_custom_op(element_dtype: str, output_dir: str) -> None:
    graph_def = helper.make_graph(
        nodes=[
            helper.make_node(
                COP_MX_OP_NAME,
                ["input"],
                ["out"],
                domain=COP_DOMAIN,
                # scale_dtype='e8m0',
                element_dtype=element_dtype,
                axis=1,
                block_size=32,
                rounding_mode=2,
            )
        ],
        name="test-model",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, shape=None)],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, shape=None)],
    )
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        ir_version=9,  # Specify the IR version here
        opset_imports=[helper.make_operatorsetid("", 19)],
    )
    onnx_model_path = Path(output_dir, "test.onnx").as_posix()
    onnx.save(model_def, onnx_model_path)


quark_supported_elem_dtype = {
    "fp8_e4m3": Dtype.fp8_e4m3,
    "fp8_e5m2": Dtype.fp8_e5m2,
    "fp6_e3m2": Dtype.fp6_e3m2,
    "fp6_e2m3": Dtype.fp6_e2m3,
    "fp4": Dtype.fp4,
    "int8": Dtype.int8,
}


def generate_test_case_input_normal():
    normal_input = torch.tensor(
        [
            [-406.0, -881.0, 227.0, -676.0, 404.0, 291.0, 267.0, 286.0],
            [-557.0, 505.0, -175.0, -252.0, -518.0, -136.0, -134.0, 990.0],
            [-148.0, 413.0, -346.0, -954.0, 748.0, -877.0, -201.0, 338.0],
            [-262.0, 437.0, 755.0, 756.0, 772.0, 767.0, 968.0, -989.0],
            [336.0, -38.0, 601.0, -162.0, 585.0, -686.0, 610.0, 98.0],
            [-718.0, 957.0, -854.0, -985.0, -570.0, 989.0, -774.0, 695.0],
            [708.0, -71.0, 863.0, 260.0, -252.0, 558.0, -750.0, -849.0],
            [-396.0, -986.0, -329.0, 814.0, -340.0, -291.0, 141.0, 784.0],
        ],
        dtype=torch.float32,
    )
    return normal_input


def load_test_case_result():
    result = {
        "input": generate_test_case_input_normal(),
        "fp8_e4m3": {
            "torchao_result": torch.tensor(
                [
                    [-416.0, -896.0, 224.0, -704.0, 416.0, 288.0, 256.0, 288.0],
                    [-576.0, 512.0, -176.0, -256.0, -512.0, -128.0, -128.0, 896.0],
                    [-144.0, 416.0, -352.0, -896.0, 768.0, -896.0, -208.0, 352.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 896.0, -896.0],
                    [320.0, -40.0, 576.0, -160.0, 576.0, -704.0, 640.0, 96.0],
                    [-704.0, 896.0, -832.0, -896.0, -576.0, 896.0, -768.0, 704.0],
                    [704.0, -72.0, 832.0, 256.0, -256.0, 576.0, -768.0, -832.0],
                    [-384.0, -896.0, -320.0, 832.0, -352.0, -288.0, 144.0, 768.0],
                ],
                dtype=torch.float32,
            ),
            "MX_result": torch.tensor(
                [
                    [-416.0, -896.0, 224.0, -704.0, 416.0, 288.0, 256.0, 288.0],
                    [-576.0, 512.0, -176.0, -256.0, -512.0, -128.0, -128.0, 896.0],
                    [-144.0, 416.0, -352.0, -896.0, 768.0, -896.0, -208.0, 352.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 896.0, -896.0],
                    [320.0, -40.0, 576.0, -160.0, 576.0, -704.0, 640.0, 96.0],
                    [-704.0, 896.0, -832.0, -896.0, -576.0, 896.0, -768.0, 704.0],
                    [704.0, -72.0, 832.0, 256.0, -256.0, 576.0, -768.0, -832.0],
                    [-384.0, -896.0, -320.0, 832.0, -352.0, -288.0, 144.0, 768.0],
                ],
                dtype=torch.float32,
            ),
        },
        "fp8_e5m2": {
            "torchao_result": torch.tensor(
                [
                    [-384.0, -896.0, 224.0, -640.0, 384.0, 320.0, 256.0, 256.0],
                    [-512.0, 512.0, -160.0, -256.0, -512.0, -128.0, -128.0, 896.0],
                    [-160.0, 384.0, -320.0, -896.0, 768.0, -896.0, -192.0, 320.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 896.0, -896.0],
                    [320.0, -40.0, 640.0, -160.0, 640.0, -640.0, 640.0, 96.0],
                    [-768.0, 896.0, -896.0, -896.0, -512.0, 896.0, -768.0, 640.0],
                    [768.0, -64.0, 896.0, 256.0, -256.0, 512.0, -768.0, -896.0],
                    [-384.0, -896.0, -320.0, 768.0, -320.0, -320.0, 128.0, 768.0],
                ],
                dtype=torch.float32,
            ),
            "MX_result": torch.tensor(
                [
                    [-384.0, -896.0, 224.0, -640.0, 384.0, 320.0, 256.0, 256.0],
                    [-512.0, 512.0, -160.0, -256.0, -512.0, -128.0, -128.0, 896.0],
                    [-160.0, 384.0, -320.0, -896.0, 768.0, -896.0, -192.0, 320.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 896.0, -896.0],
                    [320.0, -40.0, 640.0, -160.0, 640.0, -640.0, 640.0, 96.0],
                    [-768.0, 896.0, -896.0, -896.0, -512.0, 896.0, -768.0, 640.0],
                    [768.0, -64.0, 896.0, 256.0, -256.0, 512.0, -768.0, -896.0],
                    [-384.0, -896.0, -320.0, 768.0, -320.0, -320.0, 128.0, 768.0],
                ],
                dtype=torch.float32,
            ),
        },
        "fp6_e3m2": {
            "torchao_result": torch.tensor(
                [
                    [-384.0, -896.0, 224.0, -640.0, 384.0, 320.0, 256.0, 256.0],
                    [-512.0, 512.0, -160.0, -256.0, -512.0, -128.0, -128.0, 896.0],
                    [-160.0, 384.0, -320.0, -896.0, 768.0, -896.0, -192.0, 320.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 896.0, -896.0],
                    [320.0, -40.0, 640.0, -160.0, 640.0, -640.0, 640.0, 96.0],
                    [-768.0, 896.0, -896.0, -896.0, -512.0, 896.0, -768.0, 640.0],
                    [768.0, -64.0, 896.0, 256.0, -256.0, 512.0, -768.0, -896.0],
                    [-384.0, -896.0, -320.0, 768.0, -320.0, -320.0, 128.0, 768.0],
                ],
                dtype=torch.float32,
            ),
            "MX_result": torch.tensor(
                [
                    [-384.0, -896.0, 224.0, -640.0, 384.0, 320.0, 256.0, 256.0],
                    [-512.0, 512.0, -160.0, -256.0, -512.0, -128.0, -128.0, 896.0],
                    [-160.0, 384.0, -320.0, -896.0, 768.0, -896.0, -192.0, 320.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 896.0, -896.0],
                    [320.0, -40.0, 640.0, -160.0, 640.0, -640.0, 640.0, 96.0],
                    [-768.0, 896.0, -896.0, -896.0, -512.0, 896.0, -768.0, 640.0],
                    [768.0, -64.0, 896.0, 256.0, -256.0, 512.0, -768.0, -896.0],
                    [-384.0, -896.0, -320.0, 768.0, -320.0, -320.0, 128.0, 768.0],
                ],
                dtype=torch.float32,
            ),
        },
        "fp6_e2m3": {
            "torchao_result": torch.tensor(
                [
                    [-416.0, -896.0, 224.0, -704.0, 416.0, 288.0, 256.0, 288.0],
                    [-576.0, 512.0, -176.0, -256.0, -512.0, -128.0, -128.0, 960.0],
                    [-144.0, 416.0, -352.0, -960.0, 768.0, -896.0, -208.0, 352.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 960.0, -960.0],
                    [320.0, -32.0, 576.0, -160.0, 576.0, -704.0, 640.0, 96.0],
                    [-704.0, 960.0, -832.0, -960.0, -576.0, 960.0, -768.0, 704.0],
                    [704.0, -64.0, 832.0, 256.0, -256.0, 576.0, -768.0, -832.0],
                    [-384.0, -960.0, -320.0, 832.0, -352.0, -288.0, 144.0, 768.0],
                ],
                dtype=torch.float32,
            ),
            "MX_result": torch.tensor(
                [
                    [-416.0, -896.0, 224.0, -704.0, 416.0, 288.0, 256.0, 288.0],
                    [-576.0, 512.0, -176.0, -256.0, -512.0, -128.0, -128.0, 960.0],
                    [-144.0, 416.0, -352.0, -960.0, 768.0, -896.0, -208.0, 352.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 960.0, -960.0],
                    [320.0, -32.0, 576.0, -160.0, 576.0, -704.0, 640.0, 96.0],
                    [-704.0, 960.0, -832.0, -960.0, -576.0, 960.0, -768.0, 704.0],
                    [704.0, -64.0, 832.0, 256.0, -256.0, 576.0, -768.0, -832.0],
                    [-384.0, -960.0, -320.0, 832.0, -352.0, -288.0, 144.0, 768.0],
                ],
                dtype=torch.float32,
            ),
        },
        "fp4": {
            "torchao_result": torch.tensor(
                [
                    [-384.0, -768.0, 256.0, -768.0, 384.0, 256.0, 256.0, 256.0],
                    [-512.0, 512.0, -192.0, -256.0, -512.0, -128.0, -128.0, 768.0],
                    [-128.0, 384.0, -384.0, -768.0, 768.0, -768.0, -192.0, 384.0],
                    [-256.0, 384.0, 768.0, 768.0, 768.0, 768.0, 768.0, -768.0],
                    [384.0, -64.0, 512.0, -192.0, 512.0, -768.0, 512.0, 128.0],
                    [-768.0, 768.0, -768.0, -768.0, -512.0, 768.0, -768.0, 768.0],
                    [768.0, -64.0, 768.0, 256.0, -256.0, 512.0, -768.0, -768.0],
                    [-384.0, -768.0, -384.0, 768.0, -384.0, -256.0, 128.0, 768.0],
                ],
                dtype=torch.float32,
            ),
            "MX_result": torch.tensor(
                [
                    [-384.0, -768.0, 256.0, -768.0, 384.0, 256.0, 256.0, 256.0],
                    [-512.0, 512.0, -192.0, -256.0, -512.0, -128.0, -128.0, 768.0],
                    [-128.0, 384.0, -384.0, -768.0, 768.0, -768.0, -192.0, 384.0],
                    [-256.0, 384.0, 768.0, 768.0, 768.0, 768.0, 768.0, -768.0],
                    [384.0, -64.0, 512.0, -192.0, 512.0, -768.0, 512.0, 128.0],
                    [-768.0, 768.0, -768.0, -768.0, -512.0, 768.0, -768.0, 768.0],
                    [768.0, -64.0, 768.0, 256.0, -256.0, 512.0, -768.0, -768.0],
                    [-384.0, -768.0, -384.0, 768.0, -384.0, -256.0, 128.0, 768.0],
                ],
                dtype=torch.float32,
            ),
        },
    }
    return result


def compare_fixed_data(output_dir: str, elem_dtype: str, device_type: str = "CPU") -> None:
    result = load_test_case_result()
    test_tensor = result["input"]

    block_size, axis = 32, 1
    mx_element_dtype = quark_supported_elem_dtype[elem_dtype]
    _, _, emax = get_dtype_params(mx_element_dtype)
    block_x = reshape_to_blocks(test_tensor, block_size, axis)
    scale, _ = torch.max(torch.abs(block_x), dim=axis + 1, keepdim=True)
    scale = torch.pow(2, torch.floor(torch.log2(scale)) - emax)

    quark_output_tensor = fake_quantize_mx(
        input_tensor=test_tensor.clone(),
        scale=scale,
        mx_element_dtype=mx_element_dtype,
        axis=axis,
        block_size=block_size,
        scale_calculation_mode="floor",
    )

    torchao_result = result[elem_dtype]["torchao_result"]
    MX_result = result[elem_dtype]["MX_result"]

    max_diff_ao = torch.max(abs(quark_output_tensor - torchao_result))
    max_diff_MX = torch.max(abs(quark_output_tensor - MX_result))
    assert max_diff_ao == 0, f"The {elem_dtype} quantization result of quark and torchao is different"
    assert max_diff_MX == 0, f"The {elem_dtype} quantization result of quark and MX is different"

    # Inference ONNX model and verify the results
    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path(device_type))
    onnx_model_path = Path(output_dir, "test.onnx").as_posix()
    ort_session = onnxruntime.InferenceSession(onnx_model_path, so, providers=[device_type + "ExecutionProvider"])
    ort_inputs = {"input": test_tensor.numpy()}

    start_time = time.perf_counter() * 1000
    onnx_output_tensor = torch.from_numpy(ort_session.run(None, ort_inputs)[0])
    end_time = time.perf_counter() * 1000

    max_diff = torch.max(abs(quark_output_tensor - onnx_output_tensor))
    assert max_diff == 0, (
        f"The {elem_dtype} quantization result has a difference {max_diff} between quark torch and onnx"
    )

    print(
        f"Verified {elem_dtype} on {device_type} for MX fixneuron with fixed data, "
        f"the differnece is {max_diff} and it costs {end_time - start_time:.2f}ms"
    )


def compare_random_data(output_dir: str, elem_dtype: str, device_type: str = "CPU") -> None:
    onnx_model_path = Path(output_dir, "test.onnx").as_posix()
    test_tensor = torch.randn(1, 32, 8, 8)

    block_size, axis = 32, 1
    mx_element_dtype = quark_supported_elem_dtype[elem_dtype]
    _, _, emax = get_dtype_params(mx_element_dtype)
    block_x = reshape_to_blocks(test_tensor, block_size, axis)
    scale, _ = torch.max(torch.abs(block_x), dim=axis + 1, keepdim=True)
    scale = torch.pow(2, torch.floor(torch.log2(scale)) - emax)

    quark_output_tensor = fake_quantize_mx(
        input_tensor=test_tensor.clone(),
        scale=scale,
        mx_element_dtype=mx_element_dtype,
        axis=axis,
        block_size=block_size,
        scale_calculation_mode="floor",
    )

    # Inference ONNX model and verify the results
    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path(device_type))
    ort_session = onnxruntime.InferenceSession(onnx_model_path, so, providers=[device_type + "ExecutionProvider"])
    ort_inputs = {"input": test_tensor.numpy()}

    start_time = time.perf_counter() * 1000
    onnx_output_tensor = torch.from_numpy(ort_session.run(None, ort_inputs)[0])
    end_time = time.perf_counter() * 1000

    max_diff = torch.max(abs(quark_output_tensor - onnx_output_tensor))
    assert max_diff == 0, (
        f"The {elem_dtype} quantization result has a difference {max_diff} between quark torch and onnx"
    )

    print(
        f"Verified {elem_dtype} on {device_type} for MX fixneuron with random data, "
        f"the differnece is {max_diff} and it costs {end_time - start_time:.2f}ms"
    )


def verify_mx_fixneuron(output_dir: str) -> None:
    for key in quark_supported_elem_dtype:
        if key == "fp4":
            create_custom_op("fp4_e2m1", output_dir)  # Different name for fp4
        else:
            create_custom_op(key, output_dir)

        if key == "int8":  # No fixed groundtruth for int8, so exclude it here
            continue

        compare_fixed_data(output_dir, key)
        if "ROCMExecutionProvider" in onnxruntime.get_available_providers():
            compare_fixed_data(output_dir, key, "ROCM")
        elif "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            compare_fixed_data(output_dir, key, "CUDA")

        compare_random_data(output_dir, key)
        if "ROCMExecutionProvider" in onnxruntime.get_available_providers():
            compare_random_data(output_dir, key, "ROCM")
        elif "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            compare_random_data(output_dir, key, "CUDA")


# ==========================================================================

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
                [0.1875, 0.046875, 0.01171875, -0.0625],
                [0.09375, 0.046875, 0.09375, 0.046875],
                [0.09375, 0.1875, 0.375, 0.09375],
                [0.09375, 0.09375, 0.1875, 0.25],
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


class SimpleConvModel(nn.Module):
    def __init__(self):
        super(SimpleConvModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = SimpleConvModel()

    dummy_input = torch.randn(1, 3, 4, 4)

    onnx_model_path = Path(output_dir, "simple_conv_model.onnx").as_posix()
    onnx_quantized_model_path = Path(output_dir, "simple_conv_model_quantized.onnx").as_posix()
    torch.onnx.export(
        model, dummy_input, onnx_model_path, input_names=["input"], output_names=["output"], opset_version=17
    )

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_quantized_model_path


def prepare_config(element_dtype="int8"):
    quant_config = QConfig(
        global_config=QLayerConfig(activation=MXInt8Spec(symmetric=False), weight=MXInt8Spec()),
        MXAttributes={
            "element_dtype": element_dtype,
            "axis": 1,
            "block_size": 32,
            "rounding_mode": 2,
        },
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
    output = sess.run([output_name], {input_name: input_tensor})

    print(f"Model output: {output}")
    return output


def tensor_quantize(output_dir):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config("fp4_e2m1")
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    # By default, we test the MXINT8 quantization running on CPU
    @use_temporary_directory
    def test_quantize_MX(self, tmpdir: str):
        output = tensor_quantize(tmpdir)
        comp_equal = np.allclose(output, output_tensor, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)


if __name__ == "__main__":
    # Verify the function of mxfixneuron, we should ensure
    # its output is consistent with torch API
    with tempfile.TemporaryDirectory() as tmpdir:
        verify_mx_fixneuron(output_dir=tmpdir)

    # Verify the quantized model
    unittest.main()
