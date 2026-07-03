#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import Config, ModelQuantizer
from quark.onnx.quantization.config.custom_config import A16W8_CONFIG
from quark.onnx.quantization.quant_utils import is_version_below

np.random.seed(42)
tensor_1 = np.random.rand(1, 2, 4, 4).astype(np.float32)
tensor_2 = np.random.rand(1, 1, 4, 4).astype(np.float32)
output_golden = np.array(
    [[257.29266, 268.34863, 233.25618, 237.59668, 246.36774, 248.39056, 239.04623, 263.8935, 243.58327, 243.37852]],
    dtype=np.float32,
)


class DataReader(CalibrationDataReader):
    def __init__(self, tensor_1, tensor_2):
        self.tensor_1 = tensor_1
        self.tensor_2 = tensor_2
        self.data_iter = iter([{"input": self.tensor_1, "mul_input": self.tensor_2}])

    def get_next(self):
        return next(self.data_iter, None)


def prepare_model(output_dir):
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 4, 4])
    mul_input_tensor = helper.make_tensor_value_info("mul_input", TensorProto.FLOAT, [1, 1, 4, 4])

    split_node = helper.make_node("Split", inputs=["input"], outputs=["x1", "x2"], name="split_node", axis=1)

    conv_weight = np.random.rand(16, 1, 3, 3).astype(np.float32)
    conv_weight_tensor = helper.make_tensor(
        name="conv_weight", data_type=TensorProto.FLOAT, dims=conv_weight.shape, vals=conv_weight.flatten().tolist()
    )

    conv_bias = np.random.rand(16).astype(np.float32)
    conv_bias_tensor = helper.make_tensor(
        name="conv_bias", data_type=TensorProto.FLOAT, dims=conv_bias.shape, vals=conv_bias.tolist()
    )

    conv_node = helper.make_node(
        "Conv",
        inputs=["x1", "conv_weight", "conv_bias"],
        outputs=["x1_conv"],
        kernel_shape=[3, 3],
        name="conv_node",
        pads=[1, 1, 1, 1],
    )

    mul_weight = np.random.rand(1, 1, 4, 4).astype(np.float32)
    mul_weight_tensor = helper.make_tensor(
        name="mul_weight", data_type=TensorProto.FLOAT, dims=mul_weight.shape, vals=mul_weight.flatten().tolist()
    )
    mul_node = helper.make_node("Mul", inputs=["mul_input", "mul_weight"], name="mul_node", outputs=["mul_output"])

    concat_node = helper.make_node(
        "Concat", inputs=["x2", "mul_output", "x1_conv"], outputs=["concat_output"], name="concat_node", axis=1
    )

    reshape_shape = np.array([1, -1], dtype=np.int64)
    reshape_shape_tensor = helper.make_tensor(
        name="reshape_shape", data_type=TensorProto.INT64, dims=reshape_shape.shape, vals=reshape_shape.tolist()
    )
    reshape_node = helper.make_node(
        "Reshape", inputs=["concat_output", "reshape_shape"], outputs=["reshaped_output"], name="reshape_node"
    )

    gemm_weight = np.random.rand(288, 10).astype(np.float32)
    gemm_weight_tensor = helper.make_tensor(
        name="gemm_weight", data_type=TensorProto.FLOAT, dims=gemm_weight.shape, vals=gemm_weight.flatten().tolist()
    )
    gemm_node = helper.make_node(
        "Gemm", inputs=["reshaped_output", "gemm_weight"], outputs=["output"], name="gemm_node"
    )

    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    graph = helper.make_graph(
        nodes=[split_node, conv_node, mul_node, concat_node, reshape_node, gemm_node],
        name="CustomModelGraph",
        inputs=[input_tensor, mul_input_tensor],
        outputs=[output_tensor],
        initializer=[conv_weight_tensor, conv_bias_tensor, mul_weight_tensor, reshape_shape_tensor, gemm_weight_tensor],
    )

    model = helper.make_model(
        graph, producer_name="onnx-example", opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )

    onnx_model_path = Path(output_dir, "float_model.onnx").as_posix()
    onnx_quantized_model_path = Path(output_dir, "quantized_model.onnx").as_posix()
    onnx.save(model, onnx_model_path)

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_quantized_model_path


def prepare_config():
    config_copy = copy.deepcopy(A16W8_CONFIG)
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_data():
    data_reader = DataReader(tensor_1, tensor_2)
    return data_reader


def prepare_quantizer(quant_config):
    quantizer = ModelQuantizer(quant_config)
    return quantizer


def quantize_static(quantizer, input_model_path, output_model_path, data_reader):
    quantizer.quantize_model(input_model_path, output_model_path, data_reader)
    print("Quantized the ONNX model and saved it at:", output_model_path)
    return output_model_path


def infer_quantized_model(quantized_model_path):
    sess = onnxruntime.InferenceSession(quantized_model_path)
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {"input": tensor_1, "mul_input": tensor_2})
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


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_adjust_bias_scale(self, tmpdir: str):
        if not is_version_below(onnxruntime, "1.18.0"):
            output = tensor_quantize(tmpdir)
            comp_equal = np.allclose(output, output_golden, atol=1e2)
            self.assertEqual(comp_equal, True)


if __name__ == "__main__":
    unittest.main()
