#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import tempfile
import argparse
from typing import Any
import onnx
from onnx import helper
from onnx.onnx_ml_pb2 import TensorProto
import onnxruntime
import numpy as np
from pathlib import Path
from quark.onnx.operators.custom_ops import get_library_path, _COP_DOMAIN, _COP_QUANT_OP_NAME, _COP_DEQUANT_OP_NAME

from quark.onnx.quant_utils import register_custom_ops_library
try:
    register_custom_ops_library(onnxruntime.SessionOptions(), "ROCM")
except Exception as e:
    print("Did not compile custom operations library on ROCm")


def run(inputs: Any, output_dir: str) -> Any:
    onnx_model_path = Path(output_dir, 'test.onnx').as_posix()
    # Load library and create session
    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path("CPU"))
    ort_session = onnxruntime.InferenceSession(onnx_model_path, so, providers=['CPUExecutionProvider'])

    # Session run 5 cycles
    for _ in range(5):
        out = ort_session.run(None, inputs)[0]

    return out


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_channel",
                        required=False,
                        type=bool,
                        help="per channel or not, default not",
                        default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    data_shape = (1, 4, 6)
    if args.per_channel:
        quant_shape = (4, )
    else:
        quant_shape = (1, )

    data_type = TensorProto.FLOAT
    if data_type == TensorProto.FLOAT:
        x = np.random.random(data_shape).astype(np.float32) * 255 - 127
    else:
        raise ValueError("data type {} is not supported yet.".format(data_type))

    scale = np.random.random(quant_shape).astype(np.float32)

    # def _cast_data(arr: np.ndarray, qType: Union[float, int]) -> np.ndarray:
    #     from onnx.reference import ReferenceEvaluator
    #     onnx_model = helper.make_model(
    #         helper.make_graph(
    #             [helper.make_node("Cast", ["X"], ["Y"], to=qType)],
    #             "qu",
    #             [helper.make_tensor_value_info("X", TensorProto.FLOAT, None)],
    #             [helper.make_tensor_value_info("Y", qType, None)],
    #         ))
    #     ref = ReferenceEvaluator(onnx_model)
    #     return ref.run(None, {"X": arr.astype(np.float32)})[0]

    def _create_model(data_type: TensorProto.DataType, scale: Any, zero_point: Any, quant_type: Any, output_dir: str) -> None:
        '''
        y_scale_initializer = onnx.numpy_helper.from_array(scale, name="y_scale")
        y_zp_initializer = onnx.numpy_helper.from_array(zero_point,
                                                        name="y_zero_point")
        x_scale_initializer = onnx.numpy_helper.from_array(scale, name="x_scale")
        x_zp_initializer = onnx.numpy_helper.from_array(zero_point,
                                                        name="x_zero_point")
        initializers = [
            y_scale_initializer, y_zp_initializer, x_scale_initializer,
            x_zp_initializer
        ]
        '''

        # quantize params as constant
        quantize_param_nodes = [
            helper.make_node("Constant", [], ["y_scale"],
                             value=onnx.helper.make_tensor("y_scale", data_type, scale.shape, scale)),
            helper.make_node("Constant", [], ["y_zero_point"],
                             value=onnx.helper.make_tensor("y_zero_point", quant_type, zero_point.shape, zero_point)),
            helper.make_node("Constant", [], ["x_scale"],
                             value=onnx.helper.make_tensor("x_scale", data_type, scale.shape, scale)),
            helper.make_node("Constant", [], ["x_zero_point"],
                             value=onnx.helper.make_tensor("zero_point", quant_type, zero_point.shape, zero_point))
        ]

        graph_def = helper.make_graph(
            nodes=[
                helper.make_node(
                    _COP_QUANT_OP_NAME,
                    ["x", "y_scale", "y_zero_point"],
                    ["q_out"],
                    domain=_COP_DOMAIN,
                ),
                helper.make_node(
                    _COP_DEQUANT_OP_NAME,
                    ["q_out", "x_scale", "x_zero_point"],
                    ["y"],
                    domain=_COP_DOMAIN,
                )
            ] + quantize_param_nodes,
            name="test-qdq",
            inputs=[helper.make_tensor_value_info("x", data_type, shape=None)],
            outputs=[helper.make_tensor_value_info("y", data_type, shape=None)],
            # initializer=initializers
        )

        produce_opset_version = 11  # you could set any opset here
        opset_imports = [onnx.helper.make_operatorsetid("", produce_opset_version)]
        model_def = helper.make_model(graph_def, producer_name="quark.onnx", ir_version=8, opset_imports=opset_imports)
        onnx_model_path = Path(output_dir, 'test.onnx').as_posix()
        onnx.save(model_def, onnx_model_path)

    quant_type = TensorProto.BFLOAT16
    if quant_type == TensorProto.UINT8:
        zero_point = np.random.random(quant_shape).astype(np.uint8)
    elif quant_type == TensorProto.INT8:
        zero_point = np.random.random(quant_shape).astype(np.int8)
    elif quant_type == TensorProto.UINT16:
        zero_point = np.random.random(quant_shape).astype(np.uint16)
    elif quant_type == TensorProto.INT16:
        zero_point = np.random.random(quant_shape).astype(np.int16)
    elif quant_type == TensorProto.FLOAT16:
        zero_point = np.zeros(quant_shape).astype(np.float16)
    elif quant_type == TensorProto.BFLOAT16:
        # numpy does not support bfloat16, so we cast it to bfloat16.
        # unfortunately, onnx.numpy_helper.from_array can't accept bfloat16 array,
        # unless we insert additional Cast node between initializer and functional node.
        # zero_point = _cast_data(np.zeros(quant_shape).astype(np.float32), quant_type)

        # considering let quantize parameter to be constant instead.
        zero_point = np.zeros(quant_shape).astype(np.float32)
    else:
        raise ValueError("quant type {} is not supported yet.".format(quant_type))

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_model(data_type, scale, zero_point, quant_type, output_dir=tmpdir)

        y = run({"x": x}, output_dir=tmpdir)
        print("x : {} \nscale : {} \nzero_point : {} \ny : {}".format(x, scale, zero_point, y))
