#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
from onnx import helper
from onnx.onnx_ml_pb2 import TensorProto

from quark.onnx.operators.custom_ops import _COP_DOMAIN, _COP_IN_OP_NAME, get_library_path

data_type_mapping = {"float32": TensorProto.FLOAT, "bfloat16": TensorProto.BFLOAT16}


def run(inputs: Any, output_dir: str) -> Any:
    onnx_model_path = Path(output_dir, "test.onnx").as_posix()
    # Load library and create session
    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path("CPU"))
    ort_session = onnxruntime.InferenceSession(onnx_model_path, so, providers=["CPUExecutionProvider"])

    # Session run 5 cycles
    for _ in range(5):
        out = ort_session.run(None, inputs)[0]

    return out


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type",
        required=False,
        type=str,
        help="data type for instance norm's input and output",
        default="float32",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    data_type = data_type_mapping[args.data_type]
    data_shape = (1, 2, 2, 2)

    def _cast_data(arr: Any, qType: TensorProto.DataType) -> Any:
        from onnx.reference import ReferenceEvaluator

        onnx_model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Cast", ["X"], ["Y"], to=qType)],
                "cast",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, None)],
                [helper.make_tensor_value_info("Y", qType, None)],
            )
        )
        ref = ReferenceEvaluator(onnx_model)
        return ref.run(None, {"X": arr.astype(np.float32)})[0]

    def _create_model(data_type: TensorProto.DataType, data_shape: tuple[int, ...], output_dir: str) -> None:
        gamma = np.ones(data_shape).astype(np.float32)
        beta = np.zeros(data_shape).astype(np.float32)

        in_param_nodes = [
            helper.make_node(
                "Constant", [], ["gamma"], value=onnx.helper.make_tensor("y_scale", data_type, data_shape, gamma)
            ),
            helper.make_node(
                "Constant", [], ["beta"], value=onnx.helper.make_tensor("y_zero_point", data_type, data_shape, beta)
            ),
        ]

        graph_def = helper.make_graph(
            nodes=[
                helper.make_node(
                    _COP_IN_OP_NAME,
                    ["x", "gamma", "beta"],
                    ["y"],
                    domain=_COP_DOMAIN,
                ),
            ]
            + in_param_nodes,
            name="test-in",
            inputs=[helper.make_tensor_value_info("x", data_type, shape=None)],
            outputs=[helper.make_tensor_value_info("y", data_type, shape=None)],
        )

        produce_opset_version = 11  # you could set any opset here
        opset_imports = [onnx.helper.make_operatorsetid("", produce_opset_version)]
        model_def = helper.make_model(graph_def, producer_name="quark.onnx", ir_version=8, opset_imports=opset_imports)
        onnx_model_path = Path(output_dir, "test.onnx").as_posix()
        onnx.save(model_def, onnx_model_path)

    if data_type == TensorProto.FLOAT:
        # x = np.ones(data_shape).astype(np.float32)
        x = np.random.random(data_shape).astype(np.float32)
    elif data_type == TensorProto.BFLOAT16:
        # numpy does not support bfloat16, so we cast it to bfloat16.
        # x = _cast_data(np.ones(data_shape).astype(np.float32), data_type)
        x = _cast_data(np.random.random(data_shape).astype(np.float32), data_type)
    else:
        raise ValueError(f"data type {data_type} is not supported yet.")

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_model(data_type, data_shape[1:2], output_dir=tmpdir)

        y = run({"x": x}, output_dir=tmpdir)
        print(f"x : {x} \ny : {y}")
