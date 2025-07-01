#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
import numpy as np
import onnx
import onnxruntime

from pathlib import Path
from onnx import helper
from onnx.onnx_ml_pb2 import TensorProto
from quark.onnx.operators.custom_ops import get_library_path, _COP_DOMAIN, _COP_BFP_OP_NAME
from quark.shares.utils.testing_utils import use_temporary_directory


def create_custom_op(output_dir: str) -> None:
    graph_def = helper.make_graph(nodes=[
        helper.make_node(_COP_BFP_OP_NAME, ["input"], ["out"],
                         domain=_COP_DOMAIN,
                         bit_width=13,
                         block_size=16,
                         rounding_mode=0,
                         bfp_method="to_bfp_prime")
    ],
                                  name="test-model",
                                  inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, shape=None)],
                                  outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, shape=None)])
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        ir_version=9,  # Specify the IR version here
        opset_imports=[helper.make_operatorsetid('', 19)])
    onnx_model_path = Path(output_dir, 'test.onnx').as_posix()
    onnx.save(model_def, onnx_model_path)


def run(output_dir: str) -> None:
    onnx_model_path = Path(output_dir, 'test.onnx').as_posix()
    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path())
    ort_session = onnxruntime.InferenceSession(onnx_model_path, so)
    inpt = np.random.rand(6).astype(np.float32)
    for _ in range(5):
        ort_inputs = {"input": inpt}
        out = ort_session.run(None, ort_inputs)[0]
    print(inpt - out)


class TestCustomOpsBFP(unittest.TestCase):
    @use_temporary_directory
    def test_custom_ops_bfp(self, tmpdir: str):
        create_custom_op(tmpdir)
        run(tmpdir)


if __name__ == "__main__":
    unittest.main()
