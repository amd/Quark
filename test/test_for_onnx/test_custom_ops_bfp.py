#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import helper
from onnx.onnx_ml_pb2 import TensorProto

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx.operators.custom_ops import _COP_BFP_OP_NAME, _COP_DOMAIN, get_library_path


def create_custom_op(output_dir: str, use_fp16: bool = False) -> None:
    """Create an ONNX model with a single BFP custom op and save it to output_dir.

    Args:
        output_dir: Directory to save the generated ONNX model.
        use_fp16: If True, use float16 input/output tensors; otherwise use float32.
    """
    tensor_type = TensorProto.FLOAT16 if use_fp16 else TensorProto.FLOAT
    graph_def = helper.make_graph(
        nodes=[
            helper.make_node(
                _COP_BFP_OP_NAME,
                ["input"],
                ["out"],
                domain=_COP_DOMAIN,
                bit_width=13,
                block_size=16,
                rounding_mode=0,
                bfp_method="to_bfp_prime",
            )
        ],
        name="test-model",
        inputs=[helper.make_tensor_value_info("input", tensor_type, shape=None)],
        outputs=[helper.make_tensor_value_info("out", tensor_type, shape=None)],
    )
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        ir_version=9,  # Specify the IR version here
        opset_imports=[helper.make_operatorsetid("", 19)],
    )
    onnx_model_path = Path(output_dir, "test.onnx").as_posix()
    onnx.save(model_def, onnx_model_path)


def run(output_dir: str, use_fp16: bool = False) -> None:
    """Load the ONNX model from output_dir and run inference with random data.

    Args:
        output_dir: Directory containing the ONNX model to load.
        use_fp16: If True, feed float16 input and verify float16 output; otherwise use float32.
    """
    onnx_model_path = Path(output_dir, "test.onnx").as_posix()
    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path())
    ort_session = onnxruntime.InferenceSession(onnx_model_path, so)
    inpt = np.random.rand(6).astype(np.float16 if use_fp16 else np.float32)
    for _ in range(5):
        ort_inputs = {"input": inpt}
        out = ort_session.run(None, ort_inputs)[0]
    assert out.dtype == inpt.dtype, f"Expected output dtype {inpt.dtype}, got {out.dtype}"
    print(inpt - out)


class TestCustomOpsBFP(unittest.TestCase):
    @use_temporary_directory
    def test_custom_ops_bfp(self, tmpdir: str):
        create_custom_op(tmpdir)
        run(tmpdir)

    @use_temporary_directory
    def test_custom_ops_bfp_fp16(self, tmpdir: str):
        """Test BFP custom op with float16 input/output tensors."""
        create_custom_op(tmpdir, use_fp16=True)
        run(tmpdir, use_fp16=True)


if __name__ == "__main__":
    unittest.main()
