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
from quark.onnx.operators.custom_ops import _COP_DOMAIN, _COP_MX_OP_NAME, get_library_path


def create_custom_op(output_dir: str, use_fp16: bool = False) -> None:
    """Create an ONNX model with a single MX custom op and save it to output_dir.

    Args:
        output_dir: Directory to save the generated ONNX model.
        use_fp16: If True, use float16 input/output tensors; otherwise use float32.
    """
    tensor_type = TensorProto.FLOAT16 if use_fp16 else TensorProto.FLOAT
    graph_def = helper.make_graph(
        nodes=[
            helper.make_node(
                _COP_MX_OP_NAME,
                ["input"],
                ["out"],
                domain=_COP_DOMAIN,
                scale_dtype="e8m0",
                # element_dtype='fp8_e5m2',
                # element_dtype='fp8_e4m3',
                # element_dtype='fp6_e3m2',
                # element_dtype='fp6_e2m3',
                # element_dtype='fp4_e2m1',
                element_dtype="int8",
                axis=1,
                block_size=8,
                rounding_mode=2,
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
    """Load the ONNX model from output_dir and run inference with fixed data.

    Args:
        output_dir: Directory containing the ONNX model to load.
        use_fp16: If True, feed float16 input and verify float16 output; otherwise use float32.
    """
    onnx_model_path = Path(output_dir, "test.onnx").as_posix()
    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path("CPU"))
    ort_session = onnxruntime.InferenceSession(onnx_model_path, so, providers=["CPUExecutionProvider"])
    dtype = np.float16 if use_fp16 else np.float32
    inp = np.array([[1.1031372, 0.05104101, 0.8381394, 0.5155692, 0.64676553, 0.36488876]]).astype(dtype)
    for _ in range(5):
        ort_inputs = {"input": inp}
        out = ort_session.run(None, ort_inputs)[0]
    assert out.dtype == inp.dtype, f"Expected output dtype {inp.dtype}, got {out.dtype}"
    print(f"Input {inp}")
    print(f"Output {out}")
    print(f"Diff {inp - out}")


class TestCustomOpsMX(unittest.TestCase):
    @use_temporary_directory
    def test_custom_ops_mx(self, tmpdir: str):
        create_custom_op(tmpdir)
        run(tmpdir)

    @use_temporary_directory
    def test_custom_ops_mx_fp16(self, tmpdir: str):
        """Test MX custom op with float16 input/output tensors."""
        create_custom_op(tmpdir, use_fp16=True)
        run(tmpdir, use_fp16=True)


if __name__ == "__main__":
    unittest.main()
