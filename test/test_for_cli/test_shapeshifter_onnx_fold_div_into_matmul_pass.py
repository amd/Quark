#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import TensorProto, helper, numpy_helper

from quark.common.utils.testing_utils import use_temporary_directory
from quark.experimental.cli.main import main as cli


def prepare_attention_gemm_model(output_dir):
    """Fused-QKV attention pattern: Gemm(transB=1) -> Split -> (Q @ K^T) -> Div(8)."""
    np.random.seed(0)
    N, K, H = 2, 4, 3  # seq, hidden, head dim; fused QKV out = 3 * H
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, N])

    W = np.random.randn(3 * H, K).astype(np.float32)  # transB=1 -> [3H, K]
    B = np.random.randn(3 * H).astype(np.float32)
    split_sizes = np.array([H, H, H], dtype=np.int64)
    div_const = np.array(8.0, dtype=np.float32)

    inits = [
        numpy_helper.from_array(W, "W"),
        numpy_helper.from_array(B, "B"),
        numpy_helper.from_array(split_sizes, "split_sizes"),
        numpy_helper.from_array(div_const, "div_const"),
    ]

    nodes = [
        helper.make_node("Gemm", ["X", "W", "B"], ["proj"], name="proj_gemm", transB=1),
        helper.make_node("Split", ["proj", "split_sizes"], ["q", "k", "v"], name="qkv_split", axis=1),
        helper.make_node("Transpose", ["k"], ["kt"], name="k_transpose", perm=[1, 0]),
        helper.make_node("MatMul", ["q", "kt"], ["score"], name="qk_matmul"),
        helper.make_node("Div", ["score", "div_const"], ["Y"], name="score_div"),
    ]

    graph = helper.make_graph(nodes, "AttnGemmGraph", [X], [Y], initializer=inits)
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_operatorsetid("", 18)])
    onnx.checker.check_model(model)

    model_path = Path(output_dir, "attn_gemm.onnx").as_posix()
    optimized_path = Path(output_dir, "attn_gemm_optimized.onnx").as_posix()
    onnx.save(model, model_path)
    return model_path, optimized_path


def prepare_matmul_add_model(output_dir):
    """Pattern A: (MatMul + Add) projection -> Div(4) -> MatMul consumer."""
    np.random.seed(1)
    N, K, M, P = 2, 4, 3, 5
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, P])

    W1 = np.random.randn(K, M).astype(np.float32)
    B1 = np.random.randn(M).astype(np.float32)
    W2 = np.random.randn(M, P).astype(np.float32)
    div_const = np.array(4.0, dtype=np.float32)

    inits = [
        numpy_helper.from_array(W1, "W1"),
        numpy_helper.from_array(B1, "B1"),
        numpy_helper.from_array(W2, "W2"),
        numpy_helper.from_array(div_const, "div_const"),
    ]

    nodes = [
        helper.make_node("MatMul", ["X", "W1"], ["mm1"], name="proj_matmul"),
        helper.make_node("Add", ["mm1", "B1"], ["proj"], name="proj_add"),
        helper.make_node("Div", ["proj", "div_const"], ["scaled"], name="proj_div"),
        helper.make_node("MatMul", ["scaled", "W2"], ["Y"], name="out_matmul"),
    ]

    graph = helper.make_graph(nodes, "MatMulAddGraph", [X], [Y], initializer=inits)
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_operatorsetid("", 18)])
    onnx.checker.check_model(model)

    model_path = Path(output_dir, "matmul_add.onnx").as_posix()
    optimized_path = Path(output_dir, "matmul_add_optimized.onnx").as_posix()
    onnx.save(model, model_path)
    return model_path, optimized_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "fold_div_into_matmul.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {"onnx_fold_div_into_matmul": {"fold_div_into_matmul": True}},
        "output_model_path": onnx_optimized_model_path,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def count_div(model_path):
    model = onnx.load(model_path)
    return sum(1 for n in model.graph.node if n.op_type == "Div")


def run_model(model_path, feed):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess.run(None, feed)


class TestFoldDivIntoMatMulPass(unittest.TestCase):
    @use_temporary_directory
    def test_attention_gemm_split(self, tmpdir: str):
        model_path, optimized_path = prepare_attention_gemm_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, model_path, optimized_path)

        feed = {"X": np.random.RandomState(2).randn(2, 4).astype(np.float32)}
        before = run_model(model_path, feed)

        cli(["shapeshifter", yaml_path])

        # The Div node is folded away.
        self.assertEqual(count_div(optimized_path), 0)
        # Output is unchanged (bit-exact for the power-of-two divisor 8).
        after = run_model(optimized_path, feed)
        np.testing.assert_allclose(before[0], after[0], rtol=0, atol=0)

    @use_temporary_directory
    def test_matmul_add_projection(self, tmpdir: str):
        model_path, optimized_path = prepare_matmul_add_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, model_path, optimized_path)

        feed = {"X": np.random.RandomState(3).randn(2, 4).astype(np.float32)}
        before = run_model(model_path, feed)

        cli(["shapeshifter", yaml_path])

        self.assertEqual(count_div(optimized_path), 0)
        after = run_model(optimized_path, feed)
        np.testing.assert_allclose(before[0], after[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
