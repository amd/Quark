#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import unittest
from pathlib import Path

import onnx
import yaml
from onnx import TensorProto, helper

from quark.common.utils.testing_utils import use_temporary_directory
from quark.experimental.cli.main import main as cli


def prepare_model(output_dir):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])

    epsilon = helper.make_tensor(
        name="eps",
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=[1e-12],
    )

    axes_uns1 = helper.make_tensor(
        name="axes_uns1",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[-1],
    )

    axes_uns2 = helper.make_tensor(
        name="axes_uns2",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[-1],
    )

    axes_rsum = helper.make_tensor(
        name="axes_rsum",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1],
    )

    uns1 = helper.make_node("Unsqueeze", inputs=["x", "axes_uns1"], outputs=["uns_out"], name="Unsqueeze_1")

    mul1 = helper.make_node("Mul", inputs=["uns_out", "x"], outputs=["mul1_out"], name="Mul_1")

    rsum = helper.make_node("ReduceSum", inputs=["mul1_out", "axes_rsum"], outputs=["rsum_out"], name="ReduceSum_1")

    maxnode = helper.make_node("Max", inputs=["rsum_out", "eps"], outputs=["max_out"], name="Max_1")

    sqrt = helper.make_node("Sqrt", inputs=["max_out"], outputs=["sqrt_out"], name="Sqrt_1")

    rec = helper.make_node("Reciprocal", inputs=["sqrt_out"], outputs=["rec_out"], name="Reciprocal_1")

    uns2 = helper.make_node("Unsqueeze", inputs=["x", "axes_uns2"], outputs=["uns2_out"], name="Unsqueeze_2")

    mul2 = helper.make_node("Mul", inputs=["uns2_out", "rec_out"], outputs=["y"], name="Mul_2")

    graph = helper.make_graph(
        nodes=[uns1, mul1, rsum, maxnode, sqrt, rec, uns2, mul2],
        name="L2NormGraph",
        inputs=[x],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=[epsilon, axes_uns1, axes_uns2, axes_rsum],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    onnx.checker.check_model(model)

    onnx_model_path = Path(output_dir, "l2_norm_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "l2_norm_model_optimized.onnx").as_posix()
    onnx.save(model, onnx_model_path)

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "fuse_l2_norm.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_fuse_l2_norm": {
                "fuse_l2_norm": True,
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_l2_norm(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)

    l2_norm_count = 0
    for node in model.graph.node:
        if node.op_type == "LpNormalization":
            l2_norm_count += 1

    return l2_norm_count > 0


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_fuse_l2_norm(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["shapeshifter", yaml_path])
        flag = check_l2_norm(onnx_optimized_model_path)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
