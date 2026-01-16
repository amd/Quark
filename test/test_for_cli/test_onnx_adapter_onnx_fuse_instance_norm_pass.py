#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import unittest
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper

from quark.experimental.cli.main import main as cli
from quark.shares.utils.testing_utils import use_temporary_directory


def prepare_model(output_dir):
    # ---- Model inputs ----
    # assume input is NCHW, channels = 3
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])

    # ---- Initializers that will be used as bias/weight/eps and constants ----
    C = 3
    bias_init = helper.make_tensor(
        name="bias_init",
        data_type=TensorProto.FLOAT,
        dims=[1, C],
        vals=np.random.randn(1, C).astype(np.float32).flatten(),
    )
    weight_init = helper.make_tensor(
        name="weight_init",
        data_type=TensorProto.FLOAT,
        dims=[1, C],
        vals=np.random.randn(1, C).astype(np.float32).flatten(),
    )
    eps_init = helper.make_tensor(name="eps", data_type=TensorProto.FLOAT, dims=[], vals=[1e-5])

    # some other small initializers used as multipliers/constants
    const_w = helper.make_tensor(name="const_w", data_type=TensorProto.FLOAT, dims=[1], vals=[0.5])
    const_w2 = helper.make_tensor(name="const_w2", data_type=TensorProto.FLOAT, dims=[1], vals=[0.25])

    # ---- Node construction following the required pattern ----
    # We'll build the graph so that the top-level node (add0) is an Add whose inputs
    # are: add0_i0 <- mul_x (Mul) , add0_i1 <- sub0 (Sub)
    # And sub0 second input comes from mul_b, etc. The structure matches the checks.

    nodes = []

    # mul_d: Mul(x, const_w) -> mul_d_out (used as input to a GlobalAveragePool later)
    nodes.append(helper.make_node("Mul", inputs=["x", "const_w"], outputs=["mul_d_out"], name="Mul_D"))

    # gap3: GlobalAveragePool(mul_d_out) -> gap3_out
    nodes.append(helper.make_node("GlobalAveragePool", inputs=["mul_d_out"], outputs=["gap3_out"], name="GAP_3"))

    # sub1: Sub(sub1_i0, gap3_out) -> sub1_out
    # make sub1_i0 a direct input from x to keep it simple
    nodes.append(helper.make_node("Sub", inputs=["x", "gap3_out"], outputs=["sub1_out"], name="Sub_1"))

    # mul2: Mul(sub1_out, const_w2) -> mul2_out
    nodes.append(helper.make_node("Mul", inputs=["sub1_out", "const_w2"], outputs=["mul2_out"], name="Mul_2"))

    # gap2: GlobalAveragePool(mul2_out) -> gap2_out
    nodes.append(helper.make_node("GlobalAveragePool", inputs=["mul2_out"], outputs=["gap2_out"], name="GAP_2"))

    # add1: Add(gap2_out, eps) -> add1_out  (this is add1_node in your code)
    nodes.append(helper.make_node("Add", inputs=["gap2_out", "eps"], outputs=["add1_out"], name="Add_1"))

    # sqrt: Sqrt(add1_out) -> sqrt_out
    nodes.append(helper.make_node("Sqrt", inputs=["add1_out"], outputs=["sqrt_out"], name="Sqrt_1"))

    # rec: Reciprocal(sqrt_out) -> rec_out
    nodes.append(helper.make_node("Reciprocal", inputs=["sqrt_out"], outputs=["rec_out"], name="Reciprocal_1"))

    # mul1: Mul(rec_out, weight_init) -> mul1_out  (this is mul1_node)
    nodes.append(helper.make_node("Mul", inputs=["rec_out", "weight_init"], outputs=["mul1_out"], name="Mul_1"))

    # gap1: GlobalAveragePool(x) -> gap1_out  (this will serve as mul0_i0_node)
    nodes.append(helper.make_node("GlobalAveragePool", inputs=["x"], outputs=["gap1_out"], name="GAP_1"))

    # mul0: Mul(gap1_out, mul1_out) -> mul0_out  (this is mul0_node)
    nodes.append(helper.make_node("Mul", inputs=["gap1_out", "mul1_out"], outputs=["mul0_out"], name="Mul_0"))

    # sub0: Sub(bias_init, mul0_out) -> sub0_out  (this is sub0_node)
    # note: sub0_i0 is bias_init (initializer), sub0_i1 is mul0_out (producer is mul0_node)
    nodes.append(helper.make_node("Sub", inputs=["bias_init", "mul0_out"], outputs=["sub0_out"], name="Sub_0"))

    # mul_x: Mul(x, const_w) -> mul_x_out  (this will be add0_i0_node)
    nodes.append(helper.make_node("Mul", inputs=["x", "const_w"], outputs=["mul_x_out"], name="Mul_X"))

    # add0: Add(mul_x_out, sub0_out) -> final_out  (this is the top-level node 'node' in the matcher)
    nodes.append(helper.make_node("Add", inputs=["mul_x_out", "sub0_out"], outputs=["final_out"], name="TopAdd"))

    # For the other path required by your matcher: gap0_i0_node.op_type == "Mul" and gap0 is a GAP node whose input produced by Mul.
    # We already created mul2 -> gap2, which satisfies: gap2 input producer is mul2 (Mul) and add1_i0_node is gap2 (GlobalAveragePool).
    # Also mul2 input0 is sub1 (Sub) and sub1 second input is GAP_3 (GlobalAveragePool) — satisfies sub1_i1_node.op_type == "GlobalAveragePool".

    # ---- Graph assembly ----
    graph = helper.make_graph(
        nodes=nodes,
        name="InstanceNormMatchGraph",
        inputs=[x],
        outputs=[helper.make_tensor_value_info("final_out", TensorProto.FLOAT, [1, 3, 4, 4])],
        initializer=[bias_init, weight_init, eps_init, const_w, const_w2],
    )

    # ---- Model ----
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    # Validate and save
    onnx.checker.check_model(model)
    onnx.save(model, "instancenorm_match.onnx")
    print("Saved instancenorm_match.onnx")

    onnx.checker.check_model(model)

    onnx_model_path = Path(output_dir, "l2_norm_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "l2_norm_model_optimized.onnx").as_posix()
    onnx.save(model, onnx_model_path)

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "fuse_instance_norm.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_fuse_instance_norm": {
                "fuse_instance_norm": True,
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_instance_norm(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)

    instance_norm_count = 0
    for node in model.graph.node:
        if node.op_type == "InstanceNormalization":
            instance_norm_count += 1

    return instance_norm_count > 0


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_fuse_instance_norm(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["onnx-adapter", yaml_path])
        flag = check_instance_norm(onnx_optimized_model_path)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
