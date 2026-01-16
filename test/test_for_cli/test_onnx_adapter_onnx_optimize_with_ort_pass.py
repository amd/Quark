#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import unittest
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper, numpy_helper

from quark.experimental.cli.main import main as cli
from quark.shares.utils.testing_utils import use_temporary_directory


def make_constant_node(name, np_array):
    return helper.make_node(
        "Constant", inputs=[], outputs=[name], value=numpy_helper.from_array(np_array, name + "_val")
    )


def prepare_model(output_dir):
    input_shape = (1, 3)
    num_constants = 8
    identity_chain_len = 12
    nodes = []
    initializers = []

    input_name = "input_X"
    input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, list(input_shape))

    const_names = []
    for i in range(num_constants):
        arr = (np.ones(input_shape, dtype=np.float32) * (i + 1)).astype(np.float32)
        cname = f"const_{i}"
        const_node = make_constant_node(cname, arr)
        nodes.append(const_node)
        const_names.append(cname)

    prev = input_name
    for i in range(identity_chain_len):
        out = f"ident_{i}"
        id_node = helper.make_node("Identity", inputs=[prev], outputs=[out], name=f"Identity_{i}")
        nodes.append(id_node)
        prev = out

    combin_outputs = []
    add_count = 0
    for i, cname in enumerate(const_names):
        add_out = f"add_out_{i}"
        add_node = helper.make_node("Add", inputs=[prev, cname], outputs=[add_out], name=f"Add_with_const_{i}")
        nodes.append(add_node)
        combin_outputs.append(add_out)
        add_count += 1

    final_outputs = []
    for i, name_in in enumerate(combin_outputs):
        out_name = f"final_ident_{i}"
        nodes.append(helper.make_node("Identity", inputs=[name_in], outputs=[out_name], name=f"FinalIdentity_{i}"))
        final_outputs.append(out_name)

    output_tensors = []
    for i, out_name in enumerate(final_outputs):
        output_tensors.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT, list(input_shape)))

    graph = helper.make_graph(
        nodes=nodes,
        name="IdentityConstantHeavyGraph",
        inputs=[input_tensor],
        outputs=output_tensors,
        initializer=initializers,
    )

    opset = [helper.make_operatorsetid("", 13)]

    model = helper.make_model(graph, opset_imports=opset)
    model.ir_version = 10

    onnx_model_path = Path(output_dir, "double_conv_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "double_conv_model_optimized.onnx").as_posix()

    onnx.save(model, onnx_model_path)
    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "optimize_with_ort.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {"onnx_optimize_with_ort": {"optimize_with_ort": True}},
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_identity(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)
    identity_count = 0
    for node in model.graph.node:
        if node.op_type == "Identity":
            identity_count += 1
    return identity_count == 0


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_optimize_with_ort_pass(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["onnx-adapter", yaml_path])
        flag = check_identity(onnx_optimized_model_path)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
