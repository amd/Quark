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

from quark.common.utils.testing_utils import use_temporary_directory
from quark.experimental.cli.main import main as cli


def prepare_model(output_dir):
    onnx_model_path = Path(output_dir, "shared_bias_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "shared_bias_model_optimized.onnx").as_posix()

    bias = helper.make_tensor(
        name="shared_bias",
        data_type=TensorProto.FLOAT,
        dims=[4],
        vals=np.random.randn(4).astype(np.float32),
    )

    W1 = helper.make_tensor("W1", TensorProto.FLOAT, [4, 3], np.random.randn(4, 3).astype(np.float32))
    W2 = helper.make_tensor("W2", TensorProto.FLOAT, [4, 3], np.random.randn(4, 3).astype(np.float32))

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])

    node1 = helper.make_node("Gemm", ["input", "W1", "shared_bias"], ["Y1"])
    node2 = helper.make_node("Gemm", ["input", "W2", "shared_bias"], ["Y2"])
    add_node = helper.make_node("Add", ["Y1", "Y2"], ["output"])

    graph = helper.make_graph(
        [node1, node2, add_node],
        "SharedBiasGraph",
        [input_tensor],
        [output_tensor],
        initializer=[W1, W2, bias],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.save(model, onnx_model_path)
    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "copy_bias_init.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_copy_bias_init": {
                "shared_bias_op_types": ["Conv", "ConvTranspose", "Gemm"],
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_shared_bias_init(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)

    bias_usage_count = {}
    for node in model.graph.node:
        if len(node.input) > 2:
            bias_name = node.input[2]
            bias_usage_count[bias_name] = bias_usage_count.get(bias_name, 0) + 1

    return all(count <= 1 for name, count in bias_usage_count.items())


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_copy_bias_init_pass(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["shapeshifter", yaml_path])
        flag = check_shared_bias_init(onnx_optimized_model_path)
        self.assertEqual(flag, True)


if __name__ == "__main__":
    unittest.main()
