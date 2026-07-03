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
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 40, 20])

    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 1, 1])

    gap_node = helper.make_node("GlobalAveragePool", inputs=["input"], outputs=["output"], name="MyGlobalAvgPool")

    graph = helper.make_graph(
        nodes=[gap_node],
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[],
        value_info=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 40, 20])],
    )

    model = helper.make_model(graph, producer_name="test")

    onnx_model_path = Path(output_dir, "large_kernel_pool_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "large_kernel_pool_model_optimized.onnx").as_posix()
    onnx.save(model, onnx_model_path)

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "split_large_kernel_pool.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_split_large_kernel_pool": {
                "split_large_kernel_pool": True,
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_average_pool(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)

    average_pool_count = 0
    for node in model.graph.node:
        if node.op_type == "AveragePool":
            average_pool_count += 1

    return average_pool_count > 0


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_split_large_kernel_pool(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["shapeshifter", yaml_path])
        flag = check_average_pool(onnx_optimized_model_path)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
