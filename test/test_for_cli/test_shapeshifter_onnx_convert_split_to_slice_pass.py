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
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 6])

    Y0 = helper.make_tensor_value_info("Y0", TensorProto.FLOAT, None)
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, None)
    Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, None)

    split_sizes = np.array([2, 3, 1], dtype=np.int64)

    split_init = helper.make_tensor(
        name="split_sizes",
        data_type=TensorProto.INT64,
        dims=[3],
        vals=split_sizes,
    )

    split_node = helper.make_node(
        "Split", inputs=["X", "split_sizes"], outputs=["Y0", "Y1", "Y2"], axis=1, name="MySplit"
    )

    graph = helper.make_graph(
        nodes=[split_node],
        name="SplitToSliceTest",
        inputs=[X],
        outputs=[Y0, Y1, Y2],
        initializer=[split_init],
    )

    model = helper.make_model(graph)
    model.opset_import[0].version = 13

    onnx_model_path = Path(output_dir, "split_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "split_model_optimized.onnx").as_posix()
    onnx.save(model, onnx_model_path)

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "convert_split_to_slice.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_convert_split_to_slice": {
                "convert_split_to_slice": True,
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_split(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)

    split_count = 0
    for node in model.graph.node:
        if node.op_type == "Split":
            split_count += 1

    return split_count == 0


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_convert_split_to_slice(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["shapeshifter", yaml_path])
        flag = check_split(onnx_optimized_model_path)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
