#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import unittest
from pathlib import Path

import onnx
import yaml
from onnx import TensorProto, helper

from quark.experimental.cli.main import main as cli
from quark.shares.utils.testing_utils import use_temporary_directory


def prepare_model(output_dir):
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4])

    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4])

    min_init = helper.make_tensor(
        name="clip_min",
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=[0.0],
    )

    max_init = helper.make_tensor(
        name="clip_max",
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=[6.0],
    )

    clip_node = helper.make_node("Clip", inputs=["input", "clip_min", "clip_max"], outputs=["output"], name="MyClip")

    graph = helper.make_graph(
        nodes=[clip_node],
        name="ClipToReluGraph",
        inputs=[X],
        outputs=[Y],
        initializer=[min_init, max_init],
    )

    model = helper.make_model(graph, producer_name="test")

    onnx_model_path = Path(output_dir, "clip_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "clip_model_optimized.onnx").as_posix()
    onnx.save(model, onnx_model_path)

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "convert_clip_to_relu.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_convert_clip_to_relu": {
                "convert_clip_to_relu": True,
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_clip(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)

    clip_count = 0
    for node in model.graph.node:
        if node.op_type == "Clip":
            clip_count += 1

    return clip_count == 0


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_convert_clip_to_relu(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["onnx-adapter", yaml_path])
        flag = check_clip(onnx_optimized_model_path)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
