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
from onnx import TensorProto, helper

from quark.common.utils.testing_utils import use_temporary_directory
from quark.experimental.cli.main import main as cli


def prepare_model(output_dir):
    """Build a tiny Conv -> Relu -> Conv model with imbalanced weight ranges.

    The first Conv has deliberately different per-output-channel magnitudes so that
    Cross-Layer Equalization has something to rebalance against the second Conv.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 8, 8])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])

    rng = np.random.RandomState(0)

    # Conv1 weight [oc=4, ic=2, 3, 3] with per-oc scale imbalance.
    w1 = rng.randn(4, 2, 3, 3).astype(np.float32)
    w1[0] *= 0.01
    w1[1] *= 10.0
    w1[2] *= 0.1
    w1[3] *= 5.0
    b1 = rng.randn(4).astype(np.float32)

    # Conv2 weight [oc=3, ic=4, 3, 3].
    w2 = (rng.randn(3, 4, 3, 3) * 2.0).astype(np.float32)
    b2 = rng.randn(3).astype(np.float32)

    w1_init = helper.make_tensor("w1", TensorProto.FLOAT, w1.shape, w1.flatten().tolist())
    b1_init = helper.make_tensor("b1", TensorProto.FLOAT, b1.shape, b1.flatten().tolist())
    w2_init = helper.make_tensor("w2", TensorProto.FLOAT, w2.shape, w2.flatten().tolist())
    b2_init = helper.make_tensor("b2", TensorProto.FLOAT, b2.shape, b2.flatten().tolist())

    # CLE only supports Conv nodes that carry an explicit `group` attribute.
    conv1 = helper.make_node("Conv", inputs=["input", "w1", "b1"], outputs=["conv1_out"], name="Conv1", group=1)
    relu = helper.make_node("Relu", inputs=["conv1_out"], outputs=["relu_out"], name="Relu1")
    conv2 = helper.make_node("Conv", inputs=["relu_out", "w2", "b2"], outputs=["output"], name="Conv2", group=1)

    graph = helper.make_graph(
        nodes=[conv1, relu, conv2],
        name="CLEGraph",
        inputs=[X],
        outputs=[Y],
        initializer=[w1_init, b1_init, w2_init, b2_init],
    )
    model = helper.make_model(graph, producer_name="test", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    onnx_model_path = Path(output_dir, "cle_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "cle_model_optimized.onnx").as_posix()
    onnx.save(model, onnx_model_path)
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "cross_layer_equalization.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_cross_layer_equalization": {
                "cross_layer_equalization": True,
                "cle_steps": 1,
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def _run(model_path, x):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess.run(None, {"input": x})[0]


def _get_weight(model_path, name):
    model = onnx.load(model_path)
    for init in model.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    raise KeyError(name)


class TestShapeshifterCLE(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_cross_layer_equalization(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["shapeshifter", yaml_path])

        # The pass must have produced an output model.
        self.assertTrue(Path(onnx_optimized_model_path).exists())
        onnx.checker.check_model(onnx.load(onnx_optimized_model_path))

        # CLE must have modified the weights (per-channel rescaling applied).
        w1_before = _get_weight(onnx_model_path, "w1")
        w1_after = _get_weight(onnx_optimized_model_path, "w1")
        self.assertFalse(np.allclose(w1_before, w1_after), "CLE did not modify Conv1 weights.")

        # CLE is function-preserving: outputs must match within tolerance.
        x = np.random.RandomState(1).randn(1, 2, 8, 8).astype(np.float32)
        y_before = _run(onnx_model_path, x)
        y_after = _run(onnx_optimized_model_path, x)
        np.testing.assert_allclose(y_before, y_after, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
