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


def prepare_bn_model(output_dir):
    input_shape = [1, 3, 4, 4]

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)

    input_value_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)

    C = 3

    gamma = np.ones((C,), dtype=np.float32)
    beta = np.zeros((C,), dtype=np.float32)
    mean = np.zeros((C,), dtype=np.float32)
    var = np.ones((C,), dtype=np.float32)

    gamma_init = helper.make_tensor("gamma", TensorProto.FLOAT, gamma.shape, gamma)
    beta_init = helper.make_tensor("beta", TensorProto.FLOAT, beta.shape, beta)
    mean_init = helper.make_tensor("mean", TensorProto.FLOAT, mean.shape, mean)
    var_init = helper.make_tensor("var", TensorProto.FLOAT, var.shape, var)

    bn_node = helper.make_node(
        "BatchNormalization",
        inputs=["input", "gamma", "beta", "mean", "var"],
        outputs=["bn_out"],
        epsilon=1e-5,
        name="BatchNorm_ToBeConverted",
    )

    output_tensor = helper.make_tensor_value_info("bn_out", TensorProto.FLOAT, input_shape)

    output_value_info = helper.make_tensor_value_info("bn_out", TensorProto.FLOAT, input_shape)

    graph = helper.make_graph(
        nodes=[bn_node],
        name="BN_to_Conv_Graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        value_info=[input_value_info, output_value_info],
        initializer=[gamma_init, beta_init, mean_init, var_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    onnx.checker.check_model(model)

    onnx_model_path = Path(output_dir, "bn_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "bn_model_optimized.onnx").as_posix()
    onnx.save(model, onnx_model_path)

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "convert_bn_to_conv.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_convert_bn_to_conv": {
                "convert_bn_to_conv": True,
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_batch_norm(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)

    bn_count = 0
    for node in model.graph.node:
        if node.op_type == "BatchNormalization":
            bn_count += 1

    return bn_count == 0


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_convert_bn_to_conv(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_bn_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["onnx-adapter", yaml_path])
        flag = check_batch_norm(onnx_optimized_model_path)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
