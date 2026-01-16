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


def prepare_convT_bn_model(output_dir):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])

    W = np.random.randn(3, 4, 3, 3).astype(np.float32)
    W_initializer = numpy_helper.from_array(W, "W")

    scale = np.ones(4, dtype=np.float32)
    bias = np.zeros(4, dtype=np.float32)
    mean = np.zeros(4, dtype=np.float32)
    var = np.ones(4, dtype=np.float32)
    scale_init = numpy_helper.from_array(scale, "scale")
    bias_init = numpy_helper.from_array(bias, "bias")
    mean_init = numpy_helper.from_array(mean, "mean")
    var_init = numpy_helper.from_array(var, "var")

    convT_node = helper.make_node("ConvTranspose", ["X", "W"], ["convT_out"], name="ConvTranspose")
    bn_node = helper.make_node("BatchNormalization", ["convT_out", "scale", "bias", "mean", "var"], ["Y"], name="BN")

    graph = helper.make_graph(
        [convT_node, bn_node],
        "ConvTransposeBNGraph",
        [X],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 10, 10])],
        initializer=[W_initializer, scale_init, bias_init, mean_init, var_init],
    )

    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_operatorsetid("", 22)])
    onnx.checker.check_model(model)

    onnx_model_path = Path(output_dir, "convT_bn_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "convT_bn_model_optimized.onnx").as_posix()
    onnx.save(model, onnx_model_path)

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_gemm_bn_model(output_dir):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])

    W = np.random.randn(4, 8).astype(np.float32)
    B = np.random.randn(4).astype(np.float32)

    W_init = numpy_helper.from_array(W, "W")
    B_init = numpy_helper.from_array(B, "B")

    scale = np.ones(4, dtype=np.float32)
    bias = np.zeros(4, dtype=np.float32)
    mean = np.zeros(4, dtype=np.float32)
    var = np.ones(4, dtype=np.float32)

    scale_init = numpy_helper.from_array(scale, "scale")
    bias_init = numpy_helper.from_array(bias, "bias")
    mean_init = numpy_helper.from_array(mean, "mean")
    var_init = numpy_helper.from_array(var, "var")

    gemm_node = helper.make_node("Gemm", ["X", "W", "B"], ["gemm_out"], name="Gemm", transB=1)

    bn_node = helper.make_node("BatchNormalization", ["gemm_out", "scale", "bias", "mean", "var"], ["Y"], name="BN")

    graph = helper.make_graph(
        [gemm_node, bn_node],
        "GemmBNGraph_transB1",
        [X],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
        initializer=[W_init, B_init, scale_init, bias_init, mean_init, var_init],
    )

    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_operatorsetid("", 22)])

    onnx.checker.check_model(model)

    onnx_model_path = Path(output_dir, "gemm_bn_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "gemm_bn_model_optimized.onnx").as_posix()
    onnx.save(model, onnx_model_path)

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "fold_batch_norm.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_fold_batch_norm": {
                "fold_batch_norm": True,
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
    def test_onnx_adapter_onnx_fold_batch_norm_pass_convT_bn(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_convT_bn_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["onnx-adapter", yaml_path])
        flag = check_batch_norm(onnx_optimized_model_path)
        self.assertTrue(flag)

    @use_temporary_directory
    def test_onnx_adapter_onnx_fold_batch_norm_pass_gemm_bn(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_gemm_bn_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["onnx-adapter", yaml_path])
        flag = check_batch_norm(onnx_optimized_model_path)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
