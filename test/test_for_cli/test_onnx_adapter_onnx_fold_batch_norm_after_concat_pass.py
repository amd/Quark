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


def prepare_convtranspose_conv_model(output_dir):
    onnx_model_path = Path(output_dir, "convtranspose_conv_bn_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "convtranspose_conv_bn_model_optimized.onnx").as_posix()

    N = 1
    H = 4
    W = 4

    conv_C_in = 3
    conv_C_out = 3
    conv_kernel = (1, 1)

    ct_C_in = 2
    ct_C_out = 4
    ct_kernel = (1, 1)

    total_C = conv_C_out + ct_C_out  # 3 + 4 = 7

    X_conv = helper.make_tensor_value_info("X_conv", TensorProto.FLOAT, [N, conv_C_in, H, W])
    X_ct = helper.make_tensor_value_info("X_ct", TensorProto.FLOAT, [N, ct_C_in, H, W])

    conv_W = np.random.randn(conv_C_out, conv_C_in, conv_kernel[0], conv_kernel[1]).astype(np.float32)
    conv_W_init = numpy_helper.from_array(conv_W, name="conv_W")

    ct_W = np.random.randn(ct_C_in, ct_C_out, ct_kernel[0], ct_kernel[1]).astype(np.float32)
    ct_W_init = numpy_helper.from_array(ct_W, name="ct_W")

    bn_scale = np.ones((total_C,), dtype=np.float32)
    bn_bias = np.zeros((total_C,), dtype=np.float32)
    bn_mean = np.zeros((total_C,), dtype=np.float32)
    bn_var = np.ones((total_C,), dtype=np.float32)

    bn_scale_init = numpy_helper.from_array(bn_scale, name="bn_scale")
    bn_bias_init = numpy_helper.from_array(bn_bias, name="bn_bias")
    bn_mean_init = numpy_helper.from_array(bn_mean, name="bn_mean")
    bn_var_init = numpy_helper.from_array(bn_var, name="bn_var")

    conv_node = helper.make_node(
        "Conv",
        inputs=["X_conv", "conv_W"],
        outputs=["conv_out"],
        name="Conv_left",
    )

    convT_node = helper.make_node(
        "ConvTranspose",
        inputs=["X_ct", "ct_W"],
        outputs=["ct_out"],
        name="ConvTranspose_right",
    )

    concat_node = helper.make_node(
        "Concat",
        inputs=["conv_out", "ct_out"],
        outputs=["concat_out"],
        axis=1,
        name="Concat_channels",
    )

    bn_node = helper.make_node(
        "BatchNormalization",
        inputs=["concat_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        outputs=["Y"],
        name="BN_after_concat",
    )

    graph = helper.make_graph(
        nodes=[conv_node, convT_node, concat_node, bn_node],
        name="ConcatBNGraph",
        inputs=[X_conv, X_ct],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, total_C, H, W])],
        initializer=[conv_W_init, ct_W_init, bn_scale_init, bn_bias_init, bn_mean_init, bn_var_init],
    )

    model = helper.make_model(
        graph,
        producer_name="minimal_concat_bn_for_foldpass",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 22)],
    )

    onnx.checker.check_model(model)

    onnx.save_model(model, onnx_model_path)

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_gemm_model(output_dir):
    onnx_model_path = Path(output_dir, "gemm_bn_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "gemm_bn_model_optimized.onnx").as_posix()

    N = 1
    H = 4
    W = 4

    conv_C_in = 3
    conv_C_out = 3
    conv_kernel = (1, 1)

    ct_C_in = 2
    ct_C_out = 4
    ct_kernel = (1, 1)

    total_C = conv_C_out + ct_C_out

    X_conv = helper.make_tensor_value_info("X_conv", TensorProto.FLOAT, [N, conv_C_in, H, W])
    X_ct = helper.make_tensor_value_info("X_ct", TensorProto.FLOAT, [N, ct_C_in, H, W])

    conv_W = np.random.randn(conv_C_out, conv_C_in, conv_kernel[0], conv_kernel[1]).astype(np.float32)
    conv_W_init = numpy_helper.from_array(conv_W, name="conv_W")

    ct_W = np.random.randn(ct_C_in, ct_C_out, ct_kernel[0], ct_kernel[1]).astype(np.float32)
    ct_W_init = numpy_helper.from_array(ct_W, name="ct_W")

    bn_scale = np.ones((total_C,), dtype=np.float32)
    bn_bias = np.zeros((total_C,), dtype=np.float32)
    bn_mean = np.zeros((total_C,), dtype=np.float32)
    bn_var = np.ones((total_C,), dtype=np.float32)

    bn_scale_init = numpy_helper.from_array(bn_scale, name="bn_scale")
    bn_bias_init = numpy_helper.from_array(bn_bias, name="bn_bias")
    bn_mean_init = numpy_helper.from_array(bn_mean, name="bn_mean")
    bn_var_init = numpy_helper.from_array(bn_var, name="bn_var")

    conv_node = helper.make_node(
        "Conv",
        inputs=["X_conv", "conv_W"],
        outputs=["conv_out"],
        name="Conv_left",
    )

    convT_node = helper.make_node(
        "ConvTranspose",
        inputs=["X_ct", "ct_W"],
        outputs=["ct_out"],
        name="ConvTranspose_right",
    )

    concat_node = helper.make_node(
        "Concat",
        inputs=["conv_out", "ct_out"],
        outputs=["concat_out"],
        axis=1,
        name="Concat_channels",
    )

    bn_node = helper.make_node(
        "BatchNormalization",
        inputs=["concat_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        outputs=["Y"],
        name="BN_after_concat",
    )

    graph = helper.make_graph(
        nodes=[conv_node, convT_node, concat_node, bn_node],
        name="ConcatBNGraph",
        inputs=[X_conv, X_ct],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, total_C, H, W])],
        initializer=[conv_W_init, ct_W_init, bn_scale_init, bn_bias_init, bn_mean_init, bn_var_init],
    )

    model = helper.make_model(
        graph,
        producer_name="minimal_concat_bn_for_foldpass",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 22)],
    )

    onnx.checker.check_model(model)

    onnx.save_model(model, onnx_model_path)

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "fold_batch_norm_after_concat.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_fold_batch_norm_after_concat": {
                "fold_batch_norm_after_concat": True,
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
    def test_onnx_adapter_onnx_fold_batch_norm_after_concat_pass_convtranspose_conv(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_convtranspose_conv_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["onnx-adapter", yaml_path])
        flag = check_batch_norm(onnx_optimized_model_path)
        self.assertTrue(flag)

    @use_temporary_directory
    def test_onnx_adapter_onnx_fold_batch_norm_after_concat_pass_gemm(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_gemm_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["onnx-adapter", yaml_path])
        flag = check_batch_norm(onnx_optimized_model_path)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
