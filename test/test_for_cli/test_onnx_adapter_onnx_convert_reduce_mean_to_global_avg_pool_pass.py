#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import unittest
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import yaml

from quark.experimental.cli.main import main as cli
from quark.shares.utils.testing_utils import use_temporary_directory


class ReduceMeanModel(nn.Module):
    def __init__(self):
        super(ReduceMeanModel, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        return x


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = ReduceMeanModel()

    dummy_input = torch.randn(1, 3, 4, 4)
    onnx_model_path = Path(output_dir, "reduce_mean_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "reduce_mean_model_optimized.onnx").as_posix()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
    )

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "convert_reduce_mean_to_global_avg_pool.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_convert_reduce_mean_to_global_avg_pool": {
                "convert_reduce_mean_to_global_avg_pool": True,
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_reduce_mean(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)

    reduce_mean_count = 0
    for node in model.graph.node:
        if node.op_type == "ReduceMean":
            reduce_mean_count += 1

    return reduce_mean_count == 0


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_convert_reduce_mean_to_global_avg_pool(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["onnx-adapter", yaml_path])
        flag = check_reduce_mean(onnx_optimized_model_path)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
