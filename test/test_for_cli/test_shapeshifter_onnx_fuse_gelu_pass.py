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
from onnx import version_converter

from quark.common.utils.testing_utils import use_temporary_directory
from quark.experimental.cli.main import main as cli


class GeluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        return self.gelu(x)


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = GeluModel()

    dummy_input = torch.randn([1])
    onnx_model_path = Path(output_dir, "gelu_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "gelu_optimized.onnx").as_posix()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        dynamo=False,
    )

    print(f"Model has been saved to {onnx_model_path}")
    model = onnx.load(onnx_model_path)
    converted_model = version_converter.convert_version(model, 20)
    onnx.save(converted_model, onnx_model_path)
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "fuse_gelu.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_fuse_gelu": {
                "fuse_gelu": True,
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_gelu(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)

    gelu_count = 0
    for node in model.graph.node:
        if node.op_type == "Gelu":
            gelu_count += 1

    return gelu_count > 0


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_fuse_gelu(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["shapeshifter", yaml_path])
        flag = check_gelu(onnx_optimized_model_path)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
