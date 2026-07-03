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

from quark.common.utils.testing_utils import use_temporary_directory
from quark.experimental.cli.main import main as cli


class DoubleConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1, 1)

        with torch.no_grad():
            self.conv2.weight *= 100.0

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.clip(x, 0, 6)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = DoubleConvModel()

    dummy_input = torch.randn(1, 3, 4, 4)
    onnx_model_path = Path(output_dir, "double_conv_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "double_conv_model_optimized.onnx").as_posix()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "opset.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {"onnx_convert_opset_version": {"target_opset_version": 20}},
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_opset_version(onnx_optimized_model_path, opset_version):
    model = onnx.load(onnx_optimized_model_path)
    model_opset_version = model.opset_import[0].version
    return model_opset_version == opset_version


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_convert_opset_version_pass(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["shapeshifter", yaml_path])
        flag = check_opset_version(onnx_optimized_model_path, 20)
        self.assertEqual(flag, True)


if __name__ == "__main__":
    unittest.main()
