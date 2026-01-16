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


class DoubleConvModel(nn.Module):
    def __init__(self):
        super(DoubleConvModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1, 1)

        with torch.no_grad():
            self.conv4.weight *= 100.0
            self.conv3.weight = self.conv2.weight

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
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
    yaml_path = Path(output_dir, "copy_shared_init.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_copy_shared_init": {
                "shared_init_op_types": ["Conv", "ConvTranspose", "Gemm"],
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_shared_init(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)
    initializer_usage_count = {init.name: 0 for init in model.graph.initializer}

    for node in model.graph.node:
        for input_name in node.input:
            if input_name in initializer_usage_count:
                initializer_usage_count[input_name] += 1

    for count in initializer_usage_count.values():
        if count > 1:
            return False

    return True


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_copy_shared_init_pass(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["onnx-adapter", yaml_path])
        flag = check_shared_init(onnx_optimized_model_path)
        self.assertEqual(flag, True)


if __name__ == "__main__":
    unittest.main()
