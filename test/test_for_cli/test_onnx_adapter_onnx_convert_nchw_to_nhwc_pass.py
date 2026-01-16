#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
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
        model, dummy_input, onnx_model_path, input_names=["input"], output_names=["output"], opset_version=17
    )

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "convert_nchw_to_nhwc.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {"onnx_convert_nchw_to_nhwc": {"convert_nchw_to_nhwc": True}},
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_nhwc(onnx_optimized_model_path):
    _ = onnx.load(onnx_optimized_model_path)
    input_data = np.random.randn(1, 4, 4, 3).astype(np.float32)
    sess = ort.InferenceSession(onnx_optimized_model_path)
    input_meta = sess.get_inputs()
    input_name = input_meta[0].name
    _ = sess.run(None, {input_name: input_data})
    return True


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_convert_nchw_to_nhwc_pass(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["onnx-adapter", yaml_path])
        flag = check_nhwc(onnx_optimized_model_path)
        self.assertEqual(flag, True)

    @use_temporary_directory
    def test_onnx_adapter_selective_nodes_conversion(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)

        # Configure selective nodes conversion
        yaml_path = Path(tmpdir, "selective_nodes.yaml").as_posix()
        config = {
            "input_model_path": onnx_model_path,
            "passes": {"onnx_convert_nchw_to_nhwc": {"convert_nchw_to_nhwc": ["input", "output"]}},
            "output_model_path": onnx_optimized_model_path,
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

        cli(["onnx-adapter", yaml_path])
        flag = check_nhwc(onnx_optimized_model_path)
        self.assertEqual(flag, True)

    @use_temporary_directory
    def test_onnx_adapter_nonexistent_nodes_no_conversion(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)

        yaml_path = Path(tmpdir, "nonexistent_nodes.yaml").as_posix()
        config = {
            "input_model_path": onnx_model_path,
            "passes": {"onnx_convert_nchw_to_nhwc": {"convert_nchw_to_nhwc": ["foo", "bar"]}},
            "output_model_path": onnx_optimized_model_path,
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

        cli(["onnx-adapter", yaml_path])
        model = onnx.load(onnx_optimized_model_path)

        inp_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        self.assertEqual(inp_shape, [1, 3, 4, 4])


if __name__ == "__main__":
    unittest.main()
