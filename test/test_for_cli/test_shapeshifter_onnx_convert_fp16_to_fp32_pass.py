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
from onnx import TensorProto

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
    model.half()

    dummy_input = torch.randn(1, 3, 4, 4).half()
    onnx_model_path = Path(output_dir, "double_conv_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "double_conv_model_optimized.onnx").as_posix()
    torch.onnx.export(
        model, dummy_input, onnx_model_path, input_names=["input"], output_names=["output"], opset_version=17
    )

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path, subgraphs_to_include=None):
    yaml_path = Path(output_dir, "convert_fp16_to_fp32.yaml").as_posix()
    pass_config = {"convert_fp16_to_fp32": True}
    if subgraphs_to_include is not None:
        pass_config["subgraphs_to_include"] = subgraphs_to_include
    config = {
        "input_model_path": onnx_model_path,
        "passes": {"onnx_convert_fp16_to_fp32": pass_config},
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_fp32(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)
    fp32_count = 0
    fp16_count = 0

    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT:
            fp32_count += 1
        elif initializer.data_type == onnx.TensorProto.FLOAT16:
            fp16_count += 1

    def _check_value_info_dtype(value_info):
        if value_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
            return "fp32"
        elif value_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            return "fp16"
        return None

    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        dtype = _check_value_info_dtype(vi)
        if dtype == "fp32":
            fp32_count += 1
        elif dtype == "fp16":
            fp16_count += 1
    if fp32_count > 0 and fp16_count == 0:
        return True
    else:
        return False


def check_partial_fp32(onnx_optimized_model_path):
    """Verify that the model has both fp32 and fp16 tensors (partial conversion)
    and contains Cast nodes at the subgraph boundaries."""
    model = onnx.load(onnx_optimized_model_path)
    has_fp32 = False
    has_fp16 = False

    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            has_fp32 = True
        elif init.data_type == TensorProto.FLOAT16:
            has_fp16 = True

    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        elem_type = vi.type.tensor_type.elem_type
        if elem_type == TensorProto.FLOAT:
            has_fp32 = True
        elif elem_type == TensorProto.FLOAT16:
            has_fp16 = True

    has_boundary_casts = any(n.op_type == "Cast" for n in model.graph.node)
    return has_fp32 and has_fp16 and has_boundary_casts


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_convert_fp16_to_fp32_pass(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["shapeshifter", yaml_path])
        flag = check_fp32(onnx_optimized_model_path)
        self.assertEqual(flag, True)

    @use_temporary_directory
    def test_onnx_adapter_onnx_convert_fp16_to_fp32_pass_subgraphs_to_include(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(
            tmpdir,
            onnx_model_path,
            onnx_optimized_model_path,
            subgraphs_to_include=[[["node_conv2d_1"], ["node_view"]]],
        )
        cli(["shapeshifter", yaml_path])
        flag = check_partial_fp32(onnx_optimized_model_path)
        self.assertTrue(flag, "Partial conversion should produce both fp16 and fp32 tensors with boundary Cast nodes")


if __name__ == "__main__":
    unittest.main()
