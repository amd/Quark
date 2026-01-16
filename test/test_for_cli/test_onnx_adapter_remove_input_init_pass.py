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


class AddModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("w", nn.Parameter(torch.tensor([1.0, 2.0, 3.0])))

    def forward(self, x):
        return x + self.w


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = AddModel()
    model.eval()

    x = torch.randn(3)

    onnx_model_path = Path(output_dir, "input_init_model.onnx").as_posix()

    torch.onnx.export(
        model,
        (x,),
        onnx_model_path,
        input_names=["x"],
        output_names=["y"],
        opset_version=14,
        do_constant_folding=False,
        keep_initializers_as_inputs=True,
    )

    onnx_optimized_model_path = Path(output_dir, "input_init_model_optimized.onnx").as_posix()
    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "remove_input_init.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {"onnx_remove_input_init": {"remove_input_init": True}},
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_input_init(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)
    input_list = [i.name for i in model.graph.input]
    init_list = [i.name for i in model.graph.initializer]
    return not (set(input_list) & set(init_list))


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_remove_input_init_pass(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["onnx-adapter", yaml_path])
        flag = check_input_init(onnx_optimized_model_path)
        self.assertEqual(flag, True)


if __name__ == "__main__":
    unittest.main()
