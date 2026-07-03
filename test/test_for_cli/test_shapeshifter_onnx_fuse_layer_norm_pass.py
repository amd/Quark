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


class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, emb_size), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class LayerNormModel(nn.Module):
    def __init__(self, img_size=4, patch_size=2, in_channels=3, emb_size=64, num_heads=4, mlp_dim=128, num_classes=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, emb_size))
        self.transformer = TransformerBlock(emb_size, num_heads, mlp_dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(emb_size), nn.Linear(emb_size, num_classes))

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0])
        return x


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = LayerNormModel()

    dummy_input = torch.randn([1, 3, 4, 4])
    onnx_model_path = Path(output_dir, "layer_norm_model.onnx").as_posix()
    onnx_optimized_model_path = Path(output_dir, "layer_norm_optimized.onnx").as_posix()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        dynamo=False,
    )

    print(f"Model has been saved to {onnx_model_path}")
    model = onnx.load(onnx_model_path)
    converted_model = version_converter.convert_version(model, 17)
    onnx.save(converted_model, onnx_model_path)
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path, onnx_optimized_model_path):
    yaml_path = Path(output_dir, "fuse_layer_norm.yaml").as_posix()
    config = {
        "input_model_path": onnx_model_path,
        "passes": {
            "onnx_fuse_layer_norm": {
                "fuse_layer_norm": True,
            }
        },
        "output_model_path": onnx_optimized_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def check_layer_norm(onnx_optimized_model_path):
    model = onnx.load(onnx_optimized_model_path)

    layer_norm_count = 0
    for node in model.graph.node:
        if node.op_type == "LayerNormalization":
            layer_norm_count += 1

    return layer_norm_count > 0


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_onnx_adapter_onnx_fuse_layer_norm(self, tmpdir: str):
        onnx_model_path, onnx_optimized_model_path = prepare_model(tmpdir)
        yaml_path = prepare_yaml(tmpdir, onnx_model_path, onnx_optimized_model_path)
        cli(["shapeshifter", yaml_path])
        flag = check_layer_norm(onnx_optimized_model_path)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
