#!/usr/bin/env python
# Created Time: 2024-10-25 13:25

#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from quark.torch import ModelQuantizer, load_params, save_params
from quark.torch.quantization import (
    Int8PerChannelSpec,
    Int8PerTensorSpec,
    QConfig,
    QLayerConfig,
    Uint4PerChannelSpec,
    Uint4PerTensorSpec,
    Uint8PerTensorSpec,
)
from quark.torch.quantization.config.type import QuantizationMode
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

input_tensor = torch.tensor([0, 1], dtype=torch.long)
offsets = torch.tensor([0], dtype=torch.long)


class SimpleDLRM(nn.Module):
    def __init__(self, padding_idx):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(4, 128, mode="sum", padding_idx=padding_idx)
        self.fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=10), nn.ReLU(), nn.Linear(in_features=10, out_features=10)
        )

    def forward(self, indices, offsets):
        x = self.embedding_bag(indices, offsets)
        x = self.fc(x)
        return x


class SimpleEmbed(nn.Module):
    def __init__(self, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(4, 128, padding_idx=padding_idx)
        self.fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=10), nn.ReLU(), nn.Linear(in_features=10, out_features=10)
        )

    def forward(self, indices):
        x = self.embedding(indices)
        x = self.fc(x)
        return x


class EmbeddingDataset(Dataset):
    def __init__(self):
        return

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return input_tensor


class EmbeddingBagDataset(Dataset):
    def __init__(self):
        return

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return input_tensor, offsets


def test_net():
    INT8_PER_TENSOR_SPEC = Int8PerTensorSpec(
        observer_method="histogrampro", symmetric=False, is_dynamic=False
    ).to_quantization_spec()

    UINT8_PER_TENSOR_SPEC = Uint8PerTensorSpec(observer_method="histogrampro", is_dynamic=False).to_quantization_spec()

    UINT4_PER_TENSOR_SPEC = Uint4PerTensorSpec(observer_method="histogrampro", is_dynamic=True).to_quantization_spec()
    params = []
    for observal_val in [INT8_PER_TENSOR_SPEC, UINT8_PER_TENSOR_SPEC, UINT4_PER_TENSOR_SPEC]:
        for zero_point_type in ["int32", "float32"]:
            for padding_idx in [-1, 0, 1]:
                params.append((observal_val, zero_point_type, padding_idx))

    for para in params:
        observal_val, zero_point_type, padding_idx = para
        print(f"observal_val is {observal_val}, zero_point_type is {zero_point_type}, padding_idx is {padding_idx}")
        INT8_PER_CHANNEL_SPEC = Int8PerChannelSpec(ch_axis=0, is_dynamic=False).to_quantization_spec()
        quant_config = QLayerConfig(
            input_tensors=observal_val, weight=INT8_PER_CHANNEL_SPEC, output_tensors=observal_val, bias=observal_val
        )

        INT4_PER_TENSOR_SPEC = Uint4PerChannelSpec(
            ch_axis=0,
            is_dynamic=False,
            zero_point_type=zero_point_type,
        ).to_quantization_spec()
        layer_type_quant_config = {
            torch.nn.modules.sparse.EmbeddingBag: QLayerConfig(weight=INT4_PER_TENSOR_SPEC),
            torch.nn.modules.sparse.Embedding: QLayerConfig(weight=INT4_PER_TENSOR_SPEC),
        }
        quant_config = QConfig(
            global_quant_config=quant_config,
            layer_type_quant_config=layer_type_quant_config,
            quant_mode=QuantizationMode.eager_mode,
        )

        print("==================================Begin test embeddingbag =================================")
        model = SimpleDLRM(padding_idx)

        quantizer = ModelQuantizer(quant_config)
        quant_model = quantizer.quantize_model(model, [])

        for module in quant_model.modules():
            if isinstance(module, ScaledFakeQuantize):
                module.enable_observer()
                module.enable_observer()

        quant_model(input_tensor, offsets)

        for module in quant_model.modules():
            if isinstance(module, ScaledFakeQuantize):
                module.disable_observer()
                module.enable_fake_quant()

        assert hasattr(quant_model.embedding_bag, "_weight_quantizer")

        quantized_model = quantizer.freeze(quant_model.eval())

        quant_model(input_tensor, offsets)
        quantized_model(input_tensor, offsets)

        print("==============================Begin test embedding ========================================")
        model = SimpleEmbed(padding_idx)

        quantizer = ModelQuantizer(quant_config)
        quant_model = quantizer.quantize_model(model, [])

        for module in quant_model.modules():
            if isinstance(module, ScaledFakeQuantize):
                module.enable_observer()
                module.enable_observer()

        quant_model(input_tensor)

        for module in quant_model.modules():
            if isinstance(module, ScaledFakeQuantize):
                module.disable_observer()
                module.enable_fake_quant()

        assert hasattr(quant_model.embedding, "_weight_quantizer")
        quantized_model = quantizer.freeze(quant_model.eval())

        quant_model(input_tensor)
        quantized_model(input_tensor)

        with tempfile.TemporaryDirectory() as tmpdirname:
            export_dir = Path(tmpdirname, "qembed_output")

            save_params(quantized_model, model_type="embed", export_dir=export_dir, compressed=True)

            json_path = Path(export_dir, "embed.json")
            safetensors_path = Path(export_dir, "embed.safetensors")
            load_params(model, json_path=json_path, safetensors_path=safetensors_path, compressed=True)


if __name__ == "__main__":
    test_net()
