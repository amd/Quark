#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from quark.shares.utils.testing_utils import torch_device
from quark.torch import ModelQuantizer
from quark.torch.quantization import AutoSmoothQuantConfig, Int8PerTensorSpec, QConfig, QLayerConfig
from quark.torch.quantization.nn.modules.quantize_linear import QLoRaQuantLinear, QuantLinear
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase

INT8_PER_TENSOR_SPEC = Int8PerTensorSpec(is_dynamic=False).to_quantization_spec()

DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG = QLayerConfig(
    input_tensors=INT8_PER_TENSOR_SPEC,
    weight=INT8_PER_TENSOR_SPEC,
    bias=INT8_PER_TENSOR_SPEC,
    output_tensors=INT8_PER_TENSOR_SPEC,
)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


input_tensor = torch.randn(1, 64, 64)


def test_net():
    class MyDataset(Dataset):
        def __init__(self):
            return

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return input_tensor

    model = SimpleCNN(num_classes=10)
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    quant_config = QConfig(global_quant_config=DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG)
    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(model, dataloader)

    assert (
        hasattr(quant_model.fc, "_input_quantizer")
        and hasattr(quant_model.fc, "_weight_quantizer")
        and hasattr(quant_model.fc, "_bias_quantizer")
        and hasattr(quant_model.fc, "_output_quantizer")
    )


def test_quantlinear_eq_qloraquantlinear():
    empty_quant_config = QLayerConfig()
    in_feature = 128
    out_feature = 256
    qlinear = (
        QuantLinear(
            in_feature,
            out_feature,
            device=torch_device,
            bias=True,
            quant_config=empty_quant_config,
        )
        .to(torch_device)
        .to(torch.bfloat16)
    )
    qlinear.weight.data = torch.ones_like(qlinear.weight.data)

    qloralinear = (
        QLoRaQuantLinear(
            in_feature,
            out_feature,
            device=torch_device,
            bias=True,
            quant_config=empty_quant_config,
        )
        .to(torch_device)
        .to(torch.bfloat16)
    )

    qloralinear.weight.data = qlinear.weight.data
    qloralinear.bias.data = qlinear.bias.data

    example_inputs = torch.ones(1, 128).to(torch_device).to(torch.bfloat16)

    qlinear(example_inputs)
    qloralinear(example_inputs)
    qloralinear.active_adapters = True
    qloralinear(example_inputs)


@pytest.mark.parametrize("algo", [pytest.param(val, id=f"algo:{val}") for val in ["autosmoothquant", None]])
def test_quantizer_enabled_disabled(monkeypatch, algo: None | str):
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-random-LlamaForCausalLM")
    model = AutoModelForCausalLM.from_config(config)

    int8_act_spec = Int8PerTensorSpec(is_dynamic=True).to_quantization_spec()
    int8_weight_spec = Int8PerTensorSpec(is_dynamic=False).to_quantization_spec()

    layer_config = QLayerConfig(
        input_tensors=int8_act_spec,
        weight=int8_weight_spec,
    )

    if algo == "autosmoothquant":
        algo_config = [AutoSmoothQuantConfig(model_decoder_layers="model.layers", scaling_layers=[])]
    else:
        algo_config = []

    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-random-LlamaForCausalLM")
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"])

    original_method = ModelQuantizer._do_calibration

    def patched_method(self, model, dataloader):
        model = original_method(self, model, dataloader)
        for name, module in model.named_modules():
            if isinstance(module, FakeQuantizeBase):
                assert module.is_fake_quant_enabled

                if "_weight_quantizer" not in name:
                    assert module.is_dynamic
                    assert module.observer_enabled

        return model

    monkeypatch.setattr(ModelQuantizer, "_do_calibration", patched_method)

    quant_config = QConfig(global_quant_config=layer_config, algo_config=algo_config)
    quantizer = ModelQuantizer(quant_config)
    _ = quantizer.quantize_model(model, calib_dataloader)
