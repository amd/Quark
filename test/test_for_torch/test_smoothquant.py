#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Config, QuantizationConfig, QuantizationSpec, SmoothQuantConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver

in_feat = 32 * 128
out_feat = 64 * 128


class MySubModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_norm = nn.LayerNorm(in_feat, bias=False)
        self.lin1 = nn.Linear(in_feat, out_feat, bias=False)
        self.lin1.weight.data = torch.normal(0, 1, (out_feat, in_feat))

    def forward(self, x, y):
        x = self.layer_norm(x)
        x = self.lin1(x)
        return x + y


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # We put the Linear + LayerNorm in a ModuleList, which is expected by Quark,
        # as the implementation is tailored for multi-layer transformer models.
        self.layers = nn.ModuleList([MySubModule() for i in range(1)])
        self.device = "cpu"

    def forward(self, x):
        y = torch.rand(1, out_feat)
        for layer in self.layers:
            x = layer(x, y=y)
        return x


def test_vanilla_nn_module():
    model = MyModel()
    model = model.eval()

    # Create reference tensor with long tail.
    inp = torch.empty(1, in_feat)
    inp.cauchy_(sigma=5e-3)

    # Quantize the model using smoothquant.
    quant_spec = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=False,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
        ch_axis=None,
        group_size=None,
    )
    global_config = QuantizationConfig(weight=quant_spec, input_tensors=quant_spec)
    quant_config = Config(global_quant_config=global_config, algo_config=[])

    pre_quant_optimization = SmoothQuantConfig(
        scaling_layers=[{"prev_op": "layer_norm", "layers": ["lin1"], "inp": "lin1"}],
        model_decoder_layers="layers",
        alpha=0.5,
        scale_clamp_min=1e-12,
    )
    quant_config.algo_config.append(pre_quant_optimization)

    quantizer = ModelQuantizer(quant_config)
    calib_dataloader = DataLoader([{"x": inp}])

    quant_model_smooth = quantizer.quantize_model(model, calib_dataloader)
    quant_model_smooth = quant_model_smooth.eval()

    with torch.no_grad():
        _ = quant_model_smooth(inp)
