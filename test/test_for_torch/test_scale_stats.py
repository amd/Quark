#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import tempfile
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quark.common.utils.testing_utils import torch_device
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Int8PerTensorSpec, QConfig, QLayerConfig


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 12
        self.intermediate_size = 24
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


INT8_PER_TENSOR_SYM_SPEC = Int8PerTensorSpec(is_dynamic=False).to_quantization_spec()


tempdir = tempfile.TemporaryDirectory()


@patch("quark.torch.quantization.debug.SCALE_DEBUG_DIR", tempdir.name)
@patch("quark.torch.quantization.api.QUARK_CHECK_SCALE", True)
def test_smoke_check_scale_stats():
    model = TestModel()
    model = model.to(torch.float16).to(torch_device)
    global_quant_config = QLayerConfig(
        input_tensors=INT8_PER_TENSOR_SYM_SPEC, output_tensors=INT8_PER_TENSOR_SYM_SPEC, weight=INT8_PER_TENSOR_SYM_SPEC
    )
    config = QConfig(global_quant_config=global_quant_config)
    quantizer = ModelQuantizer(config)
    dataloader = DataLoader([torch.rand((12, 12), dtype=torch.float16).to(torch_device)] * 2)

    try:
        quantizer.quantize_model(model, dataloader=dataloader)
    except BaseException as e:
        assert isinstance(e, SystemExit)

    files = os.listdir(tempdir.name)
    assert "scale_stats.json" in files

    tempdir.cleanup()
