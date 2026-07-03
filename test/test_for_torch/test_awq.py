#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import importlib
import os
import tempfile
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from quark.torch.algorithm.awq.scale import apply_scale


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(2 * output_size, 2 * output_size)

    def forward(self, x):
        x = self.gelu(x)
        x = self.fc1(x)
        x = torch.cat((x, x), dim=1)
        x = self.fc2(x)
        return x


def test_apply_scale_for_gelu_fc():
    model = SimpleNN(input_size=6, output_size=6)
    scale = torch.Tensor([0.8687, 1.0146, 0.8218, 0.8765, 0.8521, 0.9272])
    scales_list = [("gelu", ("fc1",), scale)]
    weight_old = model.fc1.weight.data
    weight_golden = weight_old * scale
    apply_scale(model, scales_list)
    assert torch.equal(model.fc1.weight.data, weight_golden)


def test_apply_scale_for_fc_fc():
    model = SimpleNN(input_size=3, output_size=3)
    scale = torch.Tensor([0.8687, 1.0146, 0.8218, 0.8765, 0.8521, 0.9272])
    scales_list = [("fc1", ("fc2",), scale)]
    weight_old = model.fc2.weight.data
    weight_old.mul_(scale.to(model.fc2.weight.device).view(1, -1))
    apply_scale(model, scales_list, num_attention_heads=2, num_key_value_heads=1)
    assert torch.equal(model.fc2.weight.data, weight_old)


def test_awq_save_activation_scales():
    """Test that AwqProcessor.apply() saves activation scales when QUARK_SAVE_ACTIVATION_SCALES is enabled."""
    from quark.torch.algorithm.awq.awq import AwqProcessor

    # Bypass __init__ and set only the attributes needed by apply()
    processor = object.__new__(AwqProcessor)
    processor.using_accelerate = False
    processor.modules = []  # empty so the AWQ loop doesn't execute
    processor.model = MagicMock()
    processor.model.config._attn_implementation = "eager"
    processor.recover_attn_implementation = "eager"
    processor.global_scales_list = [("layer.0", ("fc1",), torch.ones(10))]

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        scales_file = f.name

    try:
        with (
            patch("quark.torch.algorithm.awq.awq.QUARK_SAVE_ACTIVATION_SCALES", True),
            patch("quark.torch.algorithm.awq.awq.QUARK_ACTIVATION_SCALES_FILENAME", scales_file),
        ):
            processor.apply()

        # Verify the scales were saved correctly
        loaded = torch.load(scales_file, weights_only=False)
        assert len(loaded) == 1
        assert loaded[0][0] == "layer.0"
        assert loaded[0][1] == ("fc1",)
        assert torch.equal(loaded[0][2], torch.ones(10))
    finally:
        os.unlink(scales_file)


def test_awq_memory_optimization_constant():
    """Test that QUARK_AWQ_MEMORY_OPTIMIZATION constant is correctly read from the environment."""
    import quark.torch.utils.constants as constants_module

    # Verify default (env var not set) is False
    assert constants_module.QUARK_AWQ_MEMORY_OPTIMIZATION is False

    # Reload with env var set to verify the constant and its log message are exercised
    with patch.dict(os.environ, {"QUARK_AWQ_MEMORY_OPTIMIZATION": "1"}):
        importlib.reload(constants_module)
        assert constants_module.QUARK_AWQ_MEMORY_OPTIMIZATION is True

    # Restore original state
    importlib.reload(constants_module)
    assert constants_module.QUARK_AWQ_MEMORY_OPTIMIZATION is False
