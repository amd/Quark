#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn

from quark.torch.algorithm.awq.scale import apply_scale

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
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
    scales_list = [('gelu', ('fc1',), scale)]
    weight_old = model.fc1.weight.data
    weight_golden = weight_old * scale
    apply_scale(model, scales_list)
    assert torch.equal(model.fc1.weight.data, weight_golden)

def test_apply_scale_for_fc_fc():
    model = SimpleNN(input_size=3, output_size=3)
    scale = torch.Tensor([0.8687, 1.0146, 0.8218, 0.8765, 0.8521, 0.9272])
    scales_list = [('fc1', ('fc2',), scale)]
    weight_old = model.fc2.weight.data
    weight_old.mul_(scale.to(model.fc2.weight.device).view(1, -1))
    apply_scale(model, scales_list, num_attention_heads = 2, num_key_value_heads = 1)
    assert torch.equal(model.fc2.weight.data, weight_old)
