#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver, PlaceholderObserver
from dataclasses import replace
from torch.utils.data import Dataset, DataLoader

from quark.torch.algorithm.awq.scale import scale_fc_fc
from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig, AWQConfig, SmoothQuantConfig
from quark.shares.utils.log import ScreenLogger
from quark.shares.utils.testing_utils import torch_device

logger = ScreenLogger(__name__)

INT8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                        qscheme=QSchemeType.per_tensor,
                                        observer_cls=PerTensorMinMaxObserver,
                                        symmetric=True,
                                        scale_type=ScaleType.float,
                                        round_method=RoundType.half_even,
                                        is_dynamic=False)

DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC,
                                                             weight=INT8_PER_TENSOR_SPEC,
                                                             bias=INT8_PER_TENSOR_SPEC,
                                                             output_tensors=INT8_PER_TENSOR_SPEC)

FLOAT16_SPEC = QuantizationSpec(dtype=Dtype.float16, observer_cls=PlaceholderObserver)

FLOAT16_CONFIG = QuantizationConfig(input_tensors=FLOAT16_SPEC, weight=FLOAT16_SPEC)

hidden_size = 32
num_attention_heads = 16
num_key_value_heads = 4

class SimpleLM(nn.Module):
    def __init__(self):
        super(SimpleLM, self).__init__()
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.device = torch_device

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(self, x):
        bsz, q_len, _ = x.size()

        # v_proj
        value_states = self.v_proj(x)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)
        value_states = value_states.transpose(1, 2).contiguous()
        value_states = value_states.reshape(bsz, q_len, -1)

        # o_proj
        x = self.o_proj(value_states)
        return x


def test_gqa_smoothquant():
    class MyDataset(Dataset):

        def __init__(self):
            return

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return input_tensor

    torch.cuda.empty_cache()
    # dataset
    input_tensor = torch.randn(1, 4, hidden_size, device=torch_device)
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # model
    model = SimpleLM().to(torch_device)
    model.num_key_value_groups = num_attention_heads // num_key_value_heads
    output_original = model(input_tensor)

    # algorithm config
    quant_config = Config(global_quant_config=FLOAT16_CONFIG)
    quant_config = replace(quant_config, algo_config=SmoothQuantConfig())
    quant_config.algo_config.num_attention_heads = num_attention_heads
    quant_config.algo_config.num_key_value_heads = num_key_value_heads
    quant_config.algo_config.alpha = 0.85

    # smoothing
    scale = torch.rand(hidden_size, device=torch_device) + 1.0
    scale_fc_fc(model.v_proj, model.o_proj, scale, num_attention_heads, num_key_value_heads)
    output_smooth = model(input_tensor)

    # check SmoothQuant results
    assert torch.norm(output_original - output_smooth) < 1e-6
    logger.info("GQA for SmoothQuant is checked valid!")


def test_gqa_awq():
    class MyDataset(Dataset):

        def __init__(self):
            return

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return input_tensor

    torch.cuda.empty_cache()
    # dataset
    input_tensor = torch.randn(1, 4, hidden_size, device=torch_device)
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # model
    model = SimpleLM().to(torch_device)
    model.num_key_value_groups = num_attention_heads // num_key_value_heads
    output_original = model(input_tensor)

    # algorithm config
    quant_config = Config(global_quant_config=FLOAT16_CONFIG)
    quant_config = replace(quant_config, algo_config=AWQConfig())
    quant_config.algo_config.num_attention_heads = num_attention_heads
    quant_config.algo_config.num_key_value_heads = num_key_value_heads
    quant_config.algo_config.alpha = 0.85

    # smoothing
    scale = torch.rand(hidden_size, device=torch_device) + 1.0
    scale_fc_fc(model.v_proj, model.o_proj, scale, num_attention_heads, num_key_value_heads)
    output_smooth = model(input_tensor)

    # check AWQ results
    assert torch.norm(output_original - output_smooth) < 1e-5
    logger.info("GQA for AWQ is checked valid!")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    test_gqa_smoothquant()
    test_gqa_awq()
    torch.cuda.empty_cache()
