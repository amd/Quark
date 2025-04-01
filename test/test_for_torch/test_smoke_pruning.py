#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.torch import ModelPruner
from quark.torch.pruning.config import Config, OSSCARConfig, BlockwiseTuningConfig

from quark.shares.utils.testing_utils import torch_device
from quark.torch.algorithm.utils.module import get_dtype
from quark.torch.pruning.model_transformation import prune_layer

sys.path.append("..")

hidden_size = 32
intermediate_size = 64


class SimpleMLP(nn.Module):

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.intermediate_size = intermediate_size

        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.gate_up_proj.weight.data.fill_(2.0)
        self.down_proj.weight.data.fill_(1.0)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * gate

        return self.down_proj(up_states)


def get_dataloader(model_name: str, device: torch.device):
    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs['input_ids'].to(device))
    return calib_dataloader


def llm_pruning_model(quant_config, dtype, model_name="Qwen/Qwen1.5-0.5B", multi_gpu=False):
    # Get pruner
    pruner = ModelPruner(quant_config)

    if multi_gpu:

        model_kwargs = {"torch_dtype": "auto", "max_memory": {0: "0.1GB", "cpu": "100GB"}}

        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map="auto",
                                                     **model_kwargs,
                                                     trust_remote_code=True)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        model.eval()
        model = model.to(torch_device)

    # Get dataloader, if multi_gpu, give the first layer's device
    calib_dataloader = get_dataloader(model_name, model.device)

    pruned_model = pruner.pruning_model(model, calib_dataloader)

    assert get_dtype(pruned_model) == dtype

    # Inference with pruned model
    for i in calib_dataloader:
        pruned_model(i)

    return pruned_model


def test_smoke_osscar():
    '''
        Pruning Algorithm: OSSCAR
    '''

    mlp_module = SimpleMLP(hidden_size, intermediate_size)

    pruning_list = [False] * int(intermediate_size * 0.75) + [True] * int(intermediate_size * 0.25)

    random.shuffle(pruning_list)

    pruned_layer = prune_layer(mlp_module.gate_up_proj, torch.tensor(pruning_list))

    assert pruned_layer.out_features == int(intermediate_size * 0.75) * 2

    pruning_config = Config()

    pruning_config.algo_config = OSSCARConfig()

    pruning_config.algo_config.inside_layer_modules = [
        "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.gate_proj",
        "mlp.down_proj"
    ]

    pruning_config.algo_config.mlp_pruning_modules = ["mlp.down_proj"]
    pruning_config.algo_config.mlp_scaling_layers = {
        "mlp.down_proj": ["mlp.up_proj", "mlp.gate_proj"],
    }

    pruning_config.algo_config.mlp_pruning_ratio = 0.1

    pruning_config.algo_config.mlp_intermediate_size_name = "intermediate_size"

    pruning_config.algo_config.model_decoder_layers = "model.layers"

    pruning_config.blockwise_tuning_config = BlockwiseTuningConfig(
        model_decoder_layers="model.layers",
        trainable_modules=["mlp.down_proj"],
        epochs=1,
    )

    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        llm_pruning_model(pruning_config, dtype)
