#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import random
import sys

import pytest
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.shares.utils.testing_utils import torch_device
from quark.testing import skip_if_no_gpu, slow_test
from quark.torch import ModelPruner
from quark.torch.algorithm.utils.module import get_dtype
from quark.torch.pruning.config import BlockwiseTuningConfig, Config, LayerImportancePruneConfig, OSSCARConfig
from quark.torch.pruning.model_transformation import prune_layer

sys.path.append("..")


# ========================== OSSCAR pruning method unittest ================
# this method is a channel wise pruning method
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
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(device))
    return calib_dataloader


def llm_pruning_model(quant_config, dtype, model_name="Qwen/Qwen1.5-0.5B", multi_gpu=False):
    # Get pruner
    pruner = ModelPruner(quant_config)

    if multi_gpu:
        model_kwargs = {"torch_dtype": "auto", "max_memory": {0: "0.1GB", "cpu": "100GB"}}

        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", **model_kwargs, trust_remote_code=True
        )
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


@slow_test
@skip_if_no_gpu
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_smoke_osscar(dtype):
    """
    Pruning Algorithm: OSSCAR
    """

    if not torch.cuda.is_available() and dtype in (torch.float16, torch.bfloat16):
        pytest.skip(f"This test with dtype {dtype} requires GPU support!")

    mlp_module = SimpleMLP(hidden_size, intermediate_size)

    pruning_list = [False] * int(intermediate_size * 0.75) + [True] * int(intermediate_size * 0.25)

    random.shuffle(pruning_list)

    pruned_layer = prune_layer(mlp_module.gate_up_proj, torch.tensor(pruning_list))

    assert pruned_layer.out_features == int(intermediate_size * 0.75) * 2

    pruning_config = Config()

    pruning_config.algo_config = OSSCARConfig()

    pruning_config.algo_config.inside_layer_modules = [
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
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

    llm_pruning_model(pruning_config, dtype)


# ----------------- depth wise pruning method ---------------------------


def get_llm_model(model_name="facebook/opt-125m", multi_gpu=False):
    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto", trust_remote_code=False
        )
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        model.eval()
        model = model.to(torch_device)
    return model


def get_wikitext_dataset(model_name: str = "facebook/opt-125m", dev: torch.device = None) -> list[torch.Tensor]:
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    seqlen_for_eval = 2048
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen_for_eval
    batch_data = []
    testenc = testenc.to(dev)
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * seqlen_for_eval) : ((i + 1) * seqlen_for_eval)].to(dev)
        batch_data.append(batch)
    return batch_data


@torch.no_grad()
def eval_ppl(model, test_dataset):
    torch.cuda.empty_cache()
    seqlen_for_eval = len(test_dataset[0])
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss()
    for i in tqdm(range(len(test_dataset))):
        batch = test_dataset[i]
        lm_logits = model(batch)["logits"]
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_dataset[i][:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen_for_eval
        nlls.append(neg_log_likelihood)
        torch.cuda.empty_cache()
    ppl = torch.exp(torch.stack(nlls).sum() / (len(test_dataset) * seqlen_for_eval))
    torch.cuda.empty_cache()
    return ppl


@slow_test
@skip_if_no_gpu
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_smoke_depth_wise_pruning(dtype):
    """
    Pruning Algorithm: depth-wise pruning, layer importance is evaluated by PPL influence
    """

    if not torch.cuda.is_available() and dtype in (torch.float16, torch.bfloat16):
        pytest.skip(f"This test with dtype {dtype} requires GPU support!")

    pruning_config = Config()

    pruning_config.algo_config = LayerImportancePruneConfig()
    # NOTE the config may change, as the API may be refactored.
    pruning_config.algo_config.delete_layer_num = 7
    pruning_config.algo_config.model_decoder_layers = "model.decoder.layers"
    pruning_config.algo_config.layer_norm_field = "model.decoder.final_layer_norm"
    pruning_config.algo_config.layer_num_field = "num_hidden_layers"
    # depth pruning needs not blockwise_tuning_config
    pruning_config.blockwise_tuning_config = None

    ppl_best = None
    model_name = "facebook/opt-125m"
    # model_name = "/group/amdneuralopt/huggingface/pretrained_models/facebook/opt-125m" # NOTE local debug
    for save_memory in [True, False]:
        model = get_llm_model(model_name, multi_gpu=True)
        main_device = model.device
        evaluate_data = get_wikitext_dataset(model_name=model_name, dev=main_device)
        pruning_config.algo_config.save_gpu_memory = save_memory
        model_pruner = ModelPruner(pruning_config)
        model = model_pruner.pruning_model(model, evaluate_data)

        assert len(model.model.decoder.layers) == 5
        # after pruning test the param size and forward
        ppl_rst = eval_ppl(model, evaluate_data)
        if ppl_best is None:
            ppl_best = int(ppl_rst.item())
        else:
            assert ppl_best == int(ppl_rst.item()), "save mem or not should get the same ppl results"

    # test assigned the layers to perform trim
    pruning_config.algo_config.delete_layers_index = [3, 4]
    model = get_llm_model(model_name, multi_gpu=False)
    main_device = model.device
    evaluate_data = get_wikitext_dataset(model_name=model_name, dev=main_device)
    pruning_config.algo_config.save_gpu_memory = False
    model_pruner = ModelPruner(pruning_config)
    model = model_pruner.pruning_model(model, evaluate_data)

    assert len(model.model.decoder.layers) == 10

    return


if __name__ == "__main__":
    test_smoke_depth_wise_pruning(torch.float16)
