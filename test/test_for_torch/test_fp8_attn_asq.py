#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM

from quark.shares.utils.log import ScreenLogger
from quark.shares.utils.testing_utils import torch_device
from quark.torch import ModelQuantizer
from quark.torch.quantization import FP8E4M3PerTensorSpec, QConfig, QLayerConfig, load_quant_algo_config_from_file

logger = ScreenLogger(__name__)

# quant spec
FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(is_dynamic=False).to_quantization_spec()
global_quant_config = QLayerConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
sys.path.append("..")


def get_dataloader(model_name="facebook/opt-125m", device=torch_device):
    seq_length = 4
    tokenized_outputs = {}
    torch.manual_seed(42)
    tokenized_outputs["input_ids"] = torch.randint(10, (1, seq_length), device=device)
    tokenized_outputs["attention_mask"] = torch.ones((1, seq_length), dtype=torch.int64, device=device)
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(device))
    return calib_dataloader


def test_fp8_attn_asq():
    # dataset
    config_path = "./test/test_for_torch/configs/fp8_attn_asq_model/llama"
    dataloader = get_dataloader(config_path, torch_device)

    # model
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config=config).to(torch_device).eval()
    model.config._attn_implementation = "eager"
    logits_original = model(dataloader.dataset).logits

    # kv cache
    kv_cache_quant_config = {}
    kv_layers_name = ["*k_proj", "*v_proj"]
    for layer_name in kv_layers_name:
        kv_cache_quant_config[layer_name] = QLayerConfig(
            input_tensors=global_quant_config.input_tensors,
            weight=global_quant_config.weight,
            output_tensors=FP8_PER_TENSOR_SPEC,
        )
    layer_quant_config = kv_cache_quant_config.copy()
    # fp8 attn
    q_layers_name = "*q_proj"
    attn_qspec = FP8_PER_TENSOR_SPEC
    layer_quant_config[q_layers_name] = QLayerConfig(
        input_tensors=global_quant_config.input_tensors, weight=global_quant_config.weight, output_tensors=attn_qspec
    )

    # algorithm config
    algo_config = load_quant_algo_config_from_file(config_path + "/autosmoothquant_config.json")
    quant_config = QConfig(
        global_quant_config=global_quant_config,
        layer_quant_config=layer_quant_config,
        kv_cache_quant_config=kv_cache_quant_config,
        softmax_quant_spec=attn_qspec,
        algo_config=[algo_config],
        exclude=["lm_head"],
    )

    # apply algorithm
    quantizer = ModelQuantizer(quant_config)
    quantizer._generate_complete_config_by_model(model, dataloader)
    model = quantizer._prepare_model(model)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 5e-3
    logger.info("fp8_attn with asq is checked valid!")


if __name__ == "__main__":
    test_fp8_attn_asq()
