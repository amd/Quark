#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, Llama4Config, Llama4ForConditionalGeneration

from quark.shares.utils.log import ScreenLogger
from quark.shares.utils.testing_utils import torch_device
from quark.torch import ModelQuantizer
from quark.torch.quantization import (
    Int8PerTensorSpec,
    QConfig,
    QLayerConfig,
    Uint4PerChannelSpec,
    load_pre_optimization_config_from_file,
    load_quant_algo_config_from_file,
)

logger = ScreenLogger(__name__)

# quant spec
INT8_PER_TENSOR_SPEC = Int8PerTensorSpec(is_dynamic=False).to_quantization_spec()
W8A8_CONFIG = QLayerConfig(input_tensors=INT8_PER_TENSOR_SPEC, weight=INT8_PER_TENSOR_SPEC)
UINT4_PER_CHANNEL_ASYM_SPEC = Uint4PerChannelSpec(ch_axis=0, is_dynamic=False).to_quantization_spec()
W_UINT4_PER_CHANNEL_ASYM_CONFIG = QLayerConfig(weight=UINT4_PER_CHANNEL_ASYM_SPEC)
sys.path.append("..")


def get_dataloader(model_name="facebook/opt-125m", device=torch_device):
    seq_length = 4
    tokenized_outputs = {}
    torch.manual_seed(42)
    tokenized_outputs["input_ids"] = torch.randint(10, (1, seq_length), device=device)
    tokenized_outputs["attention_mask"] = torch.ones((1, seq_length), dtype=torch.int64, device=device)
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(device))
    return calib_dataloader


def test_dbrx_num_head_sq():
    # dataset
    config_path = "./test/test_for_torch/configs/gqa_model/dbrx"
    dataloader = get_dataloader(config_path, torch_device)

    # model
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16).to(torch_device).eval()
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_pre_optimization_config_from_file(config_path + "/smoothquant_config.json")
    quant_config = QConfig(global_quant_config=W8A8_CONFIG, algo_config=[algo_config], exclude=["lm_head", "*ffn*"])

    # apply algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 1e-2
    logger.info("GQA head_numbers from model.config for SmoothQuant is checked valid!")


def test_opt_num_head_awq():
    # dataset
    config_path = "./test/test_for_torch/configs/gqa_model/opt"
    dataloader = get_dataloader(config_path, torch_device)

    # model
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16).to(torch_device).eval()
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_quant_algo_config_from_file(config_path + "/awq_config.json")
    quant_config = QConfig(
        global_quant_config=W_UINT4_PER_CHANNEL_ASYM_CONFIG, algo_config=[algo_config], exclude=["lm_head"]
    )

    # apply algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._prepare_model(model)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 1e-2
    logger.info("GQA head_numbers from model.config for AWQ is checked valid!")


def test_vlm_num_head_asq():
    # dataset
    config_path = "./test/test_for_torch/configs/gqa_model/llama4"
    dataloader = get_dataloader(config_path, torch_device)

    # model
    config = Llama4Config.from_pretrained(config_path)
    config.text_config.moe_layers[0] = 1  # dense layer
    model = Llama4ForConditionalGeneration(config).to(torch_device).to(torch.bfloat16).eval()
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_pre_optimization_config_from_file(config_path + "/smoothquant_config.json")
    quant_config = QConfig(
        global_quant_config=W8A8_CONFIG, algo_config=[algo_config], exclude=["lm_head", "vision_model*"]
    )
    # algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 1e-2
    logger.info("GQA head_numbers from model.config for AutoSmoothQuant is checked valid!")


if __name__ == "__main__":
    test_dbrx_num_head_sq()
    test_opt_num_head_awq()
    test_vlm_num_head_asq()
