#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os

os.environ["QUARK_ALGO_DEBUG"] = "1"
import sys

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, Llama4Config, Llama4ForConditionalGeneration

from quark.shares.utils.log import ScreenLogger
from quark.shares.utils.testing_utils import torch_device
from quark.torch import ModelQuantizer
from quark.torch.quantization import (
    FP8E4M3PerTensorSpec,
    OCP_MXFP4Spec,
    Uint4PerChannelSpec,
    load_pre_optimization_config_from_file,
    load_quant_algo_config_from_file,
)
from quark.torch.quantization.config.config import Config, QuantizationConfig, QuantizationSpec
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.observer.observer import PlaceholderObserver

logger = ScreenLogger(__name__)

FLOAT16_SPEC = QuantizationSpec(dtype=Dtype.float16, observer_cls=PlaceholderObserver)
FLOAT16_CONFIG = QuantizationConfig(input_tensors=FLOAT16_SPEC, weight=FLOAT16_SPEC)
FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()
W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
W_MXFP4_A_DYN_MXFP4_CONFIG = QuantizationConfig(
    input_tensors=OCP_MXFP4Spec().to_quantization_spec(), weight=OCP_MXFP4Spec(is_dynamic=False).to_quantization_spec()
)
UINT4_PER_CHANNEL_ASYM_SPEC = Uint4PerChannelSpec(
    symmetric=False, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
).to_quantization_spec()
W_UINT4_PER_CHANNEL_ASYM_CONFIG = QuantizationConfig(weight=UINT4_PER_CHANNEL_ASYM_SPEC)
sys.path.append("..")


def get_dataloader(model_name="facebook/opt-125m", device=torch_device):
    seq_length = 4
    tokenized_outputs = {}
    torch.manual_seed(42)
    tokenized_outputs["input_ids"] = torch.randint(10, (1, seq_length), device=device)
    tokenized_outputs["attention_mask"] = torch.ones((1, seq_length), dtype=torch.int64, device=device)
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(device))
    return calib_dataloader


def test_moe_gptq():
    # dataset
    config_path = "./test/test_for_torch/configs/moe_model/mixtral"
    dataloader = get_dataloader(config_path, torch_device)

    # original results
    config = AutoConfig.from_pretrained(config_path + "/config.json", trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16).to(torch_device).eval()
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_quant_algo_config_from_file(config_path + "/gptq_config.json")
    quant_config = Config(
        global_quant_config=W_UINT4_PER_CHANNEL_ASYM_CONFIG, algo_config=[algo_config], exclude=["lm_head", "*gate"]
    )

    # apply algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._prepare_model(model)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 5e-3
    logger.info("MoE GPTQ is checked valid!")


def test_moe_qronos():
    # dataset
    config_path = "./test/test_for_torch/configs/moe_model/mixtral"
    dataloader = get_dataloader(config_path, torch_device)

    # original results
    config = AutoConfig.from_pretrained(config_path + "/config.json", trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16).to(torch_device).eval()
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_quant_algo_config_from_file(config_path + "/qronos_config.json")
    quant_config = Config(
        global_quant_config=W_UINT4_PER_CHANNEL_ASYM_CONFIG, algo_config=[algo_config], exclude=["lm_head", "*gate"]
    )

    # apply algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._prepare_model(model)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 5e-3


def test_moe_awq():
    # dataset
    config_path = "./test/test_for_torch/configs/moe_model/mixtral"
    dataloader = get_dataloader(config_path, torch_device)

    # original results
    config = AutoConfig.from_pretrained(config_path + "/config.json", trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16).to(torch_device).eval()
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_quant_algo_config_from_file(config_path + "/awq_config.json")
    quant_config = Config(
        global_quant_config=W_UINT4_PER_CHANNEL_ASYM_CONFIG, algo_config=[algo_config], exclude=["lm_head", "*gate"]
    )

    # apply algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._prepare_model(model)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 5e-3
    logger.info("MoE AWQ is checked valid!")


@pytest.mark.parametrize(
    "global_quant_config,model_config_name",
    [(W_FP8_A_FP8_PER_TENSOR_CONFIG, "config.json"), (W_MXFP4_A_DYN_MXFP4_CONFIG, "config_128.json")],
)
def test_moe_autosmoothquant(global_quant_config, model_config_name):
    # dataset
    config_path = "./test/test_for_torch/configs/moe_model/mixtral"
    dataloader = get_dataloader(config_path, torch_device)

    # original results
    config = AutoConfig.from_pretrained(config_path + "/" + model_config_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16).to(torch_device).eval()
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_quant_algo_config_from_file(config_path + "/autosmoothquant_config.json")
    quant_config = Config(
        global_quant_config=global_quant_config, algo_config=[algo_config], exclude=["lm_head", "*gate"]
    )

    # apply algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._prepare_model(model)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 5e-3
    logger.info("MoE AutoSmoothQuant is checked valid!")


def test_moe_smoothquant():
    # dataset
    config_path = "./test/test_for_torch/configs/moe_model/llama4"
    dataloader = get_dataloader(config_path, torch_device)

    # model
    config = Llama4Config.from_pretrained(config_path)
    model = Llama4ForConditionalGeneration(config).to(torch_device).to(torch.bfloat16).eval()
    # init parameters defined using torch.empty
    gate_up_proj_init = model.language_model.model.layers[0].feed_forward.experts.gate_up_proj
    down_proj_init = model.language_model.model.layers[0].feed_forward.experts.down_proj
    gate_up_proj_init.data = torch.rand(
        gate_up_proj_init.shape, device=gate_up_proj_init.device, dtype=gate_up_proj_init.dtype
    )
    down_proj_init.data = torch.rand(down_proj_init.shape, device=down_proj_init.device, dtype=down_proj_init.dtype)
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_pre_optimization_config_from_file(config_path + "/smoothquant_config.json")
    quant_config = Config(global_quant_config=FLOAT16_CONFIG, algo_config=[algo_config], exclude=["lm_head", "*gate"])

    # algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 5e-3
    logger.info("MoE SmoothQuant is checked valid!")


if __name__ == "__main__":
    # test gptq
    test_moe_gptq()

    # test qronos
    test_moe_qronos()

    # test awq
    test_moe_awq()

    # test asq
    test_moe_autosmoothquant()

    # test sq
    test_moe_smoothquant()
