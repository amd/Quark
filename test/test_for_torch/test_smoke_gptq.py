#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.shares.utils.testing_utils import torch_device
from quark.testing import slow_test
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import (
    Config,
    GPTQConfig,
    Int2PerGroupSpec,
    QuantizationConfig,
    QuantizationSpec,
)
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import (
    PerBlockMXObserver,
    PerChannelMinMaxObserver,
    PerGroupMinMaxObserver,
)

INT2_PER_GROUP_ASYM_SPEC = Int2PerGroupSpec(
    symmetric=False, scale_type="float", round_method="half_even", ch_axis=1, is_dynamic=False, group_size=128
).to_quantization_spec()

DEFAULT_GPTQ_W_INT2_PER_GROUP_CONFIG = QuantizationConfig(weight=INT2_PER_GROUP_ASYM_SPEC)

INT4_PER_CHANNEL_SPEC = QuantizationSpec(
    dtype=Dtype.int4,
    observer_cls=PerChannelMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    qscheme=QSchemeType.per_channel,
    ch_axis=0,
    is_dynamic=False,
)
DEFAULT_W_INT4_PER_CHANNEL_CONFIG = QuantizationConfig(weight=INT4_PER_CHANNEL_SPEC)

DEFAULT_UINT4_PER_GROUP_ASYM_NEG_ONE_GROUPSIZE_SPEC = QuantizationSpec(
    dtype=Dtype.uint4,
    observer_cls=PerGroupMinMaxObserver,
    symmetric=False,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    qscheme=QSchemeType.per_group,
    ch_axis=1,
    is_dynamic=False,
    group_size=-1,
)

DEFAULT_GPTQ_NEG_ONE_GROUPSIZE_CONFIG = QuantizationConfig(weight=DEFAULT_UINT4_PER_GROUP_ASYM_NEG_ONE_GROUPSIZE_SPEC)

DEFAULT_UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(
    dtype=Dtype.uint4,
    observer_cls=PerGroupMinMaxObserver,
    symmetric=False,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    qscheme=QSchemeType.per_group,
    ch_axis=1,
    is_dynamic=False,
    group_size=128,
)

DEFAULT_W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=DEFAULT_UINT4_PER_GROUP_ASYM_SPEC)


def MXFP4_SPEC(is_dynamic):
    return QuantizationSpec(
        dtype=Dtype.fp4,
        observer_cls=PerBlockMXObserver,
        symmetric=None,
        scale_type=ScaleType.float,
        scale_format="e8m0",
        scale_calculation_mode="even",
        round_method=RoundType.half_even,
        qscheme=QSchemeType.per_group,
        group_size=32,
        ch_axis=0,
        is_dynamic=is_dynamic,
    )


DEFAULT_W_MXFP4_A_MXFP4_KV_MXFP4_CONFIG = QuantizationConfig(
    input_tensors=MXFP4_SPEC(True), weight=MXFP4_SPEC(False), output_tensors=MXFP4_SPEC(True)
)

# Per channel GPTQ Config.
PERCHANNEL_GPTQ_CONFIG = Config(global_quant_config=DEFAULT_W_INT4_PER_CHANNEL_CONFIG, algo_config=[GPTQConfig()])

# Per channel GPTQ Config by group_size == -1.
PERGROUP_GPTQ_NEG_ONE_GROUPSIZE_CONFIG = Config(
    global_quant_config=DEFAULT_GPTQ_NEG_ONE_GROUPSIZE_CONFIG, algo_config=[GPTQConfig()]
)

# Per group dynamic group GPTQ Config.
PERGROUP_DYNAMIC_GROUP_GPTQ_CONFIG = Config(
    global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG, algo_config=[GPTQConfig(static_groups=False, desc_act=False)]
)

# MX FP4 dynamic group GPTQ Config.
MXFP4_DYNAMIC_GROUP_GPTQ_CONFIG = Config(
    global_quant_config=DEFAULT_W_MXFP4_A_MXFP4_KV_MXFP4_CONFIG,
    algo_config=[GPTQConfig(static_groups=False, desc_act=False)],
)

# Default GPTQ Config
DEFAULT_GPTQ_CONFIG = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG, algo_config=[GPTQConfig()])

DEFAULT_GPTQ_INT2_CONFIG = Config(global_quant_config=DEFAULT_GPTQ_W_INT2_PER_GROUP_CONFIG, algo_config=[GPTQConfig()])

EXCLUDE_LAYERS = ["lm_head"]

sys.path.append("..")


def get_dataloader(model_name: str, device: torch.device):
    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(device))
    return calib_dataloader


def quantize_model(quant_config, model_name="facebook/opt-125m", multi_gpu=False, multi_device=False):
    # Get quantizer
    if not multi_device:
        quantizer = ModelQuantizer(quant_config)
        if multi_gpu:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True
            )
            model.eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()
            model = model.to(torch_device)
    else:
        max_memory = {"cpu": "100GB"}
        # cpu and one gpu; This one is the best for opt
        if torch.cuda.is_available():
            first_gpu = 0
            max_memory[first_gpu] = "0.2GB"
            if torch.cuda.device_count() > 1:
                for i in range(1, torch.cuda.device_count()):
                    max_memory[i] = "0GB"

        quantizer = ModelQuantizer(quant_config, multi_device=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto", max_memory=max_memory, trust_remote_code=True
        )
        print(model.hf_device_map)

    model.model.decoder.layers = model.model.decoder.layers[:2]
    model.config.num_hidden_layers = 2

    # Get dataloader, if multi_gpu, give the first layer's device
    calib_dataloader = get_dataloader(model_name, model.device)

    quant_model = quantizer.quantize_model(model, calib_dataloader)
    # Inference with quantized model
    for i in calib_dataloader:
        quant_model(i)

    return quant_model


@slow_test
@pytest.mark.parametrize(
    "quant_config",
    [
        DEFAULT_GPTQ_CONFIG,
        PERCHANNEL_GPTQ_CONFIG,
        PERGROUP_GPTQ_NEG_ONE_GROUPSIZE_CONFIG,
        PERGROUP_DYNAMIC_GROUP_GPTQ_CONFIG,
        MXFP4_DYNAMIC_GROUP_GPTQ_CONFIG,
    ],
)
def test_smoke_gptq_quantization(quant_config: Config):
    quant_config.algo_config[0].inside_layer_modules = [
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.out_proj",
        "fc1",
        "fc2",
    ]
    quant_config.algo_config[0].model_decoder_layers = "model.decoder.layers"
    quant_config.algo_config[0].embedding_layers = ["model.decoder.embed_tokens", "model.decoder.embed_positions"]

    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)
    quantize_model(quant_config)
    # quantize_model(quant_config, multi_gpu=True) # TODO: uncomment after ROCM support multi-GPU


@slow_test
def test_smoke_gptq_multi_device_quantization():
    """
    Quant Algorithm:          GPTQ
    """
    quant_config = DEFAULT_GPTQ_INT2_CONFIG
    quant_config.algo_config[0].inside_layer_modules = [
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.out_proj",
        "fc1",
        "fc2",
    ]
    quant_config.algo_config[0].model_decoder_layers = "model.decoder.layers"
    quant_config.algo_config[0].embedding_layers = ["model.decoder.embed_tokens", "model.decoder.embed_positions"]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)
    quantize_model(quant_config, multi_device=True)


if __name__ == "__main__":
    test_smoke_gptq_quantization()
    test_smoke_gptq_multi_device_quantization()
