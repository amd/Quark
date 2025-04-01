#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
import torch
from torch.utils.data import DataLoader
from dataclasses import replace
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig, GPTQConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
from quark.torch.quantization.observer.observer import PerChannelMinMaxObserver, PerGroupMinMaxObserver
from quark.shares.utils.testing_utils import torch_device

INT4_PER_CHANNEL_SPEC = QuantizationSpec(dtype=Dtype.int4,
                                         observer_cls=PerChannelMinMaxObserver,
                                         symmetric=True,
                                         scale_type=ScaleType.float,
                                         round_method=RoundType.half_even,
                                         qscheme=QSchemeType.per_channel,
                                         ch_axis=0,
                                         is_dynamic=False)
DEFAULT_W_INT4_PER_CHANNEL_CONFIG = QuantizationConfig(weight=INT4_PER_CHANNEL_SPEC)

DEFAULT_UINT4_PER_GROUP_ASYM_NEG_ONE_GROUPSIZE_SPEC = QuantizationSpec(dtype=Dtype.uint4,
                                                                       observer_cls=PerGroupMinMaxObserver,
                                                                       symmetric=False,
                                                                       scale_type=ScaleType.float,
                                                                       round_method=RoundType.half_even,
                                                                       qscheme=QSchemeType.per_group,
                                                                       ch_axis=1,
                                                                       is_dynamic=False,
                                                                       group_size=-1)

DEFAULT_GPTQ_NEG_ONE_GROUPSIZE_CONFIG = QuantizationConfig(weight=DEFAULT_UINT4_PER_GROUP_ASYM_NEG_ONE_GROUPSIZE_SPEC)

DEFAULT_UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(dtype=Dtype.uint4,
                                                     observer_cls=PerGroupMinMaxObserver,
                                                     symmetric=False,
                                                     scale_type=ScaleType.float,
                                                     round_method=RoundType.half_even,
                                                     qscheme=QSchemeType.per_group,
                                                     ch_axis=1,
                                                     is_dynamic=False,
                                                     group_size=128)

DEFAULT_W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=DEFAULT_UINT4_PER_GROUP_ASYM_SPEC)


# Per channel GPTQ Config.
PERCHANNEL_GPTQ_CONFIG = Config(global_quant_config=DEFAULT_W_INT4_PER_CHANNEL_CONFIG, algo_config=GPTQConfig())

# Per channel GPTQ Config by group_size == -1.
PERGROUP_GPTQ_NEG_ONE_GROUPSIZE_CONFIG = Config(global_quant_config=DEFAULT_GPTQ_NEG_ONE_GROUPSIZE_CONFIG, algo_config=GPTQConfig())

# Per group dynamic group GPTQ Config.
PERGROUP_DYNAMIC_GROUP_GPTQ_CONFIG = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG, algo_config=GPTQConfig(static_groups=False))

# Default GPTQ Config
DEFAULT_GPTQ_CONFIG = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG, algo_config=GPTQConfig())

EXCLUDE_LAYERS = ["lm_head"]

sys.path.append("..")

def get_dataloader(model_name: str, device: torch.device):
    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs['input_ids'].to(device))
    return calib_dataloader

def quantize_model(quant_config, model_name="facebook/opt-125m", multi_gpu=False):

    # Get quantizer
    quantizer = ModelQuantizer(quant_config)

    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        model = model.to(torch_device)
    # Get dataloader, if multi_gpu, give the first layer's device
    calib_dataloader = get_dataloader(model_name, model.device)

    quant_model = quantizer.quantize_model(model, calib_dataloader)
    # Inference with quantized model
    for i in calib_dataloader:
        quant_model(i)

    return quant_model



def test_smoke_gptq_quantization():
    '''
        Quant Algorithm:          GPTQ
    '''
    for quant_config in [DEFAULT_GPTQ_CONFIG, PERCHANNEL_GPTQ_CONFIG, PERGROUP_GPTQ_NEG_ONE_GROUPSIZE_CONFIG, PERGROUP_DYNAMIC_GROUP_GPTQ_CONFIG]:
        quant_config.algo_config.inside_layer_modules = ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.out_proj", "fc1", "fc2"]
        quant_config.algo_config.model_decoder_layers = "model.decoder.layers"
        quant_config.algo_config.embedding_layers = ["model.decoder.embed_tokens", "model.decoder.embed_positions"]
        quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)
        quantize_model(quant_config)
        # quantize_model(quant_config, multi_gpu=True) # TODO: uncomment after ROCM support multi-GPU
