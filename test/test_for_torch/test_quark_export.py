#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import replace
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig, AWQConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver, PerChannelMinMaxObserver, PerGroupMinMaxObserver
from quark.torch.export.main_export.quant_config_parser import QuantConfigParser
from quark.torch import ModelExporter
from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
from quark.shares.utils.testing_utils import torch_device
from quark.shares.utils.testing_utils import use_temporary_directory

import pytest
import tempfile


from pathlib import Path
import copy

INT8_PER_GROUP_SYM_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                           observer_cls=PerGroupMinMaxObserver,
                                           symmetric=True,
                                           scale_type=ScaleType.float,
                                           round_method=RoundType.half_even,
                                           qscheme=QSchemeType.per_group,
                                           ch_axis=1,
                                           is_dynamic=False,
                                           group_size=128)

INT4_PER_GROUP_SYM_SPEC = QuantizationSpec(dtype=Dtype.int4,
                                           observer_cls=PerGroupMinMaxObserver,
                                           symmetric=True,
                                           scale_type=ScaleType.float,
                                           round_method=RoundType.half_even,
                                           qscheme=QSchemeType.per_group,
                                           ch_axis=1,
                                           is_dynamic=False,
                                           group_size=128)

INT4_PER_CHANNEL_SPEC = QuantizationSpec(dtype=Dtype.int4,
                                         observer_cls=PerChannelMinMaxObserver,
                                         symmetric=True,
                                         scale_type=ScaleType.float,
                                         round_method=RoundType.half_even,
                                         qscheme=QSchemeType.per_channel,
                                         ch_axis=0,
                                         is_dynamic=False)

UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(dtype=Dtype.uint4,
                                             observer_cls=PerGroupMinMaxObserver,
                                             symmetric=False,
                                             scale_type=ScaleType.float,
                                             round_method=RoundType.half_even,
                                             qscheme=QSchemeType.per_group,
                                             ch_axis=1,
                                             is_dynamic=False,
                                             group_size=4)

W_INT8_PER_GROUP_CONFIG = QuantizationConfig(weight=INT8_PER_GROUP_SYM_SPEC)

W_INT4_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=INT4_PER_GROUP_SYM_SPEC)

W_INT4_PER_CHANNEL_CONFIG = QuantizationConfig(weight=INT4_PER_CHANNEL_SPEC)

W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)

FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                       qscheme=QSchemeType.per_tensor,
                                       observer_cls=PerTensorMinMaxObserver,
                                       is_dynamic=False)

W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                                   weight=FP8_PER_TENSOR_SPEC)

AWQ_CONFIG = AWQConfig(
    scaling_layers=[
        {
            "prev_op": "self_attn_layer_norm",
            "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "inp": "self_attn.q_proj",
            "module2inspect": "self_attn"
        },
        {
            "prev_op": "self_attn.v_proj",
            "layers": ["self_attn.out_proj"],
            "inp": "self_attn.out_proj"
        },
        {
            "prev_op": "final_layer_norm",
            "layers": ["fc1"],
            "inp": "fc1"
        },
        {
            "prev_op": "fc1",
            "layers": ["fc2"],
            "inp": "fc2"
        }
    ],
    model_decoder_layers="model.decoder.layers"
)


EXCLUDE_LAYERS = ["lm_head", "*.gate", "*.shared_expert_gate"]
sys.path.append("..")

def set_config_for_awq_or_smooth(algo_config):
    algo_config.scaling_layers = [{
        "prev_op": "self_attn_layer_norm",
        "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "inp": "self_attn.q_proj",
        "module2inspect": "self_attn",
        "has_kwargs": True,
        "help": "attention input"
    }, {
        "prev_op": "self_attn.v_proj",
        "layers": ["self_attn.out_proj"],
        "inp": "self_attn.out_proj",
        "module2inspect": None,
        "has_kwargs": False,
        "help": "attention out"
    }, {
        "prev_op": "final_layer_norm",
        "layers": ["fc1"],
        "inp": "fc1",
        "module2inspect": None,
        "has_kwargs": False,
        "help": "linear 1"
    }, {
        "prev_op": "fc1",
        "layers": ["fc2"],
        "inp": "fc2",
        "module2inspect": None,
        "has_kwargs": False,
        "help": "linear 2"
    }]
    algo_config.model_decoder_layers = "model.decoder.layers"
    algo_config.embedding_layers = ["model.decoder.embed_tokens", "model.decoder.embed_positions"]
    return algo_config

def get_dataloader(model_name="facebook/opt-125m", device=torch_device):
    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs['input_ids'].to(device))
    return calib_dataloader

def quantize_model(quant_config, export_path: str, model_name="facebook/opt-125m", multi_gpu=False, custom_mode: str = "quark"):
    # Get quantizer
    quantizer = ModelQuantizer(quant_config)

    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map="auto",
                                                     torch_dtype="auto",
                                                     trust_remote_code=True)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        model = model.to(torch_device)
    # Get dataloader, if multi_gpu, give the first layer's device
    calib_dataloader = get_dataloader(model_name, model.device)
    quant_model = quantizer.quantize_model(model, calib_dataloader)
    # Freeze model
    model = quantizer.freeze(model)
    # Export model
    with torch.no_grad():
        NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized",
                                                   pack_method="reorder")
        config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)
        config = replace(config, json_export_config=replace(config.json_export_config, pack_method="order"))
        exporter = ModelExporter(config=config, export_dir=export_path)
        exporter.export_quark_model(model, quant_config=quant_config, custom_mode=custom_mode)
    # Inference with quantized model
    for i in calib_dataloader:
        quant_model(i)

    return quant_model



LAYER_QUANT_CONFIG = Config(global_quant_config=W_INT8_PER_GROUP_CONFIG, layer_quant_config={"model.model.decoder.layers.0.self_attn": W_INT4_PER_GROUP_SYM_CONFIG, "model.model.decoder.layers[2].fc1": W_INT4_PER_CHANNEL_CONFIG}, exclude=EXCLUDE_LAYERS)

LAYER_TYPE_QUANT_CONFIG = Config(global_quant_config=W_INT8_PER_GROUP_CONFIG, layer_type_quant_config={nn.Linear: W_INT4_PER_GROUP_SYM_CONFIG}, exclude=EXCLUDE_LAYERS)

DEFAULT_AWQ_CONFIG = Config(global_quant_config=W_INT4_PER_GROUP_SYM_CONFIG, algo_config=AWQConfig(), exclude=EXCLUDE_LAYERS)

DEFAULT_AWQ_CONFIG.algo_config = set_config_for_awq_or_smooth(DEFAULT_AWQ_CONFIG.algo_config)

DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG = Config(global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG, exclude=EXCLUDE_LAYERS)

def test_fp8_kv_cache_check():
    layer_quant_config = {
            "*v_proj":
            QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                               weight=FP8_PER_TENSOR_SPEC,
                               output_tensors=FP8_PER_TENSOR_SPEC),
            "*k_proj":
            QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                               weight=FP8_PER_TENSOR_SPEC,
                               output_tensors=FP8_PER_TENSOR_SPEC)
    }

    parser = QuantConfigParser(Config(global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG), JsonExporterConfig())
    parser._kv_cache_group = None

    assert parser.fp8_kv_cache_check(layer_quant_config) is None
    assert parser._fp8_kv_cache_scheme is None

    parser._kv_cache_group = ["*q_proj", "*k_proj"]
    assert parser.fp8_kv_cache_check(layer_quant_config) is None
    assert parser._fp8_kv_cache_scheme is None

    parser._kv_cache_group = ["*k_proj", "*v_proj"]
    layer_quant_config = {
            "*v_proj":
            QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                               weight=FP8_PER_TENSOR_SPEC,
                               output_tensors=FP8_PER_TENSOR_SPEC),
            "*k_proj": None
    }
    assert parser.fp8_kv_cache_check(layer_quant_config) is None
    assert parser._fp8_kv_cache_scheme is None

    layer_quant_config = {
            "*v_proj":
            QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                               weight=None,
                               output_tensors=FP8_PER_TENSOR_SPEC),
            "*k_proj":
            QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                               weight=None,
                               output_tensors=FP8_PER_TENSOR_SPEC)
    }
    assert parser.fp8_kv_cache_check(layer_quant_config) is None
    assert parser._fp8_kv_cache_scheme is None

    layer_quant_config = {
            "*v_proj":
            QuantizationConfig(input_tensors=INT4_PER_GROUP_SYM_SPEC,
                               weight=FP8_PER_TENSOR_SPEC,
                               output_tensors=FP8_PER_TENSOR_SPEC),
            "*k_proj":
            QuantizationConfig(input_tensors=INT4_PER_GROUP_SYM_SPEC,
                               weight=FP8_PER_TENSOR_SPEC,
                               output_tensors=FP8_PER_TENSOR_SPEC)
    }
    assert parser.fp8_kv_cache_check(layer_quant_config) is None
    assert parser._fp8_kv_cache_scheme is None

    layer_quant_config = {
            "*v_proj":
            QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                               weight=FP8_PER_TENSOR_SPEC,
                               output_tensors=None),
            "*k_proj":
            QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                               weight=FP8_PER_TENSOR_SPEC,
                               output_tensors=None)
    }
    assert parser.fp8_kv_cache_check(layer_quant_config) is None
    assert parser._fp8_kv_cache_scheme is None





@pytest.mark.parametrize("model_dtype", [
    pytest.param(model_dtype, id=str(model_dtype)) for model_dtype in [torch.float32, torch.float16, torch.bfloat16]
])
@pytest.mark.parametrize("scale_type", [
    pytest.param(scale_type, id=str(scale_type)) for scale_type in [ScaleType.float, ScaleType.float32, ScaleType.float16, ScaleType.bfloat16]
])
def test_scale_type(model_dtype: torch.dtype, scale_type: ScaleType):
    # Test that exported scale dtype match with what was specified in the quantization config.

    scale_type_to_torch = {
        ScaleType.float32: torch.float32,
        ScaleType.float16: torch.float16,
        ScaleType.bfloat16: torch.bfloat16
    }

    model = AutoModelForCausalLM.from_pretrained("fxmarty/tiny-llama-fast-tokenizer", torch_dtype=model_dtype)
    model = model.eval()

    quant_spec = copy.deepcopy(INT4_PER_CHANNEL_SPEC)
    quant_spec.scale_type = scale_type
    quant_config = Config(global_quant_config=QuantizationConfig(weight=quant_spec))

    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(model)

    for name, param in quant_model.named_parameters():
        if "scale" in name:
            if scale_type == ScaleType.float:
                assert param.dtype == model_dtype
            else:
                assert param.dtype == scale_type_to_torch[scale_type]

    for name, param in quant_model.named_buffers():
        if "scale" in name:
            if scale_type == ScaleType.float:
                assert param.dtype == model_dtype
            else:
                assert param.dtype == scale_type_to_torch[scale_type]

    # Freeze model.
    quant_model = quantizer.freeze(quant_model)

    # Export model.
    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.no_grad():
            NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized",
                                                       pack_method="reorder")
            config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)


            exporter = ModelExporter(config=config, export_dir=tmpdir)
            exporter.export_quark_model(model, quant_config=quant_config)

        # Check that the serialized scales are of correct dtype.
        model_state_dict = torch.load(Path(tmpdir) / "model_state_dict.pth")
        for name, param in model_state_dict.items():
            if name == "scale":
                if scale_type == ScaleType.float:
                    assert param.dtype == model_dtype
                else:
                    assert param.dtype == scale_type_to_torch[scale_type]

@use_temporary_directory
def test_int8_per_group(tmpdir: str):
    quant_spec = copy.deepcopy(INT8_PER_GROUP_SYM_SPEC)
    quant_spec.group_size = 8
    quant_config = Config(global_quant_config=QuantizationConfig(weight=quant_spec))

    quantize_model(quant_config, export_path=tmpdir, model_name="fxmarty/tiny-llama-fast-tokenizer")

    model_state_dict = torch.load(Path(tmpdir) / "model_state_dict.pth")


    # Original gate_proj shape: [64, 16]
    gate_proj = model_state_dict["model.layers.0.mlp.gate_proj.weight"]
    gate_proj_scale = model_state_dict["model.layers.0.mlp.gate_proj.weight_scale"]

    assert gate_proj.shape == (16, 64)
    assert gate_proj_scale.shape == (2, 64)
