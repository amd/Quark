#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
import torch.nn as nn

from quark.torch import LLMTemplate
from quark.torch.quantization.config.config import (
    AutoSmoothQuantConfig,
    AWQConfig,
    Config,
    GPTQConfig,
    Int4PerGroupSpec,
    Int8PerTensorSpec,
    QLayerConfig,
    QronosConfig,
    RotationConfig,
    SmoothQuantConfig,
)
from quark.torch.quantization.config.type import Dtype, QSchemeType


def test_llm_template_basic_initialization():
    """Test LLMTemplate basic initialization"""
    template = LLMTemplate(
        model_type="test_model",
        kv_layers_name=["*k_proj", "*v_proj"],
        q_layer_name="*q_proj",
        exclude_layers_name=["lm_head"],
    )

    assert template.model_type == "test_model"
    assert template.kv_layers_name == ["*k_proj", "*v_proj"]
    assert template.q_layer_name == "*q_proj"
    assert template.exclude_layers_name == ["lm_head"]
    # Check algo_config dictionary structure
    assert isinstance(template.algo_config, dict)
    assert template.algo_config["awq"] is None
    assert template.algo_config["gptq"] is None
    assert template.algo_config["smoothquant"] is None
    assert template.algo_config["rotation"] is None


def test_register_template_method():
    """Test explicit template registration method"""
    initial_count = len(LLMTemplate._templates)

    # Create a template
    test_template = LLMTemplate(
        model_type="test_register_template",
        kv_layers_name=["*k_proj", "*v_proj"],
        q_layer_name="*q_proj",
        exclude_layers_name=["lm_head"],
    )

    # Should not be registered yet
    assert "test_register_template" not in LLMTemplate._templates
    assert len(LLMTemplate._templates) == initial_count

    # Register explicitly
    LLMTemplate.register_template(test_template)

    # Should now be registered
    assert "test_register_template" in LLMTemplate._templates
    assert LLMTemplate._templates["test_register_template"] is test_template
    assert len(LLMTemplate._templates) == initial_count + 1

    # Can retrieve and use the template
    retrieved_template = LLMTemplate.get("test_register_template")
    assert retrieved_template is test_template

    # Template should work normally
    config = retrieved_template.get_config("int4_wo_128")
    assert isinstance(config, Config)

    # Clean up
    del LLMTemplate._templates["test_register_template"]


def test_list_available_templates():
    """Test listing available templates"""
    available = LLMTemplate.list_available()
    assert isinstance(available, list)
    assert len(available) > 0
    assert "llama" in available
    assert "opt" in available


def test_get_existing_template():
    """Test getting existing template"""
    template = LLMTemplate.get("llama")
    assert isinstance(template, LLMTemplate)
    assert template.model_type == "llama"


def test_get_nonexistent_template():
    """Test getting non-existent template raises error"""
    with pytest.raises(ValueError):
        LLMTemplate.get("nonexistent")


def test_supported_schemes():
    """Test all supported quantization schemes"""
    template = LLMTemplate.get("llama")
    expected_schemes = [
        "fp8",
        "ptpc_fp8",
        "int4_wo_32",
        "int4_wo_64",
        "int4_wo_128",
        "int4_wo_per_channel",
        "uint4_wo_32",
        "uint4_wo_64",
        "uint4_wo_128",
        "uint4_wo_per_channel",
        "mxfp4",
        "mxfp4_mxfp6_e2m3",
        "mxfp4_fp8",
        "mxfp6_e3m2",
        "mxfp6_e2m3",
        "mx6",
        "bfp16",
        "int8",
    ]
    assert sorted(LLMTemplate._SUPPORTED_SCHEMES) == sorted(expected_schemes)

    # Test actually supported schemes in implementation
    working_schemes = [
        "int4_wo_32",
        "int4_wo_64",
        "int4_wo_128",
        "int4_wo_per_channel",
        "uint4_wo_32",
        "uint4_wo_64",
        "uint4_wo_128",
        "uint4_wo_per_channel",
        "fp8",
        "ptpc_fp8",
        "mxfp4",
        "mxfp6_e3m2",
        "mxfp6_e2m3",
        "mx6",
        "bfp16",
        "int8",
    ]
    for scheme in working_schemes:
        config = template.get_config(scheme)
        assert isinstance(config, Config)
        assert config.global_quant_config.weight is not None


def test_int4_wo_32_scheme():
    """Test INT4 weight-only quantization scheme with group size 32"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_32")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.int4
    assert config.global_quant_config.weight.group_size == 32
    assert config.global_quant_config.input_tensors is None


def test_int4_wo_64_scheme():
    """Test INT4 weight-only quantization scheme with group size 64"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_64")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.int4
    assert config.global_quant_config.weight.group_size == 64
    assert config.global_quant_config.input_tensors is None


def test_int4_wo_128_scheme():
    """Test INT4 weight-only quantization scheme with group size 128"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_128")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.int4
    assert config.global_quant_config.weight.group_size == 128
    assert config.global_quant_config.input_tensors is None


def test_int4_wo_per_channel_scheme():
    """Test INT4 weight-only per-channel quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_per_channel")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.int4
    assert config.global_quant_config.weight.group_size is None
    assert config.global_quant_config.input_tensors is None
    assert config.global_quant_config.weight.symmetric
    assert config.global_quant_config.weight.qscheme.value == "per_channel"


def test_uint4_wo_32_scheme():
    """Test UINT4 weight-only quantization scheme with group size 32"""
    template = LLMTemplate.get("llama")
    config = template.get_config("uint4_wo_32")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.uint4
    assert config.global_quant_config.weight.group_size == 32
    assert config.global_quant_config.input_tensors is None


def test_uint4_wo_64_scheme():
    """Test UINT4 weight-only quantization scheme with group size 64"""
    template = LLMTemplate.get("llama")
    config = template.get_config("uint4_wo_64")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.uint4
    assert config.global_quant_config.weight.group_size == 64
    assert config.global_quant_config.input_tensors is None


def test_uint4_wo_128_scheme():
    """Test UINT4 weight-only quantization scheme with group size 128"""
    template = LLMTemplate.get("llama")
    config = template.get_config("uint4_wo_128")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.uint4
    assert config.global_quant_config.weight.group_size == 128
    assert config.global_quant_config.input_tensors is None


def test_uint4_wo_per_channel_scheme():
    """Test UINT4 weight-only per-channel quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("uint4_wo_per_channel")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.uint4
    assert config.global_quant_config.weight.group_size is None
    assert config.global_quant_config.input_tensors is None
    assert not config.global_quant_config.weight.symmetric
    assert config.global_quant_config.weight.qscheme.value == "per_channel"


def test_fp8_scheme():
    """Test FP8 quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("fp8")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.fp8_e4m3
    assert config.global_quant_config.input_tensors.dtype == Dtype.fp8_e4m3


def test_ptpc_fp8_scheme():
    """Test PTPC FP8 quantization scheme (Per-Token Per-Channel)"""
    template = LLMTemplate.get("llama")
    config = template.get_config("ptpc_fp8")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    # Weight: FP8 Per-Channel Static
    assert config.global_quant_config.weight.qscheme == QSchemeType.per_channel
    assert config.global_quant_config.weight.dtype == Dtype.fp8_e4m3
    assert config.global_quant_config.weight.is_dynamic is False
    assert config.global_quant_config.weight.ch_axis == 0
    # Activation: FP8 Per-Token Dynamic
    assert config.global_quant_config.input_tensors.qscheme == QSchemeType.per_channel
    assert config.global_quant_config.input_tensors.dtype == Dtype.fp8_e4m3
    assert config.global_quant_config.input_tensors.is_dynamic is True
    assert config.global_quant_config.input_tensors.ch_axis == 1


def test_mxfp4_scheme():
    """Test MXFP4 quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("mxfp4")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.fp4
    assert config.global_quant_config.input_tensors.dtype == Dtype.fp4


def test_mxfp6_e3m2_scheme():
    """Test MXFP6E3M2 quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("mxfp6_e3m2")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.fp6_e3m2


def test_mxfp6_e2m3_scheme():
    """Test MXFP6E2M3 quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("mxfp6_e2m3")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.fp6_e2m3


def test_int8_scheme():
    """Test INT8 quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int8")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.int8
    assert config.global_quant_config.input_tensors.dtype == Dtype.int8


def test_mx6_scheme():
    """Test MX6 quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("mx6")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.mx6
    assert config.global_quant_config.input_tensors.dtype == Dtype.mx6


def test_bfp16_scheme():
    """Test BFP16 quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("bfp16")
    assert isinstance(config, Config)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.bfp16
    assert config.global_quant_config.input_tensors.dtype == Dtype.bfp16


def test_unsupported_scheme():
    """Test unsupported quantization scheme raises error"""
    template = LLMTemplate.get("llama")
    with pytest.raises(ValueError, match="Unsupported quantization scheme: int8_wo"):
        template.get_config("int8_wo")

    with pytest.raises(ValueError, match="Unsupported quantization scheme: invalid_scheme"):
        template.get_config("invalid_scheme")


def test_awq_algorithm():
    """Test AWQ algorithm with custom configs"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_128", algorithm="awq")
    assert isinstance(config, Config)
    assert len(config.algo_config) > 0
    assert isinstance(config.algo_config[0], AWQConfig)
    assert config.algo_config[0].name == "awq"


def test_gptq_algorithm():
    """Test GPTQ algorithm with custom configs"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_128", algorithm="gptq")
    assert isinstance(config, Config)
    assert len(config.algo_config) > 0
    assert isinstance(config.algo_config[0], GPTQConfig)
    assert config.algo_config[0].name == "gptq"


def test_qronos_algorithm():
    """Test Qronos algorithm with custom configs"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_128", algorithm="qronos")
    assert isinstance(config, Config)
    assert len(config.algo_config) > 0
    assert isinstance(config.algo_config[0], QronosConfig)

    qronos_config = config.algo_config[0]
    assert qronos_config.name == "qronos"
    assert hasattr(qronos_config, "inside_layer_modules")
    assert hasattr(qronos_config, "model_decoder_layers")


def test_smoothquant_algorithm():
    """Test SmoothQuant algorithm with custom configs"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_32", algorithm="smoothquant")
    assert isinstance(config, Config)
    assert len(config.algo_config) > 0
    assert isinstance(config.algo_config[0], SmoothQuantConfig)
    assert config.algo_config[0].name == "smooth"


def test_autosmoothquant_algorithm():
    """Test AutoSmoothQuant algorithm with custom configs"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_32", algorithm="autosmoothquant")
    assert isinstance(config, Config)
    assert len(config.algo_config) > 0
    assert isinstance(config.algo_config[0], AutoSmoothQuantConfig)


def test_rotation_algorithm():
    """Test Rotation algorithm with custom configs"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_128", algorithm="rotation")
    assert isinstance(config, Config)
    assert len(config.algo_config) > 0
    assert isinstance(config.algo_config[0], RotationConfig)
    assert config.algo_config[0].name == "rotation"


def test_algorithm_config_missing_raises_error():
    """Test that missing algorithm configs raise appropriate errors"""
    # Create a template without any algorithm configs
    template = LLMTemplate(
        model_type="test_missing_configs",
        kv_layers_name=["*k_proj", "*v_proj"],
        q_layer_name="*q_proj",
        exclude_layers_name=["lm_head"],
        # No algorithm configs provided
    )

    # Test that missing Qronos config raises error
    with pytest.raises(ValueError, match="No Qronos config provided for test_missing_configs"):
        template.get_config("int4_wo_128", algorithm="qronos")


def test_autosmoothquant_algorithm_fallback():
    """Test AutoSmoothQuant algorithm fallback to default config when no custom config provided"""
    # Create a template without autosmoothquant_config
    template = LLMTemplate(
        model_type="test_fallback_autosq",
        kv_layers_name=["*k_proj", "*v_proj"],
        q_layer_name="*q_proj",
        exclude_layers_name=["lm_head"],
        # No autosmoothquant_config provided
    )

    config = template.get_config("int4_wo_128", algorithm="autosmoothquant")
    assert isinstance(config, Config)
    assert len(config.algo_config) > 0
    assert isinstance(config.algo_config[0], AutoSmoothQuantConfig)
    assert config.algo_config[0].name == "autosmoothquant"


def test_rotation_algorithm_no_config_warning():
    """Test Rotation algorithm warning when no rotation config provided"""
    # Create a template without rotation_config
    template = LLMTemplate(
        model_type="test_no_rotation",
        kv_layers_name=["*k_proj", "*v_proj"],
        q_layer_name="*q_proj",
        exclude_layers_name=["lm_head"],
        # No rotation_config provided
    )

    # Test that no config is added when rotation config is missing
    config = template.get_config("int4_wo_128", algorithm="rotation")
    assert isinstance(config, Config)
    # Should have no algo_config since rotation doesn't fallback to default
    assert config.algo_config is None or len(config.algo_config) == 0


def test_unsupported_algorithm():
    """Test unsupported algorithm raises error"""
    template = LLMTemplate.get("llama")
    with pytest.raises(ValueError, match="Unsupported algorithm: invalid_algo"):
        template.get_config("int4_wo_128", algorithm="invalid_algo")


def test_fp8_kv_cache_scheme():
    """Test FP8 KV cache quantization"""
    template = LLMTemplate.get("llama")
    config = template.get_config("fp8", kv_cache_scheme="fp8")

    assert isinstance(config, Config)
    assert len(config.layer_quant_config) > 0
    assert len(config.kv_cache_quant_config) > 0


def test_unsupported_kv_cache_scheme():
    """Test unsupported KV cache scheme raises error"""
    template = LLMTemplate.get("llama")
    with pytest.raises(ValueError, match="Unsupported KV cache scheme: invalid_kv"):
        template.get_config("fp8", kv_cache_scheme="invalid_kv")


def test_min_kv_scale():
    """Test min_kv_scale"""
    template = LLMTemplate.get("llama")
    config = template.get_config("fp8", kv_cache_scheme="fp8", min_kv_scale=1.0)
    assert isinstance(config, Config)
    assert config.min_kv_scale == 1.0


def test_fp8_attention_scheme():
    """Test FP8 attention quantization"""
    template = LLMTemplate.get("llama")
    config = template.get_config("fp8", attention_scheme="fp8")

    assert isinstance(config, Config)
    assert config.softmax_quant_spec is not None
    assert config.softmax_quant_spec.dtype == Dtype.fp8_e4m3


def test_unsupported_attention_scheme():
    """Test unsupported attention scheme raises error"""
    template = LLMTemplate.get("llama")
    with pytest.raises(ValueError, match="Unsupported attention scheme: invalid_attn"):
        template.get_config("fp8", attention_scheme="invalid_attn")


def test_layer_config_with_quantization_config():
    """Test per-layer config with QLayerConfig objects"""
    template = LLMTemplate.get("llama")

    per_layer_config = {"layer1": "int4_wo_64", "layer2": "int4_wo_128"}

    config = template.get_config("int4_wo_32", layer_config=per_layer_config)

    assert isinstance(config, Config)
    assert "layer1" in config.layer_quant_config
    assert "layer2" in config.layer_quant_config
    assert config.layer_quant_config["layer1"].weight.group_size == 64
    assert config.layer_quant_config["layer2"].weight.group_size == 128


def test_layer_type_config():
    """Test layer type config"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_32", layer_type_config={nn.Linear: "int4_wo_64"})
    assert isinstance(config, Config)
    assert len(config.layer_type_quant_config) > 0
    assert nn.Linear in config.layer_type_quant_config
    assert config.layer_type_quant_config[nn.Linear].weight.group_size == 64


def test_exclude_layers():
    """Test exclude layers"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_32", exclude_layers=["*.mlp.gate_proj"])
    assert isinstance(config, Config)
    assert "*.mlp.gate_proj" in config.exclude


def test_full_feature_combination():
    """Test using all features together"""
    template = LLMTemplate.get("llama")

    per_layer_config = {"special_layer": "mxfp4"}

    layer_type_config = {nn.Linear: "int4_wo_64"}

    config = template.get_config(
        scheme="fp8",
        algorithm="awq",
        kv_cache_scheme="fp8",
        min_kv_scale=1.0,
        attention_scheme="fp8",
        layer_config=per_layer_config,
        layer_type_config=layer_type_config,
        exclude_layers=["*.mlp.gate_proj"],
    )

    assert isinstance(config, Config)
    assert config.global_quant_config.weight.dtype == Dtype.fp8_e4m3
    assert len(config.algo_config) > 0
    assert config.softmax_quant_spec is not None
    assert len(config.layer_quant_config) > 0
    assert len(config.kv_cache_quant_config) > 0
    assert "special_layer" in config.layer_quant_config
    assert config.layer_quant_config["special_layer"].weight.dtype == Dtype.fp4
    assert nn.Linear in config.layer_type_quant_config
    assert config.layer_type_quant_config[nn.Linear].weight.group_size == 64
    assert "*.mlp.gate_proj" in config.exclude
    assert config.min_kv_scale == 1.0


def test_with_llm_template_all_params():
    """Test with_llm_template with all parameters"""
    template = LLMTemplate.get("llama")

    per_layer_config = {"test_layer": "uint4_wo_64"}

    layer_type_config = {nn.Linear: "uint4_wo_128"}

    config = Config.with_llm_template(
        template=template,
        scheme="fp8",
        algorithm="awq",
        kv_cache_scheme="fp8",
        min_kv_scale=1.0,
        attention_scheme="fp8",
        layer_config=per_layer_config,
        layer_type_config=layer_type_config,
        exclude_layers=["*.mlp.gate_proj"],
    )

    assert isinstance(config, Config)
    assert config.global_quant_config.weight.dtype == Dtype.fp8_e4m3
    assert len(config.algo_config) > 0
    assert config.softmax_quant_spec is not None
    assert "test_layer" in config.layer_quant_config
    assert config.layer_quant_config["test_layer"].weight.dtype == Dtype.uint4
    assert config.layer_quant_config["test_layer"].weight.group_size == 64
    assert nn.Linear in config.layer_type_quant_config
    assert config.layer_type_quant_config[nn.Linear].weight.group_size == 128
    assert "*.mlp.gate_proj" in config.exclude
    assert config.min_kv_scale == 1.0


def test_builtin_templates_exist():
    """Test that all expected built-in templates exist"""
    expected_models = [
        "chatglm",
        "cohere",
        "dbrx",
        "deepseek",
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v32",
        "deepseek_vl_v2",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "glm4_moe",
        "gptj",
        "gpt_oss",
        "granitemoehybrid",
        "grok-1",
        "instella",
        "kimi_k25",
        "llama",
        "llama4",
        "minimax_m2",
        "mistral",
        "mixtral",
        "mllama",
        "olmo",
        "opt",
        "phi",
        "phi3",
        "qwen",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "qwen3_next",
        "qwen3_vl_moe",
    ]

    available_models = LLMTemplate.list_available()

    assert set(expected_models) == set(available_models), (
        "expected_models and available_models should contain the same model types"
    )


def test_builtin_templates_create_valid_configs():
    """Test that built-in templates can create valid configurations"""
    test_models = ["llama", "opt", "qwen", "mllama", "gemma2"]

    for model_type in test_models:
        template = LLMTemplate.get(model_type)
        config = template.get_config("int4_wo_128")
        assert isinstance(config, Config)
        assert config.global_quant_config.weight is not None


def test_template_with_algorithm_configs():
    """Test template with custom algorithm configurations"""
    custom_awq = AWQConfig(name="awq", scaling_layers=[], model_decoder_layers="test.layers")
    custom_gptq = GPTQConfig(name="gptq", block_size=256, inside_layer_modules=["test_module"])
    custom_smoothquant = SmoothQuantConfig(name="smooth", alpha=0.8, scaling_layers=[])
    custom_autosmoothquant = AutoSmoothQuantConfig(name="autosmoothquant", scaling_layers=[], compute_scale_loss="MSE")
    custom_rotation = RotationConfig(
        name="rotation",
        backbone="model",
        model_decoder_layers="test.layers",
        v_proj="self_attn.v_proj",
        o_proj="self_attn.o_proj",
        self_attn="self_attn",
        mlp="mlp",
        r1=True,
        r2=False,
        scaling_layers={
            "first_layer": [
                {
                    "prev_modules": ["model.embed_tokens"],
                    "norm_module": "model.layers.layer_id.input_layernorm",
                    "next_modules": [
                        "model.layers.layer_id.self_attn.q_proj",
                        "model.layers.layer_id.self_attn.k_proj",
                        "model.layers.layer_id.self_attn.v_proj",
                    ],
                }
            ],
            "middle_layers": [
                {
                    "prev_modules": ["model.layers.pre_layer_id.mlp.down_proj"],
                    "norm_module": "model.layers.layer_id.input_layernorm",
                    "next_modules": [
                        "model.layers.layer_id.self_attn.q_proj",
                        "model.layers.layer_id.self_attn.k_proj",
                        "model.layers.layer_id.self_attn.v_proj",
                    ],
                }
            ],
            "last_layer": [
                {
                    "prev_modules": ["model.layers.layer_id.mlp.down_proj"],
                    "norm_module": "model.norm",
                    "next_modules": ["lm_head"],
                }
            ],
        },
    )

    template = LLMTemplate(
        model_type="test_custom_algos",
        kv_layers_name=["*k_proj"],
        q_layer_name="*q_proj",
        exclude_layers_name=["lm_head"],
        awq_config=custom_awq,
        gptq_config=custom_gptq,
        smoothquant_config=custom_smoothquant,
        autosmoothquant_config=custom_autosmoothquant,
        rotation_config=custom_rotation,
    )

    # Verify algo_config dictionary structure
    assert isinstance(template.algo_config, dict)
    assert template.algo_config["awq"] is custom_awq
    assert template.algo_config["gptq"] is custom_gptq
    assert template.algo_config["smoothquant"] is custom_smoothquant
    assert template.algo_config["autosmoothquant"] is custom_autosmoothquant
    assert template.algo_config["rotation"] is custom_rotation

    # Test AWQ custom config
    config = template.get_config("int4_wo_128", algorithm="awq")
    assert config.algo_config[0] is custom_awq

    # Test GPTQ custom config
    config = template.get_config("int4_wo_128", algorithm="gptq")
    assert config.algo_config[0] is custom_gptq

    # Test SQ custom config
    config = template.get_config("int4_wo_128", algorithm="smoothquant")
    assert config.algo_config[0] is custom_smoothquant

    # Test AutoSmoothQuant custom config
    config = template.get_config("int4_wo_128", algorithm="autosmoothquant")
    assert config.algo_config[0] is custom_autosmoothquant

    # Test Rotation custom config
    config = template.get_config("int4_wo_128", algorithm="rotation")
    assert config.algo_config[0] is custom_rotation


def test_multiple_layers_kv_cache():
    """Test KV cache config with multiple layers"""
    template = LLMTemplate(
        model_type="test_multi_kv",
        kv_layers_name=["*k_proj", "*v_proj", "*other_proj"],
        q_layer_name="*q_proj",
        exclude_layers_name=["lm_head"],
    )

    config = Config(
        global_quant_config=QLayerConfig(
            weight=Int4PerGroupSpec(
                ch_axis=-1, group_size=32, is_dynamic=False, scale_type="float"
            ).to_quantization_spec(),
            input_tensors=Int4PerGroupSpec(
                ch_axis=-1, group_size=32, is_dynamic=True, scale_type="float"
            ).to_quantization_spec(),
        )
    )

    config = template._set_kv_cache_config(config, "fp8")

    # Should have configs for all KV layers
    assert len(config.layer_quant_config) == 3
    assert len(config.kv_cache_quant_config) == 3


def test_register_scheme():
    """Test register scheme"""
    template = LLMTemplate.get("llama")
    quant_spec = Int8PerTensorSpec(is_dynamic=False).to_quantization_spec()
    template.register_scheme("int8_wo", QLayerConfig(weight=quant_spec))
    assert "int8_wo" in template._SUPPORTED_SCHEMES
    assert template.get_config("int8_wo") is not None
    # Clean up
    template.unregister_scheme("int8_wo")
    assert "int8_wo" not in template._SUPPORTED_SCHEMES
    with pytest.raises(ValueError, match="Unsupported quantization scheme: int8_wo"):
        template.get_config("int8_wo")


def test_get_config_with_algo_configs():
    """Test get_config with algo_configs parameter"""
    template = LLMTemplate.get("llama")

    custom_awq_config = AWQConfig(
        name="awq",
        scaling_layers=[],
        model_decoder_layers="custom.layers",
    )

    # Get config with custom algo_configs
    config = template.get_config("int4_wo_128", algorithm="awq", algo_configs={"awq": custom_awq_config})

    # Verify the custom config is used
    assert config.algo_config is not None
    assert len(config.algo_config) == 1
    assert config.algo_config[0] is custom_awq_config
    assert config.algo_config[0].model_decoder_layers == "custom.layers"

    # Verify that the template's original config is not modified
    default_config = template.get_config("int4_wo_128", algorithm="awq")
    assert default_config.algo_config[0] is not custom_awq_config
