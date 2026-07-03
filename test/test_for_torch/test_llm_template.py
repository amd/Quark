#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


import pytest
import torch.nn as nn

from quark.torch import LLMTemplate
from quark.torch.quantization.config import template as template_module
from quark.torch.quantization.config.config import (
    AutoSmoothQuantConfig,
    AWQConfig,
    FP8E4M3PerTensorSpec,
    GPTQConfig,
    Int4PerChannelSpec,
    Int4PerGroupSpec,
    Int8PerTensorSpec,
    ProgressiveSpec,
    QConfig,
    QLayerConfig,
    QronosConfig,
    RotationConfig,
    SmoothQuantConfig,
)
from quark.torch.quantization.config.config_verification import ConfigVerifier
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType


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
    assert isinstance(config, QConfig)

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
        "int4_wa_64",
        "uint4_wo_32",
        "uint4_wo_64",
        "uint4_wo_128",
        "uint4_wo_per_channel",
        "mxfp4",
        "mxfp4_mxfp6_e2m3",
        "mxfp4_fp8",
        "mxfp4_weight_only",
        "mxfp6_e3m2",
        "mxfp6_e2m3",
        "mx6",
        "bfp16",
        "int8",
        "nvfp4",
        "fp4_block16_scale_e4m3",  # legacy alias of nvfp4
        "amdfp4",
        "amdfp4_g32",
        "amdfp4_global16",
        "amdfp4_global32",
        "int4_fp8",
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
        "nvfp4",
        "fp4_block16_scale_e4m3",  # legacy alias of nvfp4
    ]
    for scheme in working_schemes:
        config = template.get_config(scheme)
        assert isinstance(config, QConfig)
        assert config.global_quant_config.weight is not None


def test_int4_wo_32_scheme():
    """Test INT4 weight-only quantization scheme with group size 32"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_32")
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.int4
    assert config.global_quant_config.weight.group_size == 32
    assert config.global_quant_config.input_tensors is None


def test_int4_wo_64_scheme():
    """Test INT4 weight-only quantization scheme with group size 64"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_64")
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.int4
    assert config.global_quant_config.weight.group_size == 64
    assert config.global_quant_config.input_tensors is None


def test_int4_wo_128_scheme():
    """Test INT4 weight-only quantization scheme with group size 128"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_128")
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.int4
    assert config.global_quant_config.weight.group_size == 128
    assert config.global_quant_config.input_tensors is None


def test_int4_wo_per_channel_scheme():
    """Test INT4 weight-only per-channel quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_per_channel")
    assert isinstance(config, QConfig)
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
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.uint4
    assert config.global_quant_config.weight.group_size == 32
    assert config.global_quant_config.input_tensors is None


def test_uint4_wo_64_scheme():
    """Test UINT4 weight-only quantization scheme with group size 64"""
    template = LLMTemplate.get("llama")
    config = template.get_config("uint4_wo_64")
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.uint4
    assert config.global_quant_config.weight.group_size == 64
    assert config.global_quant_config.input_tensors is None


def test_uint4_wo_128_scheme():
    """Test UINT4 weight-only quantization scheme with group size 128"""
    template = LLMTemplate.get("llama")
    config = template.get_config("uint4_wo_128")
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.weight.dtype == Dtype.uint4
    assert config.global_quant_config.weight.group_size == 128
    assert config.global_quant_config.input_tensors is None


def test_uint4_wo_per_channel_scheme():
    """Test UINT4 weight-only per-channel quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("uint4_wo_per_channel")
    assert isinstance(config, QConfig)
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
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.fp8_e4m3
    assert config.global_quant_config.input_tensors.dtype == Dtype.fp8_e4m3


def test_ptpc_fp8_scheme():
    """Test PTPC FP8 quantization scheme (Per-Token Per-Channel)"""
    template = LLMTemplate.get("llama")
    config = template.get_config("ptpc_fp8")
    assert isinstance(config, QConfig)
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


def _build_kimi_k2_w4a8_reference_config():
    """Build the W4A8 reference config from the amd/Kimi-K2-Thinking-W4A8 model card.

    This is the exact ``get_config()`` published in the model card's quantization script
    (https://huggingface.co/amd/Kimi-K2-Thinking-W4A8), reproduced verbatim so the
    ``int4_fp8`` template scheme can be asserted equivalent to it field by field.
    """
    input_spec = FP8E4M3PerTensorSpec(
        observer_method="min_max", scale_type="float32", is_dynamic=True
    ).to_quantization_spec()
    weight_spec = ProgressiveSpec(
        first_stage=FP8E4M3PerTensorSpec(observer_method="min_max", scale_type="float32", is_dynamic=False),
        second_stage=Int4PerChannelSpec(
            symmetric=True, scale_type="float32", round_method="half_even", is_dynamic=False, ch_axis=0
        ),
    ).to_quantization_spec()
    return QConfig(global_quant_config=QLayerConfig(input_tensors=input_spec, weight=weight_spec))


def test_int4_fp8_scheme():
    """Test the int4_fp8 scheme is field-for-field equivalent to the amd/Kimi-K2-Thinking-W4A8 model card.

    The generated config is compared against the model card's published ``get_config()`` so that
    every quantization spec field (dtype, qscheme, ch_axis, is_dynamic, symmetric, round_method,
    scale_type, observer_cls, etc.) is guaranteed to match exactly.
    """
    reference_config = _build_kimi_k2_w4a8_reference_config()
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_fp8")
    assert isinstance(config, QConfig)

    reference_weight_stages = reference_config.global_quant_config.weight
    generated_weight_stages = config.global_quant_config.weight
    assert isinstance(generated_weight_stages, list)
    assert len(generated_weight_stages) == len(reference_weight_stages) == 2

    # Compare the full serialized spec so any field divergence from the model card is caught.
    assert generated_weight_stages[0].to_dict() == reference_weight_stages[0].to_dict()
    assert generated_weight_stages[1].to_dict() == reference_weight_stages[1].to_dict()
    assert (
        config.global_quant_config.input_tensors.to_dict()
        == reference_config.global_quant_config.input_tensors.to_dict()
    )


def test_mxfp4_scheme():
    """Test MXFP4 quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("mxfp4")
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.fp4
    assert config.global_quant_config.input_tensors.dtype == Dtype.fp4


def test_mxfp6_e3m2_scheme():
    """Test MXFP6E3M2 quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("mxfp6_e3m2")
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.fp6_e3m2


def test_mxfp6_e2m3_scheme():
    """Test MXFP6E2M3 quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("mxfp6_e2m3")
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.fp6_e2m3


def test_int8_scheme():
    """Test INT8 quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int8")
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.int8
    assert config.global_quant_config.input_tensors.dtype == Dtype.int8


def test_mx6_scheme():
    """Test MX6 quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("mx6")
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.mx6
    assert config.global_quant_config.input_tensors.dtype == Dtype.mx6


def test_bfp16_scheme():
    """Test BFP16 quantization scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config("bfp16")
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None
    assert config.global_quant_config.weight.dtype == Dtype.bfp16
    assert config.global_quant_config.input_tensors.dtype == Dtype.bfp16


def test_nvfp4_scheme():
    """Test NVFP4 (FP4 Block16 with FP8 E4M3 scale) quantization scheme."""
    template = LLMTemplate.get("llama")
    config = template.get_config("nvfp4")
    assert isinstance(config, QConfig)
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None

    # Weight: Two-stage quantization (FP4 per-group with FP8 E4M3 scale, static)
    assert isinstance(config.global_quant_config.weight, list)
    assert len(config.global_quant_config.weight) == 2

    # First stage: FP4 per-group
    weight_first_stage = config.global_quant_config.weight[0]
    assert weight_first_stage.dtype == Dtype.fp4
    assert weight_first_stage.qscheme == QSchemeType.per_group
    assert weight_first_stage.group_size == 16
    assert weight_first_stage.is_dynamic is False
    assert weight_first_stage.ch_axis == -1
    assert weight_first_stage.is_scale_quant is False

    # Second stage: FP8 E4M3 scale quantization
    weight_second_stage = config.global_quant_config.weight[1]
    assert weight_second_stage.dtype == Dtype.fp8_e4m3
    assert weight_second_stage.qscheme == QSchemeType.per_tensor
    assert weight_second_stage.scale_type == ScaleType.float32
    assert weight_second_stage.is_dynamic is False
    assert weight_second_stage.is_scale_quant is True

    # Activation: Two-stage quantization (FP4 per-group with FP8 E4M3 scale)
    assert isinstance(config.global_quant_config.input_tensors, list)
    assert len(config.global_quant_config.input_tensors) == 2

    # First stage: FP4 per-group (dynamic)
    input_first_stage = config.global_quant_config.input_tensors[0]
    assert input_first_stage.dtype == Dtype.fp4
    assert input_first_stage.qscheme == QSchemeType.per_group
    assert input_first_stage.group_size == 16
    assert input_first_stage.is_dynamic is True
    assert input_first_stage.ch_axis == -1
    assert input_first_stage.is_scale_quant is False

    # Second stage: FP8 E4M3 scale quantization
    input_second_stage = config.global_quant_config.input_tensors[1]
    assert input_second_stage.dtype == Dtype.fp8_e4m3
    assert input_second_stage.qscheme == QSchemeType.per_tensor
    assert input_second_stage.is_dynamic is False
    assert input_second_stage.is_scale_quant is True
    assert input_second_stage.scale_type == ScaleType.float32

    config_verifier = ConfigVerifier(config)
    assert config_verifier.is_act_dynamic is True


def test_nvfp4_default_shared_scale_groups():
    """Test that nvfp4 scheme sets the expected shared-scale defaults."""
    template = LLMTemplate.get("llama")
    config = template.get_config("nvfp4")

    # NVFP4 should have default shared_scale_groups
    assert config.shared_scale_groups == [["q_proj", "k_proj", "v_proj"], ["gate_proj", "up_proj"]]
    assert config.sync_moe_expert_input_amax is True


def test_nvfp4_shared_scale_groups_explicit_override():
    """Test that shared_scale_groups can be explicitly overridden for nvfp4."""
    template = LLMTemplate.get("llama")

    # Override with custom groups
    config = template.get_config(
        "nvfp4",
        shared_scale_groups=[["q_proj", "k_proj"]],
    )
    assert config.shared_scale_groups == [["q_proj", "k_proj"]]

    # Disable with empty list
    config_disabled = template.get_config(
        "nvfp4",
        shared_scale_groups=[],
    )
    assert config_disabled.shared_scale_groups == []


def test_nvfp4_legacy_alias():
    """The legacy ``fp4_block16_scale_e4m3`` name must keep returning the same scheme."""
    template = LLMTemplate.get("llama")
    new_config = template.get_config("nvfp4")
    legacy_config = template.get_config("fp4_block16_scale_e4m3")
    assert legacy_config.shared_scale_groups == new_config.shared_scale_groups
    assert legacy_config.sync_moe_expert_input_amax == new_config.sync_moe_expert_input_amax
    assert type(legacy_config.global_quant_config.weight) is type(new_config.global_quant_config.weight)


def test_non_nvfp4_no_default_shared_scale_groups():
    """Test that non-NVFP4 schemes do not have default shared_scale_groups"""
    template = LLMTemplate.get("llama")

    for scheme in ["fp8", "int4_wo_128", "mxfp4", "bfp16", "ptpc_fp8"]:
        config = template.get_config(scheme)
        assert config.shared_scale_groups == [], f"{scheme} should not have default shared_scale_groups"
        assert config.sync_moe_expert_input_amax is False, f"{scheme} should not enable MoE expert input sync"


def test_non_nvfp4_explicit_shared_scale_groups():
    """Test that shared_scale_groups can be explicitly set for any scheme"""
    template = LLMTemplate.get("llama")
    config = template.get_config(
        "fp8",
        shared_scale_groups=[["q_proj", "k_proj", "v_proj"], ["gate_proj", "up_proj"]],
    )
    assert config.shared_scale_groups == [["q_proj", "k_proj", "v_proj"], ["gate_proj", "up_proj"]]


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
    assert isinstance(config, QConfig)
    assert len(config.algo_config) > 0
    assert isinstance(config.algo_config[0], AWQConfig)
    assert config.algo_config[0].name == "awq"


def test_gptq_algorithm():
    """Test GPTQ algorithm with custom configs"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_128", algorithm="gptq")
    assert isinstance(config, QConfig)
    assert len(config.algo_config) > 0
    assert isinstance(config.algo_config[0], GPTQConfig)
    assert config.algo_config[0].name == "gptq"


def test_qronos_algorithm():
    """Test Qronos algorithm with custom configs"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_128", algorithm="qronos")
    assert isinstance(config, QConfig)
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
    assert isinstance(config, QConfig)
    assert len(config.algo_config) > 0
    assert isinstance(config.algo_config[0], SmoothQuantConfig)
    assert config.algo_config[0].name == "smooth"


def test_autosmoothquant_algorithm():
    """Test AutoSmoothQuant algorithm with custom configs"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_32", algorithm="autosmoothquant")
    assert isinstance(config, QConfig)
    assert len(config.algo_config) > 0
    assert isinstance(config.algo_config[0], AutoSmoothQuantConfig)


def test_rotation_algorithm():
    """Test Rotation algorithm with custom configs"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_128", algorithm="rotation")
    assert isinstance(config, QConfig)
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

    # Test that missing Qronos config raises NotImplementedError
    with pytest.raises(
        NotImplementedError,
        match="No built-in qronos configuration is available for the 'test_missing_configs' architecture",
    ):
        template.get_config("int4_wo_128", algorithm="qronos")


def test_unsupported_algorithm():
    """Test unsupported algorithm raises error"""
    template = LLMTemplate.get("llama")
    with pytest.raises(ValueError, match="Unsupported algorithm: invalid_algo"):
        template.get_config("int4_wo_128", algorithm="invalid_algo")


def test_fp8_kv_cache_scheme():
    """Test FP8 KV cache quantization"""
    template = LLMTemplate.get("llama")
    config = template.get_config("fp8", kv_cache_scheme="fp8")

    assert isinstance(config, QConfig)
    assert len(config.layer_quant_config) > 0
    assert len(config.kv_cache_quant_config) > 0


def test_kv_cache_inherits_layer_override_spec():
    """KV projection layers must inherit weight/input spec from a user layer override.

    Regression for: when global=mxfp4, user passes ``layer_config={'*self_attn*': 'fp8'}``
    and ``kv_cache_scheme='fp8'``, the resulting ``layer_quant_config['*k_proj']`` and
    ``kv_cache_quant_config['*k_proj']`` weight/input must be fp8, not the global mxfp4.
    """
    template = LLMTemplate.get("qwen3")
    config = template.get_config(
        scheme="mxfp4",
        kv_cache_scheme="fp8",
        layer_config={"*self_attn*": "fp8"},
        exclude_layers=["lm_head"],
    )

    for pattern in ("*k_proj", "*v_proj"):
        layer_cfg = config.layer_quant_config[pattern]
        assert layer_cfg.weight.dtype == Dtype.fp8_e4m3
        assert layer_cfg.input_tensors.dtype == Dtype.fp8_e4m3
        assert layer_cfg.output_tensors.dtype == Dtype.fp8_e4m3
        kv_cfg = config.kv_cache_quant_config[pattern]
        assert kv_cfg.weight.dtype == Dtype.fp8_e4m3
        assert kv_cfg.input_tensors.dtype == Dtype.fp8_e4m3
        assert kv_cfg.output_tensors.dtype == Dtype.fp8_e4m3


def test_resolve_projection_base_spec_matches_self_attn_override():
    """User override targeting the self_attn container must propagate to kv projections."""
    template = LLMTemplate.get("qwen3")
    config = template.get_config(
        scheme="mxfp4",
        layer_config={"*self_attn*": "fp8"},
        exclude_layers=["lm_head"],
    )
    # Strip the kv entries that get_config already synthesised (`*k_proj`, `*v_proj`);
    # we want to exercise the helper against the user override only.
    config.layer_quant_config = {k: v for k, v in config.layer_quant_config.items() if k == "*self_attn*"}
    spec = template_module._resolve_projection_base_spec(config, "*k_proj")
    assert spec.weight.dtype == Dtype.fp8_e4m3
    assert spec.input_tensors.dtype == Dtype.fp8_e4m3


def test_resolve_projection_base_spec_empty_layer_config_falls_back_to_global():
    """Empty ``layer_quant_config`` is the no-override case — fall back to global."""
    template = LLMTemplate.get("qwen3")
    config = template.get_config(scheme="mxfp4", exclude_layers=["lm_head"])
    config.layer_quant_config = {}
    spec = template_module._resolve_projection_base_spec(config, "*k_proj")
    assert spec is config.global_quant_config


def test_resolve_projection_base_spec_ignores_non_attention_override():
    """An MLP-only override must NOT be picked up as the kv projection base spec."""
    template = LLMTemplate.get("qwen3")
    config = template.get_config(scheme="mxfp4", exclude_layers=["lm_head"])
    fp8_spec = template_module.FP8Scheme().config
    config.layer_quant_config = {"*mlp*": fp8_spec}
    spec = template_module._resolve_projection_base_spec(config, "*k_proj")
    assert spec is config.global_quant_config


def test_resolve_projection_base_spec_matches_direct_leaf_override():
    """User writing ``*k_proj`` directly (no self_attn token) must still be the base."""
    template = LLMTemplate.get("qwen3")
    config = template.get_config(scheme="mxfp4", exclude_layers=["lm_head"])
    fp8_spec = template_module.FP8Scheme().config
    config.layer_quant_config = {"*k_proj": fp8_spec, "*v_proj": fp8_spec}
    spec = template_module._resolve_projection_base_spec(config, "*k_proj")
    assert spec.weight.dtype == Dtype.fp8_e4m3
    assert spec.input_tensors.dtype == Dtype.fp8_e4m3


def test_resolve_projection_base_spec_leaf_only_v_falls_back_for_k():
    """User overriding only ``*v_proj`` must NOT bleed into ``*k_proj`` resolution."""
    template = LLMTemplate.get("qwen3")
    config = template.get_config(scheme="mxfp4", exclude_layers=["lm_head"])
    fp8_spec = template_module.FP8Scheme().config
    config.layer_quant_config = {"*v_proj": fp8_spec}
    spec = template_module._resolve_projection_base_spec(config, "*k_proj")
    assert spec is config.global_quant_config


def test_kv_cache_applies_even_when_kv_pattern_is_excluded():
    """exclude applies to weight/input only; kv-cache output is still wired
    (matches runtime exclude-with-kv branch in get_layer_quant_config)."""
    template = LLMTemplate.get("qwen3")
    config = template.get_config(
        scheme="mxfp4",
        kv_cache_scheme="fp8",
        exclude_layers=["lm_head", "*k_proj"],
    )
    # Weight quant is excluded → no layer_quant_config entry.
    assert "*k_proj" not in config.layer_quant_config
    # But kv-cache (output) is still wired with weight/input cleared.
    assert "*k_proj" in config.kv_cache_quant_config
    kv = config.kv_cache_quant_config["*k_proj"]
    assert kv.weight is None
    assert kv.input_tensors is None
    assert kv.output_tensors.dtype == Dtype.fp8_e4m3
    # Non-excluded sibling still gets the full kv entry.
    assert "*v_proj" in config.kv_cache_quant_config
    assert "*v_proj" in config.layer_quant_config


def test_kv_cache_inherits_self_attn_override_spec():
    """User sets *self_attn*→fp8 atop an mxfp4 base; k_proj/v_proj kv-cache
    entries must inherit weight+input from the self_attn override (fp8), not
    from the mxfp4 global. Locks the bug fixed in _set_kv_cache_config /
    _resolve_projection_base_spec."""
    template = LLMTemplate.get("qwen3")
    config = template.get_config(
        scheme="mxfp4",
        kv_cache_scheme="fp8",
        layer_config={"*self_attn*": "fp8"},
    )
    for pattern in ("*k_proj", "*v_proj"):
        assert pattern in config.layer_quant_config, pattern
        entry = config.layer_quant_config[pattern]
        # Weight/input must come from self_attn fp8 override, NOT mxfp4 global.
        assert entry.weight.dtype == Dtype.fp8_e4m3, pattern
        assert entry.input_tensors.dtype == Dtype.fp8_e4m3, pattern
        # Output (kv-cache) is fp8 per_tensor.
        assert entry.output_tensors.dtype == Dtype.fp8_e4m3, pattern
        # kv_cache_quant_config mirrors the same spec.
        kv = config.kv_cache_quant_config[pattern]
        assert kv.weight.dtype == Dtype.fp8_e4m3, pattern
        assert kv.output_tensors.dtype == Dtype.fp8_e4m3, pattern


def test_unsupported_kv_cache_scheme():
    """Test unsupported KV cache scheme raises error"""
    template = LLMTemplate.get("llama")
    with pytest.raises(ValueError, match="Unsupported KV cache scheme: invalid_kv"):
        template.get_config("fp8", kv_cache_scheme="invalid_kv")


def test_min_kv_scale():
    """Test min_kv_scale"""
    template = LLMTemplate.get("llama")
    config = template.get_config("fp8", kv_cache_scheme="fp8", min_kv_scale=1.0)
    assert isinstance(config, QConfig)
    assert config.min_kv_scale == 1.0


def test_fp8_attention_scheme():
    """Test FP8 attention quantization"""
    template = LLMTemplate.get("llama")
    config = template.get_config("fp8", attention_scheme="fp8")

    assert isinstance(config, QConfig)
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

    assert isinstance(config, QConfig)
    assert "layer1" in config.layer_quant_config
    assert "layer2" in config.layer_quant_config
    assert config.layer_quant_config["layer1"].weight.group_size == 64
    assert config.layer_quant_config["layer2"].weight.group_size == 128


def test_layer_type_config():
    """Test layer type config"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_32", layer_type_config={nn.Linear: "int4_wo_64"})
    assert isinstance(config, QConfig)
    assert len(config.layer_type_quant_config) > 0
    assert nn.Linear in config.layer_type_quant_config
    assert config.layer_type_quant_config[nn.Linear].weight.group_size == 64


def test_exclude_layers():
    """Test exclude layers"""
    template = LLMTemplate.get("llama")
    config = template.get_config("int4_wo_32", exclude_layers=["*.mlp.gate_proj"])
    assert isinstance(config, QConfig)
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

    assert isinstance(config, QConfig)
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

    config = QConfig.with_llm_template(
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

    assert isinstance(config, QConfig)
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
        "deepseek_v4",
        "deepseek_vl_v2",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "glm4_moe",
        "glm4_moe_lite",
        "glm_moe_dsa",
        "gptj",
        "gpt_oss",
        "granitemoehybrid",
        "grok-1",
        "instella",
        "kimi_k2",
        "kimi_k25",
        "llama",
        "llama4",
        "minimax_m2",
        "minimax_m3_vl",
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
        "qwen3_5_moe",
    ]

    available_models = LLMTemplate.list_available()

    assert set(expected_models) == set(available_models), (
        "expected_models and available_models should contain the same model types"
    )


def test_builtin_templates_create_valid_configs():
    """Test that built-in templates can create valid configurations"""
    test_models = ["llama", "opt", "qwen", "mllama", "gemma2", "glm4_moe_lite"]

    for model_type in test_models:
        template = LLMTemplate.get(model_type)
        config = template.get_config("int4_wo_128")
        assert isinstance(config, QConfig)
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
        algorithm_configs={
            "awq": custom_awq,
            "gptq": custom_gptq,
            "smoothquant": custom_smoothquant,
            "autosmoothquant": custom_autosmoothquant,
            "rotation": custom_rotation,
        },
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


def test_template_with_algorithm_configs_dictionary_parameter():
    """Test template initialization using algorithm_configs dictionary."""
    custom_awq = AWQConfig(name="awq", scaling_layers=[], model_decoder_layers="dictionary.layers")
    custom_gptq = GPTQConfig(name="gptq", block_size=64, inside_layer_modules=["dictionary_module"])

    template = LLMTemplate(
        model_type="test_dictionary_algos",
        kv_layers_name=["*k_proj"],
        q_layer_name="*q_proj",
        exclude_layers_name=["lm_head"],
        algorithm_configs={"awq": custom_awq, "gptq": custom_gptq},
    )

    awq_quantization_config = template.get_config("int4_wo_128", algorithm="awq")
    assert awq_quantization_config.algo_config[0] is custom_awq

    gptq_quantization_config = template.get_config("int4_wo_128", algorithm="gptq")
    assert gptq_quantization_config.algo_config[0] is custom_gptq


def test_template_with_deprecated_algorithm_keyword_argument_warns(monkeypatch: pytest.MonkeyPatch):
    """Test deprecated algorithm keyword arguments are still accepted with warning logs."""
    custom_awq = AWQConfig(name="awq", scaling_layers=[], model_decoder_layers="deprecated.layers")
    warning_messages: list[str] = []

    def _capture_warning(message: str, *args: object, **kwargs: object) -> None:
        warning_messages.append(message)

    monkeypatch.setattr(template_module.logger, "warning", _capture_warning)

    template = LLMTemplate(
        model_type="test_deprecated_algo_kwarg",
        kv_layers_name=["*k_proj"],
        q_layer_name="*q_proj",
        exclude_layers_name=["lm_head"],
        awq_config=custom_awq,
    )

    assert warning_messages
    assert "Deprecated keyword arguments were used when initializing `LLMTemplate`" in warning_messages[0]

    awq_quantization_config = template.get_config("int4_wo_128", algorithm="awq")
    assert awq_quantization_config.algo_config[0] is custom_awq


def test_template_with_invalid_legacy_algorithm_keyword_raises_error():
    """Test invalid legacy algorithm keyword argument raises helpful error."""
    with pytest.raises(ValueError, match="Unsupported legacy algorithm keyword 'invalid_config'"):
        LLMTemplate(
            model_type="test_invalid_legacy_keyword",
            kv_layers_name=["*k_proj"],
            q_layer_name="*q_proj",
            exclude_layers_name=["lm_head"],
            invalid_config=None,
        )


def test_multiple_layers_kv_cache():
    """Test KV cache config with multiple layers"""
    template = LLMTemplate(
        model_type="test_multi_kv",
        kv_layers_name=["*k_proj", "*v_proj", "*other_proj"],
        q_layer_name="*q_proj",
        exclude_layers_name=["lm_head"],
    )

    config = QConfig(
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


def test_int4_weight_and_activation_scheme():
    from quark.torch.quantization.config.template import Int4WeightAndActivationScheme

    scheme = Int4WeightAndActivationScheme(group_size=64)
    cfg = scheme.config
    assert cfg is not None
    assert cfg.weight is not None
    assert cfg.input_tensors is not None


def test_scheme_collection_w4a4():
    from quark.torch.quantization.config.template import QuantizationSchemeCollection

    coll = QuantizationSchemeCollection()
    scheme = coll.get_scheme("int4_wa_64")
    cfg = scheme.config
    assert cfg is not None
