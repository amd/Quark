#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


import pytest
import torch
import torch.nn as nn

from quark.torch.algorithm.utils.prepare import get_layers_for_scaling
from quark.torch.quantization.config.algo_configs import (
    AUTOROUND_MAP,
    AUTOSMOOTHQUANT_MAP,
    AWQ_MAP,
    GPTAQ_MAP,
    GPTQ_MAP,
    QRONOS_MAP,
    ROTATION_MAP,
    SQ_MAP,
    get_algo_config,
    get_supported_algorithm_types,
)
from quark.torch.quantization.config.config import (
    AutoRoundConfig,
    AutoSmoothQuantConfig,
    AWQConfig,
    GPTAQConfig,
    GPTQConfig,
    QronosConfig,
    RotationConfig,
    SmoothQuantConfig,
)

# Qwen3.5 dense model type key, shared by the AWQ / GPTQ / Qronos / SmoothQuant / AutoRound maps.
QWEN3_5_MODEL_TYPE = "qwen3_5"
# Linear module names that are quantized inside a Qwen3.5 hybrid decoder layer. Standard
# attention layers expose "self_attn.*" and gated-delta linear-attention layers expose
# "linear_attn.*"; both sets are listed in the GPTQ/Qronos configs so a single config
# covers every decoder layer.
QWEN3_5_LINEAR_ATTENTION_MODULES = [
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.out_proj",
]


def test_awq_map_basic():
    """Test basic AWQ_MAP functionality"""
    # Test some known models exist
    assert "llama" in AWQ_MAP
    assert "qwen" in AWQ_MAP
    assert "opt" in AWQ_MAP

    # Test configurations are AWQConfig instances
    for _, config in list(AWQ_MAP.items())[:3]:  # Test first 3 models
        assert isinstance(config, AWQConfig)
        assert config.name == "awq"
        assert hasattr(config, "scaling_layers")
        assert hasattr(config, "model_decoder_layers")


def test_gptq_map_basic():
    """Test basic GPTQ_MAP functionality"""
    # Test some known models exist
    assert "llama" in GPTQ_MAP
    assert "qwen" in GPTQ_MAP
    assert "opt" in GPTQ_MAP

    # Test configurations are GPTQConfig instances
    for _, config in list(GPTQ_MAP.items())[:3]:  # Test first 3 models
        assert isinstance(config, GPTQConfig)
        assert config.name == "gptq"
        assert hasattr(config, "block_size")
        assert hasattr(config, "inside_layer_modules")


def test_sq_map_basic():
    """Test basic SQ_MAP functionality"""
    # Test some known models exist
    assert "llama" in SQ_MAP
    assert "qwen" in SQ_MAP
    assert "opt" in SQ_MAP

    # Test configurations are SmoothQuantConfig instances
    for _, config in list(SQ_MAP.items())[:3]:  # Test first 3 models
        assert isinstance(config, SmoothQuantConfig)
        assert config.name == "smooth"
        assert hasattr(config, "alpha")
        assert hasattr(config, "scaling_layers")


def test_autosmoothquant_map_basic():
    """Test basic AUTOSMOOTHQUANT_MAP functionality"""
    # Test some known models exist
    assert "llama" in AUTOSMOOTHQUANT_MAP
    assert "mixtral" in AUTOSMOOTHQUANT_MAP
    assert "deepseek_v2" in AUTOSMOOTHQUANT_MAP

    # Test configurations are AutoSmoothQuantConfig instances
    for _, config in list(AUTOSMOOTHQUANT_MAP.items())[:3]:  # Test first 3 models
        assert isinstance(config, AutoSmoothQuantConfig)
        assert config.name == "autosmoothquant"
        assert hasattr(config, "scaling_layers")
        assert hasattr(config, "model_decoder_layers")
        assert hasattr(config, "compute_scale_loss")


def test_rotation_map_basic():
    """Test basic ROTATION_MAP functionality"""
    # Test some known models exist
    assert "llama" in ROTATION_MAP

    # Test configurations are RotationConfig instances
    for _, config in ROTATION_MAP.items():
        assert isinstance(config, RotationConfig)
        assert config.name == "rotation"
        assert hasattr(config, "model_decoder_layers")
        assert hasattr(config, "v_proj")
        assert hasattr(config, "o_proj")
        assert hasattr(config, "self_attn")
        assert hasattr(config, "mlp")
        assert hasattr(config, "r1")
        assert hasattr(config, "r2")
        assert hasattr(config, "r3")
        assert hasattr(config, "r4")


def test_get_algo_config_existing():
    """Test getting algorithm configs for existing models"""
    # Test AWQ
    awq_config = get_algo_config("awq", "llama")
    assert awq_config is not None
    assert isinstance(awq_config, AWQConfig)
    assert awq_config.name == "awq"

    # Test GPTQ
    gptq_config = get_algo_config("gptq", "llama")
    assert gptq_config is not None
    assert isinstance(gptq_config, GPTQConfig)
    assert gptq_config.name == "gptq"

    # Test SmoothQuant
    sq_config = get_algo_config("smoothquant", "llama")
    assert sq_config is not None
    assert isinstance(sq_config, SmoothQuantConfig)
    assert sq_config.name == "smooth"

    # Test AutoSmoothQuant
    autosq_config = get_algo_config("autosmoothquant", "llama")
    assert autosq_config is not None
    assert isinstance(autosq_config, AutoSmoothQuantConfig)
    assert autosq_config.name == "autosmoothquant"

    # Test Rotation
    rotation_config = get_algo_config("rotation", "llama")
    assert rotation_config is not None
    assert isinstance(rotation_config, RotationConfig)
    assert rotation_config.name == "rotation"


def test_qwen3_5_present_in_all_maps():
    """Verify qwen3_5 is registered for every supporting algorithm with the right config type."""
    assert isinstance(AWQ_MAP[QWEN3_5_MODEL_TYPE], AWQConfig)
    assert isinstance(GPTQ_MAP[QWEN3_5_MODEL_TYPE], GPTQConfig)
    assert isinstance(GPTAQ_MAP[QWEN3_5_MODEL_TYPE], GPTAQConfig)
    assert isinstance(QRONOS_MAP[QWEN3_5_MODEL_TYPE], QronosConfig)
    assert isinstance(SQ_MAP[QWEN3_5_MODEL_TYPE], SmoothQuantConfig)
    assert isinstance(AUTOSMOOTHQUANT_MAP[QWEN3_5_MODEL_TYPE], AutoSmoothQuantConfig)
    assert isinstance(AUTOROUND_MAP[QWEN3_5_MODEL_TYPE], AutoRoundConfig)


def test_qwen4_exp_present_in_all_maps():
    """Both qwen4_exp keys are registered for every algorithm that supports the architecture.

    file2file mode reads the raw top-level model_type ("qwen4_exp"); live model loading resolves
    the nested text_config instead ("qwen4_exp_text"). Both must be registered, and must select
    the same quantization scope -- only the decoder path differs (VL wrapper vs text-only).
    """
    for model_type in ("qwen4_exp", "qwen4_exp_text"):
        assert isinstance(AWQ_MAP[model_type], AWQConfig)
        assert isinstance(GPTQ_MAP[model_type], GPTQConfig)
        assert isinstance(GPTAQ_MAP[model_type], GPTAQConfig)
        assert isinstance(QRONOS_MAP[model_type], QronosConfig)
        assert isinstance(AUTOSMOOTHQUANT_MAP[model_type], AutoSmoothQuantConfig)

    for algo_map in (GPTQ_MAP, GPTAQ_MAP, QRONOS_MAP):
        assert algo_map["qwen4_exp"].inside_layer_modules == algo_map["qwen4_exp_text"].inside_layer_modules


def test_no_algo_map_still_keys_on_qwen3_5_text():
    """
    Guard against re-introducing the unreachable "qwen3_5_text" key.

    "qwen3_5_text" is the model_type of the nested text_config; Quark resolves algorithm
    configs against the top-level model_type, which is "qwen3_5" for every released
    Qwen3.5 dense checkpoint. An entry keyed on "qwen3_5_text" can never be looked up.
    """
    for map_name, algo_map in [
        ("AWQ_MAP", AWQ_MAP),
        ("GPTQ_MAP", GPTQ_MAP),
        ("QRONOS_MAP", QRONOS_MAP),
        ("SQ_MAP", SQ_MAP),
        ("AUTOROUND_MAP", AUTOROUND_MAP),
    ]:
        assert "qwen3_5_text" not in algo_map, (
            f"{map_name} still keys on the unreachable 'qwen3_5_text'; use 'qwen3_5' instead"
        )


def test_qwen3_5_model_decoder_layers_path():
    """
    Verify the qwen3_5 configs point at the actual decoder layer path.

    Qwen3_5ForConditionalGeneration wraps its decoder stack at
    model.language_model.layers, not model.layers (the latter has no `.layers`
    attribute and raises AttributeError when resolved against a loaded model).
    """
    for config in [
        AWQ_MAP[QWEN3_5_MODEL_TYPE],
        GPTQ_MAP[QWEN3_5_MODEL_TYPE],
        QRONOS_MAP[QWEN3_5_MODEL_TYPE],
        SQ_MAP[QWEN3_5_MODEL_TYPE],
        AUTOROUND_MAP[QWEN3_5_MODEL_TYPE],
    ]:
        assert config.model_decoder_layers == "model.language_model.layers"


def test_qwen3_5_via_get_algo_config():
    """Verify get_algo_config resolves qwen3_5 for every newly supported algorithm."""
    for algorithm_type, expected_config_type in [
        ("awq", AWQConfig),
        ("gptq", GPTQConfig),
        ("qronos", QronosConfig),
        ("smoothquant", SmoothQuantConfig),
        ("autoround", AutoRoundConfig),
    ]:
        config = get_algo_config(algorithm_type, QWEN3_5_MODEL_TYPE)
        assert config is not None, f"{algorithm_type} has no qwen3_5 config"
        assert isinstance(config, expected_config_type)


def test_qwen3_5_hybrid_layer_coverage():
    """
    Verify the blockwise qwen3_5 configs cover both attention variants of the
    hybrid decoder.

    Qwen3.5 dense layers alternate between standard attention (self_attn.*) and
    gated-delta linear attention (linear_attn.*); both module-name sets plus the shared
    MLP must appear in inside_layer_modules so a single config quantizes every layer.
    """
    for config in [
        GPTQ_MAP[QWEN3_5_MODEL_TYPE],
        QRONOS_MAP[QWEN3_5_MODEL_TYPE],
        AUTOROUND_MAP[QWEN3_5_MODEL_TYPE],
    ]:
        inside_layer_modules = config.inside_layer_modules
        assert "self_attn.q_proj" in inside_layer_modules
        assert "self_attn.o_proj" in inside_layer_modules
        for linear_attention_module in QWEN3_5_LINEAR_ATTENTION_MODULES:
            assert linear_attention_module in inside_layer_modules
        assert "mlp.gate_proj" in inside_layer_modules
        assert "mlp.up_proj" in inside_layer_modules
        assert "mlp.down_proj" in inside_layer_modules


def test_qwen3_5_smoothquant_alpha():
    """
    Verify the qwen3_5 SmoothQuant config uses alpha=0.5.

    The default alpha=1 pushes all quantization difficulty onto the weights and badly
    hurts W8A8 accuracy on this model; alpha=0.5 is the validated value, so guard against
    a regression back to the default.
    """
    smoothquant_config = SQ_MAP[QWEN3_5_MODEL_TYPE]
    assert smoothquant_config.alpha == 0.5


def test_get_algo_config_unsupported_model():
    """Test getting algorithm configs for unsupported models returns None"""
    # Test AWQ with unsupported model
    awq_config = get_algo_config("awq", "unsupported_model")
    assert awq_config is None

    # Test GPTQ with unsupported model
    gptq_config = get_algo_config("gptq", "unsupported_model")
    assert gptq_config is None

    # Test SmoothQuant with unsupported model
    sq_config = get_algo_config("smoothquant", "unsupported_model")
    assert sq_config is None

    # Test AutoSmoothQuant with unsupported model
    autosq_config = get_algo_config("autosmoothquant", "unsupported_model")
    assert autosq_config is None

    # Test Rotation with unsupported model
    rotation_config = get_algo_config("rotation", "unsupported_model")
    assert rotation_config is None


def test_get_algo_config_invalid_type():
    """Test getting algorithm configs with invalid algorithm type"""
    with pytest.raises(ValueError, match="Unsupported algorithm type"):
        get_algo_config("invalid_algo", "llama")


def test_get_supported_algorithm_types():
    """Test listing supported algorithm type names."""
    supported_algorithm_types = get_supported_algorithm_types()

    assert "awq" in supported_algorithm_types
    assert "gptq" in supported_algorithm_types
    assert "gptaq" in supported_algorithm_types
    assert "qronos" in supported_algorithm_types
    assert "smoothquant" in supported_algorithm_types
    assert "autosmoothquant" in supported_algorithm_types
    assert "rotation" in supported_algorithm_types


def test_config_structure_validation():
    """Test that configurations have expected structure"""
    # Test AWQ config structure
    awq_config = AWQ_MAP["llama"]
    assert isinstance(awq_config.scaling_layers, list)
    assert isinstance(awq_config.model_decoder_layers, str)
    for layer in awq_config.scaling_layers:
        assert isinstance(layer, dict)
        assert "layers" in layer
        assert "inp" in layer

    # Test GPTQ config structure
    gptq_config = GPTQ_MAP["llama"]
    assert isinstance(gptq_config.block_size, int)
    assert gptq_config.block_size > 0
    assert isinstance(gptq_config.inside_layer_modules, list)

    # Test SQ config structure
    sq_config = SQ_MAP["llama"]
    assert isinstance(sq_config.alpha, int | float)
    assert sq_config.alpha > 0
    assert isinstance(sq_config.scale_clamp_min, float)
    assert sq_config.scale_clamp_min > 0

    # Test AutoSmoothQuant config structure
    autosq_config = AUTOSMOOTHQUANT_MAP["llama"]
    assert isinstance(autosq_config.scaling_layers, list)
    assert isinstance(autosq_config.model_decoder_layers, str)
    assert isinstance(autosq_config.compute_scale_loss, str)
    assert autosq_config.compute_scale_loss == "MAE"

    # Test Rotation config structure
    rotation_config = ROTATION_MAP["llama"]
    assert isinstance(rotation_config.model_decoder_layers, str)
    assert isinstance(rotation_config.v_proj, str)
    assert isinstance(rotation_config.o_proj, str)
    assert isinstance(rotation_config.self_attn, str)
    assert isinstance(rotation_config.mlp, str)
    assert isinstance(rotation_config.r1, bool)
    assert isinstance(rotation_config.r2, bool)
    assert isinstance(rotation_config.r3, bool)
    assert isinstance(rotation_config.r4, bool)
    assert hasattr(rotation_config, "scaling_layers")
    assert isinstance(rotation_config.scaling_layers, dict)


def test_error_message():
    """Test that error messages contain expected specific text"""
    with pytest.raises(ValueError) as exc_info:
        get_algo_config("invalid", "llama")
    assert "Unsupported algorithm type: invalid" in str(exc_info.value)
    assert "Supported types: awq, gptq, gptaq, qronos, smoothquant, autosmoothquant, rotation" in str(exc_info.value)


def test_config_consistency_across_maps():
    """Test consistency of configuration properties across maps"""
    # Test that all AWQ configs have required properties
    for model_type, config in AWQ_MAP.items():
        assert hasattr(config, "scaling_layers"), f"AWQ {model_type} missing scaling_layers"
        assert hasattr(config, "model_decoder_layers"), f"AWQ {model_type} missing model_decoder_layers"
        assert isinstance(config.scaling_layers, list), f"AWQ {model_type} scaling_layers not list"
        assert isinstance(config.model_decoder_layers, str), f"AWQ {model_type} model_decoder_layers not str"

    # Test that all GPTQ configs have required properties
    for model_type, config in GPTQ_MAP.items():
        assert hasattr(config, "inside_layer_modules"), f"GPTQ {model_type} missing inside_layer_modules"
        assert hasattr(config, "model_decoder_layers"), f"GPTQ {model_type} missing model_decoder_layers"
        assert isinstance(config.inside_layer_modules, list), f"GPTQ {model_type} inside_layer_modules not list"

    # Test that all SQ configs have required properties
    for model_type, config in SQ_MAP.items():
        assert hasattr(config, "scaling_layers"), f"SQ {model_type} missing scaling_layers"
        assert hasattr(config, "model_decoder_layers"), f"SQ {model_type} missing model_decoder_layers"
        assert hasattr(config, "alpha"), f"SQ {model_type} missing alpha"
        assert hasattr(config, "scale_clamp_min"), f"SQ {model_type} missing scale_clamp_min"

    # Test that all AutoSmoothQuant configs have required properties
    for model_type, config in AUTOSMOOTHQUANT_MAP.items():
        assert hasattr(config, "scaling_layers"), f"AutoSQ {model_type} missing scaling_layers"
        assert hasattr(config, "model_decoder_layers"), f"AutoSQ {model_type} missing model_decoder_layers"
        assert hasattr(config, "compute_scale_loss"), f"AutoSQ {model_type} missing compute_scale_loss"
        assert isinstance(config.scaling_layers, list), f"AutoSQ {model_type} scaling_layers not list"
        assert isinstance(config.model_decoder_layers, str), f"AutoSQ {model_type} model_decoder_layers not str"
        assert isinstance(config.compute_scale_loss, str), f"AutoSQ {model_type} compute_scale_loss not str"

    # Test that all Rotation configs have required properties
    for model_type, config in ROTATION_MAP.items():
        assert hasattr(config, "model_decoder_layers"), f"Rotation {model_type} missing model_decoder_layers"
        assert hasattr(config, "v_proj"), f"Rotation {model_type} missing v_proj"
        assert hasattr(config, "o_proj"), f"Rotation {model_type} missing o_proj"
        assert hasattr(config, "self_attn"), f"Rotation {model_type} missing self_attn"
        assert hasattr(config, "mlp"), f"Rotation {model_type} missing mlp"
        assert hasattr(config, "r1"), f"Rotation {model_type} missing r1"
        assert hasattr(config, "r2"), f"Rotation {model_type} missing r2"
        assert hasattr(config, "r3"), f"Rotation {model_type} missing r3"
        assert hasattr(config, "r4"), f"Rotation {model_type} missing r4"
        assert hasattr(config, "scaling_layers"), f"Rotation {model_type} missing scaling_layers"
        assert isinstance(config.model_decoder_layers, str), f"Rotation {model_type} model_decoder_layers not str"
        assert isinstance(config.v_proj, str), f"Rotation {model_type} v_proj not str"
        assert isinstance(config.o_proj, str), f"Rotation {model_type} o_proj not str"
        assert isinstance(config.self_attn, str), f"Rotation {model_type} self_attn not str"
        assert isinstance(config.mlp, str), f"Rotation {model_type} mlp not str"
        assert isinstance(config.r1, bool), f"Rotation {model_type} r1 not bool"
        assert isinstance(config.r2, bool), f"Rotation {model_type} r2 not bool"
        assert isinstance(config.r3, bool), f"Rotation {model_type} r3 not bool"
        assert isinstance(config.r4, bool), f"Rotation {model_type} r4 not bool"
        assert isinstance(config.scaling_layers, dict), f"Rotation {model_type} scaling_layers not dict"


# ---------------------------------------------------------------------------
# qwen3_5_moe — GPTQ
# ---------------------------------------------------------------------------

QWEN3_5_MOE_MODEL_TYPE = "qwen3_5_moe"

QWEN3_5_LINEAR_ATTENTION_MODULES = [
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.out_proj",
]

QWEN3_5_MOE_EXPERT_MODULES = [
    "mlp.experts.*.up_proj",
    "mlp.experts.*.gate_proj",
    "mlp.experts.*.down_proj",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
]

QWEN3_5_MOE_ROUTER_MODULES = ["mlp.gate", "mlp.shared_expert_gate"]


def test_qwen3_5_moe_present_in_gptq_map():
    """Verify qwen3_5_moe is registered in GPTQ with the right config type."""
    assert isinstance(GPTQ_MAP[QWEN3_5_MOE_MODEL_TYPE], GPTQConfig)


def test_qwen3_5_moe_via_get_algo_config():
    """Verify get_algo_config resolves qwen3_5_moe for GPTQ."""
    config = get_algo_config("gptq", QWEN3_5_MOE_MODEL_TYPE)
    assert config is not None, "gptq has no qwen3_5_moe config"
    assert isinstance(config, GPTQConfig)


def test_qwen3_5_moe_model_decoder_layers_path():
    """
    Verify the qwen3_5_moe GPTQ config points at the actual decoder layer path.

    Qwen3_5MoeForConditionalGeneration wraps its decoder stack at
    model.language_model.layers, not model.layers.
    """
    assert GPTQ_MAP[QWEN3_5_MOE_MODEL_TYPE].model_decoder_layers == "model.language_model.layers"


def test_qwen3_5_moe_gptq_covers_hybrid_attention_and_experts():
    """
    Verify the GPTQ qwen3_5_moe config covers both attention variants and the MoE block.

    Qwen3.5 MoE layers alternate between standard attention (self_attn.*) and gated-delta
    linear attention (linear_attn.*), and every layer carries a sparse MoE block with routed
    experts plus a shared expert.
    """
    inside_layer_modules = GPTQ_MAP[QWEN3_5_MOE_MODEL_TYPE].inside_layer_modules
    assert "self_attn.q_proj" in inside_layer_modules
    assert "self_attn.o_proj" in inside_layer_modules
    for linear_attention_module in QWEN3_5_LINEAR_ATTENTION_MODULES:
        assert linear_attention_module in inside_layer_modules
    for expert_module in QWEN3_5_MOE_EXPERT_MODULES:
        assert expert_module in inside_layer_modules


def test_qwen3_5_moe_gptq_excludes_router_projections():
    """
    Verify the router projections stay out of the GPTQ qwen3_5_moe config.

    `mlp.gate` and `mlp.shared_expert_gate` are tiny [num_experts, hidden] projections whose
    outputs choose experts; quantizing them changes routing decisions while saving nothing.
    """
    inside_layer_modules = GPTQ_MAP[QWEN3_5_MOE_MODEL_TYPE].inside_layer_modules
    for router_module in QWEN3_5_MOE_ROUTER_MODULES:
        assert router_module not in inside_layer_modules


# qwen3_5_moe — AWQ and SmoothQuant
def test_qwen3_5_moe_present_in_awq_and_sq_maps():
    """Verify qwen3_5_moe is registered in AWQ and SmoothQuant with the right config types."""
    assert isinstance(AWQ_MAP[QWEN3_5_MOE_MODEL_TYPE], AWQConfig)
    assert isinstance(SQ_MAP[QWEN3_5_MOE_MODEL_TYPE], SmoothQuantConfig)


def test_qwen3_5_moe_awq_and_sq_via_get_algo_config():
    """Verify get_algo_config resolves qwen3_5_moe for AWQ and SmoothQuant."""
    for algorithm_type, expected_type in [("awq", AWQConfig), ("smoothquant", SmoothQuantConfig)]:
        config = get_algo_config(algorithm_type, QWEN3_5_MOE_MODEL_TYPE)
        assert config is not None, f"{algorithm_type} has no qwen3_5_moe config"
        assert isinstance(config, expected_type)


def test_qwen3_5_moe_awq_and_sq_model_decoder_layers_path():
    """Verify AWQ and SmoothQuant qwen3_5_moe configs use the correct decoder layer path."""
    for config in [AWQ_MAP[QWEN3_5_MOE_MODEL_TYPE], SQ_MAP[QWEN3_5_MOE_MODEL_TYPE]]:
        assert config.model_decoder_layers == "model.language_model.layers"


def test_qwen3_5_moe_smoothquant_alpha():
    """Verify the qwen3_5_moe SmoothQuant config uses alpha=0.5, same as the dense qwen3_5."""
    assert SQ_MAP[QWEN3_5_MOE_MODEL_TYPE].alpha == 0.5


# qwen3_5_moe — Qronos
def test_qwen3_5_moe_present_in_qronos_map():
    """Verify qwen3_5_moe is registered in Qronos with the right config type."""
    assert isinstance(QRONOS_MAP[QWEN3_5_MOE_MODEL_TYPE], QronosConfig)


def test_qwen3_5_moe_qronos_via_get_algo_config():
    """Verify get_algo_config resolves qwen3_5_moe for Qronos."""
    config = get_algo_config("qronos", QWEN3_5_MOE_MODEL_TYPE)
    assert config is not None
    assert isinstance(config, QronosConfig)


def test_qwen3_5_moe_qronos_model_decoder_layers_path():
    """Verify the Qronos qwen3_5_moe config uses the correct decoder layer path."""
    assert QRONOS_MAP[QWEN3_5_MOE_MODEL_TYPE].model_decoder_layers == "model.language_model.layers"


def test_qwen3_5_moe_qronos_covers_hybrid_attention_and_experts():
    """Verify the Qronos config covers self_attn, linear_attn and MoE expert modules."""
    inside = QRONOS_MAP[QWEN3_5_MOE_MODEL_TYPE].inside_layer_modules
    assert "self_attn.q_proj" in inside
    for m in QWEN3_5_LINEAR_ATTENTION_MODULES:
        assert m in inside
    for m in QWEN3_5_MOE_EXPERT_MODULES:
        assert m in inside


def test_qwen3_5_moe_qronos_excludes_router_projections():
    """Verify the router projections stay out of the Qronos qwen3_5_moe config."""
    inside = QRONOS_MAP[QWEN3_5_MOE_MODEL_TYPE].inside_layer_modules
    for m in QWEN3_5_MOE_ROUTER_MODULES:
        assert m not in inside


@pytest.mark.parametrize("num_experts", [2, 32, 512])
def test_qwen4_exp_scaling_layers_reference_real_modules(num_experts):
    """qwen4_exp smoothing must resolve to every expert, whatever num_experts is.

    Regression test: the original entries used `post_attention_layernorm` (this architecture has
    no layernorm in the decoder layer at all) and unindexed `mlp.experts.gate_up_proj` /
    `mlp.experts.down_proj`. After MoE preprocessing the hooked names are indexed
    (`mlp.experts.<i>.down_proj`), so `fnmatch` matched nothing and every scaling group was
    silently dropped -- AWQ/AutoSmoothQuant performed no smoothing at all.

    Asserted against a real module tree rather than by pattern-matching the config strings: the
    shipped bug was strings that looked entirely reasonable and resolved to nothing.
    """
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.ModuleList()
    for _ in range(num_experts):
        expert = nn.Module()
        expert.gate_proj = nn.Linear(2, 2)
        expert.up_proj = nn.Linear(2, 2)
        expert.down_proj = nn.Linear(2, 2)
        layer.mlp.experts.append(expert)
    names = [f"mlp.experts.{i}.{p}" for i in range(num_experts) for p in ("gate_proj", "up_proj", "down_proj")]

    for map_name, algo_map in (("AWQ_MAP", AWQ_MAP), ("AUTOSMOOTHQUANT_MAP", AUTOSMOOTHQUANT_MAP)):
        for model_type in ("qwen4_exp", "qwen4_exp_text"):
            groups = algo_map[model_type].scaling_layers
            assert groups, f"{map_name}[{model_type}] has no scaling layers"
            for entry in groups:
                for path in (entry["prev_op"], entry["inp"], *entry["layers"]):
                    assert "layernorm" not in path, (
                        f"{map_name}[{model_type}] references {path!r}; the qwen4_exp decoder layer "
                        "has no layernorm (its predecessor is a gated-residual hyper-connection)"
                    )
            # Through the production path, with every candidate hooked, so the only thing that can
            # drop a group is the pattern itself.
            input_feat = {name: torch.zeros(1, 2) for name in names}
            resolved = get_layers_for_scaling(layer, input_feat, {}, groups)
            targets = sum(len(entry["layers"]) for entry in resolved)
            assert targets == num_experts, (
                f"{map_name}[{model_type}] resolved {targets} target(s) against {num_experts} "
                f"experts; every expert's down_proj must be smoothed"
            )
            # One unindexed MoE group expands to one entry per expert, so the count above is the
            # real check; this just pins that no entry came back empty.
            assert resolved and all(entry["layers"] for entry in resolved), (
                f"{map_name}[{model_type}] has scaling group(s) that resolve to nothing"
            )


def test_qwen4_exp_inside_layer_modules_cover_routed_and_shared_experts():
    """GPTQ-family selection must reach every routed expert and the shared expert.

    Regression test: a path-like `mlp.experts.down_proj` never matches
    `mlp.experts.<i>.down_proj` under `fnmatch`, so the Hessian processors selected zero layers
    and GPTQ/GPTAQ/QRONOS silently degenerated to plain weight quantization. Leaf suffixes avoid
    that and cover the shared expert with the same entry, since selection matches
    `"*" + pattern` against the full module name.
    """
    import fnmatch

    real = [f"mlp.experts.{i}.{proj}" for i in range(4) for proj in ("gate_proj", "up_proj", "down_proj")]
    real += [f"mlp.shared_expert.{proj}" for proj in ("gate_proj", "up_proj", "down_proj")]
    for map_name, algo_map in (("GPTQ_MAP", GPTQ_MAP), ("GPTAQ_MAP", GPTAQ_MAP), ("QRONOS_MAP", QRONOS_MAP)):
        for model_type in ("qwen4_exp", "qwen4_exp_text"):
            selected = set()
            for pattern in algo_map[model_type].inside_layer_modules:
                selected |= set(fnmatch.filter(real, "*" + pattern))
            assert selected == set(real), (
                f"{map_name}[{model_type}] selects {len(selected)} of {len(real)} expert layers; "
                "missing entries are silently skipped"
            )
            # The router-like gates pick experts rather than carry activations, and must not match.
            gates = ["mlp.gate", "mlp.shared_expert_gate"]
            for pattern in algo_map[model_type].inside_layer_modules:
                assert not fnmatch.filter(gates, "*" + pattern), f"{pattern!r} matches a router gate"
