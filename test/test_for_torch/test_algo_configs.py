#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest

from quark.torch.quantization.config.algo_configs import (
    AUTOSMOOTHQUANT_MAP,
    AWQ_MAP,
    GPTQ_MAP,
    ROTATION_MAP,
    SQ_MAP,
    get_algo_config,
    get_supported_algorithm_types,
)
from quark.torch.quantization.config.config import (
    AutoSmoothQuantConfig,
    AWQConfig,
    GPTQConfig,
    RotationConfig,
    SmoothQuantConfig,
)


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
