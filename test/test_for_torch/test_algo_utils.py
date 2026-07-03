#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import tempfile

from quark.torch.quantization.config.config import (
    RotationConfig,
    _load_pre_optimization_config_from_dict,
    _load_quant_algo_config_from_dict,
    _migrate_deprecated_rotation_fields,
    load_pre_optimization_config_from_file,
    load_quant_algo_config_from_file,
)


def create_temp_file(content: str):
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as temp_file:
        temp_file.write(content)
    return temp_file.name


def test_load_pre_optimization_config_from_file():
    file_path = create_temp_file(
        '{"name":"smooth", "scaling_layers":[{"prev_op": "self_attn_layer_norm"}], "model_decoder_layers": "model.decoder.layers"}'
    )
    try:
        loaded_config = load_pre_optimization_config_from_file(file_path)
        assert loaded_config.model_decoder_layers == "model.decoder.layers"
    finally:
        os.unlink(file_path)


def test_load_quant_algo_config_from_file():
    file_path = create_temp_file(
        '{"name":"awq", "scaling_layers":[{"prev_op": "self_attn_layer_norm"}], "model_decoder_layers": "model.decoder.layers"}'
    )
    try:
        loaded_config = load_quant_algo_config_from_file(file_path)
        assert loaded_config.model_decoder_layers == "model.decoder.layers"
    finally:
        os.unlink(file_path)


def test_load_pre_optimization_config_from_dict():
    config_dict = {
        "name": "smooth",
        "scaling_layers": [{"prev_op": "self_attn_layer_norm"}],
        "model_decoder_layers": "model.decoder.layers",
    }
    loaded_config = _load_pre_optimization_config_from_dict(config_dict)
    assert loaded_config.model_decoder_layers == "model.decoder.layers"


def test_load_quant_algo_config_from_dict():
    config_dict = {
        "name": "awq",
        "scaling_layers": [{"prev_op": "self_attn_layer_norm"}],
        "model_decoder_layers": "model.decoder.layers",
    }
    loaded_config = _load_quant_algo_config_from_dict(config_dict)
    assert loaded_config.model_decoder_layers == "model.decoder.layers"


def test_load_rotation_config_from_dict():
    config_dict = {
        "name": "rotation",
        "scaling_layers": [{"prev_op": "self_attn_layer_norm"}],
        "model_decoder_layers": "model.decoder.layers",
    }
    loaded_config = _load_quant_algo_config_from_dict(config_dict)
    assert loaded_config.model_decoder_layers == "model.decoder.layers"


def test_load_quarot_config_from_dict():
    config_dict = {
        "name": "quarot",
        "scaling_layers": [{"prev_op": "self_attn_layer_norm"}],
        "model_decoder_layers": "model.decoder.layers",
    }
    loaded_config = _load_quant_algo_config_from_dict(config_dict)
    assert isinstance(loaded_config, RotationConfig)
    assert loaded_config.name == "rotation"
    assert loaded_config.r1 is True
    assert loaded_config.r2 is True
    assert loaded_config.r3 is True
    assert loaded_config.r4 is True
    assert loaded_config.model_decoder_layers == "model.decoder.layers"


def test_load_quarot_config_preserves_explicit_overrides():
    """quarot migration should not overwrite explicitly set r1-r4 values."""
    config_dict = {
        "name": "quarot",
        "r1": False,
        "r3": False,
        "scaling_layers": [],
        "model_decoder_layers": "model.layers",
    }
    loaded_config = _load_quant_algo_config_from_dict(config_dict)
    assert loaded_config.name == "rotation"
    assert loaded_config.r1 is False
    assert loaded_config.r2 is True
    assert loaded_config.r3 is False
    assert loaded_config.r4 is True


def test_load_quarot_config_strips_optimized_rotation_path():
    """quarot migration should remove the obsolete optimized_rotation_path field."""
    config_dict = {
        "name": "quarot",
        "optimized_rotation_path": "/some/path",
        "scaling_layers": [],
        "model_decoder_layers": "model.layers",
    }
    loaded_config = _load_quant_algo_config_from_dict(config_dict)
    assert loaded_config.name == "rotation"
    assert (
        not hasattr(loaded_config, "optimized_rotation_path")
        or getattr(loaded_config, "optimized_rotation_path", None) is None
    )


def test_migrate_random_field():
    """Deprecated 'random' field should be split into random_r1 and random_r2."""
    config_dict = {
        "name": "rotation",
        "random": True,
        "scaling_layers": [],
        "model_decoder_layers": "model.layers",
    }
    loaded_config = _load_quant_algo_config_from_dict(config_dict)
    assert loaded_config.random_r1 is True
    assert loaded_config.random_r2 is True


def test_migrate_random_field_does_not_overwrite_explicit():
    """If random_r1/random_r2 are already set, 'random' should not overwrite them."""
    config_dict = {
        "name": "rotation",
        "random": True,
        "random_r1": False,
        "scaling_layers": [],
        "model_decoder_layers": "model.layers",
    }
    _migrate_deprecated_rotation_fields(config_dict)
    assert config_dict["random_r1"] is False
    assert config_dict["random_r2"] is True
    assert "random" not in config_dict


def test_migrate_quarot_with_random():
    """Both quarot and random migrations should compose correctly."""
    config_dict = {
        "name": "quarot",
        "random": True,
        "scaling_layers": [],
        "model_decoder_layers": "model.layers",
    }
    loaded_config = _load_quant_algo_config_from_dict(config_dict)
    assert loaded_config.name == "rotation"
    assert loaded_config.r1 is True
    assert loaded_config.r2 is True
    assert loaded_config.r3 is True
    assert loaded_config.r4 is True
    assert loaded_config.random_r1 is True
    assert loaded_config.random_r2 is True


def test_quarot_migration_via_pre_optimization_path():
    """quarot deserialization should also work via _load_pre_optimization_config_from_dict."""
    config_dict = {
        "name": "quarot",
        "scaling_layers": [{"prev_op": "self_attn_layer_norm"}],
        "model_decoder_layers": "model.decoder.layers",
    }
    loaded_config = _load_pre_optimization_config_from_dict(config_dict)
    assert isinstance(loaded_config, RotationConfig)
    assert loaded_config.name == "rotation"
    assert loaded_config.r1 is True
    assert loaded_config.r2 is True


def test_load_smoothquant_config_from_dict():
    config_dict = {
        "name": "smooth",
        "scaling_layers": [{"prev_op": "self_attn_layer_norm"}],
        "model_decoder_layers": "model.decoder.layers",
    }
    loaded_config = _load_quant_algo_config_from_dict(config_dict)
    assert loaded_config.model_decoder_layers == "model.decoder.layers"
