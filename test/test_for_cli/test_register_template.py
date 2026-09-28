#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json

import pytest

from quark.experimental.cli import register_template
from quark.experimental.torch.twobitscalar.config import TwoBitScalarConfig
from quark.torch import LLMTemplate
from quark.torch.quantization.config.config import (
    AutoSmoothQuantConfig,
    AWQConfig,
    GPTAQConfig,
    GPTQConfig,
    QronosConfig,
    RotationConfig,
    SmoothQuantConfig,
)

UNIQUE_MODEL_TYPE = "quark_cli_test_custom_model"


@pytest.fixture(autouse=True)
def _cleanup_template():
    yield
    LLMTemplate._templates.pop(UNIQUE_MODEL_TYPE, None)


def _write_template_file(tmp_path, data=None):
    if data is None:
        data = {
            "model_type": UNIQUE_MODEL_TYPE,
            "kv_layers_name": ["*k_proj", "*v_proj"],
            "q_layer_name": "*q_proj",
            "exclude_layers_name": ["lm_head"],
        }
    path = tmp_path / "template.json"
    path.write_text(json.dumps(data))
    return str(path)


# A single JSON string covering every template field and a configuration for every
# supported quantization algorithm (awq, gptq, gptaq, qronos, smoothquant,
# autosmoothquant, rotation, twobitscalar).
COMPREHENSIVE_TEMPLATE_JSON = """
{
    "model_type": "quark_cli_test_custom_model",
    "kv_layers_name": ["*k_proj", "*v_proj"],
    "q_layer_name": "*q_proj",
    "gate_up_layers_name": ["gate_proj", "up_proj"],
    "exclude_layers_name": ["lm_head"],
    "algorithm_configs": {
        "awq": {
            "name": "awq",
            "model_decoder_layers": "model.layers",
            "scaling_layers": [
                {
                    "prev_op": "input_layernorm",
                    "layers": ["self_attn.q_proj"],
                    "inp": "self_attn.q_proj",
                    "module2inspect": "self_attn"
                }
            ]
        },
        "gptq": {
            "name": "gptq",
            "block_size": 64,
            "damp_percent": 0.1
        },
        "gptaq": {
            "name": "gptaq",
            "block_size": 32,
            "alpha": 0.5
        },
        "qronos": {
            "name": "qronos",
            "inside_layer_modules": ["mlp.gate_proj"],
            "model_decoder_layers": "model.layers",
            "block_size": 128
        },
        "smoothquant": {
            "name": "smooth",
            "alpha": 0.5,
            "scale_clamp_min": 0.01
        },
        "autosmoothquant": {
            "name": "autosmoothquant",
            "compute_scale_loss": "MSE"
        },
        "rotation": {
            "name": "rotation",
            "scaling_layers": {
                "self_attn.q_proj": [
                    {
                        "prev_op": "input_layernorm",
                        "layers": ["self_attn.q_proj"],
                        "inp": "self_attn.q_proj",
                        "module2inspect": "self_attn"
                    }
                ]
            }
        },
        "twobitscalar": {
            "name": "twobitscalar",
            "group_size": 32
        }
    }
}
"""


def test_load_template_from_json_file_registers(tmp_path):
    path = _write_template_file(tmp_path)
    template = register_template.load_template_from_json_file(path)
    assert LLMTemplate.get(UNIQUE_MODEL_TYPE) is template
    assert template.kv_layers_name == ["*k_proj", "*v_proj"]
    assert template.q_layer_name == "*q_proj"
    assert template.exclude_layers_name == ["lm_head"]
    assert template.gate_up_layers_name == ["gate_proj", "up_proj"]


def test_load_template_from_json_file_minimal_config(tmp_path):
    path = _write_template_file(tmp_path, data={"model_type": UNIQUE_MODEL_TYPE})
    template = register_template.load_template_from_json_file(path)
    assert template.kv_layers_name is None
    assert template.q_layer_name is None
    assert template.exclude_layers_name == []


def test_registered_template_can_build_config(tmp_path):
    path = _write_template_file(tmp_path)
    template = register_template.load_template_from_json_file(path)
    config = template.get_config(scheme="fp8")
    assert config is not None


def test_register_template_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        register_template.load_template_from_json_file(str(tmp_path / "does_not_exist.json"))


def test_register_template_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{this is not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        register_template.load_template_from_json_file(str(path))


def test_register_template_missing_model_type_raises(tmp_path):
    path = _write_template_file(tmp_path, data={"kv_layers_name": ["*k_proj"]})
    with pytest.raises(ValueError, match="model_type"):
        register_template.load_template_from_json_file(path)


def test_register_template_unknown_key_raises(tmp_path):
    path = _write_template_file(tmp_path, data={"model_type": UNIQUE_MODEL_TYPE, "bogus_key": 1})
    with pytest.raises(ValueError, match="Unknown key"):
        register_template.load_template_from_json_file(path)


def test_create_template_from_json_dict_wrong_types():
    with pytest.raises(ValueError, match="model_type"):
        register_template.create_template_from_json_dict({"model_type": 123})

    with pytest.raises(ValueError, match="kv_layers_name"):
        register_template.create_template_from_json_dict(
            {"model_type": UNIQUE_MODEL_TYPE, "kv_layers_name": "not-a-list"}
        )

    with pytest.raises(ValueError, match="q_layer_name"):
        register_template.create_template_from_json_dict({"model_type": UNIQUE_MODEL_TYPE, "q_layer_name": ["ok", 42]})


def test_create_template_from_json_dict_rejects_non_dict():
    with pytest.raises(ValueError, match="JSON object"):
        register_template.create_template_from_json_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_algorithm_configs_are_parsed_into_template(tmp_path):
    data = {
        "model_type": UNIQUE_MODEL_TYPE,
        "algorithm_configs": {
            "awq": {"name": "awq", "model_decoder_layers": "model.layers"},
            "gptq": {"name": "gptq", "block_size": 64},
        },
    }
    path = _write_template_file(tmp_path, data=data)
    template = register_template.load_template_from_json_file(path)
    assert isinstance(template.algo_config["awq"], AWQConfig)
    assert template.algo_config["awq"].model_decoder_layers == "model.layers"
    assert isinstance(template.algo_config["gptq"], GPTQConfig)
    assert template.algo_config["gptq"].block_size == 64


def test_algorithm_configs_enable_get_config_with_algorithm(tmp_path):
    data = {
        "model_type": UNIQUE_MODEL_TYPE,
        "algorithm_configs": {"awq": {"name": "awq"}},
    }
    path = _write_template_file(tmp_path, data=data)
    template = register_template.load_template_from_json_file(path)
    config = template.get_config(scheme="fp8", algorithm="awq")
    assert config is not None
    assert len(config.algo_config) == 1
    assert isinstance(config.algo_config[0], AWQConfig)


def test_algorithm_configs_unsupported_algorithm_exits(tmp_path):
    data = {
        "model_type": UNIQUE_MODEL_TYPE,
        "algorithm_configs": {"not_an_algorithm": {"name": "awq"}},
    }
    path = _write_template_file(tmp_path, data=data)
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        register_template.load_template_from_json_file(path)


def test_algorithm_configs_missing_name_field_exits(tmp_path):
    data = {
        "model_type": UNIQUE_MODEL_TYPE,
        "algorithm_configs": {"awq": {"model_decoder_layers": "model.layers"}},
    }
    path = _write_template_file(tmp_path, data=data)
    with pytest.raises(ValueError, match="missing the required 'name' field"):
        register_template.load_template_from_json_file(path)


def test_algorithm_configs_mismatched_name_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        register_template.create_template_from_json_dict(
            {
                "model_type": UNIQUE_MODEL_TYPE,
                "algorithm_configs": {"awq": {"name": "gptq", "block_size": 64}},
            }
        )


def test_algorithm_configs_invalid_fields_exits(tmp_path):
    data = {
        "model_type": UNIQUE_MODEL_TYPE,
        "algorithm_configs": {"gptq": {"name": "gptq", "bogus_field": 1}},
    }
    path = _write_template_file(tmp_path, data=data)
    with pytest.raises(ValueError, match="Invalid configuration"):
        register_template.load_template_from_json_file(path)


def test_algorithm_configs_non_dict_value_exits(tmp_path):
    data = {
        "model_type": UNIQUE_MODEL_TYPE,
        "algorithm_configs": {"awq": "not-an-object"},
    }
    path = _write_template_file(tmp_path, data=data)
    with pytest.raises(ValueError, match="must be a JSON object"):
        register_template.load_template_from_json_file(path)


def test_comprehensive_template_json_covers_all_template_and_algorithm_fields(tmp_path):
    """One JSON string covering every template field and every supported algorithm configuration."""
    path = tmp_path / "comprehensive_template.json"
    path.write_text(COMPREHENSIVE_TEMPLATE_JSON)
    register_template.load_template_from_json_file(str(path))
    template = LLMTemplate.get(UNIQUE_MODEL_TYPE)

    assert template.model_type == UNIQUE_MODEL_TYPE
    assert template.kv_layers_name == ["*k_proj", "*v_proj"]
    assert template.q_layer_name == "*q_proj"
    assert template.gate_up_layers_name == ["gate_proj", "up_proj"]
    assert template.exclude_layers_name == ["lm_head"]

    expected_algorithm_config_classes = {
        "awq": AWQConfig,
        "gptq": GPTQConfig,
        "gptaq": GPTAQConfig,
        "qronos": QronosConfig,
        "smoothquant": SmoothQuantConfig,
        "autosmoothquant": AutoSmoothQuantConfig,
        "rotation": RotationConfig,
        "twobitscalar": TwoBitScalarConfig,
    }
    for algorithm_name, expected_class in expected_algorithm_config_classes.items():
        algorithm_config = template.algo_config[algorithm_name]
        assert isinstance(algorithm_config, expected_class), f"unexpected type for {algorithm_name}"
        quantization_config = template.get_config(scheme="fp8", algorithm=algorithm_name)
        assert quantization_config is not None
        assert len(quantization_config.algo_config) == 1
        assert isinstance(quantization_config.algo_config[0], expected_class)

    assert template.algo_config["awq"].model_decoder_layers == "model.layers"
    assert len(template.algo_config["awq"].scaling_layers) == 1
    assert template.algo_config["gptq"].block_size == 64
    assert template.algo_config["gptq"].damp_percent == 0.1
    assert template.algo_config["gptaq"].block_size == 32
    assert template.algo_config["gptaq"].alpha == 0.5
    assert template.algo_config["qronos"].block_size == 128
    assert template.algo_config["qronos"].inside_layer_modules == ["mlp.gate_proj"]
    assert template.algo_config["smoothquant"].alpha == 0.5
    assert template.algo_config["smoothquant"].scale_clamp_min == 0.01
    assert template.algo_config["autosmoothquant"].compute_scale_loss == "MSE"
    assert "self_attn.q_proj" in template.algo_config["rotation"].scaling_layers
    assert template.algo_config["twobitscalar"].group_size == 32
