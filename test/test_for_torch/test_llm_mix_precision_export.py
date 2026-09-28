#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Regression tests for mixed-precision export preparation and quantization metadata."""

from __future__ import annotations

import copy
import fnmatch
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from quark.experimental.torch.mix_precision import export_utils as export_module


def test_export_preprocesses_loaded_model(monkeypatch, tmp_path) -> None:
    import quark.experimental.torch.mix_precision.quantizer as quantizer_module
    from quark.experimental.torch.mix_precision.config import MixPrecisionConfig
    from quark.experimental.torch.mix_precision.quantizer import MixPrecisionQuantizer

    model = MagicMock()
    tokenizer = MagicMock()
    quantized_model = MagicMock()
    preprocess = MagicMock()

    monkeypatch.setattr(quantizer_module, "load_transformers_model", MagicMock(return_value=model))
    monkeypatch.setattr(
        quantizer_module.AutoTokenizer,
        "from_pretrained",
        MagicMock(return_value=tokenizer),
    )
    monkeypatch.setattr(quantizer_module, "preprocess_for_quantization", preprocess)
    apply_quant = MagicMock(return_value=quantized_model)
    manager = MagicMock()
    manager.attach_mock(preprocess, "preprocess")
    manager.attach_mock(apply_quant, "apply_quant")
    monkeypatch.setattr(quantizer_module, "apply_quant_config", apply_quant)
    monkeypatch.setattr(quantizer_module.ModelQuantizer, "freeze", MagicMock(return_value=quantized_model))
    monkeypatch.setattr(quantizer_module, "export_safetensors", MagicMock())

    quantizer = MixPrecisionQuantizer(MixPrecisionConfig())
    quantizer.model_path = "/models/example"
    quantizer.result = SimpleNamespace(best_config={"mlp_mode": "fp8"})

    quantizer.export_best(str(tmp_path / "export"))

    preprocess.assert_called_once_with(model)
    assert [c[0] for c in manager.mock_calls] == ["preprocess", "apply_quant"]


def _config():
    return {
        "quant_method": "quark",
        "global_quant_config": {"weight": {"dtype": "fp8_e4m3"}},
        "layer_quant_config": {"model.layers.*.mlp.experts.*": {"weight": {"dtype": "fp4"}}},
        "exclude": ["lm_head"],
    }


def _weights(projections=("gate_up_proj", "down_proj")):
    return {
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight": None
        for layer in range(2)
        for expert in range(2)
        for proj in projections
    }


def _matched(config, name):
    """Model vLLM's matching contract; real-vLLM tests below check it as well."""
    return next(
        (
            value
            for pattern, value in config["layer_quant_config"].items()
            if (fnmatch.fnmatch(name, pattern) if "*" in pattern else name in pattern)
        ),
        config["global_quant_config"],
    )


def _substring_rule_config(container, existing_alias):
    from quark.experimental.torch.mix_precision.config import get_layer_config

    mxfp4 = get_layer_config("mxfp4").to_dict()
    config = {
        "quant_method": "quark",
        "global_quant_config": mxfp4,
        "layer_quant_config": {
            container + ".routed_input_transform": get_layer_config("fp8").to_dict(),
            "model.layers.0.mlp.experts.*.*": mxfp4,
        },
        "layer_type_quant_config": {},
        "exclude": [],
        "export": {"kv_cache_group": [], "pack_method": "reorder"},
    }
    if existing_alias:
        config["layer_quant_config"][container] = copy.deepcopy(mxfp4)
    return config


@pytest.mark.parametrize("suffix", ["", ".routed_experts"])
@pytest.mark.parametrize("existing_alias", [False, True])
def test_container_alias_precedes_non_glob_substring_rules(suffix, existing_alias):
    container = "model.layers.0.mlp.experts" + suffix
    config = _substring_rule_config(container, existing_alias)
    original = copy.deepcopy(config)

    updated = export_module._add_vllm_quantization_aliases(config, _weights())

    assert config == original
    assert updated["global_quant_config"] == original["global_quant_config"]
    assert updated["layer_quant_config"][container] == config["global_quant_config"]
    rules = list(updated["layer_quant_config"])
    assert rules.index(container) < rules.index(container + ".routed_input_transform")
    assert "model.layers.1.mlp.experts" not in updated["layer_quant_config"]
    assert export_module._add_vllm_quantization_aliases(updated, _weights()) == updated


def test_non_glob_descendant_rule_does_not_change_hf_projection_config():
    container = "model.layers.0.mlp.experts"
    config = _substring_rule_config(container, False)
    fp8 = config["layer_quant_config"].pop(container + ".routed_input_transform")
    # vLLM matches the parent container by substring, but Quark must not use
    # a child's explicit config to quantize its parent HF projection.
    config["layer_quant_config"] = {
        container + ".0.gate_up_proj.input_transform": fp8,
        **config["layer_quant_config"],
    }

    updated = export_module._add_vllm_quantization_aliases(config, _weights())

    assert updated["layer_quant_config"][container] == config["global_quant_config"]


@pytest.mark.parametrize("suffix", ["", ".routed_experts"])
@pytest.mark.parametrize("existing_alias", [False, True])
def test_export_aliases_match_real_vllm_container_lookup(suffix, existing_alias):
    vllm = pytest.importorskip("vllm")
    # The integrations conftest provides importable stubs when vLLM is absent.
    if getattr(vllm, "_quark_test_vllm_stub", False):
        pytest.skip("requires real vLLM, not the test stub")
    runtime_module = pytest.importorskip("vllm.model_executor.layers.quantization.quark.quark")
    import torch.nn as nn
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.model_executor.models.gpt_oss import GptOssForCausalLM

    container = "model.layers.0.mlp.experts" + suffix
    config = _substring_rule_config(container, existing_alias)
    updated = export_module._add_vllm_quantization_aliases(config, _weights())
    for candidate, expected in (
        (config, config["layer_quant_config"][container + ".routed_input_transform"]),
        (updated, config["global_quant_config"]),
    ):
        runtime = runtime_module.QuarkConfig.from_config(copy.deepcopy(candidate))
        configure_quant_config(runtime, GptOssForCausalLM)
        assert runtime._find_matched_config(container, nn.Module()) == expected


def _tiny_moe_model():
    import torch.nn as nn

    model = nn.Module()
    model.self_attn = nn.Module()
    model.self_attn.q_proj = nn.Linear(32, 32)
    model.dense_mlp = nn.Module()
    model.dense_mlp.gate_proj = nn.Linear(32, 64)
    model.mlp = nn.Module()
    expert = nn.Module()
    expert.gate_up_proj = nn.Linear(32, 64)
    expert.down_proj = nn.Linear(32, 32)
    model.mlp.experts = nn.ModuleList([expert])
    model.lm_head = nn.Linear(32, 32)
    return model


def test_moe_default_uses_container_projections_not_routed_transforms():
    import torch.nn as nn

    from quark.experimental.torch.mix_precision.config import get_layer_config
    from quark.experimental.torch.mix_precision.utils import _moe_global_config_from_prefixes

    model = _tiny_moe_model()
    model.mlp.routed_input_transform = nn.Linear(32, 32)
    model.mlp.shared_experts = nn.Linear(32, 32)
    fp8, mxfp4 = get_layer_config("fp8"), get_layer_config("mxfp4")
    configs = {
        "mlp.routed_input_transform": fp8,
        "mlp.shared_experts": fp8,
        "mlp.experts.0.gate_up_proj": mxfp4,
        "mlp.experts.0.down_proj": mxfp4,
    }
    assert _moe_global_config_from_prefixes(model, configs) == mxfp4
    del model.mlp.experts
    assert _moe_global_config_from_prefixes(model, configs) is None


@pytest.mark.parametrize("down", ["fp8", "native", "activation_only"])
def test_moe_default_requires_uniform_complete_container_config(down):
    from dataclasses import replace

    from quark.experimental.torch.mix_precision.config import get_layer_config
    from quark.experimental.torch.mix_precision.utils import _moe_global_config_from_prefixes

    mxfp4 = get_layer_config("mxfp4")
    down_config = replace(mxfp4, weight=None) if down == "activation_only" else get_layer_config(down)
    configs = {"mlp.experts.0.gate_up_proj": mxfp4, "mlp.experts.0.down_proj": down_config}
    assert _moe_global_config_from_prefixes(_tiny_moe_model(), configs) is None


@pytest.mark.parametrize("attention,moe", [("fp8", "mxfp4"), ("mxfp4", "fp8")])
def test_generated_moe_default_preserves_other_partitions_without_runtime_aliases(attention, moe):
    from quark.experimental.torch.mix_precision.config import get_layer_config
    from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

    model = _tiny_moe_model()
    qconfig = create_qconfig_from_quant_config(
        model,
        {"self_attn_mode": attention, "dense_mlp_mode": "ptpc_fp8", "routed_moe_mode": moe},
        exclude_patterns=["lm_head"],
    )
    assert qconfig.global_quant_config == get_layer_config(moe)
    serialized = qconfig.to_dict()
    assert _matched(serialized, "self_attn.q_proj") == get_layer_config(attention).to_dict()
    assert _matched(serialized, "dense_mlp.gate_proj") == get_layer_config("ptpc_fp8").to_dict()
    assert "lm_head" in serialized["exclude"]
    updated = export_module._add_vllm_quantization_aliases(
        serialized, (f"{name}.weight" for name, module in model.named_modules() if hasattr(module, "weight"))
    )
    assert updated == serialized
    for suffix in ("", ".routed_experts"):
        assert "mlp.experts" + suffix not in updated["layer_quant_config"]
        assert _matched(updated, "mlp.experts" + suffix) == get_layer_config(moe).to_dict()


@pytest.mark.parametrize("case", ["absent", "native", "excluded", "source_weights_preserved"])
def test_ineligible_moe_keeps_complete_non_moe_default(case):
    from quark.experimental.torch.mix_precision.config import get_layer_config
    from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

    model = _tiny_moe_model()
    config = {"self_attn_mode": "mxfp4", "dense_mlp_mode": "fp8", "routed_moe_mode": "fp8"}
    excludes = ["lm_head"]
    source_bitwidths = None
    if case == "absent":
        del model.mlp
    elif case == "native":
        config["routed_moe_mode"] = "native"
    elif case == "excluded":
        excludes.append("mlp.experts.*")
    else:
        source_bitwidths = {"mlp.experts.0.gate_up_proj": 4, "mlp.experts.0.down_proj": 4}
    qconfig = create_qconfig_from_quant_config(
        model, config, exclude_patterns=excludes, source_weight_bitwidth_by_layer=source_bitwidths
    )
    assert qconfig.global_quant_config == get_layer_config("mxfp4")
    if case == "source_weights_preserved":
        assert _matched(qconfig.to_dict(), "mlp.experts.0.down_proj")["weight"] is None


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("live_config", [False, True])
def test_mix_precision_export_adapts_saved_metadata_only(monkeypatch, tmp_path, sharded, live_config):
    import torch
    from safetensors.torch import save_file

    config = _config()
    original = copy.deepcopy(config)
    tensors = {name: torch.zeros(2, 2) for name in _weights()}
    shard = tmp_path / "model.safetensors"
    saved_weights = []

    def export_model(model, path):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt_oss", "quantization_config": config}))
        save_file(tensors, shard)
        saved_weights.append(shard.read_bytes())
        if sharded:
            (tmp_path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(tensors, shard.name)})
            )

    monkeypatch.setattr(export_module, "export_safetensors", export_model)
    model = object()
    if live_config:
        model = SimpleNamespace(
            quant_config=SimpleNamespace(to_dict=lambda: config),
            named_modules=lambda: (
                (name.removesuffix(".weight"), SimpleNamespace(weight=t)) for name, t in tensors.items()
            ),
        )
    export_module.export_safetensors_for_vllm(model, str(tmp_path))
    saved = json.loads((tmp_path / "config.json").read_text())["quantization_config"]
    for layer in range(2):
        for suffix in ("", ".routed_experts"):
            assert _matched(saved, f"model.layers.{layer}.mlp.experts{suffix}")["weight"]["dtype"] == "fp4"
    assert _matched(saved, "model.layers.0.attn.q_proj")["weight"]["dtype"] == "fp8_e4m3"
    assert config == original
    assert shard.read_bytes() == saved_weights[0]


def test_exact_projection_overrides_earlier_wildcard():
    config = _config()
    config["layer_quant_config"] = {
        "model.layers.*.mlp.experts.*": {"weight": {"dtype": "fp8_e4m3"}},
        **{name.removesuffix(".weight"): {"weight": {"dtype": "fp4"}} for name in _weights()},
    }
    updated = export_module._add_vllm_quantization_aliases(config, _weights())
    assert _matched(updated, "model.layers.0.mlp.experts")["weight"]["dtype"] == "fp4"


def test_fused_aliases_preserve_layer_overrides_and_native_exclusions():
    config = _config()
    config["exclude"].append("model.layers.1.mlp.experts.*")
    updated = export_module._add_vllm_quantization_aliases(config, _weights())
    assert _matched(updated, "model.layers.0.mlp.experts")["weight"]["dtype"] == "fp4"
    assert "model.layers.1.mlp.experts" in updated["exclude"]
    config["exclude"] = []
    config["layer_quant_config"] = {
        "model.layers.1.mlp.experts.*": {"weight": {"dtype": "fp8_e4m3"}},
        **config["layer_quant_config"],
    }
    updated = export_module._add_vllm_quantization_aliases(config, _weights())
    assert _matched(updated, "model.layers.1.mlp.experts")["weight"]["dtype"] == "fp8_e4m3"
    assert _matched(updated, "model.layers.0.mlp.experts")["weight"]["dtype"] == "fp4"


@pytest.mark.parametrize("conflict", ["expert_weight", "projection_weight", "activation", "excluded"])
def test_nonuniform_expert_configs_are_rejected_without_mutating_input(conflict):
    config = _config()
    projection = "model.layers.0.mlp.experts.0.gate_up_proj"
    if conflict == "excluded":
        config["exclude"].append(projection)
    else:
        override = {"weight": {"dtype": "fp8_e4m3"}}
        pattern = "model.layers.0.mlp.experts.1.*" if conflict == "expert_weight" else projection
        if conflict == "activation":
            override = {"weight": {"dtype": "fp4"}, "input_tensors": {"dtype": "fp8_e4m3"}}
        config["layer_quant_config"] = {
            pattern: override,
            **config["layer_quant_config"],
        }
    original = copy.deepcopy(config)
    with pytest.raises(ValueError, match="Cannot export fused MoE container") as exc:
        export_module._add_vllm_quantization_aliases(config, _weights())
    assert "model.layers.0.mlp.experts" in str(exc.value)
    assert "different quantization configurations" in str(exc.value)
    assert config == original


def test_conflicting_fused_config_is_rejected_before_writing_checkpoint(monkeypatch, tmp_path):
    import torch

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    expert = torch.nn.Module()
    expert.gate_up_proj = torch.nn.Linear(2, 2)
    expert.down_proj = torch.nn.Linear(2, 2)
    model.model.layers[0].mlp.experts = torch.nn.ModuleList([expert])
    config = _config()
    config["layer_quant_config"]["model.layers.0.mlp.experts.0.down_proj"] = {"weight": {"dtype": "fp8_e4m3"}}
    model.quant_config = SimpleNamespace(to_dict=lambda: config)
    export_model = Mock()
    monkeypatch.setattr(export_module, "export_safetensors", export_model)

    with pytest.raises(ValueError, match="Cannot export fused MoE container"):
        export_module.export_safetensors_for_vllm(model, str(tmp_path))
    export_model.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_final_export_metadata_conflicts_are_rejected(monkeypatch, tmp_path):
    config = _config()
    config["exclude"].append("model.layers.0.mlp.experts.0.down_proj")

    def export_model(model, path):
        (tmp_path / "config.json").write_text(json.dumps({"quantization_config": config}))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": dict.fromkeys(_weights(), "model.safetensors")})
        )

    monkeypatch.setattr(export_module, "export_safetensors", export_model)
    with pytest.raises(ValueError, match="Cannot export fused MoE container"):
        export_module.export_safetensors_for_vllm(object(), str(tmp_path))
    assert json.loads((tmp_path / "config.json").read_text())["quantization_config"] == config


@pytest.mark.parametrize("suffix", ["", ".routed_experts"])
@pytest.mark.parametrize("excluded", [True, False])
def test_conflicting_explicit_container_settings_are_rejected(suffix, excluded):
    config = _config()
    container = "model.layers.0.mlp.experts" + suffix
    if excluded:
        config["exclude"].append(container)
    else:
        config["layer_quant_config"][container] = {"weight": {"dtype": "fp8_e4m3"}}
    with pytest.raises(ValueError, match="excluded vLLM container|conflicts with the exported projection"):
        export_module._add_vllm_quantization_aliases(config, _weights())


def test_other_quantization_formats_are_unchanged():
    config = {"quant_method": "fp8", "ignored_layers": ["lm_head"]}
    assert export_module._add_vllm_quantization_aliases(config, _weights()) == config


def test_fused_alias_uses_linear_type_config_before_global_default():
    config = _config()
    config["layer_quant_config"] = {}
    config["layer_type_quant_config"] = {"Linear": {"weight": {"dtype": "fp4"}}}
    updated = export_module._add_vllm_quantization_aliases(config, _weights())
    assert _matched(updated, "model.layers.0.mlp.experts")["weight"]["dtype"] == "fp4"
