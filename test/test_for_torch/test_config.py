#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import AlgoConfig, QConfig, QLayerConfig, QTensorConfig, SVDQuantConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, QuantFlow, RoundType, ScaleType
from quark.torch.quantization.model_transformation import setup_config_per_layer
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
from quark.torch.utils.llm.config import get_quantization_config


def test_reload_config():
    quantization_spec = QTensorConfig(
        dtype=Dtype.int8,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
        qscheme=QSchemeType.per_tensor,
        ch_axis=None,
        group_size=None,
        symmetric=False,
        round_method=RoundType.round,
        scale_type=ScaleType.float,
    )
    quantization_config = QLayerConfig(weight=quantization_spec)
    config = QConfig(global_quant_config=quantization_config, layer_type_quant_config={nn.Linear: quantization_config})

    config_dict = config.to_dict()

    config_reloaded = QConfig.from_dict(config_dict)

    assert config == config_reloaded

    quantization_spec = QTensorConfig(
        dtype=Dtype.int8,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
        qscheme=QSchemeType.per_tensor,
        ch_axis=None,
        group_size=None,
        symmetric=False,
        round_method=RoundType.round,
        scale_type=ScaleType.float,
    )
    quantization_config = QLayerConfig(weight=quantization_spec)
    config = QConfig(
        global_quant_config=quantization_config, layer_type_quant_config={nn.LayerNorm: quantization_config}
    )

    config_dict = config.to_dict()

    with pytest.raises(Exception) as e_info:
        _ = QConfig.from_dict(config_dict)
    assert "from a dictionary using custom `layer_type_quantization_config`" in str(e_info.value)


def test_svdquant_config_instantiation():
    cfg = SVDQuantConfig()
    assert cfg.name == "svdquant"
    assert cfg.svd_rank == 32
    assert isinstance(cfg, AlgoConfig)


def test_svdquant_config_from_dict():
    d = {"name": "svdquant", "svd_rank": 16, "smooth_alpha": 0.3}
    cfg = SVDQuantConfig.from_dict(d)
    assert cfg.svd_rank == 16
    assert cfg.smooth_alpha == 0.3


def test_qconfig_exclude_patterns_are_used_as_provided():
    quantization_spec = QTensorConfig(
        dtype=Dtype.int8,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
        qscheme=QSchemeType.per_tensor,
        ch_axis=None,
        group_size=None,
        symmetric=False,
        round_method=RoundType.round,
        scale_type=ScaleType.float,
    )
    quantization_config = QLayerConfig(weight=quantization_spec)

    config = QConfig(global_quant_config=quantization_config, exclude=["*.mlp.gate_proj.*"])
    assert config.exclude == ["*.mlp.gate_proj.*"]

    config = QConfig(global_quant_config=quantization_config, exclude=["*.mlp.gate_proj"])
    assert config.exclude == ["*.mlp.gate_proj"]

    config = QConfig(global_quant_config=quantization_config, exclude=["*mlp.gate"])
    assert config.exclude == ["*mlp.gate"]

    config = QConfig(global_quant_config=quantization_config, exclude=["lm_head", "*mlp.gate", "*.self_attn.*"])
    assert config.exclude == ["lm_head", "*mlp.gate", "*.self_attn.*"]


def test_exclude_pattern_of_parent_module_does_not_cover_quantizable_children():
    """A parent-module pattern excludes only the parent, never its descendants.

    ``exclude`` patterns are matched verbatim with ``fnmatch`` against the full
    module name, so ``"*.mlp.gate"`` does not reach ``model.layers.0.mlp.gate.wg``.
    Excluding a whole subtree requires spelling the descendant pattern out.
    """
    quantization_spec = QTensorConfig(
        dtype=Dtype.int8,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
        qscheme=QSchemeType.per_tensor,
        ch_axis=None,
        group_size=None,
        symmetric=False,
        round_method=RoundType.round,
        scale_type=ScaleType.float,
    )
    quantization_config = QLayerConfig(weight=quantization_spec)
    named_modules = {
        "model.layers.0.mlp.gate": nn.Linear(8, 8, bias=False),
        "model.layers.0.mlp.gate.wg": nn.Linear(8, 8, bias=False),
        "model.layers.0.mlp.down_proj": nn.Linear(8, 8, bias=False),
    }

    config = QConfig(global_quant_config=quantization_config, exclude=["*.mlp.gate"])
    module_configs: dict[str, QLayerConfig] = {}
    setup_config_per_layer(config, named_modules, module_configs)

    assert config.exclude == ["model.layers.0.mlp.gate"]
    assert "model.layers.0.mlp.gate" not in module_configs
    assert "model.layers.0.mlp.gate.wg" in module_configs
    assert "model.layers.0.mlp.down_proj" in module_configs

    # Opting the subtree in requires the descendant pattern.
    config = QConfig(global_quant_config=quantization_config, exclude=["*.mlp.gate", "*.mlp.gate.*"])
    module_configs = {}
    setup_config_per_layer(config, named_modules, module_configs)

    assert sorted(config.exclude) == ["model.layers.0.mlp.gate", "model.layers.0.mlp.gate.wg"]
    assert module_configs.keys() == {"model.layers.0.mlp.down_proj"}


def test_qconfig_quant_flow_defaults_to_standard_and_is_never_serialized():
    quantization_spec = QTensorConfig(
        dtype=Dtype.int8,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
        qscheme=QSchemeType.per_tensor,
        ch_axis=None,
        group_size=None,
        symmetric=False,
        round_method=RoundType.round,
        scale_type=ScaleType.float,
    )
    quantization_config = QLayerConfig(weight=quantization_spec)

    config = QConfig(global_quant_config=quantization_config)
    assert config.quant_flow is QuantFlow.standard
    assert config.gpu_resident_blocks == 0

    config.quant_flow = QuantFlow.per_block
    config.gpu_resident_blocks = 4

    # Runtime-only fields: every caller of `to_dict()` writes the result to disk, so these
    # must not reach the serialized form.
    config_dict = config.to_dict()
    assert "quant_flow" not in config_dict
    assert "gpu_resident_blocks" not in config_dict

    reloaded = QConfig.from_dict(config_dict)
    assert reloaded.quant_flow is QuantFlow.standard
    assert reloaded.gpu_resident_blocks == 0


def test_qconfig_from_dict_ignores_stray_quant_flow_keys():
    quantization_spec = QTensorConfig(
        dtype=Dtype.int8,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
        qscheme=QSchemeType.per_tensor,
        ch_axis=None,
        group_size=None,
        symmetric=False,
        round_method=RoundType.round,
        scale_type=ScaleType.float,
    )
    quantization_config = QLayerConfig(weight=quantization_spec)
    config_dict = QConfig(global_quant_config=quantization_config).to_dict()

    # A config file that does carry these fields must still load, with them ignored rather
    # than letting a stale execution strategy come back from disk.
    config_dict["quant_flow"] = "per_block"
    config_dict["gpu_resident_blocks"] = 4

    reloaded = QConfig.from_dict(config_dict)
    assert reloaded.quant_flow is QuantFlow.standard
    assert reloaded.gpu_resident_blocks == 0


def _make_qconfig(**kwargs):
    quantization_spec = QTensorConfig(
        dtype=Dtype.int8,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
        qscheme=QSchemeType.per_tensor,
        ch_axis=None,
        group_size=None,
        symmetric=False,
        round_method=RoundType.round,
        scale_type=ScaleType.float,
    )
    quantization_config = QLayerConfig(weight=quantization_spec)
    return QConfig(global_quant_config=quantization_config, **kwargs)


def test_quantize_model_raises_when_quant_flow_is_file2file():
    config = _make_qconfig(quant_flow=QuantFlow.file2file)
    quantizer = ModelQuantizer(config)

    with pytest.raises(ValueError, match="direct_quantize_checkpoint"):
        quantizer.quantize_model(nn.Linear(4, 4))


def test_quantize_model_raises_when_quant_flow_is_per_block_with_weight_only_config():
    # `_make_qconfig()`'s default global_quant_config only sets `weight`, leaving
    # input_tensors/output_tensors as None, which `ConfigVerifier` classifies as
    # weight-only -- a combination the per-block lazy loader doesn't support.
    config = _make_qconfig(quant_flow=QuantFlow.per_block)
    quantizer = ModelQuantizer(config)
    assert quantizer.config_verifier.is_weight_only

    with pytest.raises(ValueError, match="per_block"):
        quantizer.quantize_model(nn.Linear(4, 4))


def test_direct_quantize_checkpoint_stamps_quant_flow(monkeypatch, tmp_path):
    config = _make_qconfig()
    assert config.quant_flow is QuantFlow.standard

    monkeypatch.setattr(
        "quark.torch.quantization.api.quantize_model_per_safetensor",
        lambda **kwargs: None,
    )

    quantizer = ModelQuantizer(config)
    quantizer.direct_quantize_checkpoint(pretrained_model_path=str(tmp_path), save_path=str(tmp_path))

    assert config.quant_flow is QuantFlow.file2file


def test_get_quantization_config():
    """Test get_quantization_config with various config structures."""
    quantization_config_data = {"quant_method": "awq", "bits": 4}

    # Test 1: Config dict with quantization_config at top level (line 28)
    config_dict = {"model_type": "llama", "quantization_config": quantization_config_data}
    result = get_quantization_config(config_dict)
    assert result == quantization_config_data

    # Test 2: Config dict with quantization_config in a sub-config (lines 33, 42)
    config_with_subconfig = {"model_type": "llama", "text_config": {"quantization_config": quantization_config_data}}
    result = get_quantization_config(config_with_subconfig)
    assert result == quantization_config_data

    # Test 3: Config dict with no quantization_config (returns None)
    config_no_quant = {"model_type": "llama", "hidden_size": 4096}
    result = get_quantization_config(config_no_quant)
    assert result is None

    # Test 4: Config dict with multiple quantization_configs in different sub-configs (line 37-40)
    config_multiple = {
        "text_config": {"quantization_config": quantization_config_data},
        "vision_config": {"quantization_config": {"quant_method": "gptq", "bits": 8}},
    }
    with pytest.raises(NotImplementedError, match="Found multiple quantization_config entries"):
        get_quantization_config(config_multiple)

    # Test 5: PretrainedConfig object (mock) - tests line 25
    mock_pretrained_config = MagicMock()
    mock_pretrained_config.to_dict.return_value = config_dict
    result = get_quantization_config(mock_pretrained_config)
    assert result == quantization_config_data
    mock_pretrained_config.to_dict.assert_called_once()
