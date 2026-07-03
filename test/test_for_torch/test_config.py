#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from quark.torch.quantization.config.config import AlgoConfig, QConfig, QLayerConfig, QTensorConfig, SVDQuantConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
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


def test_qconfig_exclude_pattern_with_trailing_wildcard_is_not_expanded():
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
    config.exclude.sort()
    assert config.exclude == ["*.mlp.gate_proj", "*.mlp.gate_proj.*"]


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
