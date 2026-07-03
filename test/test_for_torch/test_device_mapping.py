#
# Copyright (C) 2023 - 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Tests for quark.torch.utils.llm.device_mapping module."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from quark.common.utils.testing_utils import require_accelerate, require_torch_multi_gpu
from quark.torch.utils.llm import device_mapping as device_mapping_module
from quark.torch.utils.llm.device_mapping import (
    _log_device_map_distribution,
    build_skeleton_from_config,
    check_need_rebalance,
    create_auto_adjusted_device_map,
    get_no_split_modules,
    preview_device_map,
    rebalance_device_map,
)

QWEN2_5_MODEL_ID = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def qwen2_5_configuration():
    """Load Qwen2.5 config for device mapping tests."""
    transformers_module = pytest.importorskip("transformers")
    auto_configuration_class = transformers_module.AutoConfig
    return auto_configuration_class.from_pretrained(QWEN2_5_MODEL_ID, trust_remote_code=True)


# =============================================================================
# Test Models
# =============================================================================
class SimpleModel(nn.Module):
    """A simple model without _no_split_modules."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ModelWithNoSplitModules(nn.Module):
    """A model with _no_split_modules attribute."""

    _no_split_modules = ["TransformerBlock", "AttentionBlock"]

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DecoderLayer(nn.Module):
    """A mock decoder layer."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ModelWithLayers(nn.Module):
    """A model with model.layers.0 structure."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DecoderLayer() for _ in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        return x


class TransformerBlock(nn.Module):
    """A mock transformer block."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ModelWithTransformerH(nn.Module):
    """A model with transformer.h.0 structure (GPT-2 style)."""

    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TransformerBlock() for _ in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer.h:
            x = block(x)
        return x


# =============================================================================
# Tests for get_no_split_modules
# =============================================================================
@pytest.mark.parametrize(
    "model_class,expected",
    [
        (ModelWithNoSplitModules, ["TransformerBlock", "AttentionBlock"]),
        (ModelWithLayers, ["DecoderLayer"]),
        (ModelWithTransformerH, ["TransformerBlock"]),
        (SimpleModel, []),
    ],
    ids=["no_split_attr", "layers_structure", "transformer_h_structure", "simple_model"],
)
def test_get_no_split_modules(model_class, expected):
    """Test get_no_split_modules with various model structures."""
    model = model_class()
    result = get_no_split_modules(model)
    assert result == expected


# =============================================================================
# Tests for check_need_rebalance
# =============================================================================
@pytest.mark.parametrize(
    "gpu_layer_counts,max_layer_imbalance,expected_rebalance",
    [
        ({0: 32}, 1, False),  # Single GPU
        ({0: 10, 1: 11}, 1, False),  # Balanced (diff=1, equal to max_layer_imbalance)
        ({0: 10, 1: 15}, 1, True),  # Imbalanced (diff=5 > max_layer_imbalance)
        ({0: 8, 1: 8, 2: 8, 3: 16}, 1, True),  # Multi-GPU imbalanced
        ({0: 0, 1: 32}, 1, False),  # Second-last GPU has 0 layers
        ({0: 10, 1: 13}, 5, False),  # Custom max_layer_imbalance (diff=3 < 5)
        ({0: 10, 1: 13}, 2, True),  # Custom max_layer_imbalance (diff=3 > 2)
    ],
    ids=["single_gpu", "balanced", "imbalanced", "multi_gpu", "zero_layers", "max_imbalance_5", "max_imbalance_2"],
)
def test_check_need_rebalance(gpu_layer_counts, max_layer_imbalance, expected_rebalance):
    """Test check_need_rebalance with various GPU distributions."""
    need_rebalance, _ = check_need_rebalance(gpu_layer_counts, max_layer_imbalance=max_layer_imbalance)
    assert need_rebalance is expected_rebalance


# =============================================================================
# Tests for rebalance_device_map
# =============================================================================
def test_rebalance_device_map():
    """Test rebalance_device_map preserves non-layer modules and keeps layers contiguous."""
    preview_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 1,
        "model.layers.4": 1,
        "model.layers.5": 1,
        "model.layers.6": 1,
        "model.layers.7": 1,
        "model.norm": 1,
        "lm_head": 1,
    }
    gpu_layer_counts = {0: 3, 1: 5}
    rebalance_info = {
        "last_gpu": 1,
        "second_last_gpu": 0,
        "last_gpu_layers": 5,
        "second_last_gpu_layers": 3,
    }

    result = rebalance_device_map(preview_map, gpu_layer_counts, rebalance_info, num_layers=8, num_gpus=2)

    # Check non-layer modules are preserved
    assert result["model.embed_tokens"] == 0
    assert result["model.norm"] == 1
    assert result["lm_head"] == 1

    # Check all layers are present and contiguous per GPU
    layer_gpus = {i: result[f"model.layers.{i}"] for i in range(8)}
    for gpu_id in [0, 1]:
        layers = sorted([i for i, g in layer_gpus.items() if g == gpu_id])
        if layers:
            assert layers == list(range(min(layers), max(layers) + 1)), f"GPU {gpu_id} layers not contiguous"


# =============================================================================
# Tests for _log_device_map_distribution
# =============================================================================
@pytest.mark.parametrize(
    "device_map",
    [
        {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 1, "model.norm": 1},
        {"model.layers.0": 0, "model.layers.1": 0, "model.layers.2": 0, "model.layers.3": 1, "model.layers.4": 1},
        {"model.layers.0": 0, "model.layers.2": 0, "model.layers.1": 1, "model.layers.3": 1},
        {
            "model.embed_tokens": 0,
            "model.rotary_emb": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.norm": 1,
            "lm_head": 1,
        },
    ],
    ids=["single_layer", "contiguous_layers", "non_contiguous_layers", "with_other_modules"],
)
def test_log_device_map_distribution(device_map):
    """Test _log_device_map_distribution with various device map configurations."""
    _log_device_map_distribution(device_map, num_gpus=2, method_name="Test")


# =============================================================================
# Tests for preview_device_map
# =============================================================================
# =============================================================================
# Tests for create_auto_adjusted_device_map num_hidden_layers fallback
# =============================================================================
class _ConfigStub:
    """Lightweight stand-in for AutoConfig; only the attribute lookups exercised by
    create_auto_adjusted_device_map matter here."""

    def __init__(self, num_hidden_layers=None, text_config=None):
        if num_hidden_layers is not None:
            self.num_hidden_layers = num_hidden_layers
        if text_config is not None:
            self.text_config = text_config


def test_create_auto_adjusted_device_map_reads_num_hidden_layers_from_text_config(monkeypatch):
    """Wrapped multimodal configs (e.g. Mllama, Qwen3.5-MoE) put LM hyperparameters
    under config.text_config — the fallback must read it instead of bailing to 'auto'."""
    captured: dict[str, object] = {}

    def fake_preview(config, max_memory, use_no_split):  # noqa: ARG001
        captured["called"] = True
        return {"model.layers.0": 0, "model.layers.1": 1}, {0: 1, 1: 1}

    monkeypatch.setattr(device_mapping_module, "preview_device_map", fake_preview)

    config = _ConfigStub(text_config=_ConfigStub(num_hidden_layers=2))
    result = create_auto_adjusted_device_map(config, num_gpus=2)

    # Proves we fell through to text_config (and so reached preview_device_map),
    # rather than the "auto" early-return when num_hidden_layers is missing.
    assert captured.get("called") is True
    assert result != "auto"


def test_create_auto_adjusted_device_map_falls_back_to_auto_without_text_config():
    """When neither the top-level config nor text_config exposes num_hidden_layers,
    the function must still fall back to 'auto'."""
    config = _ConfigStub()  # no num_hidden_layers, no text_config
    assert create_auto_adjusted_device_map(config, num_gpus=2) == "auto"


def test_create_auto_adjusted_device_map_falls_back_when_text_config_missing_num_layers():
    """text_config exists but has no num_hidden_layers — fall back to 'auto'."""
    config = _ConfigStub(text_config=_ConfigStub())
    assert create_auto_adjusted_device_map(config, num_gpus=2) == "auto"


@require_accelerate
@require_torch_multi_gpu
def test_tied_parameters_are_mapped_to_the_same_gpu(qwen2_5_configuration):
    """Test tied parameters are mapped to the same GPU for Qwen2.5."""
    device_map, gpu_layer_counts = preview_device_map(
        qwen2_5_configuration, max_memory={0: "512MiB", 1: "512MiB"}, use_no_split=True
    )

    assert device_map["model.embed_tokens"] == device_map["lm_head"] == 0


# =============================================================================
# Tests for create_auto_adjusted_device_map num_hidden_layers fallback
# =============================================================================
@pytest.mark.parametrize(
    "sub_config_attr",
    ["text_config", "language_config"],
    ids=["text_config", "language_config"],
)
def test_create_auto_adjusted_device_map_sub_config_fallback(monkeypatch, capfd, sub_config_attr):
    """``num_hidden_layers`` is read from nested sub-configs when missing on the top-level.

    Covers both naming conventions: ``text_config`` (Kimi-K2.5, Qwen3.5-MoE, Mllama, Llama-4, Gemma-3)
    and ``language_config`` (DeepSeek-VL).
    """

    class StubSubConfig:
        num_hidden_layers = 32

    class StubConfig:
        pass

    config = StubConfig()
    setattr(config, sub_config_attr, StubSubConfig())

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    # Preview fails (no real model), so the function returns "auto" via the preview-failure path
    # rather than the missing-num_hidden_layers path. The assertion below verifies the fallback was hit.
    result = create_auto_adjusted_device_map(config, num_gpus=2)
    assert result == "auto"
    assert "Cannot determine num_hidden_layers" not in capfd.readouterr().err


def test_create_auto_adjusted_device_map_no_num_hidden_layers(monkeypatch, capfd):
    """When neither top-level nor sub-configs expose ``num_hidden_layers``, fall back to ``auto`` with a warning."""

    class StubConfig:
        pass

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    # ScreenLogger sets propagate=False and writes to stderr, so capture via capfd rather than caplog.
    result = create_auto_adjusted_device_map(StubConfig(), num_gpus=2)

    assert result == "auto"
    assert "Cannot determine num_hidden_layers" in capfd.readouterr().err


# =============================================================================
# Shared build_skeleton_from_config (preview_device_map + create_model_skeleton)
# =============================================================================


def test_build_skeleton_from_config_creates_meta_model():
    """build_skeleton_from_config (shared with create_model_skeleton) builds a meta-device model."""
    transformers_module = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")

    config = transformers_module.AutoConfig.from_pretrained(QWEN2_5_MODEL_ID, trust_remote_code=True)
    model = build_skeleton_from_config(config, trust_remote_code=True)
    assert hasattr(model, "model") and hasattr(model.model, "layers")
    # All parameters should be on meta device — zero memory.
    assert all(p.device.type == "meta" for p in model.parameters())


@pytest.mark.parametrize(
    "config_kwargs, expected_name",
    [
        # Architecture name registered for CausalLM.
        ({"architectures": ["LlamaForCausalLM"]}, "AutoModelForCausalLM"),
        # auto_map carries the AutoModelForCausalLM key (trust_remote_code path).
        ({"auto_map": {"AutoModelForCausalLM": "modeling_x.MyLM"}}, "AutoModelForCausalLM"),
        # Architecture name registered for ImageTextToText (multimodal wrapper path).
        ({"architectures": ["Llama4ForConditionalGeneration"]}, "AutoModelForImageTextToText"),
        # auto_map carries the AutoModelForImageTextToText key.
        ({"auto_map": {"AutoModelForImageTextToText": "modeling_x.MyVLM"}}, "AutoModelForImageTextToText"),
    ],
    ids=["arch_causal", "auto_map_causal", "arch_image_text", "auto_map_image_text"],
)
def test_resolve_model_class_branches(config_kwargs, expected_name):
    pytest.importorskip("transformers")
    if "ImageTextToText" in expected_name and not pytest.importorskip("transformers").__version__.startswith(
        ("4.5", "4.6", "4.7", "5.")
    ):
        pytest.skip("AutoModelForImageTextToText requires transformers >= 4.51")
    config = SimpleNamespace(**config_kwargs)
    resolved = device_mapping_module._resolve_model_class(config)
    assert resolved.__name__ == expected_name


def test_resolve_model_class_raises_when_no_match():
    pytest.importorskip("transformers")
    config = SimpleNamespace(architectures=["TotallyMadeUpArch"], auto_map={})
    with pytest.raises(ValueError, match="Cannot determine model class"):
        device_mapping_module._resolve_model_class(config)
