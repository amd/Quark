#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Utility functions for mixed precision quantization.

This module provides utilities for layer categorization, pattern matching,
and configuration conversion.
"""

from __future__ import annotations

import fnmatch
import re
from collections import defaultdict
from typing import TYPE_CHECKING

import torch.nn as nn

from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer import PerTensorMinMaxObserver

from .config import QuantConfig, get_layer_config, is_native_mode, normalize_quant_mode

if TYPE_CHECKING:
    pass


# =============================================================================
# Layer Pattern Definitions
# =============================================================================

# Standard patterns for self_attn (attention projections)
_SELF_ATTN_STANDARD = [
    "*q_proj*",
    "*k_proj*",
    "*v_proj*",
    "*qkv_proj*",
    "*q_a_proj*",
    "*q_b_proj*",  # DeepSeek MLA: Q projections (a=low-rank, b=full-rank)
    "*kv_a_proj*",
    "*kv_b_proj*",  # DeepSeek MLA: KV projections
    "*fused_qkv_a_proj*",  # DeepSeek MLA: fused A projection (vLLM runtime)
    "*o_proj*",
    "*out_proj*",
]

# Linear attention patterns (Qwen 3.5 and other hybrid models)
_LINEAR_ATTN_STANDARD = [
    "*linear_attn*in_proj*",
    "*linear_attn*out_proj*",
    "*linear_attn*q_proj*",
    "*linear_attn*k_proj*",
    "*linear_attn*v_proj*",
    "*linear_attn*o_proj*",
    "*linear_attn*qkv_proj*",
    "*linear_attn*z_proj*",
    "*linear_attn*a_proj*",
    "*linear_attn*b_proj*",
]

# KV projection layer patterns (for kv_cache_quant_config and kv_cache_group)
_KV_PROJ_PATTERNS = ["*k_proj", "*v_proj"]

# Standard patterns for mlp (feed-forward and MoE)
_MLP_STANDARD = [
    "*gate_proj*",
    "*up_proj*",
    "*down_proj*",
    "*fc1*",
    "*fc2*",
    "*gate_up_proj*",
    "*w1*",
    "*w2*",
    "*w3*",
    "*experts*gate_proj*",
    "*experts*up_proj*",
    "*experts*down_proj*",
    "*shared_expert*",
    "*shared_experts*",
]

# Model-specific patterns
LAYER_PATTERNS: dict[str, dict[str, list[str]]] = {
    "llama": {
        "self_attn": _SELF_ATTN_STANDARD,
        "mlp": _MLP_STANDARD,
    },
    "mistral": {
        "self_attn": _SELF_ATTN_STANDARD,
        "mlp": _MLP_STANDARD,
    },
    "mixtral": {
        "self_attn": _SELF_ATTN_STANDARD,
        "mlp": _MLP_STANDARD + ["*block_sparse_moe.gate*"],
    },
    "qwen2": {
        "self_attn": _SELF_ATTN_STANDARD,
        "mlp": _MLP_STANDARD,
    },
    "qwen2_moe": {
        "self_attn": _SELF_ATTN_STANDARD,
        "mlp": _MLP_STANDARD + ["*mlp.gate*"],
    },
    "qwen3_5_moe": {
        "linear_attn": _LINEAR_ATTN_STANDARD,
        "self_attn": _SELF_ATTN_STANDARD,
        "mlp": _MLP_STANDARD + ["*mlp.gate*"],
    },
    "deepseek": {
        "self_attn": _SELF_ATTN_STANDARD,
        "mlp": _MLP_STANDARD,
    },
    "deepseek_v2": {
        "self_attn": _SELF_ATTN_STANDARD,
        "mlp": _MLP_STANDARD + ["*mlp.gate*"],
    },
    "phi": {
        "self_attn": ["*Wqkv*", "*q_proj*", "*k_proj*", "*v_proj*", "*out_proj*", "*o_proj*"],
        "mlp": ["*fc1*", "*fc2*"],
    },
    "gemma": {
        "self_attn": _SELF_ATTN_STANDARD,
        "mlp": _MLP_STANDARD,
    },
    "default": {
        "linear_attn": _LINEAR_ATTN_STANDARD,
        "self_attn": _SELF_ATTN_STANDARD,
        "mlp": _MLP_STANDARD,
    },
}

# DEFAULT_EXCLUDED_LAYERS = [
#     "*lm_head*",
#     "*embed*",
#     "*router*",
#     "*.gate",
#     "*gate.weight*",
# Default layer patterns to exclude from quantization
DEFAULT_EXCLUDE_PATTERNS = [
    "lm_head",
    "model.visual.*",
    "mtp.*",
    "*mlp.gate",
    "*mlp.gate.linear",
    "*shared_expert_gate*",
    # REMOVED: "*.self_attn.*" - this would exclude ALL self_attn layers, breaking functionality
    "*.shared_expert.*",
]

# Backward-compatible alias for older imports.
DEFAULT_EXCLUDED_LAYERS = DEFAULT_EXCLUDE_PATTERNS


def _match_patterns(name: str, patterns: list[str]) -> bool:
    """Check if name matches any pattern."""
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _should_exclude(name: str, exclude_patterns: list[str] | None = None) -> bool:
    """Check if layer should be excluded."""
    patterns = DEFAULT_EXCLUDE_PATTERNS if exclude_patterns is None else exclude_patterns
    return _match_patterns(name, patterns)


def _get_layer_partition_from_config(model: nn.Module, layer_name: str) -> str | None:
    """
    Determine partition from model.config.layer_types.

    This is a fallback detection method for models with heterogeneous layer types
    (e.g., Qwen 3.5 with linear_attention and full_attention layers).

    Args:
        model: Model with config attribute
        layer_name: Full layer name (e.g., "model.layers.15.self_attn.q_proj")

    Returns:
        "linear_attn" if layer_types[idx] == "linear_attention"
        "self_attn" if layer_types[idx] == "full_attention"
        None if detection fails or config unavailable
    """
    if not hasattr(model, "config"):
        return None

    config = model.config
    # Support nested text_config (Qwen 3.5 MoE)
    if hasattr(config, "text_config"):
        config = config.text_config

    if not hasattr(config, "layer_types"):
        return None

    # Extract layer index from name
    # Example: "model.layers.15.self_attn.q_proj" -> 15
    match = re.search(r"\.layers\.(\d+)\.", layer_name)
    if not match:
        return None

    layer_idx = int(match.group(1))
    if layer_idx >= len(config.layer_types):
        return None

    layer_type = config.layer_types[layer_idx]
    if layer_type == "linear_attention":
        return "linear_attn"
    elif layer_type == "full_attention":
        return "self_attn"

    return None


def get_model_type(model: nn.Module) -> str:
    """Detect model type from model configuration."""
    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "model_type"):
            return config.model_type
        if hasattr(config, "architectures") and config.architectures:
            arch = config.architectures[0].lower()
            for known_type in LAYER_PATTERNS:
                if known_type in arch:
                    return known_type
    return "default"


def categorize_layers(
    model: nn.Module,
    model_type: str | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict[str, set[str]]:
    """
    Categorize model layers into partitions (linear_attn, self_attn, mlp).

    Detection strategy (hybrid approach):
    1. Pattern matching (fast, works for most cases)
    2. Config-based fallback (precise for models with layer_types)
    3. Backward compatibility (remove empty linear_attn for 2-partition models)

    Args:
        model: Model to categorize
        model_type: Optional model type override
        exclude_patterns: Optional exclusion patterns

    Returns:
        Dictionary mapping category to set of layer names.
        For hybrid models: {"linear_attn": {...}, "self_attn": {...}, "mlp": {...}}
        For traditional models: {"self_attn": {...}, "mlp": {...}}
    """
    if model_type is None:
        model_type = get_model_type(model)

    patterns = LAYER_PATTERNS.get(model_type, LAYER_PATTERNS["default"])

    categories: dict[str, set[str]] = {
        "linear_attn": set(),
        "self_attn": set(),
        "mlp": set(),
    }

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if _should_exclude(name, exclude_patterns):
            continue

        # Try pattern matching first (fast path)
        matched = False

        # Check linear_attn patterns if they exist
        if "linear_attn" in patterns:
            if _match_patterns(name, patterns["linear_attn"]):
                categories["linear_attn"].add(name)
                matched = True
                continue

        # Check self_attn patterns
        if _match_patterns(name, patterns["self_attn"]):
            categories["self_attn"].add(name)
            matched = True
            continue

        # Check mlp patterns
        if _match_patterns(name, patterns["mlp"]):
            categories["mlp"].add(name)
            matched = True
            continue

        # Pattern matching failed - try config-based detection
        if not matched:
            config_partition = _get_layer_partition_from_config(model, name)
            if config_partition == "linear_attn":
                categories["linear_attn"].add(name)
                matched = True
            elif config_partition == "self_attn":
                categories["self_attn"].add(name)
                matched = True

        # Final fallback: expert layers go to mlp
        if not matched and "expert" in name.lower():
            categories["mlp"].add(name)

    # Backward compatibility: remove ALL empty partitions
    # This prevents search space expansion when partitions have no layers
    categories = {k: v for k, v in categories.items() if len(v) > 0}

    return categories


def _compact_layer_configs(layer_configs: dict[str, QLayerConfig]) -> dict[str, QLayerConfig]:
    """
    Compact same layer configs into wildcard patterns.
    """
    if not layer_configs:
        return layer_configs

    grouped: dict[str, list[tuple[str, QLayerConfig]]] = defaultdict(list)
    for name, cfg in layer_configs.items():
        pattern = re.sub(r"\d+", "*", name)
        grouped[pattern].append((name, cfg))

    compact: dict[str, QLayerConfig] = {}
    for pattern, entries in grouped.items():
        if len(entries) > 1:
            first_cfg = entries[0][1]
            if all(cfg == first_cfg for _, cfg in entries):
                compact[pattern] = first_cfg
                continue
        for name, cfg in entries:
            compact[name] = cfg

    return compact


def _get_fp8_output_spec() -> QTensorConfig:
    """Create FP8 output quantization spec for KV cache."""
    return QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        qscheme=QSchemeType.per_tensor,
        is_dynamic=False,
        symmetric=True,
        observer_cls=PerTensorMinMaxObserver,
    )


def create_qconfig_from_quant_config(
    model: nn.Module,
    config: QuantConfig,
    layer_sensitivity: dict[str, int] | None = None,
    min_kv_scale: float = 0.0,
    exclude_patterns: list[str] | None = None,
) -> QConfig:
    """
    Create QConfig from QuantConfig dict for use with ModelQuantizer.

    Args:
        model: Model to quantize
        config: Quantization configuration dict, e.g.,
                {"self_attn_mode": "fp8", "mlp_mode": "native", "kv_cache_mode": "native", ...}
        layer_sensitivity: Layer sensitivity defining partitions.
                          Default: {"self_attn": 3, "mlp": 1}
        min_kv_scale: Minimum kv-cache scale.
        exclude_patterns: Optional layer-name patterns to skip during quantization.

    Returns:
        QConfig for use with ModelQuantizer
    """
    categories = categorize_layers(model, exclude_patterns=exclude_patterns)

    # Build layer configs
    layer_configs: dict[str, QLayerConfig] = {}
    exclude: list[str] = []

    # Collect all Linear layers matching the requested exclude patterns so they are kept
    # out of quantization; categorize_layers skips them but they would otherwise fall
    # through to global_config.
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _should_exclude(name, exclude_patterns):
            exclude.append(name)

    # Mode map for all layer partitions (dynamic construction based on actual categories)
    # Exclude output-only partitions (kv_cache, attention) which are handled separately
    mode_map = {}
    for partition in categories:
        mode_key = f"{partition}_mode"
        mode_map[partition] = normalize_quant_mode(config.get(mode_key, "native"))

    for cat, mode in mode_map.items():
        if is_native_mode(mode):
            # Exclude from quantization
            for name in categories[cat]:
                exclude.append(name)
        else:
            layer_cfg = get_layer_config(mode)
            if layer_cfg:
                for name in categories[cat]:
                    layer_configs[name] = layer_cfg

    # Handle KV cache quantization (key addition!)
    kv_cache_quant_config: dict[str, QLayerConfig] = {}
    kv_cache_group: list[str] = []
    kv_cache_mode = normalize_quant_mode(config.get("kv_cache_mode", "native"))

    if kv_cache_mode == "fp8":
        # Set kv_cache_group for export-time k_scale/v_scale generation
        kv_cache_group = _KV_PROJ_PATTERNS.copy()

        # Create FP8 output spec for KV cache
        fp8_output_spec = _get_fp8_output_spec()

        # Get base layer config from self_attn (k_proj/v_proj are part of self_attn)
        self_attn_mode = normalize_quant_mode(config.get("self_attn_mode", "native"))
        base_layer_cfg = get_layer_config(self_attn_mode) if not is_native_mode(self_attn_mode) else None

        # Create kv_cache_quant_config with output_tensors for k_proj and v_proj
        for pattern in _KV_PROJ_PATTERNS:
            if base_layer_cfg:
                kv_cache_cfg = QLayerConfig(
                    weight=base_layer_cfg.weight,
                    input_tensors=base_layer_cfg.input_tensors,
                    output_tensors=fp8_output_spec,
                )
            else:
                kv_cache_cfg = QLayerConfig(
                    output_tensors=fp8_output_spec,
                )
            kv_cache_quant_config[pattern] = kv_cache_cfg

        # Also update layer_quant_config for k_proj/v_proj to include output quantization
        for pattern in _KV_PROJ_PATTERNS:
            layer_configs[pattern] = kv_cache_quant_config[pattern]

    global_config = next(iter(layer_configs.values()), QLayerConfig())

    return QConfig(
        global_quant_config=global_config,
        layer_quant_config=_compact_layer_configs(layer_configs) if layer_configs else {},
        kv_cache_quant_config=kv_cache_quant_config,
        kv_cache_group=kv_cache_group,
        exclude=list(dict.fromkeys(exclude)),
        min_kv_scale=min_kv_scale,
    )


__all__ = [
    "categorize_layers",
    "create_qconfig_from_quant_config",
    "DEFAULT_EXCLUDE_PATTERNS",
]
