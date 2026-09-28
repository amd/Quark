#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

import torch.nn as nn

from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer import PerTensorMinMaxObserver

from .config import QuantConfig, get_layer_config, get_partition_mode, is_native_mode, normalize_quant_mode

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
    "*indexer*wq_b*",  # DeepSeek/GLM DSA indexer: query projection
    # HF keeps indexer.wk and indexer.weights_proj separate; vLLM fuses them into
    # wk_weights_proj at runtime. categorize_layers runs on the HF model, so match
    # the HF names here; adapt_layer_patterns_for_vllm maps them to the fused name.
    "*indexer*wk*",
    "*indexer*weights_proj*",
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

# Standard patterns for shared experts. This check runs before generic MLP
# patterns because shared-expert projections also contain gate/up/down_proj.
_SHARED_EXPERT_STANDARD = [
    "*shared_expert*",
    "*shared_experts*",
]

# Standard patterns for dense feed-forward blocks. Routed-expert paths can also
# contain these leaf names, so categorize_layers identifies routed experts first.
_DENSE_MLP_STANDARD = [
    "*gate_proj*",
    "*up_proj*",
    "*down_proj*",
    "*fc1*",
    "*fc2*",
    "*gate_up_proj*",
    "*w1*",
    "*w2*",
    "*w3*",
]

_ROUTED_MOE_STANDARD = [
    "*experts*",
    "*routed_expert*",
    "*routed_input_transform*",
    "*routed_output_transform*",
]

# Model-specific patterns
LAYER_PATTERNS: dict[str, dict[str, list[str]]] = {
    "llama": {
        "self_attn": _SELF_ATTN_STANDARD,
        "dense_mlp": _DENSE_MLP_STANDARD,
    },
    "mistral": {
        "self_attn": _SELF_ATTN_STANDARD,
        "dense_mlp": _DENSE_MLP_STANDARD,
    },
    "mixtral": {
        "self_attn": _SELF_ATTN_STANDARD,
        "dense_mlp": _DENSE_MLP_STANDARD + ["*block_sparse_moe.gate*"],
        "routed_moe": _ROUTED_MOE_STANDARD,
    },
    "qwen2": {
        "self_attn": _SELF_ATTN_STANDARD,
        "dense_mlp": _DENSE_MLP_STANDARD,
    },
    "qwen2_moe": {
        "self_attn": _SELF_ATTN_STANDARD,
        "dense_mlp": _DENSE_MLP_STANDARD + ["*mlp.gate*"],
        "routed_moe": _ROUTED_MOE_STANDARD,
        "shared_expert": _SHARED_EXPERT_STANDARD,
    },
    "qwen3_5_moe": {
        "linear_attn": _LINEAR_ATTN_STANDARD,
        "self_attn": _SELF_ATTN_STANDARD,
        "dense_mlp": _DENSE_MLP_STANDARD + ["*mlp.gate*"],
        "routed_moe": _ROUTED_MOE_STANDARD,
        "shared_expert": _SHARED_EXPERT_STANDARD,
    },
    "glm_moe_dsa": {
        "self_attn": _SELF_ATTN_STANDARD,
        "dense_mlp": _DENSE_MLP_STANDARD + ["*mlp.gate*"],
        "routed_moe": _ROUTED_MOE_STANDARD,
        "shared_expert": _SHARED_EXPERT_STANDARD,
    },
    "deepseek": {
        "self_attn": _SELF_ATTN_STANDARD,
        "dense_mlp": _DENSE_MLP_STANDARD,
        "routed_moe": _ROUTED_MOE_STANDARD,
    },
    "deepseek_v2": {
        "self_attn": _SELF_ATTN_STANDARD,
        "dense_mlp": _DENSE_MLP_STANDARD + ["*mlp.gate*"],
        "routed_moe": _ROUTED_MOE_STANDARD,
        "shared_expert": _SHARED_EXPERT_STANDARD,
    },
    "phi": {
        "self_attn": ["*Wqkv*", "*q_proj*", "*k_proj*", "*v_proj*", "*out_proj*", "*o_proj*"],
        "dense_mlp": ["*fc1*", "*fc2*"],
    },
    "gemma": {
        "self_attn": _SELF_ATTN_STANDARD,
        "dense_mlp": _DENSE_MLP_STANDARD,
    },
    "default": {
        "linear_attn": _LINEAR_ATTN_STANDARD,
        "self_attn": _SELF_ATTN_STANDARD,
        "dense_mlp": _DENSE_MLP_STANDARD,
        "routed_moe": _ROUTED_MOE_STANDARD,
        "shared_expert": _SHARED_EXPERT_STANDARD,
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
    "*shared_expert_gate*",
    # REMOVED: "*.self_attn.*" - this would exclude ALL self_attn layers, breaking functionality
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


def _is_quantizable_linear_module(module: nn.Module) -> bool:
    """Return whether ``module`` behaves as a quantizable linear layer."""
    return isinstance(module, nn.Linear) or bool(getattr(module, "_is_fp8_block_quantized_linear", False))


def _is_routed_expert_container(name: str, module: nn.Module) -> bool:
    """Recognize fused and preprocessed routed-expert containers."""
    if not name.endswith(".experts") or not hasattr(module, "num_experts"):
        return False
    if hasattr(module, "down_proj") and (hasattr(module, "gate_up_proj") or hasattr(module, "up_proj")):
        return True
    return any(
        child_name.isdigit()
        and hasattr(child, "down_proj")
        and (hasattr(child, "gate_proj") or hasattr(child, "up_proj"))
        for child_name, child in module.named_children()
    )


def _get_layer_partition_from_config(model: nn.Module, layer_name: str) -> str | None:
    """
    Determine a hybrid-attention partition from model configuration.

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

    # Extract layer index from name
    # Example: "model.layers.15.self_attn.q_proj" -> 15
    match = re.search(r"\.layers\.(\d+)\.", layer_name)
    if not match:
        return None

    layer_idx = int(match.group(1))

    # Kimi-K3 stores one-based KDA/full-attention layer numbers in a nested
    # linear_attn_config instead of exposing a zero-based layer_types list.
    linear_attn_config = getattr(config, "linear_attn_config", None)
    if linear_attn_config is not None:
        if isinstance(linear_attn_config, dict):
            kda_layers = linear_attn_config.get("kda_layers", [])
            full_attn_layers = linear_attn_config.get("full_attn_layers", [])
        else:
            kda_layers = getattr(linear_attn_config, "kda_layers", [])
            full_attn_layers = getattr(linear_attn_config, "full_attn_layers", [])
        one_based_layer = layer_idx + 1
        if one_based_layer in kda_layers:
            return "linear_attn"
        if one_based_layer in full_attn_layers:
            return "self_attn"

    if not hasattr(config, "layer_types"):
        return None
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
    Categorize model layers into partitions (linear_attn, self_attn,
    dense_mlp, routed_moe, shared_expert).

    Detection strategy (hybrid approach):
    1. Pattern matching (fast, works for most cases)
    2. Config-based fallback (precise for models with layer_types)
    3. Dense MLP, routed experts, and shared experts are separate partitions
    4. Backward compatibility (remove empty partitions)

    Args:
        model: Model to categorize
        model_type: Optional model type override
        exclude_patterns: Optional exclusion patterns

    Returns:
        Dictionary mapping category to set of layer names.
        For hybrid MoE models:
        {"linear_attn": {...}, "self_attn": {...}, "dense_mlp": {...},
         "routed_moe": {...}, "shared_expert": {...}}
        For traditional models: {"self_attn": {...}, "dense_mlp": {...}}
    """
    if model_type is None:
        model_type = get_model_type(model)

    patterns = LAYER_PATTERNS.get(model_type, LAYER_PATTERNS["default"])

    categories: dict[str, set[str]] = {
        "linear_attn": set(),
        "self_attn": set(),
        "dense_mlp": set(),
        "routed_moe": set(),
        "shared_expert": set(),
    }

    for name, module in model.named_modules():
        if _should_exclude(name, exclude_patterns):
            continue

        # Some pre-quantized MoE checkpoints keep routed expert weights in one
        # fused Experts container instead of exposing per-expert nn.Linear
        # children. Treat the structural container as the routed MLP partition;
        # shared experts remain separate through their Linear children.
        if _is_routed_expert_container(name, module):
            categories["routed_moe"].add(name)
            continue

        if not _is_quantizable_linear_module(module):
            continue

        # Try pattern matching first (fast path)
        matched = False

        # Shared experts must be checked before generic gate/up/down MLP
        # patterns so they remain an independent partition.
        shared_expert_patterns = patterns.get("shared_expert", _SHARED_EXPERT_STANDARD)
        if _match_patterns(name, shared_expert_patterns):
            categories["shared_expert"].add(name)
            continue

        # Routed experts must be recognized before broad gate/up/down patterns,
        # otherwise dense FFNs and prequantized expert weights end up sharing one
        # partition despite having different source-precision floors.
        routed_moe_patterns = patterns.get("routed_moe", _ROUTED_MOE_STANDARD)
        if _match_patterns(name, routed_moe_patterns):
            categories["routed_moe"].add(name)
            continue

        is_attention_path = any(marker in name for marker in (".self_attn.", ".attention.", ".attn."))

        # Check linear_attn patterns if they exist
        if "linear_attn" in patterns:
            if _match_patterns(name, patterns["linear_attn"]):
                categories["linear_attn"].add(name)
                matched = True
                continue

        # KDA and full-attention modules can share the same ``self_attn`` path
        # (Kimi-K3). Dispatch them from config before generic self-attention
        # patterns consume both kinds.
        if is_attention_path:
            config_partition = _get_layer_partition_from_config(model, name)
            if config_partition is not None:
                categories[config_partition].add(name)
                matched = True
                continue

        # Check self_attn patterns
        if _match_patterns(name, patterns["self_attn"]):
            categories["self_attn"].add(name)
            matched = True
            continue

        # Check dense MLP patterns after routed/shared experts have been removed.
        dense_mlp_patterns = patterns.get("dense_mlp", patterns.get("mlp", _DENSE_MLP_STANDARD))
        if _match_patterns(name, dense_mlp_patterns):
            categories["dense_mlp"].add(name)
            matched = True
            continue

        # Pattern matching failed - try config-based detection
        if not matched and is_attention_path:
            config_partition = _get_layer_partition_from_config(model, name)
            if config_partition == "linear_attn":
                categories["linear_attn"].add(name)
                matched = True
            elif config_partition == "self_attn":
                categories["self_attn"].add(name)
                matched = True

        # Final fallback: non-shared expert layers are routed MoE layers.
        if not matched and "expert" in name.lower():
            categories["routed_moe"].add(name)

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


def _compact_generated_exclude_names(names: list[str]) -> list[str]:
    """Compact generated expert excludes across numeric path segments.

    A large MoE may contain tens of thousands of expert projection leaves.
    Compacting complete numeric path components avoids creating one exact
    exclude plus one ``name.*`` glob for every expert. Other layer names stay
    exact because downstream runtimes may not interpret shell-style globs in
    their exclusion lists.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    compact: list[str] = []
    for name in names:
        if ".experts." not in f".{name}.":
            compact.append(name)
            continue
        pattern = re.sub(r"(?:(?<=\.)|^)\d+(?=\.|$)", "*", name)
        grouped[pattern].append(name)

    for pattern, entries in grouped.items():
        compact.append(pattern if len(entries) > 1 else entries[0])
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


def _with_output_tensors(
    layer_config: QLayerConfig | None,
    output_tensors: QTensorConfig,
) -> QLayerConfig:
    """Return an independent layer config with the requested output quantization."""
    return replace(layer_config or QLayerConfig(), output_tensors=output_tensors)


def _moe_global_config_from_prefixes(
    model: nn.Module, layer_configs: Mapping[str, QLayerConfig]
) -> QLayerConfig | None:
    """Resolve actual expert containers from their configured projections.

    vLLM looks up one config at the container prefix, while HF may configure
    individual experts. Only a uniform, complete config can represent that
    container. Routed input/output transforms and shared experts are not its
    expert projections. Do not infer a container's config from the old global.
    """
    resolved: list[QLayerConfig] = []
    projection_names = {"gate_up_proj", "gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}
    for prefix, container in model.named_modules():
        if prefix.rsplit(".", 1)[-1] != "experts":
            continue
        names = [
            f"{prefix}.{expert_index}.{projection}"
            for expert_index, expert in container.named_children()
            if expert_index.isdigit()
            for projection, module in expert.named_children()
            if projection in projection_names and _is_quantizable_linear_module(module)
        ]
        if not names and _is_routed_expert_container(prefix, container):
            # Fused HF weights are configured at the container itself.
            names = [prefix]
        configs = [layer_configs.get(name) for name in names]
        if not configs or any(config is None or config.weight is None for config in configs):
            continue
        first = configs[0]
        if first is not None and all(config == first for config in configs[1:]):
            resolved.append(first)
    # A single global cannot describe heterogeneous containers. Their existing
    # per-layer configs remain available to the export adapter in that case.
    if resolved and all(config == resolved[0] for config in resolved[1:]):
        return resolved[0]
    return None


def create_qconfig_from_quant_config(
    model: nn.Module,
    config: QuantConfig,
    layer_sensitivity: dict[str, int] | None = None,
    min_kv_scale: float = 0.0,
    exclude_patterns: list[str] | None = None,
    source_weight_bitwidth_by_layer: Mapping[str, int] | None = None,
) -> QConfig:
    """
    Create QConfig from QuantConfig dict for use with ModelQuantizer.

    Args:
        model: Model to quantize
        config: Quantization configuration dict, e.g.,
                {"self_attn_mode": "fp8", "dense_mlp_mode": "native",
                 "routed_moe_mode": "native", "kv_cache_mode": "native", ...}
        layer_sensitivity: Layer sensitivity defining partitions. Defaults include
                          separate ``dense_mlp`` and ``routed_moe`` entries.
        min_kv_scale: Minimum kv-cache scale.
        exclude_patterns: Optional layer-name patterns to skip during quantization.
        source_weight_bitwidth_by_layer: Source bitwidth for concrete prequantized
            layers. If a requested mode has a wider weight dtype, that layer keeps
            its source weight while still receiving the target activation QDQ.

    Returns:
        QConfig for use with ModelQuantizer
    """
    categories = categorize_layers(model, exclude_patterns=exclude_patterns)

    if "shared_expert" in categories:
        shared_expert_mode = normalize_quant_mode(config.get("shared_expert_mode", "native"))
        parent_partition = "routed_moe" if "routed_moe" in categories else "dense_mlp"
        parent_mode = get_partition_mode(config, parent_partition)
        if not is_native_mode(shared_expert_mode) and shared_expert_mode != parent_mode:
            raise ValueError(
                f"shared_expert_mode must be native or match {parent_partition}_mode "
                f"(got shared_expert_mode={shared_expert_mode!r}, "
                f"{parent_partition}_mode={parent_mode!r})"
            )

    # Build layer configs
    layer_configs: dict[str, QLayerConfig] = {}
    # Keep user globs as well as resolved HF names. Runtime-only modules such
    # as vLLM's fused router gate do not exist as nn.Linear in the meta model.
    exclude: list[str] = list(exclude_patterns or [])
    native_partition_excludes: list[str] = []

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
        mode_map[partition] = get_partition_mode(config, partition)

    def _mode_weight_bitwidth(mode: str) -> int | None:
        layer_cfg = get_layer_config(mode)
        if layer_cfg is None:
            return None
        weight = layer_cfg.weight
        if isinstance(weight, list):
            weight = weight[0] if weight else None
        to_bitwidth = getattr(getattr(weight, "dtype", None), "to_bitwidth", None)
        return int(to_bitwidth()) if callable(to_bitwidth) else None

    for cat, mode in mode_map.items():
        if is_native_mode(mode):
            # Exclude from quantization
            native_partition_excludes.extend(categories[cat])
        else:
            layer_cfg = get_layer_config(mode)
            if layer_cfg:
                target_bitwidth = _mode_weight_bitwidth(mode)
                for name in categories[cat]:
                    source_bitwidth = (
                        source_weight_bitwidth_by_layer.get(name) if source_weight_bitwidth_by_layer else None
                    )
                    if (
                        source_bitwidth is not None
                        and target_bitwidth is not None
                        and target_bitwidth > source_bitwidth
                    ):
                        # Clamp only the unrealizable weight conversion. Retain
                        # input/output specs as a final per-layer safety net for
                        # legacy configs and heterogeneous source checkpoints.
                        layer_configs[name] = replace(layer_cfg, weight=None)
                        continue
                    layer_configs[name] = layer_cfg

    exclude.extend(_compact_generated_exclude_names(native_partition_excludes))
    global_config = next(
        (layer_config for layer_config in layer_configs.values() if layer_config.weight is not None),
        next(iter(layer_configs.values()), QLayerConfig()),
    )
    # Prefer the config resolved for actual expert container prefixes, rather
    # than an arbitrary member of the broader routed_moe search partition.
    moe_global_config = _moe_global_config_from_prefixes(model, layer_configs)
    if moe_global_config is not None:
        global_config = moe_global_config

    # Handle KV cache quantization.
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

        # Keep the generic entries used by Quark and downstream runtimes, while
        # applying the same complete config to every concrete self-attention K/V
        # layer. This avoids conflicting overlapping entries after export.
        for pattern in _KV_PROJ_PATTERNS:
            kv_cache_cfg = _with_output_tensors(base_layer_cfg, fp8_output_spec)
            kv_cache_quant_config[pattern] = kv_cache_cfg
            layer_configs[pattern] = replace(kv_cache_cfg)

        kv_layer_names = sorted(
            name for name in categories.get("self_attn", set()) if _match_patterns(name, _KV_PROJ_PATTERNS)
        )
        for name in kv_layer_names:
            layer_configs[name] = _with_output_tensors(layer_configs.get(name), fp8_output_spec)

    return QConfig(
        global_quant_config=global_config,
        layer_quant_config=_compact_layer_configs(layer_configs) if layer_configs else {},
        kv_cache_quant_config=kv_cache_quant_config,
        kv_cache_group=kv_cache_group,
        exclude=list(dict.fromkeys(exclude)),
        min_kv_scale=min_kv_scale,
    )
