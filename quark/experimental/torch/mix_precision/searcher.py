#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Configuration search algorithms for mixed precision quantization.

This module provides the ConfigSearcher class which generates and sorts
quantization configurations based on sensitivity and precision hierarchy.

Module-Partition Support
------------------------
The searcher supports the following primary layer partitions:
  - `linear_attn`: Linear attention (e.g., Qwen 3.5)
  - `self_attn`: Traditional QKV + O projections (e.g., Llama)
  - `dense_mlp`: Dense feed-forward layers
  - `routed_moe`: Routed MoE experts
  - `shared_expert`: Always-active MoE shared experts

For backward compatibility:
  - Traditional models (Llama): only `self_attn` and `dense_mlp` detected
  - Hybrid MoE models: linear/self attention, dense MLP, and routed MoE partitions available
  - MoE models with shared experts add `shared_expert` as a dependent partition

The ConfigSearcher automatically adapts to model architecture via
`available_partitions` filtering (set from detector results) to prevent
search space explosion when some partitions are not present.

`shared_expert` is not an independent Cartesian-product dimension. Its mode
is constrained to either `native` or the current `routed_moe_mode` (falling
back to `dense_mlp_mode` on architectures without routed experts).

Example:
    # Traditional model (Llama-like)
    searcher = ConfigSearcher(
        available_partitions={"self_attn", "dense_mlp"}  # Only 2 partitions
    )

    # Hybrid model (Qwen 3.5)
    searcher = ConfigSearcher(
        available_partitions={"linear_attn", "self_attn", "dense_mlp", "routed_moe", "shared_expert"}
    )
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product

from .config import (
    ALL_QUANT_MODES,
    ATTENTION_MODES,
    KV_CACHE_MODES,
    PRECISION_SCORES,
    HardwareTarget,
    ModuleSearchConfig,
    QuantConfig,
    QuantMode,
    get_layer_config,
    get_partition_mode,
    get_supported_schemes,
    is_native_mode,
    normalize_quant_mode,
)

logger = logging.getLogger(__name__)


@dataclass
class ScoredConfig:
    """Configuration with its computed score for sorting."""

    config: QuantConfig
    score: float

    def __lt__(self, other: ScoredConfig) -> bool:
        return self.score > other.score  # Higher score = more conservative


class ConfigSearcher:
    """
    Prior-driven searcher for mixed precision configurations.

    Generates and sorts configurations based on:
    1. Layer sensitivity (more sensitive layers prefer higher precision)
    2. Precision hierarchy constraints
    3. Hardware support (filter unsupported modes)

    Args:
        search_config: ModuleSearchConfig with sensitivity settings
        hardware: Hardware target for filtering modes
        precision_weights: Custom precision scores for modes

    Example:
        >>> # Default partitioning (self_attn, mlp)
        >>> searcher = ConfigSearcher(hardware=HardwareTarget.MI300)
        >>>
        >>> # Custom sensitivity weights via ModuleSearchConfig
        >>> search_config = ModuleSearchConfig(
        ...     layer_sensitivity={"self_attn": 5, "dense_mlp": 2, "routed_moe": 1}
        ... )
        >>> searcher = ConfigSearcher(
        ...     search_config=search_config,
        ...     hardware=HardwareTarget.MI300,
        ... )
        >>> configs = searcher.generate_sorted_configs()
    """

    def __init__(
        self,
        search_config: ModuleSearchConfig | None = None,
        hardware: HardwareTarget | str | None = None,
        precision_weights: dict[str, int] | None = None,
        available_partitions: set[str] | None = None,
        source_weight_bitwidth: dict[str, int] | None = None,
    ):
        self.search_config = search_config or ModuleSearchConfig()
        self.hardware = hardware
        self.precision_weights = precision_weights or PRECISION_SCORES
        # Per-partition weight bitwidth of a prequantized source, e.g. ``{"routed_moe": 4}``.
        # A partition absent from the map (or a ``None`` map for a float source) is
        # not filtered. Drops that partition's layer modes whose target weight is
        # higher-precision than the source (unrealizable — the source already
        # discarded that precision).
        self.source_weight_bitwidth = dict(source_weight_bitwidth) if source_weight_bitwidth else None
        if self.source_weight_bitwidth and "mlp" in self.source_weight_bitwidth:
            legacy_floor = self.source_weight_bitwidth.pop("mlp")
            self.source_weight_bitwidth.setdefault("dense_mlp", legacy_floor)
            self.source_weight_bitwidth.setdefault("routed_moe", legacy_floor)

        # Get sensitivity from search config
        base_layer_sensitivity = self.search_config.get_layer_sensitivity()

        # Filter by available_partitions if provided (prevents search space explosion)
        if available_partitions is not None:
            available_partitions = set(available_partitions)
            if "mlp" in available_partitions:
                available_partitions.remove("mlp")
                available_partitions.update(("dense_mlp", "routed_moe"))
            self.layer_sensitivity = {k: v for k, v in base_layer_sensitivity.items() if k in available_partitions}
            if "shared_expert" in available_partitions and "shared_expert" not in self.layer_sensitivity:
                parent = "routed_moe" if "routed_moe" in self.layer_sensitivity else "dense_mlp"
                if parent in self.layer_sensitivity:
                    self.layer_sensitivity["shared_expert"] = self.layer_sensitivity[parent]
        else:
            self.layer_sensitivity = base_layer_sensitivity

        self.kv_cache_sensitivity = self.search_config.kv_cache_sensitivity
        self.attention_sensitivity = self.search_config.attention_sensitivity

        # Combined sensitivity for score calculation
        self.sensitivity_weights = {
            **self.layer_sensitivity,
            "kv_cache": self.kv_cache_sensitivity,
            "attention": self.attention_sensitivity,
        }

        # Get available modes for each partition, dropping (per partition) any
        # target weight higher-precision than that partition's source weight.
        layer_modes = self._get_hardware_modes()
        self.partition_modes: dict[str, list[str]] = {}
        for name in self.layer_sensitivity:
            if name == "shared_expert":
                # shared_expert is generated from its parent expert partition below.
                continue
            bitwidth = self.source_weight_bitwidth.get(name) if self.source_weight_bitwidth else None
            self.partition_modes[name] = self._drop_modes_above_source_weight(list(layer_modes), bitwidth)
        kv_modes = getattr(self.search_config, "kv_cache_modes", None) or list(KV_CACHE_MODES)
        self.partition_modes["kv_cache"] = list(kv_modes)
        self.partition_modes["attention"] = list(ATTENTION_MODES)

        logger.debug(f"ConfigSearcher initialized. Hardware: {self.hardware}")
        logger.debug(f"Partition modes for search: {self.partition_modes}")

    def _mode_weight_bitwidth(self, mode: str) -> int | None:
        """Weight bitwidth of a layer mode, or None if it has no weight spec."""
        layer_cfg = get_layer_config(mode)  # None for native
        if layer_cfg is None:
            return None
        weight = layer_cfg.weight
        if isinstance(weight, list):
            weight = weight[0] if weight else None
        if weight is None:
            return None
        to_bitwidth = getattr(weight.dtype, "to_bitwidth", None)
        return int(to_bitwidth()) if callable(to_bitwidth) else None

    def _drop_modes_above_source_weight(self, modes: list[str], source_bitwidth: int | None) -> list[str]:
        """Drop layer modes whose target weight is higher-precision than the source.

        A target weight bitwidth strictly greater than the source weight bitwidth is
        unrealizable (the source already discarded that precision). ``native`` is
        always kept; a ``None`` source bitwidth (float / unquantized / unknown
        partition) is a no-op.
        """
        if source_bitwidth is None:
            return modes

        kept = []
        for mode in modes:
            if is_native_mode(mode):
                kept.append(mode)
                continue
            bitwidth = self._mode_weight_bitwidth(mode)
            if bitwidth is None or bitwidth <= source_bitwidth:
                kept.append(mode)

        if not any(not is_native_mode(m) for m in kept):
            logger.warning(
                "Source weight bitwidth %s dropped every quantizable layer mode; "
                "falling back to the unfiltered mode set.",
                source_bitwidth,
            )
            return modes
        return kept

    def _get_hardware_modes(self) -> list[QuantMode]:
        """Get quantization modes for layer partitions.

        Priority: search_config.layer_modes > hardware-derived modes > all modes.
        """
        explicit = getattr(self.search_config, "layer_modes", None)
        if explicit is not None:
            return list(explicit)
        if self.hardware is None:
            return list(ALL_QUANT_MODES)
        return [m for m in ALL_QUANT_MODES if m in get_supported_schemes(self.hardware)]

    def _get_mode(self, config: QuantConfig, partition: str) -> str:
        """Get mode for a partition from config dict."""
        return get_partition_mode(config, partition)

    def _shared_expert_parent(self) -> str:
        """Partition whose mode constrains the always-active shared experts."""
        return "routed_moe" if "routed_moe" in self.layer_sensitivity else "dense_mlp"

    def _default_constraints(self, config: QuantConfig) -> bool:
        """
        Check if configuration satisfies constraints.

        Constraints:
        1. Layer partitions cannot all be native (must quantize something)
        2. Precision hierarchy based on sensitivity (higher sensitivity >= lower sensitivity)
        3. shared_expert is native or uses the same mode as routed_moe
        4. kv_cache and attention are independent
        """
        pw = self.precision_weights

        # Layer partitions cannot all keep the native model behavior
        all_layers_native = all(is_native_mode(self._get_mode(config, p)) for p in self.layer_sensitivity)
        if all_layers_native:
            return False

        if "shared_expert" in self.layer_sensitivity:
            shared_expert_mode = self._get_mode(config, "shared_expert")
            parent_mode = self._get_mode(config, self._shared_expert_parent())
            if not is_native_mode(shared_expert_mode) and shared_expert_mode != parent_mode:
                return False

        # Precision hierarchy: higher sensitivity layers should have >= precision
        # Sort partitions by sensitivity (descending)
        sorted_partitions = sorted(
            ((name, weight) for name, weight in self.layer_sensitivity.items() if name != "shared_expert"),
            key=lambda x: x[1],
            reverse=True,
        )

        prev_precision = float("inf")
        for partition, _ in sorted_partitions:
            mode = self._get_mode(config, partition)
            precision = pw.get(mode, 0)
            if precision > prev_precision:
                return False  # Lower sensitivity partition has higher precision
            prev_precision = precision

        return True

    def compute_score(self, config: QuantConfig) -> float:
        """
        Compute score = sum(sensitivity * precision).

        Higher score = more conservative (higher precision).
        """
        return sum(
            self.sensitivity_weights[p] * self.precision_weights[self._get_mode(config, p)]
            for p in self.sensitivity_weights
        )

    def generate_all_configs(self) -> list[QuantConfig]:
        """Generate all valid configurations."""
        partition_order = list(self.partition_modes.keys())

        logger.debug(f"Generating configs with partition order: {partition_order}")

        configs: list[QuantConfig] = []
        for modes in product(*[self.partition_modes[p] for p in partition_order]):
            base_config: QuantConfig = {f"{p}_mode": m for p, m in zip(partition_order, modes, strict=True)}
            if "shared_expert" in self.layer_sensitivity:
                parent_mode = self._get_mode(base_config, self._shared_expert_parent())
                shared_expert_modes: list[str | None] = ["native"]
                if not is_native_mode(parent_mode):
                    source_bitwidth = (
                        self.source_weight_bitwidth.get("shared_expert") if self.source_weight_bitwidth else None
                    )
                    target_bitwidth = self._mode_weight_bitwidth(parent_mode)
                    if source_bitwidth is None or target_bitwidth is None or target_bitwidth <= source_bitwidth:
                        shared_expert_modes.append(parent_mode)
            else:
                shared_expert_modes = [None]

            for shared_expert_mode in shared_expert_modes:
                config = dict(base_config)
                if shared_expert_mode is not None:
                    config["shared_expert_mode"] = shared_expert_mode
                if self._default_constraints(config):
                    configs.append(config)
        return configs

    def generate_sorted_configs(self) -> list[QuantConfig]:
        """Generate all configs sorted from conservative to aggressive."""
        scored = [ScoredConfig(c, self.compute_score(c)) for c in self.generate_all_configs()]
        scored.sort()
        return [s.config for s in scored]

    def iter_configs(self) -> Iterator[QuantConfig]:
        """Iterate over configs from conservative to aggressive."""
        yield from self.generate_sorted_configs()

    def get_config_count(self) -> int:
        """Get total number of valid configurations."""
        return len(self.generate_all_configs())


def print_configs(
    searcher: ConfigSearcher,
    configs: list[QuantConfig] | None = None,
    limit: int = 20,
) -> None:
    """Log configurations with scores for debugging."""
    configs = configs or searcher.generate_sorted_configs()

    all_partitions = [
        *searcher.layer_sensitivity.keys(),
        *(p for p in ("kv_cache", "attention") if p in searcher.partition_modes),
    ]
    header = " ".join(f"{p.upper():<11}" for p in all_partitions)

    logger.info(f"Hardware: {searcher.hardware or 'all'} | Total: {len(configs)}")
    logger.info("-" * (20 + len(header)))
    logger.info(f"{'Rank':<5} {'Score':<7} {header}")
    logger.info("-" * (20 + len(header)))

    for i, c in enumerate(configs[:limit]):
        values = " ".join(f"{normalize_quant_mode(c.get(f'{p}_mode', 'native')):<11}" for p in all_partitions)
        logger.info(f"{i + 1:<5} {searcher.compute_score(c):<7.1f} {values}")

    if len(configs) > limit:
        logger.info(f"\n... ({len(configs) - limit} more)")
