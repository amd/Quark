#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Configuration search algorithms for mixed precision quantization.

This module provides the ConfigSearcher class which generates and sorts
quantization configurations based on sensitivity and precision hierarchy.

Three-Partition Support
------------------------
The searcher now supports three primary layer partitions for hybrid models:
  - `linear_attn`: Linear attention (e.g., Qwen 3.5)
  - `self_attn`: Traditional QKV + O projections (e.g., Llama)
  - `mlp`: FFN / MoE layers

For backward compatibility:
  - Traditional models (Llama): only `self_attn` and `mlp` detected
  - Hybrid models (Qwen 3.5): all three partitions available

The ConfigSearcher automatically adapts to model architecture via
`available_partitions` filtering (set from detector results) to prevent
search space explosion when some partitions are not present.

Example:
    # Traditional model (Llama-like)
    searcher = ConfigSearcher(
        available_partitions={"self_attn", "mlp"}  # Only 2 partitions
    )

    # Hybrid model (Qwen 3.5)
    searcher = ConfigSearcher(
        available_partitions={"linear_attn", "self_attn", "mlp"}  # All 3
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
        ...     layer_sensitivity={"self_attn": 5, "mlp": 2}
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
    ):
        self.search_config = search_config or ModuleSearchConfig()
        self.hardware = hardware
        self.precision_weights = precision_weights or PRECISION_SCORES

        # Get sensitivity from search config
        base_layer_sensitivity = self.search_config.get_layer_sensitivity()

        # Filter by available_partitions if provided (prevents search space explosion)
        if available_partitions is not None:
            self.layer_sensitivity = {k: v for k, v in base_layer_sensitivity.items() if k in available_partitions}
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

        # Get available modes for each partition
        layer_modes = self._get_hardware_modes()
        self.partition_modes: dict[str, list[str]] = {}
        for name in self.layer_sensitivity:
            self.partition_modes[name] = list(layer_modes)
        kv_modes = getattr(self.search_config, "kv_cache_modes", None) or list(KV_CACHE_MODES)
        self.partition_modes["kv_cache"] = list(kv_modes)
        self.partition_modes["attention"] = list(ATTENTION_MODES)

        logger.debug(f"ConfigSearcher initialized. Hardware: {self.hardware}")
        logger.debug(f"Partition modes for search: {self.partition_modes}")

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
        return normalize_quant_mode(config.get(f"{partition}_mode", "native"))

    def _default_constraints(self, config: QuantConfig) -> bool:
        """
        Check if configuration satisfies constraints.

        Constraints:
        1. Layer partitions cannot all be native (must quantize something)
        2. Precision hierarchy based on sensitivity (higher sensitivity >= lower sensitivity)
        3. kv_cache and attention are independent
        """
        pw = self.precision_weights

        # Layer partitions cannot all keep the native model behavior
        all_layers_native = all(is_native_mode(self._get_mode(config, p)) for p in self.layer_sensitivity)
        if all_layers_native:
            return False

        # Precision hierarchy: higher sensitivity layers should have >= precision
        # Sort partitions by sensitivity (descending)
        sorted_partitions = sorted(self.layer_sensitivity.items(), key=lambda x: x[1], reverse=True)

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
            config: QuantConfig = {f"{p}_mode": m for p, m in zip(partition_order, modes, strict=True)}
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

    all_partitions = list(searcher.partition_modes.keys())
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


__all__ = ["ConfigSearcher"]
