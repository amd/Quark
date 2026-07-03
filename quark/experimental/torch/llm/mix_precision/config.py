#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Configuration for mixed precision quantization.

This module defines configurations for automatic mixed precision search
across different granularities and hardware targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from quark.torch.quantization.config.config import QLayerConfig
from quark.torch.quantization.config.template import (
    FP8Scheme,
    MXFP4_FP8Scheme,
    MXFP4Scheme,
    MXFP6E2M3Scheme,
    PTPCFP8Scheme,
)

# =============================================================================
# Enums
# =============================================================================


class SearchGranularity(Enum):
    """
    Defines the search granularity for mixed precision quantization.

    Attributes:
        MODULE: Search over module types (self_attn, mlp, kv_cache, attention).
                All decoder layers share the same configuration.
        DECODER_LAYER: (Future) Different configs per decoder layer.
        LINEAR_LAYER: (Future) Per-linear layer configuration.
    """

    MODULE = "module"
    DECODER_LAYER = "decoder_layer"
    LINEAR_LAYER = "linear_layer"


class HardwareTarget(Enum):
    """
    Hardware target for quantization.

    Different hardware supports different quantization schemes:
        - MI300/MI325: fp8, ptpc_fp8
        - MI355: fp8, ptpc_fp8, mxfp4, mxfp4_fp8, mxfp6_e2m3
    """

    MI300 = "mi300"
    MI325 = "mi325"
    MI355 = "mi355"


class MetricType(Enum):
    """Evaluation metric types."""

    PPL = "ppl"  # Perplexity (lower is better)
    GSM8K = "gsm8k"  # GSM8K accuracy (higher is better)


# Type alias for quantization modes
# NOTE: `native` means "keep the incoming model in its native runtime form".
# For self_attn/mlp this means following the original model path (which may be
# pre-quantized FP8). For kv_cache/attention this means "do not quantize" and
# therefore stay in bf16.
QuantMode = Literal["native", "fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8", "mxfp6_e2m3"]
OutputQuantMode = Literal["native", "fp8"]

# =============================================================================
# Hardware Support
# =============================================================================
HARDWARE_SCHEMES: dict[HardwareTarget, list[QuantMode]] = {
    HardwareTarget.MI300: ["native", "fp8", "ptpc_fp8"],
    HardwareTarget.MI325: ["native", "fp8", "ptpc_fp8"],
    HardwareTarget.MI355: ["native", "fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8", "mxfp6_e2m3"],
}

ALL_QUANT_MODES: list[QuantMode] = ["native", "fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8", "mxfp6_e2m3"]
KV_CACHE_MODES: list[OutputQuantMode] = ["native", "fp8"]
ATTENTION_MODES: list[OutputQuantMode] = [
    "native"
]  # for now, only native/bf16 is supported; fp8 attention is not supported in vLLM


def normalize_quant_mode(mode: str | None) -> str:
    """Normalize legacy aliases to the user-facing mixed-precision mode names."""
    if mode in {"native", "original", "bf16"}:
        return "native"
    return "" if mode is None else mode


def is_native_mode(mode: str | None) -> bool:
    return normalize_quant_mode(mode) == "native"


def get_supported_schemes(hardware: HardwareTarget | str | None) -> list[QuantMode]:
    """Get supported quantization schemes for hardware target."""
    if hardware is None:
        return ALL_QUANT_MODES
    if isinstance(hardware, str):
        hardware = HardwareTarget(hardware.lower())
    return HARDWARE_SCHEMES.get(hardware, ALL_QUANT_MODES)


# =============================================================================
# Precision Scores (for sorting)
# =============================================================================

# TODO: for int4/fp8 prequantized models, this score needs to be updated
PRECISION_SCORES: dict[str, int] = {
    "native": 10,
    "ptpc_fp8": 9,
    "fp8": 8,
    "mxfp6_e2m3": 6,
    "mxfp4_fp8": 4,
    "mxfp4": 2,
}


# =============================================================================
# Sensitivity Configurations
# =============================================================================

"""
Default partition sensitivity weights for mixed precision search.

Sensitivity Scale
-----------------
Higher values = more sensitive to quantization = prefer higher precision.

Partition Definitions
---------------------
- `linear_attn` (3): Linear attention layers (Qwen 3.5, etc.)
  Uses linear projections instead of traditional QKV attention

- `self_attn` (3): Traditional QKV + O projections (Llama, etc.)
  Standard transformer self-attention mechanisms

- `mlp` (1): FFN / MoE layers
  Feed-forward networks and mixture-of-experts blocks

Backward Compatibility
----------------------
For traditional models (Llama, Mistral, etc.):
  - Only `self_attn` and `mlp` are detected by layer pattern matching
  - `linear_attn` is filtered out during ConfigSearcher initialization
  - No change to existing behavior or search space

For hybrid models (Qwen 3.5):
  - All three partitions are detected
  - Search space includes configurations for all partitions
  - Example: linear_attn=fp8, self_attn=fp8, mlp=native

Custom Sensitivity
------------------
Users can override via MixPrecisionConfig:

    config = MixPrecisionConfig(
        module_search_config=ModuleSearchConfig(
            layer_sensitivity={"self_attn": 5, "mlp": 2}
        )
    )
"""
DEFAULT_PARTITION_SENSITIVITY: dict[str, int | float] = {
    "linear_attn": 3,  # Linear attention (Qwen 3.5, etc.)
    "self_attn": 3,  # QKV + O projections
    "mlp": 1,  # FFN / MoE layers
}

# Independent output partition sensitivity
KV_CACHE_SENSITIVITY: int = 2
ATTENTION_SENSITIVITY: int = 2


# =============================================================================
# Quantization Configuration Type and Helpers
# =============================================================================

# QuantConfig is a dict mapping partition names to quantization modes
# Example: {"self_attn_mode": "fp8", "mlp_mode": "native", "kv_cache_mode": "native", "attention_mode": "native"}
QuantConfig = dict[str, str]


def create_quant_config(
    layer_partitions: dict[str, str] | None = None,
    kv_cache_mode: str = "native",
    attention_mode: str = "native",
) -> QuantConfig:
    """
    Create a quantization configuration dict.

    Args:
        layer_partitions: Dict of partition -> mode, e.g., {"self_attn": "fp8", "mlp": "native"}
        kv_cache_mode: KV cache quantization mode
        attention_mode: Attention quantization mode

    Returns:
        QuantConfig dict with all partition modes
    """
    config: QuantConfig = {}
    if layer_partitions:
        for part, mode in layer_partitions.items():
            config[f"{part}_mode"] = normalize_quant_mode(mode)
    config["kv_cache_mode"] = normalize_quant_mode(kv_cache_mode)
    config["attention_mode"] = normalize_quant_mode(attention_mode)
    return config


def is_config_all_native(config: QuantConfig, layer_partitions: list[str]) -> bool:
    """Check if all layer partition modes keep the native model behavior."""
    return all(is_native_mode(config.get(f"{p}_mode")) for p in layer_partitions)


def is_config_all_bf16(config: QuantConfig, layer_partitions: list[str]) -> bool:
    """Backward-compatible alias for the legacy name."""
    return is_config_all_native(config, layer_partitions)


def is_config_valid_for_hardware(
    config: QuantConfig,
    layer_partitions: list[str],
    hardware: HardwareTarget | str,
) -> bool:
    """Check if config is valid for hardware target."""
    supported = get_supported_schemes(hardware)
    for p in layer_partitions:
        mode = normalize_quant_mode(config.get(f"{p}_mode", "native"))
        if not is_native_mode(mode) and mode not in supported:
            return False
    return True


def config_to_str(config: QuantConfig) -> str:
    """Convert config dict to readable string."""
    parts = [f"{k}={v}" for k, v in sorted(config.items())]
    return ", ".join(parts)


# =============================================================================
# Granularity-Specific Search Configurations
# =============================================================================


@dataclass
class ModuleSearchConfig:
    """
    Search configuration for MODULE granularity.

    When using MODULE granularity, all decoder layers share the same
    quantization configuration. Search is performed over module partitions.

    Attributes:
        layer_sensitivity: Sensitivity weights for layer partitions.
                          Default: {"self_attn": 3, "mlp": 1}
        kv_cache_sensitivity: Sensitivity weight for KV cache quantization
        attention_sensitivity: Sensitivity weight for attention quantization

    Example:
        # Default partitioning (self_attn, mlp)
        config = ModuleSearchConfig()

        # Custom sensitivity weights
        config = ModuleSearchConfig(
            layer_sensitivity={"self_attn": 5, "mlp": 2}
        )
    """

    layer_sensitivity: dict[str, int | float] | None = None
    kv_cache_sensitivity: int = KV_CACHE_SENSITIVITY
    attention_sensitivity: int = ATTENTION_SENSITIVITY

    # Restrict which modes are searched for layer partitions (self_attn, mlp, etc.).
    # None = all modes supported by hardware target.
    # Example: ["native", "ptpc_fp8", "mxfp4"] to search only those three.
    layer_modes: list[str] | None = None
    kv_cache_modes: list[str] | None = None  # None = KV_CACHE_MODES; ["native"] = no kv cache quant

    def get_layer_sensitivity(self) -> dict[str, int | float]:
        """Get layer sensitivity (user-provided or default)."""
        if self.layer_sensitivity is not None:
            return self.layer_sensitivity
        return DEFAULT_PARTITION_SENSITIVITY


@dataclass
class DecoderLayerSearchConfig:
    """
    Search configuration for DECODER_LAYER granularity (Future).

    Each decoder layer or layer group can have different quantization settings.
    """

    pass


@dataclass
class LinearLayerSearchConfig:
    """
    Search configuration for LINEAR_LAYER granularity (Future).

    Each linear layer can have its own quantization mode.
    Reference: NVIDIA Model-Optimizer approach.
    """

    pass


# =============================================================================
# Main Configuration
# =============================================================================


@dataclass
class MixPrecisionConfig:
    """
    Main configuration for mixed precision quantization.

    This configuration controls the search algorithm, evaluation metrics,
    and hardware constraints for finding optimal quantization settings.

    Attributes:
        granularity: Search granularity level (MODULE, DECODER_LAYER, LINEAR_LAYER)
        hardware: Target hardware (affects available schemes)

        eval_metrics: List of metrics to evaluate (ppl, gsm8k)
        eval_threshold: Metric-specific threshold ratio:
            - ppl: maximum allowed multiplicative increase over original_ppl
            - gsm8k: degradation ratio (e.g., 1.02 = allow 2% drop)
        eval_num_samples: Number of samples for evaluation
        eval_batch_size: Batch size for evaluation
        eval_max_new_tokens: Maximum new tokens to generate for tasks like GSM8K
        eval_max_length: Maximum length of the input sequence
        eval_stride: Stride for the input sequence

        num_calib_samples: Number of samples for calibration
        calib_seq_len: Sequence length for calibration

        min_kv_scale: Minimum allowed kv-cache scale (forwarded to QConfig)

        max_configs: Maximum configs to evaluate (None = all)
        early_stop: Stop when threshold exceeded

        module_search_config: Config for MODULE granularity search
        decoder_layer_config: Config for DECODER_LAYER granularity (future)
        linear_layer_config: Config for LINEAR_LAYER granularity (future)

    Example:
        # Default module search
        config = MixPrecisionConfig(
            hardware=HardwareTarget.MI300,
            eval_metrics=["ppl"],
            eval_threshold=1.02,  # Allow PPL up to original_ppl * 1.02
        )

        # Custom sensitivity weights
        config = MixPrecisionConfig(
            hardware=HardwareTarget.MI300,
            module_search_config=ModuleSearchConfig(
                layer_sensitivity={"self_attn": 5, "mlp": 2}
            ),
        )
    """

    # Search settings
    granularity: SearchGranularity = SearchGranularity.MODULE
    hardware: HardwareTarget | None = None

    # Evaluation settings
    eval_metrics: list[str] = field(default_factory=lambda: ["ppl"])
    eval_threshold: float = 1.02  # Default: allow PPL up to original_ppl * 1.02
    eval_num_samples: int = 200
    eval_batch_size: int = 1
    eval_max_new_tokens: int = 512  # For generation tasks like GSM8K
    eval_max_length: int = 2048
    eval_stride: int = 512

    # Calibration settings
    num_calib_samples: int = 128
    calib_seq_len: int = 2048
    min_kv_scale: float = 0.0
    exclude_patterns: list[str] | None = None

    # Search algorithm settings
    max_configs: int | None = None
    early_stop: bool = True
    seed: int = 42

    # Granularity-specific search configs (mutually exclusive based on granularity)
    module_search_config: ModuleSearchConfig | None = None
    decoder_layer_config: DecoderLayerSearchConfig | None = None
    linear_layer_config: LinearLayerSearchConfig | None = None

    def get_search_config(self) -> ModuleSearchConfig | DecoderLayerSearchConfig | LinearLayerSearchConfig:
        """Get the search config for the selected granularity."""
        if self.granularity == SearchGranularity.MODULE:
            return self.module_search_config or ModuleSearchConfig()
        elif self.granularity == SearchGranularity.DECODER_LAYER:
            if self.decoder_layer_config is None:
                raise ValueError("decoder_layer_config required for DECODER_LAYER granularity")
            return self.decoder_layer_config
        elif self.granularity == SearchGranularity.LINEAR_LAYER:
            if self.linear_layer_config is None:
                raise ValueError("linear_layer_config required for LINEAR_LAYER granularity")
            return self.linear_layer_config
        raise ValueError(f"Unknown granularity: {self.granularity}")

    def validate(self) -> None:
        """Validate configuration."""
        # Check granularity support
        if self.granularity == SearchGranularity.DECODER_LAYER:
            raise NotImplementedError("DECODER_LAYER granularity not yet implemented. Use MODULE granularity for now.")
        if self.granularity == SearchGranularity.LINEAR_LAYER:
            raise NotImplementedError("LINEAR_LAYER granularity not yet implemented. Use MODULE granularity for now.")

        # Validate metrics
        valid_metrics = [m.value for m in MetricType]
        for metric in self.eval_metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric '{metric}'. Valid options: {valid_metrics}")

        # Validate threshold semantics by metric.
        if self.eval_threshold < 1.0:
            raise ValueError("eval_threshold must be >= 1.0")

        if self.min_kv_scale < 0.0:
            raise ValueError("min_kv_scale must be >= 0.0")


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class ConfigEvalResult:
    """
    Evaluation result for a single configuration.

    The config type depends on search granularity:
    - MODULE: QuantConfig (dict[str, str])
    - DECODER_LAYER: dict[int, QuantConfig] (future)
    - LINEAR_LAYER: dict[str, str] (future)
    """

    config: QuantConfig
    metrics: dict[str, float]  # {"ppl": 5.23, "gsm8k": 0.45}
    relative_change: dict[str, float]  # {"ppl": 0.02, "gsm8k": -0.01}
    is_valid: bool  # Within threshold
    rank: int  # Position in search order


@dataclass
class SearchResult:
    """
    Result of the mixed precision search.

    The config type depends on search granularity.
    """

    best_config: QuantConfig | None
    all_results: list[ConfigEvalResult]
    baseline_metrics: dict[str, float]
    total_configs_evaluated: int
    total_configs_available: int
    search_time_seconds: float
    granularity: SearchGranularity
    hardware: HardwareTarget | None


# =============================================================================
# Scheme Mapping (using Quark templates)
# =============================================================================


def get_layer_config(mode: QuantMode) -> QLayerConfig | None:
    """
    Get QLayerConfig for a quantization mode.

    Uses Quark's built-in scheme definitions from template.py.
    """
    if is_native_mode(mode):
        return None

    scheme_map = {
        "fp8": FP8Scheme(),
        "ptpc_fp8": PTPCFP8Scheme(),
        "mxfp4": MXFP4Scheme(),
        "mxfp4_fp8": MXFP4_FP8Scheme(),
        "mxfp6_e2m3": MXFP6E2M3Scheme(),
    }

    scheme = scheme_map.get(mode)
    return scheme.config if scheme else None


__all__ = [
    # Enums
    "SearchGranularity",
    "HardwareTarget",
    # Type alias
    "QuantConfig",
    # Search config
    "ModuleSearchConfig",
    # Main config
    "MixPrecisionConfig",
    # Result classes
    "ConfigEvalResult",
    "SearchResult",
]
