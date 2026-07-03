#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Quantization switching for mixed precision.

This module provides utilities to dynamically apply quantization configurations
to models without full reloading. Supports both dynamic (no calibration) and
static (with calibration) quantization modes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    HardwareTarget,
    QuantConfig,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


# Modes that require calibration (static quantization)
STATIC_MODES = {"fp8", "mxfp4_fp8"}

# Modes that don't require calibration (dynamic quantization)
DYNAMIC_QUANT_MODES = {"native", "ptpc_fp8", "mxfp4", "mxfp6_e2m3"}


def needs_calibration(config: QuantConfig) -> bool:
    """
    Check if configuration requires calibration.

    Static quantization modes (fp8 per-tensor) require calibration,
    while dynamic modes (ptpc_fp8, mxfp4) do not.
    """
    return any(mode in STATIC_MODES for key, mode in config.items() if key.endswith("_mode"))


def apply_quant_config(
    model: nn.Module,
    config: QuantConfig,
    tokenizer: PreTrainedTokenizer | None = None,
    calib_dataloader: DataLoader | None = None,
    num_calib_samples: int = 128,
    calib_seq_len: int = 2048,
    hardware: HardwareTarget | str | None = None,
    layer_sensitivity: dict[str, int] | None = None,
    min_kv_scale: float = 0.0,
    exclude_patterns: list[str] | None = None,
) -> nn.Module:
    """
    Apply quantization configuration, choosing method based on config.

    Automatically selects:
    - Dynamic application for dynamic-only modes (ptpc_fp8, mxfp4, etc.)
    - Static application with calibration for static modes (fp8)

    Args:
        model: Model to quantize
        config: Quantization configuration dict
        tokenizer: Tokenizer (required for calibration)
        calib_dataloader: Optional calibration dataloader
        num_calib_samples: Number of calibration samples
        calib_seq_len: Sequence length for calibration
        hardware: Hardware target
        layer_sensitivity: Layer sensitivity for partition mapping
        min_kv_scale: Minimum kv-cache scale (forwarded to QConfig)
        exclude_patterns: Optional layer-name patterns to exclude from quantization

    Returns:
        Quantized model
    """
    from quark.torch.quantization.api import ModelQuantizer

    from .utils import create_qconfig_from_quant_config

    # Check if calibration is needed and validate inputs
    requires_calib = needs_calibration(config)
    if requires_calib:
        if tokenizer is None and calib_dataloader is None:
            raise ValueError("Tokenizer or calib_dataloader required for static quantization modes")

    # Create QConfig from QuantConfig
    qconfig = create_qconfig_from_quant_config(
        model,
        config,
        layer_sensitivity,
        exclude_patterns=exclude_patterns,
        min_kv_scale=min_kv_scale,
    )

    if requires_calib:
        # Static quantization with calibration
        logger.info(f"Applying static config with calibration: {config}")

        # Create calibration dataloader if needed
        if calib_dataloader is None:
            from quark.torch.utils.llm.data_preparation import get_calib_dataloader

            device = next(model.parameters()).device
            calib_dataloader = get_calib_dataloader(
                tokenizer=tokenizer,
                dataset_name="pileval",
                seqlen=calib_seq_len,
                num_calib_data=num_calib_samples,
                device=device,
            )

        # Use ModelQuantizer with calibration
        quantizer = ModelQuantizer(qconfig)
        model = quantizer.quantize_model(model, calib_dataloader)
    else:
        # Dynamic quantization without calibration
        logger.info(f"Applying dynamic config: {config}")

        # Use ModelQuantizer without calibration dataloader
        quantizer = ModelQuantizer(qconfig)
        model = quantizer.quantize_model(model)

    return model


__all__ = [
    "apply_quant_config",
]
