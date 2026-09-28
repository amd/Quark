#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import copy
from typing import cast

from quark.torch.quantization.config.config import QLayerConfig
from quark.torch.quantization.config.template import LLMTemplate

from .errors import SchemaValidationError
from .mixed_precision_strategy import HardwareTarget

_MI355_SCHEMES = ("native", "fp8", "ptpc_fp8")
_SUPPORTED_DENSE_MODEL_TYPES = ("qwen3", "llama")


def get_supported_schemes(target: HardwareTarget, model_type: str) -> tuple[str, ...]:
    """Return schemes supported by both the MVP target and Quark."""
    if target is not HardwareTarget.MI355:
        raise SchemaValidationError(f"Unsupported hardware target: {target}.")
    if model_type not in _SUPPORTED_DENSE_MODEL_TYPES:
        supported = ", ".join(_SUPPORTED_DENSE_MODEL_TYPES)
        raise SchemaValidationError(f"MVP only supports dense model types ({supported}), got {model_type!r}.")

    template = LLMTemplate.get(model_type)
    registered = set(template.get_supported_schemes())
    missing = sorted(set(_MI355_SCHEMES) - {"native"} - registered)
    if missing:
        raise SchemaValidationError(f"Quark does not provide the required MI355 schemes: {missing}.")
    return _MI355_SCHEMES


def get_scheme_config(model_type: str, scheme: str) -> QLayerConfig:
    """Resolve one MI355 scheme through Quark's template registry."""
    if scheme == "native":
        return QLayerConfig()
    if scheme not in _MI355_SCHEMES:
        raise SchemaValidationError(f"Unsupported MI355 scheme: {scheme!r}.")
    template = LLMTemplate.get(model_type)
    config = template.get_config(scheme, exclude_layers=[], shared_scale_groups=[])
    return cast(QLayerConfig, copy.deepcopy(config.global_quant_config))


__all__ = ["get_scheme_config", "get_supported_schemes"]
