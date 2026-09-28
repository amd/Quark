#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from quark.torch.quantization.config.config import QTensorConfig
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase

from ._serialization import StrictSchema
from .errors import SchemaValidationError
from .hardware_capability import get_scheme_config


@dataclass(frozen=True, slots=True)
class WeightMseScore(StrictSchema):
    squared_error: float
    signal_power: float
    relative_mse: float
    weighted_score: float

    def __post_init__(self) -> None:
        values = (self.squared_error, self.signal_power, self.relative_mse, self.weighted_score)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise SchemaValidationError("Weight-MSE values must be finite and non-negative.")


def quantize_weight_for_profile(weight: torch.Tensor, scheme: str, *, model_type: str) -> torch.Tensor:
    """Apply the same Quark weight QDQ recipe used by QuantLinear."""
    if scheme == "native":
        return weight.detach().clone()
    layer_config = get_scheme_config(model_type, scheme)
    weight_spec = layer_config.weight
    if not isinstance(weight_spec, QTensorConfig):
        raise SchemaValidationError(f"MVP scheme {scheme!r} must have one weight QTensorConfig.")
    quantizer = FakeQuantizeBase.get_fake_quantize(weight_spec, device=weight.device)
    with torch.no_grad():
        quantized = quantizer(weight.detach().contiguous())
    if not isinstance(quantized, torch.Tensor):
        raise SchemaValidationError(f"Quark QDQ for {scheme!r} did not return a tensor.")
    return quantized


def calculate_weight_mse(
    weight: torch.Tensor,
    scheme: str,
    *,
    total_quantizable_params: int,
    model_type: str,
) -> WeightMseScore:
    if total_quantizable_params <= 0:
        raise SchemaValidationError("total_quantizable_params must be positive.")
    if scheme == "native":
        return WeightMseScore(0.0, float(weight.detach().float().square().sum()), 0.0, 0.0)

    quantized = quantize_weight_for_profile(weight, scheme, model_type=model_type)
    reference = weight.detach().float()
    difference = reference - quantized.float()
    squared_error = float(difference.square().sum())
    signal_power = float(reference.square().sum())
    if signal_power == 0.0:
        if squared_error != 0.0:
            raise SchemaValidationError(f"QDQ scheme {scheme!r} changed an all-zero weight.")
        relative_mse = 0.0
    else:
        relative_mse = squared_error / signal_power
    weighted_score = relative_mse * weight.numel() / total_quantizable_params
    return WeightMseScore(squared_error, signal_power, relative_mse, weighted_score)


__all__ = ["WeightMseScore", "calculate_weight_mse", "quantize_weight_for_profile"]
