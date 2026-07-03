#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn


def _ste_round(X: torch.Tensor) -> torch.Tensor:
    """Round operation with straight-through gradient estimator."""
    return (X.round() - X).detach() + X


def _ste_clamp(X: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Clamp operation with straight-through gradient estimator."""
    return (X.clamp(lower, upper) - X).detach() + X


class WeightGroupQuantizer(nn.Module):
    """Per-group uniform affine quantizer for weight tensors.

    Initializes scale and zero_point from the weight distribution and
    registers them as learnable parameters for QAT fine-tuning.

    Args:
        num_bits: Quantization bit-width. Must be in [2, 16].
        group_size: Number of elements per quantization group. Use -1 to
            treat the entire last dimension as a single group.
        weight: Reference weight tensor used to initialize quantization
            parameters. Required at construction time.
    """

    def __init__(
        self,
        num_bits: int = 8,
        group_size: int | None = None,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        assert 2 <= num_bits <= 16, f"num_bits={num_bits} is outside supported range [2, 16]"
        assert weight is not None, "A reference weight tensor is required to initialize quantization parameters"
        assert group_size is not None, "group_size must be specified"

        self.num_bits = num_bits
        self.quant_min = 0
        self.quant_max = 2**num_bits - 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]

        assert weight.shape[-1] % self.group_size == 0, (
            f"Weight last dimension ({weight.shape[-1]}) must be divisible by group_size ({self.group_size})"
        )
        self.quant_enabled = True

        with torch.no_grad():
            W = weight.reshape(-1, self.group_size)
            w_min = W.amin(dim=-1, keepdim=True)
            w_max = W.amax(dim=-1, keepdim=True)
            scale = (w_max - w_min) / (self.quant_max - self.quant_min)
            scale = scale.clamp(min=1e-4, max=1e4)
            zero_point = -(w_min / scale).clamp(min=-1e4, max=1e4)
            self.scale = nn.Parameter(scale)
            self.zero_point = nn.Parameter(zero_point.round())

    def _apply_fake_quant(self, X: torch.Tensor) -> torch.Tensor:
        scale = _ste_clamp(self.scale, 1e-4, 1e4)
        zp = _ste_clamp(_ste_round(self.zero_point), self.quant_min, self.quant_max)
        orig_shape = X.shape
        X = X.reshape(-1, self.group_size)
        X_int = _ste_round(X / scale).add(zp).clamp(self.quant_min, self.quant_max)
        X_dequant = X_int.sub(zp).mul(scale)
        return X_dequant.reshape(orig_shape)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.num_bits >= 16 or not self.quant_enabled:
            return X
        return self._apply_fake_quant(X)
