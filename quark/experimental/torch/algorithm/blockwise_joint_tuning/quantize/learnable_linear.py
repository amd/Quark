#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import WeightGroupQuantizer


class ExperimentalLearnableQuantizedLinear(nn.Module):
    """Linear layer wrapper with optional per-group learnable weight quantization.

    Wraps an existing ``nn.Linear`` module and attaches a learnable
    :class:`WeightGroupQuantizer` to its weight. Scale and zero_point are
    registered as trainable parameters for QAT fine-tuning. Weight quantization
    can be toggled at runtime via :meth:`enable_weight_quant`.

    Args:
        source_module: The original ``nn.Linear`` whose weight and bias are
            adopted by this module.
        num_bits: Quantization bit-width passed to :class:`WeightGroupQuantizer`.
        group_size: Group size passed to :class:`WeightGroupQuantizer`.
    """

    def __init__(self, source_module: nn.Linear, num_bits: int = 4, group_size: int = 64) -> None:
        super().__init__()
        self.in_features = source_module.in_features
        self.out_features = source_module.out_features
        self.register_parameter("weight", source_module.weight)
        if source_module.bias is not None:
            self.register_buffer("bias", source_module.bias)
        else:
            self.bias = None
        self.weight_quant_enabled = False
        self.weight_quantizer = WeightGroupQuantizer(
            num_bits=num_bits,
            group_size=group_size,
            weight=source_module.weight,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        W = self.weight_quantizer(self.weight) if self.weight_quant_enabled else self.weight
        return F.linear(X, W, self.bias)

    def enable_weight_quant(self, enabled: bool = True) -> None:
        self.weight_quant_enabled = enabled

    def disable_weight_quant(self) -> None:
        self.enable_weight_quant(False)
