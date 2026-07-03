#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Algorithm Config API for PyTorch.

This module hosts algorithm-level configs shared across features (e.g. pruning).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quark.common.config import BaseAlgoConfig


@dataclass
class BlockwiseTuningConfig(BaseAlgoConfig):
    name: str = "blockwise_tuning"
    epochs: int = 5
    weight_lr: float = 0.0001
    weight_decay: float = 0.0
    min_lr_factor: float = 20.0
    max_grad_norm: float = 0.3
    model_decoder_layers: str = field(default_factory=str)
    trainable_modules: list[str] = field(default_factory=list)


@dataclass
class BlockwiseJointTuningConfig(BaseAlgoConfig):
    """Config for blockwise_joint_tuning.

    Difference from `blockwise_tuning`:
    - `blockwise_tuning`: tunes module parameters selected by `trainable_modules`.
    - `blockwise_joint_tuning`: is intended for joint optimization of module params
      and quant-related params (e.g. scale/zero_point) with separate controls.
    """

    name: str = "blockwise_joint_tuning"
    epochs: int = 5
    weight_lr: float = 0.0001
    qparam_lr: float = 0.0001
    weight_decay: float = 0.0
    qparam_weight_decay: float = 0.0
    min_lr_factor: float = 20.0
    max_grad_norm: float = 0.3
    model_decoder_layers: str = field(default_factory=str)
    trainable_modules: list[str] = field(default_factory=list)
    quant_trainable_modules: list[str] = field(default_factory=list)


__all__ = ["BaseAlgoConfig", "BlockwiseTuningConfig", "BlockwiseJointTuningConfig"]
