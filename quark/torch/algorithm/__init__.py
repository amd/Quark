#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quantization algorithms for Quark's PyTorch backend."""

from quark.torch.algorithm.algorithm import QuarkAlgorithm
from quark.torch.algorithm.registry import ALGORITHM_REGISTRY

__all__ = [
    "ALGORITHM_REGISTRY",
    "QuarkAlgorithm",
]
