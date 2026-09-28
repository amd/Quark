#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark quantization hooks for the target (verifier) and optional draft quant."""

from quark.experimental.speculative_decoding.quant.integrate import load_target_verifier

__all__ = ["load_target_verifier"]
