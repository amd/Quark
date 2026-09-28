#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Re-export AutoMixprecisionConfig from the canonical location.

The class is defined in ``quark.onnx.quantization.config.algorithm`` to avoid
circular-import issues (the algorithm package depends on calibration, which
depends on quantization). This module provides the canonical import path for
the new AMP redesign without duplicating the class body.
"""

from __future__ import annotations

from quark.onnx.quantization.config.algorithm import AutoMixprecisionConfig

__all__ = ["AutoMixprecisionConfig"]
