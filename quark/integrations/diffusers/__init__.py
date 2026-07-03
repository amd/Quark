#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from diffusers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING, AUTO_QUANTIZER_MAPPING

from .quantization_config import QuarkQuantizationConfig
from .quantizer import QuarkDiffusersQuantizer

AUTO_QUANTIZER_MAPPING["quark"] = QuarkDiffusersQuantizer  # type: ignore
AUTO_QUANTIZATION_CONFIG_MAPPING["quark"] = QuarkQuantizationConfig  # type: ignore

__all__ = ["QuarkDiffusersQuantizer", "QuarkQuantizationConfig"]
