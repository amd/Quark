#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Registry for preprocess: maps HF module type -> Quark/replacement class."""

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


# HF module type -> replacement class (QuarkXxx or DbrxExperts_).
PREPROCESS_REGISTRY: dict[type[nn.Module], type[nn.Module]] = {}


def register_quark_preprocess(hf_module_type: type[nn.Module]) -> Callable[[type[Any]], type[Any]]:
    """Decorator to register a Quark class for hf_module_type. model_type inferred from module path."""

    def decorator(cls: type[Any]) -> type[Any]:
        """Register cls in PREPROCESS_REGISTRY for hf_module_type and return cls unchanged."""
        PREPROCESS_REGISTRY[hf_module_type] = cls
        return cls

    return decorator
