#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from contextlib import suppress

# Import passes module to trigger automatic registration of all passes
# This ensures passes are registered before Engine.initialize() is called
from . import passes  # noqa: F401

# Import community passes if available (optional)
with suppress(ImportError):
    from quark.contrib import shapeshifter_community_passes  # noqa: F401

from .api import shapeshifter
from .engine import Engine
from .model_config import ModelConfig, ONNXModelConfig, PytorchModelConfig
from .pass_base import ONNXPass, PytorchPass
from .run_config import RunConfig
from .utils import LoadConfigFromFileOrDict

__all__ = [
    "ONNXPass",
    "PytorchPass",
    "Engine",
    "LoadConfigFromFileOrDict",
    "shapeshifter",
    "RunConfig",
    "ModelConfig",
    "ONNXModelConfig",
    "PytorchModelConfig",
]
