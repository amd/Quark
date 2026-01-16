#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pydantic import BaseModel, ConfigDict, Field

from .engine_config import EngineConfig
from .model_config import ModelConfig
from .pass_config import PassConfig


class RunConfig(BaseModel):
    """Run configuration for the workflow.

    This is the top-level configuration and includes configurations for input model, data, engine, passes, etc.
    """

    model_config = ConfigDict(extra="forbid")

    input_model_config: ModelConfig = Field(description="Input model configuration.")
    engine: EngineConfig = Field(
        default_factory=EngineConfig,
        description=("Engine configuration. If not provided, the workflow uses the default engine configuration."),
    )
    passes: dict[str, list[PassConfig]] = Field(default_factory=dict, description="Pass configurations.")
