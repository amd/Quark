#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .engine_config import EngineConfig
from .model_config import ONNXModelConfig, PytorchModelConfig


class RunConfig(BaseModel):
    """Run configuration for the Shapeshifter workflow.

    This is the top-level configuration and includes configurations for input model,
    engine, passes, and output path.

    The input_model_config field accepts either ONNXModelConfig or PytorchModelConfig,
    which are automatically distinguished by the model_type discriminator field.
    """

    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility

    input_model_config: ONNXModelConfig | PytorchModelConfig | None = Field(
        default=None,
        discriminator="model_type",
        description="Input model configuration. Use ONNXModelConfig for ONNX models, "
        "PytorchModelConfig for PyTorch models. Optional for in-memory PyTorch workflows.",
    )

    engine: EngineConfig = Field(
        default_factory=EngineConfig,
        description="Engine configuration. If not provided, uses default engine configuration.",
    )

    passes: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Pass configurations. Key: pass name, Value: pass config dict."
    )

    output_model_path: str | None = Field(default=None, description="Output model path for saving.")
