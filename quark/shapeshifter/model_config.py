#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import Literal

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field


class ModelConfig(PydanticBaseModel):
    """Base configuration for model specification.

    This is the base class for model configurations. Use ONNXModelConfig
    for ONNX models or PytorchModelConfig for PyTorch models.
    """

    input_model_path: Path = Field(description="Path to input model file.")


class ONNXModelConfig(ModelConfig):
    """Configuration for ONNX model specification.

    Use this for ONNX models (.onnx files). This class currently has no
    additional fields beyond the base ModelConfig, but provides type distinction
    and allows future ONNX-specific configuration options.
    """

    model_type: Literal["onnx"] = Field(default="onnx", description="Model type discriminator for Pydantic.")


class PytorchModelConfig(ModelConfig):
    """Configuration for PyTorch model specification.

    Use this for PyTorch models (.pt, .pth, .bin files). Provides PyTorch-specific
    loading configuration options.
    """

    model_type: Literal["pytorch"] = Field(default="pytorch", description="Model type discriminator for Pydantic.")

    weights_only: bool = Field(
        default=False,
        description="If True, only load state_dict (more secure but restrictive). "
        "If False, load full model object using pickle (required for nn.Module objects). "
        "Note: PyTorch 2.6+ defaults to True for security, but loading nn.Module requires False.",
    )

    map_location: str = Field(
        default="cpu",
        description="Device to load model on. Examples: 'cpu', 'cuda', 'cuda:0', etc. "
        "Controls where tensors are loaded in memory.",
    )
