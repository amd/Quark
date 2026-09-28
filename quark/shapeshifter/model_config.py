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
        default=True,
        description="If True (default), only load tensors/state_dict, matching PyTorch 2.6+'s "
        "secure default that refuses to unpickle arbitrary objects. If False, load the full "
        "model object using pickle (required to load nn.Module objects, but executes arbitrary "
        "code on load -- only use it for files you trust). Loading a full nn.Module fails under "
        "the safe default; set this to False explicitly to opt in.",
    )

    map_location: str = Field(
        default="cpu",
        description="Device to load model on. Examples: 'cpu', 'cuda', 'cuda:0', etc. "
        "Controls where tensors are loaded in memory.",
    )
