#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field


class ModelConfig(PydanticBaseModel):
    input_model_path: Path = Field(description="Path to input model.")
