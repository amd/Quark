#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field


class EngineConfig(PydanticBaseModel):
    output_dir: Path | str | None = Field(None, description="Path where final output get saved.")
    log_severity_level: None = None  # deprecated. TODO: remove.
