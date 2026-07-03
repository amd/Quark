#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field


class PassConfig(PydanticBaseModel):
    config: dict[str, Any] = Field(
        description=(
            "The configuration of the pass. Values for required parameters must be provided. For optional parameters, "
            "default values will be used if not provided."
        )
    )


class PassConfigParam:
    def __init__(self, type_: type, default_value: Any = None, required: bool = False, description: str = "") -> None:
        self.type_ = type_
        self.default_value = default_value
        self.required = required
        self.description = description

    def __repr__(self) -> str:
        return (
            f"PassConfigParam(type_={self.type_.__name__}, "
            f"default_value={self.default_value}, "
            f"required={self.required}, "
            f"description='{self.description}')"
        )
