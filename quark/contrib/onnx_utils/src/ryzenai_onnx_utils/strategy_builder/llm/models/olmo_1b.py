# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

from .. import LlmModel, LlmModelBase
from . import register_model


@register_model
class Olmo_1B(LlmModelBase):
    model_type = LlmModel.OLMO_1B
    default_max_seq_len = 2048

    @classmethod
    def matches(cls, input_path: str) -> bool:
        return cls.model_type.value in input_path
