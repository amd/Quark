# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

from .. import LlmMode, LlmModel, LlmModelBase
from . import register_model


@register_model
class GptOss(LlmModelBase):
    model_type = LlmModel.GPT_OSS

    def _override_eager_operators(self, operators: set[str]) -> set[str]:
        assert self.llm_mode == LlmMode.NPU_EAGER, "gpt-oss model only supports npu_eager mode"
        operators.add("qmoe")
        return operators

    @classmethod
    def matches(cls, input_path: str) -> bool:
        return cls.model_type.value in input_path
