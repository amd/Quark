# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

from .. import LlmModel, LlmModelBase
from . import register_model


@register_model
class Gemma3_4B_Text(LlmModelBase):
    model_type = LlmModel.GEMMA3_4B_TEXT

    def postprocess_attributes(self) -> None:
        attributes = super().postprocess_attributes()
        attributes["input_name"] = "inputs_embeds"
        return attributes

    def _override_token_fusion_attributes(self):
        self._set_attribute("pdi_id", 1)
        self._set_attribute("input_name", "inputs_embeds")

    def _override_prefill_fusion_attributes(self):
        self._set_attribute("input_name", "inputs_embeds")

    @classmethod
    def matches(cls, input_path: str) -> bool:
        return "gemma-3-4b-vision" in input_path and Path(input_path).stem.endswith("model")
