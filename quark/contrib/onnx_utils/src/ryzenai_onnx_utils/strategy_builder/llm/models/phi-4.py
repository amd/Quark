# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

from .. import LlmModel, LlmModelBase
from . import register_model


@register_model
class Phi4(LlmModelBase):
    model_type = LlmModel.PHI_4

    def _eager(self) -> None:
        super()._eager()
        self.genai_config.set_option("hybrid_opt_npu_pdi_name", "DPU_9")

    def _override_prefill_fusion_attributes(self):
        self._set_attribute("pdi_id", 9)

    def _override_eager_operators(self, operators: set[str]) -> set[str]:
        # the conv in this graph transforms into NhwcConv which we don't support
        # in eager for this model nor do we have a good pass to revert it back to conv
        operators.discard("conv")
        return operators

    @classmethod
    def matches(cls, input_path: str) -> bool:
        return cls.model_type.value in input_path
