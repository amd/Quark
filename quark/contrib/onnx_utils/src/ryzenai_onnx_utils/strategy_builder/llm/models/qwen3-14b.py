# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

from .. import LlmModel, LlmModelBase
from . import register_model


@register_model
class Qwen3_14B(LlmModelBase):
    model_type = LlmModel.QWEN3_14B

    def _override_prefill_fusion_attributes(self):
        # Prefill fusion for qwen3-14b has to be split in two to avoid BO size
        # limitations. Split by layer to allow weight sharing between prefill
        # and token
        self._set_attribute("split_dd_fusion", 2)
        self._set_attribute(
            "dynamic_shape_list",
            [
                {"sequence_length_padded": 128},
                {"sequence_length_padded": 256},
                {"sequence_length_padded": 512},
                {"sequence_length_padded": 768},
                {"sequence_length_padded": 1024},
                {"sequence_length_padded": 1536},
                {"sequence_length_padded": 1792},
                {"sequence_length_padded": 2048},
            ],
        )

    def _override_token_fusion_attributes(self):
        if self.args.shared_weights:
            # When weights are shared, token fusion has to be split the same
            # way as prefill fusion to allow weight sharing between prefill
            # and token
            self._set_attribute("split_dd_fusion", 2)

    @classmethod
    def matches(cls, input_path: str) -> bool:
        return cls.model_type.value in input_path
