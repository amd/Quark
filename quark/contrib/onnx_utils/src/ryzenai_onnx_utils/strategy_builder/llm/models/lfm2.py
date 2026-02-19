# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import re

from .. import LlmModel, LlmModelBase
from . import register_model


class Lfm2(LlmModelBase):
    default_jit_enabled = False

    def _override_token_fusion_attributes(self):
        self._set_attribute("xclbins", "stx_LLM_TPS_10bo_mladf_2x4x4_bfp16_gemm_mlp_flat_rms_conv_mul")
        self.args.ctrl_pkt = True if self.args.ctrl_pkt is None else self.args.ctrl_pkt
        if self.args.dynamic_attention_mask is None or self.args.dynamic_attention_mask:
            self.args.dynamic_attention_mask = True
            self._set_attribute(
                "dynamic_shape_list",
                [
                    {"attention_mask_padded": 256},
                    {"attention_mask_padded": 384},
                    {"attention_mask_padded": 512},
                    {"attention_mask_padded": 1024},
                    {"attention_mask_padded": 1536},
                    {"attention_mask_padded": 2048},
                    {"attention_mask_padded": 2304},
                    {"attention_mask_padded": 3072},
                ],
            )
        self._set_attribute("fuse_qk_mha", False)


@register_model
class Lfm2_2_6B(Lfm2):
    model_type = LlmModel.LFM2_2_6B

    @classmethod
    def matches(cls, input_path: str) -> bool:
        return re.findall(r"lfm2[-_]?2\.6b", input_path)


@register_model
class Lfm2_1_2B(Lfm2):
    model_type = LlmModel.LFM2_1_2B

    @classmethod
    def matches(cls, input_path: str) -> bool:
        return re.findall(r"lfm2[-_]?1\.2b", input_path)


@register_model
class LFM2_5_1_2B(Lfm2):
    model_type = LlmModel.LFM2_5_1_2B

    @classmethod
    def matches(cls, input_path: str) -> bool:
        return re.findall(r"lfm2.5[-_]?1\.2b", input_path)
