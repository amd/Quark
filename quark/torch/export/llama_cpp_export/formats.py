#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""llama.cpp GGUF export format registry."""

from __future__ import annotations

from dataclasses import dataclass

from quark.common.utils.import_utils import is_gguf_available_and_minimum_version

if is_gguf_available_and_minimum_version():
    from gguf import GGMLQuantizationType, LlamaFileType
else:
    GGMLQuantizationType = None  # type: ignore[misc, assignment]
    LlamaFileType = None  # type: ignore[misc, assignment]


@dataclass(frozen=True)
class LlamaCppExportFormat:
    name: str
    llama_file_type: "LlamaFileType"
    weight_qtype: "GGMLQuantizationType"
    use_libggml: bool = True


def _formats() -> dict[str, LlamaCppExportFormat]:
    if GGMLQuantizationType is None or LlamaFileType is None:
        return {}

    q = GGMLQuantizationType
    f = LlamaFileType
    entries = [
        ("f32", f.ALL_F32, q.F32, False),
        ("f16", f.MOSTLY_F16, q.F16, False),
        ("bf16", f.MOSTLY_BF16, q.BF16, False),
        ("q4_0", f.MOSTLY_Q4_0, q.Q4_0, True),
        ("q4_1", f.MOSTLY_Q4_1, q.Q4_1, True),
        ("q5_0", f.MOSTLY_Q5_0, q.Q5_0, True),
        ("q5_1", f.MOSTLY_Q5_1, q.Q5_1, True),
        ("q8_0", f.MOSTLY_Q8_0, q.Q8_0, True),
        ("q2_k", f.MOSTLY_Q2_K, q.Q2_K, True),
        ("q2_k_s", f.MOSTLY_Q2_K_S, q.Q2_K, True),
        ("q3_k_s", f.MOSTLY_Q3_K_S, q.Q3_K, True),
        ("q3_k_m", f.MOSTLY_Q3_K_M, q.Q3_K, True),
        ("q3_k_l", f.MOSTLY_Q3_K_L, q.Q3_K, True),
        ("q4_k_s", f.MOSTLY_Q4_K_S, q.Q4_K, True),
        ("q4_k_m", f.MOSTLY_Q4_K_M, q.Q4_K, True),
        ("q5_k_s", f.MOSTLY_Q5_K_S, q.Q5_K, True),
        ("q5_k_m", f.MOSTLY_Q5_K_M, q.Q5_K, True),
        ("q6_k", f.MOSTLY_Q6_K, q.Q6_K, True),
        ("tq1_0", f.MOSTLY_TQ1_0, q.TQ1_0, True),
        ("tq2_0", f.MOSTLY_TQ2_0, q.TQ2_0, True),
        ("iq2_xxs", f.MOSTLY_IQ2_XXS, q.IQ2_XXS, True),
        ("iq2_xs", f.MOSTLY_IQ2_XS, q.IQ2_XS, True),
        ("iq2_s", f.MOSTLY_IQ2_S, q.IQ2_S, True),
        ("iq3_xxs", f.MOSTLY_IQ3_XXS, q.IQ3_XXS, True),
        ("iq3_s", f.MOSTLY_IQ3_S, q.IQ3_S, True),
        ("iq3_m", f.MOSTLY_IQ3_M, q.IQ3_S, True),
        ("iq1_s", f.MOSTLY_IQ1_S, q.IQ1_S, True),
        ("iq1_m", f.MOSTLY_IQ1_M, q.IQ1_M, True),
        ("iq4_nl", f.MOSTLY_IQ4_NL, q.IQ4_NL, True),
        ("iq4_xs", f.MOSTLY_IQ4_XS, q.IQ4_XS, True),
        ("mxfp4", f.MOSTLY_MXFP4_MOE, q.MXFP4, True),
        ("nvfp4", f.MOSTLY_NVFP4, q.NVFP4, True),
    ]
    return {
        name: LlamaCppExportFormat(name, ftype, qtype, use_libggml)
        for name, ftype, qtype, use_libggml in entries
    }


LLAMA_CPP_EXPORT_FORMATS: dict[str, LlamaCppExportFormat] = _formats()


def get_export_format(name: str) -> LlamaCppExportFormat:
    key = name.lower()
    if key not in LLAMA_CPP_EXPORT_FORMATS:
        supported = ", ".join(sorted(LLAMA_CPP_EXPORT_FORMATS))
        raise ValueError(f"Unsupported llama.cpp export format {name!r}. Supported: {supported}")
    return LLAMA_CPP_EXPORT_FORMATS[key]


def list_export_formats() -> list[str]:
    return sorted(LLAMA_CPP_EXPORT_FORMATS)
