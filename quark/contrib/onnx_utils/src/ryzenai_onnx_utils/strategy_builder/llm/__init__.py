# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import enum


class LlmMode(enum.Enum):
    HYBRID = enum.auto()
    NPU_EAGER = enum.auto()
    NPU_PREFILL_FUSION = enum.auto()
    NPU_TOKEN_FUSION = enum.auto()
    FULL_FUSION_PREFILL = enum.auto()
    FULL_FUSION_TOKEN = enum.auto()


class LlmModel(enum.StrEnum):
    AUTO = "infer_from_path"
    GEMMA3_4B_TEXT = "gemma3-4b-text"
    GPT_OSS = "gpt-oss"
    LFM2_1_2B = "lfm2-1.2b"
    LFM2_2_6B = "lfm2-2.6b"
    LFM2_5_1_2B = "lfm2.5-1.2b"
    LLAMA3_8B = "llama3-8b"
    OLMO_1B = "olmo-1b"
    PHI_4 = "phi-4"
    QWEN3_14B = "qwen3-14b"
    DEFAULT = "default"


# the order is significant. These operators are replaced in the model in the same
# order so composite ops like SSMLP should come before a simpler op like Matmul
SUPPORTED_LLM_OPERATORS = [
    "embeddings",
    "ssmlp",
    "fuse_qkv_matmul",
    "fuse_qkv_matmul_concat",
    "gqo",
    "slrn",
    "matmulnbits",
    "matmul",
    "qmoe",
    "conv",
]


class GenaiMode(enum.StrEnum):
    STANDALONE = "standalone"
    EP = "ep"


from .builder import LlmModelBase, get_builder  # noqa: E402
from .genai_config import GenAiConfig  # noqa: E402

__all__ = ["LlmMode", "LlmModel", "SUPPORTED_LLM_OPERATORS", "GenaiMode", "get_builder", "LlmModelBase", "GenAiConfig"]
