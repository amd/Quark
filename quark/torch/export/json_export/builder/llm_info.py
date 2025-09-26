#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import torch

CURRENT_VERSION = 0.1


class LayerNormType(Enum):
    default = "default"
    rms = "rms"


class EmbeddingType(Enum):
    default = "default"
    rotary = "rotary"


@dataclass
class QuantInfo:
    name: str = ""
    dtype: str = ""
    qscheme: str = ""
    ch_axis: int | None = None
    scale: torch.Tensor | None = None
    zero_point: torch.Tensor | None = None
    group_size: int = 0


@dataclass
class EmbeddingInfo:
    name: str = ""
    type: str = EmbeddingType.default.value
    weight: torch.Tensor | None = None


@dataclass
class LayerNormInfo:
    name: str = ""
    type: str = LayerNormType.default.value
    weight: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    eps: float = 1e-5


@dataclass
class LinearInfo:
    name: str = ""
    weight: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    input_quant_info: QuantInfo | None = None
    weight_quant_info: QuantInfo | None = None
    output_quant_info: QuantInfo | None = None


@dataclass
class ActInfo:
    name: str = ""
    type: str = ""


@dataclass
class AttentionInfo:
    name: str = ""
    q_proj: LinearInfo | None = None
    k_proj: LinearInfo | None = None
    v_proj: LinearInfo | None = None
    o_proj: LinearInfo | None = None
    emb: EmbeddingInfo | None = None


@dataclass
class MLPInfo:
    name: str = ""
    gate_proj: LinearInfo | None = None
    up_proj: LinearInfo | None = None
    down_proj: LinearInfo | None = None
    act_fn: ActInfo | None = None


@dataclass
class DecoderInfo:
    name: str = ""
    decoder_type: str = ""
    input_layernorm: LayerNormInfo | None = None
    self_attn: AttentionInfo | None = None
    post_attention_layernorm: LayerNormInfo | None = None
    mlp: MLPInfo | None = None
    num_attention_heads: int = 0
    attention_head_size: int | None = None
    num_kv_heads: int = 0
    max_position_embeddings: int = 0
    rotary_pct: int = 0
    parallel_attention: bool = False
    apply_residual_connection_post_layernorm: bool = False
    use_cache: bool = True
    rope_ratio: float = 1.0
    seq_length: int = 0


@dataclass
class ModelInfo:
    version: float = CURRENT_VERSION
    dtype: str = "float16"
    vocab_size: int = 0
    tokens_embed: EmbeddingInfo | None = None
    positional_embed: EmbeddingInfo | None = None
    layers: list[DecoderInfo] = field(default_factory=list)
    final_norm: LayerNormInfo | None = None
    lm_head: LinearInfo | None = None
    embed_weight_share: bool = False
