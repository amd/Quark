#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

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
    ch_axis: Optional[int] = None
    scale: Optional[torch.Tensor] = None
    zero_point: Optional[torch.Tensor] = None
    group_size: int = 0


@dataclass
class EmbeddingInfo:
    name: str = ""
    type: str = EmbeddingType.default.value
    weight: Optional[torch.Tensor] = None


@dataclass
class LayerNormInfo:
    name: str = ""
    type: str = LayerNormType.default.value
    weight: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    eps: float = 1e-5


@dataclass
class LinearInfo:
    name: str = ""
    weight: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    input_quant_info: Optional[QuantInfo] = None
    weight_quant_info: Optional[QuantInfo] = None
    output_quant_info: Optional[QuantInfo] = None


@dataclass
class ActInfo:
    name: str = ""
    type: str = ""


@dataclass
class AttentionInfo:
    name: str = ""
    q_proj: Optional[LinearInfo] = None
    k_proj: Optional[LinearInfo] = None
    v_proj: Optional[LinearInfo] = None
    o_proj: Optional[LinearInfo] = None
    emb: Optional[EmbeddingInfo] = None


@dataclass
class MLPInfo:
    name: str = ""
    gate_proj: Optional[LinearInfo] = None
    up_proj: Optional[LinearInfo] = None
    down_proj: Optional[LinearInfo] = None
    act_fn: Optional[ActInfo] = None


@dataclass
class DecoderInfo:
    name: str = ""
    decoder_type: str = ""
    input_layernorm: Optional[LayerNormInfo] = None
    self_attn: Optional[AttentionInfo] = None
    post_attention_layernorm: Optional[LayerNormInfo] = None
    mlp: Optional[MLPInfo] = None
    num_attention_heads: int = 0
    attention_head_size: Optional[int] = None
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
    tokens_embed: Optional[EmbeddingInfo] = None
    positional_embed: Optional[EmbeddingInfo] = None
    layers: List[DecoderInfo] = field(default_factory=list)
    final_norm: Optional[LayerNormInfo] = None
    lm_head: Optional[LinearInfo] = None
    embed_weight_share: bool = False
