#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any, Optional
import torch
from torch import nn
from torch.nn import functional as F
from quark.torch.quantization.config.config import QuantizationConfig
from .mixin import QuantMixin
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ["QuantEmbedding", "QuantEmbeddingBag"]


class QuantEmbedding(nn.Embedding, QuantMixin):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 _weight: Optional[torch.Tensor] = None,
                 quant_config: QuantizationConfig = QuantizationConfig(),
                 device: torch.device = torch.device("cpu"),
                 **kwargs: Any) -> None:
        super().__init__(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        if _weight is not None:
            self.weight = _weight

        self.init_quantizer(quant_config, device, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq,
                           self.sparse)

    @classmethod
    def from_float(cls,
                   float_module: nn.Module,
                   quant_config: QuantizationConfig,
                   reload: bool = False,
                   weight_tensor: Optional[torch.Tensor] = None,
                   **kwargs: Any) -> nn.Module:
        quant_embedding = cls(float_module.num_embeddings,
                              float_module.embedding_dim,
                              float_module.padding_idx,
                              float_module.max_norm,
                              float_module.norm_type,
                              float_module.scale_grad_by_freq,
                              float_module.sparse,
                              float_module.weight,
                              quant_config,
                              reload=reload)
        if reload is True and weight_tensor is not None:
            quant_embedding.weight.data = weight_tensor.to(float_module.weight.device)
        else:
            quant_weight = quant_embedding.get_quant_weight(float_module.weight)
            quant_embedding.weight.data = quant_weight
        return quant_embedding


class QuantEmbeddingBag(nn.EmbeddingBag, QuantMixin):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False,
                 mode: str = "mean",
                 sparse: bool = False,
                 _weight: Optional[torch.Tensor] = None,
                 include_last_offset: bool = False,
                 padding_idx: Optional[int] = None,
                 quant_config: QuantizationConfig = QuantizationConfig(),
                 device: torch.device = torch.device("cpu"),
                 **kwargs: Any) -> None:
        super().__init__(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset
        self.padding_idx = padding_idx

        if _weight is not None:
            self.weight = _weight

        self.init_quantizer(quant_config, device, **kwargs)

    def forward(self,
                input: torch.Tensor,
                offsets: Optional[torch.Tensor] = None,
                per_sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.embedding_bag(
            input,
            self.weight,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights,
            self.include_last_offset,
            self.padding_idx,
        )

    @classmethod
    def from_float(cls,
                   float_module: nn.Module,
                   quant_config: QuantizationConfig,
                   reload: bool = False,
                   weight_tensor: Optional[torch.Tensor] = None,
                   **kwargs: Any) -> nn.Module:
        quant_embeddingbag = cls(
            float_module.num_embeddings,
            float_module.embedding_dim,
            float_module.max_norm,
            float_module.norm_type,
            float_module.scale_grad_by_freq,
            float_module.mode,
            float_module.sparse,
            float_module.weight,
            float_module.include_last_offset,
            float_module.padding_idx,
            quant_config,
            reload=reload,
        )
        if reload is True and weight_tensor is not None:
            quant_embeddingbag.weight.data = weight_tensor.to(float_module.weight.device)
        else:
            quant_weight = quant_embeddingbag.get_quant_weight(float_module.weight)
            quant_embeddingbag.weight.data = quant_weight
        return quant_embeddingbag
