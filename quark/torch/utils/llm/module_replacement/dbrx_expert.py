#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from quark.common.utils.import_utils import (
    is_torch_available,
    is_transformers_available,
)
from quark.torch.utils.llm.module_replacement.preprocess_registry import register_quark_preprocess

if is_torch_available():
    import torch
    import torch.nn as nn

if is_transformers_available():
    from transformers.models.dbrx.modeling_dbrx import DbrxExpertGLU, DbrxExperts


class DbrxExpertGLU_(nn.Module):
    def __init__(self, mlp: DbrxExpertGLU):
        super().__init__()
        self.w1 = nn.Linear(mlp.hidden_size, mlp.ffn_hidden_size, device=mlp.w1.device, bias=False)
        self.v1 = nn.Linear(mlp.hidden_size, mlp.ffn_hidden_size, device=mlp.v1.device, bias=False)
        self.w2 = nn.Linear(mlp.ffn_hidden_size, mlp.hidden_size, device=mlp.w2.device, bias=False)
        self.activation_fn = mlp.activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_proj = self.activation_fn(self.w1(x))
        up_proj = self.v1(x)
        intermediate_states = gate_proj * up_proj
        down_proj = self.w2(intermediate_states)
        return down_proj


class DbrxExperts_(nn.Module):
    def __init__(self, experts_module: DbrxExperts):
        super().__init__()
        self.num_experts = experts_module.num_experts
        self.mlp = nn.ModuleList([DbrxExpertGLU_(experts_module.mlp) for _ in range(self.num_experts)])
        w1_chunked = experts_module.mlp.w1.view(
            experts_module.mlp.moe_num_experts, experts_module.mlp.ffn_hidden_size, experts_module.mlp.hidden_size
        )
        v1_chunked = experts_module.mlp.v1.view(
            experts_module.mlp.moe_num_experts, experts_module.mlp.ffn_hidden_size, experts_module.mlp.hidden_size
        )
        w2_chunked = experts_module.mlp.w2.view(
            experts_module.mlp.moe_num_experts, experts_module.mlp.ffn_hidden_size, experts_module.mlp.hidden_size
        )
        for idx in range(self.num_experts):
            self.mlp[idx].w1.weight.data = w1_chunked[idx].contiguous()
            self.mlp[idx].v1.weight.data = v1_chunked[idx].contiguous()
            self.mlp[idx].w2.weight.data = w2_chunked[idx].t().contiguous()
        experts_module.mlp.w1 = None
        experts_module.mlp.v1 = None
        experts_module.mlp.w2 = None

    @classmethod
    def from_float(cls, float_module: DbrxExperts) -> DbrxExperts:
        return cls(experts_module=float_module)

    def forward(
        self,
        x: torch.Tensor,
        weights_or_top_k_index: torch.Tensor,
        top_weights_or_top_k_weights: torch.Tensor | None = None,
        top_experts: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass supporting both transformers v4 and v5 signatures.

        Transformers v4.57: forward(x, weights, top_weights, top_experts)
        Transformers v5+:   forward(x, top_k_index, top_k_weights)
        """
        # Detect which version based on the number of arguments
        if top_experts is None:
            # v5 signature: (x, top_k_index, top_k_weights)
            top_experts = weights_or_top_k_index
            top_weights = top_weights_or_top_k_weights
        else:
            # v4 signature: (x, weights, top_weights, top_experts)
            # weights_or_top_k_index is actually 'weights' (unused in this implementation)
            top_weights = top_weights_or_top_k_weights

        bsz, q_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        out = torch.zeros_like(x)
        expert_mask = nn.functional.one_hot(top_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(0, self.num_experts):
            topk_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.shape[0] == 0:
                continue
            token_list = token_idx
            topk_list = topk_idx
            expert_tokens = x[None, token_list].reshape(-1, hidden_size)
            expert_out = self.mlp[expert_idx](expert_tokens) * top_weights[token_list, topk_list, None]
            out.index_add_(0, token_idx, expert_out)
        out = out.reshape(bsz, q_len, hidden_size)
        return out


@register_quark_preprocess(DbrxExperts)
class QuarkDbrxExperts(DbrxExperts_):
    def __init__(self, experts_module: DbrxExperts):
        super().__init__(experts_module=experts_module)

    @classmethod
    def from_hf(cls, float_module: DbrxExperts, reload: bool = False) -> DbrxExperts:
        return cls.from_float(float_module=float_module)
