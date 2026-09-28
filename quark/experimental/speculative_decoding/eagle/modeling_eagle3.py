#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""EAGLE-3 draft head, structured to be **vLLM-loadable** as
``LlamaForCausalLMEagle3``.

The module names and weight shapes match exactly what vLLM's
``vllm/model_executor/models/llama_eagle3.py`` expects on disk, so the exported
draft (see ``export/export_hf.py``) loads directly into a vLLM speculative
serve. Reference (a working Qwen3-8B EAGLE-3 draft) state dict::

    embed_tokens.weight                       [vocab, hidden]
    fc.weight                                 [hidden, num_aux * target_hidden]
    layers.0.input_layernorm.weight           [hidden]
    layers.0.hidden_norm.weight               [hidden]
    layers.0.self_attn.q_proj.weight          [n_heads*head_dim, 2*hidden]   # 1st layer: 2*hidden in
    layers.0.self_attn.k_proj.weight          [n_kv*head_dim,   2*hidden]
    layers.0.self_attn.v_proj.weight          [n_kv*head_dim,   2*hidden]
    layers.0.self_attn.o_proj.weight          [hidden, n_heads*head_dim]
    layers.0.post_attention_layernorm.weight  [hidden]
    layers.0.mlp.gate_proj.weight             [intermediate, hidden]
    layers.0.mlp.up_proj.weight               [intermediate, hidden]
    layers.0.mlp.down_proj.weight             [intermediate, hidden]  (down: [hidden, intermediate])
    norm.weight                               [hidden]
    lm_head.weight                            [draft_vocab, hidden]
    d2t                                       [draft_vocab] int64, only when draft_vocab < vocab
    t2d                                       [vocab] bool,   only when draft_vocab < vocab

The runtime data flow mirrors vLLM:
  feature = fc(cat(aux_hidden_states))          # combine_hidden_states
  embeds  = input_layernorm(embed_tokens(ids))
  h, res  = hidden_norm(feature), feature       # norm-after-residual
  h       = attn(cat([embeds, h]))              # 1st layer sees 2*hidden
  h, res  = post_attention_layernorm(h + res)
  h       = mlp(h)
  out, prenorm = norm(h + res)                  # returns (post-norm, pre-norm)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quark.experimental.speculative_decoding.eagle.config import Eagle3Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class Eagle3Attention(nn.Module):
    """Llama-style GQA attention; first layer consumes 2*hidden (embeds|hidden)."""

    def __init__(self, cfg: Eagle3Config, in_features: int) -> None:
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        bias = cfg.attention_bias
        self.q_proj = nn.Linear(in_features, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(in_features, self.n_kv * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(in_features, self.n_kv * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(o)


class Eagle3MLP(nn.Module):
    def __init__(self, cfg: Eagle3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3DecoderLayer(nn.Module):
    """The single EAGLE-3 draft layer (exported as ``layers.0``)."""

    def __init__(self, cfg: Eagle3Config) -> None:
        super().__init__()
        h = cfg.hidden_size
        self.input_layernorm = RMSNorm(h, cfg.rms_norm_eps)  # applied to embeds
        self.hidden_norm = RMSNorm(h, cfg.rms_norm_eps)  # applied to feature
        self.self_attn = Eagle3Attention(cfg, in_features=2 * h)
        self.post_attention_layernorm = RMSNorm(h, cfg.rms_norm_eps)
        self.mlp = Eagle3MLP(cfg)

    def forward(
        self, embeds: torch.Tensor, feature: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.input_layernorm(embeds)
        residual = feature
        h = self.hidden_norm(feature)
        x = torch.cat([embeds, h], dim=-1)  # [B, T, 2*hidden]
        attn = self.self_attn(x, cos, sin)
        residual = attn + residual
        h = self.post_attention_layernorm(residual)
        h = self.mlp(h)
        return h, residual


class Eagle3DraftModel(nn.Module):
    """vLLM-compatible EAGLE-3 draft (single layer)."""

    def __init__(self, config: Eagle3Config) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, h)
        self.fc = nn.Linear(config.num_aux_layers * config.target_hidden_size, h, bias=False)
        self.fc_norm = (
            nn.ModuleList(
                [RMSNorm(config.target_hidden_size, config.rms_norm_eps) for _ in range(config.num_aux_layers)]
            )
            if config.fc_norm
            else None
        )
        # The draft is a single EAGLE-3 decoder layer: layer 0 consumes 2*hidden
        # (cat[embeds, feature]) and step() indexes layers[0]. This matches vLLM's
        # single-layer LlamaForCausalLMEagle3 serving format, so deeper drafts are not
        # supported -- fail loudly instead of silently training a 1-layer model.
        if config.num_hidden_layers != 1:
            raise ValueError(
                f"Eagle3DraftModel supports a single draft layer only, but got "
                f"num_hidden_layers={config.num_hidden_layers}. Multi-layer drafts are not "
                f"implemented (the vLLM LlamaForCausalLMEagle3 format is single-layer)."
            )
        self.layers = nn.ModuleList([Eagle3DecoderLayer(config)])
        self.norm = RMSNorm(h, config.rms_norm_eps)
        self.lm_head = nn.Linear(h, config.draft_vocab_size, bias=False)

        # rotary tables
        inv = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim))
        self.register_buffer("_inv_freq", inv, persistent=False)

        # Draft<->target vocab mapping, in the on-disk EAGLE-3 convention that
        # serving engines read: ``d2t`` is an *offset* (target_id = draft_id +
        # d2t[draft_id]) and ``t2d`` is a bool "is this target id representable"
        # mask. vLLM's Eagle3LlamaForCausalLM computes `arange(draft_vocab) + d2t`
        # directly, so storing absolute target ids here would double-count.
        # Identity mapping = all-zero offsets over the first draft_vocab_size ids.
        self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long), persistent=True)
        t2d = torch.zeros(config.vocab_size, dtype=torch.bool)
        t2d[: config.draft_vocab_size] = True
        self.register_buffer("t2d", t2d, persistent=True)
        # Training needs target_id -> draft_id, which the bool mask cannot answer
        # on its own. Derived from d2t, and non-persistent so it never reaches a
        # checkpoint and cannot drift from the exported convention.
        self.register_buffer("t2d_index", torch.empty(config.vocab_size, dtype=torch.long), persistent=False)
        self._refresh_t2d_index()
        self.register_load_state_dict_post_hook(lambda module, _keys: module._refresh_t2d_index())

    # -- draft vocab ------------------------------------------------------
    @torch.no_grad()
    def _refresh_t2d_index(self) -> None:
        """Rebuild the target_id -> draft_id lookup (-1 = outside the draft vocab)."""
        draft_ids = torch.arange(self.config.draft_vocab_size, device=self.d2t.device)
        target_ids = draft_ids + self.d2t
        index = torch.full_like(self.t2d_index, -1)
        index[target_ids] = draft_ids
        self.t2d_index.copy_(index)

    @torch.no_grad()
    def set_draft_vocab_mapping(self, d2t: torch.Tensor, t2d: torch.Tensor) -> None:
        """Install a calibrated draft-vocab mapping from :func:`calibrate_draft_vocab`.

        ``d2t`` is the offset form (``target_id - draft_id``) and ``t2d`` the bool
        mask, matching what :meth:`export_state_dict` writes.
        """
        for name, incoming in (("d2t", d2t), ("t2d", t2d)):
            buf = getattr(self, name)
            if incoming.shape != buf.shape:
                raise ValueError(
                    f"draft-vocab mapping '{name}' shape {tuple(incoming.shape)} does not match the "
                    f"draft's {tuple(buf.shape)}. This usually means the tokenizer length used by "
                    f"calibrate_draft_vocab differs from the target config vocab_size "
                    f"(vocab_size={self.config.vocab_size}, draft_vocab_size={self.config.draft_vocab_size}). "
                    f"Re-run calibrate_draft_vocab against the target's vocab_size, or reconcile the two."
                )
            if incoming.dtype != buf.dtype:
                raise ValueError(
                    f"draft-vocab mapping '{name}' dtype {incoming.dtype} does not match the draft's "
                    f"{buf.dtype}. d2t is an int64 offset (target_id - draft_id) and t2d a bool mask; "
                    f"a cache built before this convention stored absolute ids and must be rebuilt."
                )
        self.d2t.copy_(d2t.to(self.d2t.device))
        self.t2d.copy_(t2d.to(self.t2d.device))
        self._refresh_t2d_index()

    # -- helpers ----------------------------------------------------------
    def _rope(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids[:, :, None].float() * self._inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def combine_hidden_states(self, aux_hidden: torch.Tensor) -> torch.Tensor:
        """Reduce concatenated aux hidden states [B,T,num_aux*target_h] -> [B,T,hidden]."""
        if self.fc_norm is not None:
            chunks = aux_hidden.chunk(self.config.num_aux_layers, dim=-1)
            aux_hidden = torch.cat([n(c) for n, c in zip(self.fc_norm, chunks, strict=False)], dim=-1)
        return self.fc(aux_hidden)

    def step(
        self, input_ids: torch.Tensor, feature: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One draft step. Returns (draft_logits, next_feature)."""
        cos, sin = self._rope(position_ids, feature.dtype)
        embeds = self.embed_tokens(input_ids)
        h, residual = self.layers[0](embeds, feature, cos, sin)
        residual = h + residual
        out = self.norm(residual)
        aux_output = out if self.config.norm_output else residual
        logits = self.lm_head(out)
        return logits, aux_output

    def forward(
        self, input_ids: torch.Tensor, aux_hidden: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        feature = self.combine_hidden_states(aux_hidden)
        return self.step(input_ids, feature, position_ids)

    @torch.no_grad()
    def tie_embeddings(self, target_embed_weight: torch.Tensor) -> None:
        if target_embed_weight.shape != self.embed_tokens.weight.shape:
            raise ValueError(
                f"target embedding shape {tuple(target_embed_weight.shape)} does not match the "
                f"draft's {tuple(self.embed_tokens.weight.shape)}. If the recipe sets "
                f"model.embedding_key, check it names the target's input embedding (not the LM head)."
            )
        self.embed_tokens.weight.copy_(target_embed_weight)
        self.embed_tokens.weight.requires_grad_(False)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict with exactly the vLLM ``LlamaForCausalLMEagle3`` key names."""
        sd: dict[str, torch.Tensor] = {}
        sd["embed_tokens.weight"] = self.embed_tokens.weight
        sd["fc.weight"] = self.fc.weight
        if self.fc_norm is not None:
            for i, n in enumerate(self.fc_norm):
                sd[f"fc_norm.{i}.weight"] = n.weight
        L = self.layers[0]
        sd["layers.0.input_layernorm.weight"] = L.input_layernorm.weight
        sd["layers.0.hidden_norm.weight"] = L.hidden_norm.weight
        sd["layers.0.self_attn.q_proj.weight"] = L.self_attn.q_proj.weight
        sd["layers.0.self_attn.k_proj.weight"] = L.self_attn.k_proj.weight
        sd["layers.0.self_attn.v_proj.weight"] = L.self_attn.v_proj.weight
        sd["layers.0.self_attn.o_proj.weight"] = L.self_attn.o_proj.weight
        sd["layers.0.post_attention_layernorm.weight"] = L.post_attention_layernorm.weight
        sd["layers.0.mlp.gate_proj.weight"] = L.mlp.gate_proj.weight
        sd["layers.0.mlp.up_proj.weight"] = L.mlp.up_proj.weight
        sd["layers.0.mlp.down_proj.weight"] = L.mlp.down_proj.weight
        sd["norm.weight"] = self.norm.weight
        sd["lm_head.weight"] = self.lm_head.weight
        if self.config.draft_vocab_size != self.config.vocab_size:
            sd["d2t"] = self.d2t
            sd["t2d"] = self.t2d
        return sd
