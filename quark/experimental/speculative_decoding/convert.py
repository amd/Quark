#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""``convert(target, spec_cfg) -> SpecModel``.

Turns a (optionally Quark-quantized) target model into a trainable EAGLE-3
speculative-decoding model. The returned :class:`SpecModel` bundles:

* the frozen target (the *verifier* and the source of hidden states), and
* a fresh EAGLE-3 draft head sized from the target + recipe.

The target is only needed on-device for the ``online`` extraction regime; for
``offline``/``streaming`` the draft trains against pre-extracted / streamed
hidden states and the target can live in a separate serve process.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from quark.experimental.speculative_decoding.config import SpecConfig
from quark.experimental.speculative_decoding.eagle.config import Eagle3Config
from quark.experimental.speculative_decoding.eagle.modeling_eagle3 import Eagle3DraftModel


def _resolve_aux_layer_ids(aux: list[int], num_layers: int) -> list[int]:
    """Normalize (possibly negative) aux layer indices against the target depth."""
    return [i if i >= 0 else num_layers + i for i in aux]


def _weight_by_key(module: nn.Module, key: str) -> torch.Tensor:
    """Fetch an explicitly named weight (dotted state-dict path) from ``module``.

    ``get_input_embeddings()`` is the right default, but nested targets (a
    multimodal wrapper whose text tower owns ``language_model.model.embed_tokens``,
    or a ``trust_remote_code`` model that does not implement the accessor) may not
    return the tensor the draft has to tie to. Such recipes name it directly.
    """
    sd = module.state_dict()
    for candidate in (key, f"{key}.weight"):
        if candidate in sd:
            return sd[candidate]
    hints = [k for k in sd if "embed" in k][:5]
    raise KeyError(
        f"embedding_key {key!r} is not in the target state dict. Candidate embedding keys: {hints or 'none found'}."
    )


class SpecModel(nn.Module):
    """A target verifier paired with an EAGLE-3 draft head."""

    def __init__(
        self,
        draft: Eagle3DraftModel,
        eagle_config: Eagle3Config,
        aux_layer_ids: list[int],
        ttt_length: int,
        target: nn.Module | None = None,
        target_model_path: str | None = None,
        embedding_key: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        self.draft = draft
        self.eagle_config = eagle_config
        self.aux_layer_ids = aux_layer_ids
        self.ttt_length = ttt_length
        self.target = target  # may be None (offline/streaming)
        self.target_model_path = target_model_path
        self.embedding_key = embedding_key  # explicit override for nested targets
        self.trust_remote_code = trust_remote_code

    # -- target hidden-state extraction (online regime) --------------------
    @torch.no_grad()
    def target_hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the target and return (aux_hidden [B,T,A*Ht], target_logits [B,T,V]).

        Requires a co-located target (``online``). Hidden states are gathered
        from ``aux_layer_ids`` and concatenated along the feature dim.
        """
        if self.target is None:
            raise RuntimeError(
                "SpecModel.target is not loaded; online extraction needs a co-located target. "
                "Use extraction='offline'/'streaming' or pass a loaded target to convert()."
            )
        self.target.eval()
        out = self.target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states  # tuple: embeddings + one per layer
        aux = [hs[i + 1] for i in self.aux_layer_ids]  # +1 to skip the embedding entry
        aux_hidden = torch.cat(aux, dim=-1)
        return aux_hidden, out.logits

    def load_target_embeddings_into_draft(self) -> None:
        if self.target is None:
            return
        if self.embedding_key:
            weight = _weight_by_key(self.target, self.embedding_key)
        else:
            weight = self.target.get_input_embeddings().weight.data
        self.draft.tie_embeddings(weight)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.draft(*args, **kwargs)


def convert(
    target: str | nn.Module,
    spec_cfg: dict[str, Any] | SpecConfig,
    *,
    load_target: bool | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = "auto",
    trust_remote_code: bool = True,
    embedding_key: str | None = None,
) -> SpecModel:
    """Build a :class:`SpecModel` from a target and a spec recipe.

    Args:
        target: a loaded HF/Quark model, or a path/repo id to load.
        spec_cfg: a :class:`SpecConfig` or an equivalent dict.
        load_target: force-load the target even when a path is given lazily;
            defaults to loading when ``target`` is a string.
        embedding_key: dotted state-dict path of the target input-embedding
            weight, for targets whose ``get_input_embeddings()`` does not reach
            it. Defaults to the standard accessor.
    """
    cfg = spec_cfg if isinstance(spec_cfg, SpecConfig) else SpecConfig(**spec_cfg)
    if cfg.method != "eagle3":
        raise NotImplementedError(f"Only method='eagle3' is supported (got {cfg.method!r}).")

    target_model: nn.Module | None = None
    target_path: str | None = None

    if isinstance(target, str):
        target_path = target
        if load_target is None:
            load_target = True
        if load_target:
            from transformers import AutoModelForCausalLM

            target_model = AutoModelForCausalLM.from_pretrained(
                target, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust_remote_code
            )
    else:
        target_model = target

    # Resolve the target HF config (from the loaded model or from the path).
    if target_model is not None:
        target_hf_cfg = target_model.config
    else:
        from transformers import AutoConfig

        target_hf_cfg = AutoConfig.from_pretrained(target_path, trust_remote_code=trust_remote_code)

    # Some multimodal targets nest the text config.
    text_cfg = getattr(target_hf_cfg, "text_config", target_hf_cfg)
    num_layers = getattr(text_cfg, "num_hidden_layers", target_hf_cfg.num_hidden_layers)
    target_hidden = getattr(text_cfg, "hidden_size", target_hf_cfg.hidden_size)
    target_vocab = getattr(text_cfg, "vocab_size", target_hf_cfg.vocab_size)

    arch = cfg.eagle_config().merged_with_target(text_cfg)
    aux_ids = _resolve_aux_layer_ids(cfg.aux_hidden_layers, num_layers)

    hidden_size = arch.hidden_size or target_hidden
    num_heads = arch.num_attention_heads
    if num_heads is None:
        raise ValueError(
            "num_attention_heads could not be resolved: it is unset in the recipe and the target "
            f"config ({type(text_cfg).__name__}) does not expose it. Set "
            "eagle.eagle_architecture_config.num_attention_heads explicitly."
        )

    eagle_config = Eagle3Config(
        vocab_size=target_vocab,
        draft_vocab_size=arch.draft_vocab_size or target_vocab,
        hidden_size=hidden_size,
        intermediate_size=arch.intermediate_size,
        num_hidden_layers=arch.num_hidden_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=arch.num_key_value_heads,
        head_dim=getattr(text_cfg, "head_dim", None) or (hidden_size // num_heads),
        attention_bias=getattr(text_cfg, "attention_bias", False),
        max_position_embeddings=arch.max_position_embeddings,
        rms_norm_eps=arch.rms_norm_eps,
        rope_theta=getattr(text_cfg, "rope_theta", 10000.0),
        fc_norm=arch.fc_norm,
        norm_output=arch.norm_output,
        aux_hidden_state_layer_ids=aux_ids,
        ttt_length=cfg.ttt_length,
        target_hidden_size=target_hidden,
        target_model_path=target_path,
    )

    draft = Eagle3DraftModel(eagle_config).to(dtype)
    spec = SpecModel(
        draft=draft,
        eagle_config=eagle_config,
        aux_layer_ids=aux_ids,
        ttt_length=cfg.ttt_length,
        target=target_model,
        target_model_path=target_path,
        embedding_key=embedding_key,
        trust_remote_code=trust_remote_code,
    )
    if target_model is not None:
        for p in target_model.parameters():
            p.requires_grad_(False)
        spec.load_target_embeddings_into_draft()
    return spec
