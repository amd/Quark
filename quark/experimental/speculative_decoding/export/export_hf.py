#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Export a trained EAGLE-3 draft to a **vLLM-loadable** Hugging Face directory.

The output loads directly into vLLM speculative decoding as
``LlamaForCausalLMEagle3`` (``architectures: ["LlamaForCausalLMEagle3"]``,
``model_type: "llama"``) with the weight key names vLLM's loader expects. Serve
it with::

    vllm serve <target> --speculative-config \
        '{"method":"eagle3","model":"<out_dir>","num_speculative_tokens":3}'
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch


def _load_best_into_draft(spec_model: Any, best_ckpt: str | None) -> None:
    if best_ckpt is None:
        return
    state_path = os.path.join(best_ckpt, "trainer_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        spec_model.draft.load_state_dict(state["draft"], strict=False)


def export_hf(
    spec_model: Any,
    out_dir: str,
    tokenizer: Any | None = None,
    best_ckpt: str | None = None,
    share_embeddings: bool = False,
) -> str:
    """Serialize ``spec_model.draft`` to ``out_dir`` in vLLM EAGLE-3 format.

    Args:
        share_embeddings: if True, drop ``embed_tokens.weight`` from the export
            (vLLM will source it from the target). Default False keeps it, which
            is the most robust for serving.
    """
    os.makedirs(out_dir, exist_ok=True)
    _load_best_into_draft(spec_model, best_ckpt)

    draft = spec_model.draft
    cfg = spec_model.eagle_config

    state = {k: v.detach().contiguous().cpu() for k, v in draft.export_state_dict().items()}
    if share_embeddings:
        state.pop("embed_tokens.weight", None)

    try:
        from safetensors.torch import save_file

        save_file(state, os.path.join(out_dir, "model.safetensors"))
    except Exception:
        torch.save(state, os.path.join(out_dir, "pytorch_model.bin"))

    # config.json in the exact shape vLLM's LlamaForCausalLMEagle3 reads.
    config = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "model_type": "llama",
        "attention_bias": cfg.attention_bias,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": False,
        "torch_dtype": str(next(iter(state.values())).dtype).replace("torch.", ""),
        "vocab_size": cfg.vocab_size,
        "draft_vocab_size": cfg.draft_vocab_size,
        "target_hidden_size": cfg.target_hidden_size,
        "fc_norm": cfg.fc_norm,
        "norm_output": cfg.norm_output,
        "eagle_aux_hidden_state_layer_ids": cfg.aux_hidden_state_layer_ids,
        "num_aux_hidden_states": cfg.num_aux_layers,
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)

    print(f"[export_hf] wrote vLLM-loadable EAGLE-3 draft -> {out_dir}")
    return out_dir
