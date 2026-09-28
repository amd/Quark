#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Hugging Face ``PretrainedConfig`` for the EAGLE-3 draft.

Keeping the draft config as a real ``PretrainedConfig`` means the exported draft
loads with ``AutoConfig``/``AutoModel`` and carries the extra EAGLE-3 fields
(``aux_hidden_state_layer_ids``, ``draft_vocab_size``, ``fc_norm`` ...) that a
serving engine's EAGLE-3 loader reads.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class Eagle3Config(PretrainedConfig):
    """Configuration of the EAGLE-3 draft head.

    Attributes mirror :class:`~quark.experimental.speculative_decoding.config.EagleArchitectureConfig`
    plus the target-derived fields needed to rebuild the module and to serve it.
    """

    model_type = "eagle3"

    def __init__(
        self,
        vocab_size: int = 32000,
        draft_vocab_size: int | None = None,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        attention_bias: bool = False,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        fc_norm: bool = False,
        norm_output: bool = True,
        aux_hidden_state_layer_ids: list[int] | None = None,
        ttt_length: int = 7,
        target_hidden_size: int | None = None,
        target_model_path: str | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: object,
    ) -> None:
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size or vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.attention_bias = attention_bias
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.fc_norm = fc_norm
        self.norm_output = norm_output
        self.aux_hidden_state_layer_ids = aux_hidden_state_layer_ids or [2, -3, -1]
        self.ttt_length = ttt_length
        # EAGLE-3 reduces the concatenation of ``len(aux)`` target hidden states,
        # so the fc input dim is ``num_aux * target_hidden_size``.
        self.target_hidden_size = target_hidden_size or hidden_size
        self.target_model_path = target_model_path
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def num_aux_layers(self) -> int:
        return len(self.aux_hidden_state_layer_ids)
