#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Online extraction: target co-located with the trainer.

For small/medium targets (e.g. Qwen3-8B) the target fits alongside the draft on
the same GPU(s). We run the target forward once per batch with
``output_hidden_states=True`` and hand the aux hidden states straight to the
draft. This is the simplest and most reproducible regime and is what the
Qwen3-8B quick-start uses.
"""

from __future__ import annotations

from typing import Any

import torch


class OnlineExtractor:
    def __init__(self, spec_model: Any) -> None:
        self.spec = spec_model

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return the target aux hidden states for a batch.

        Output ``aux_hidden`` is ``[B, T, A*Ht]`` aligned position-for-position
        with ``input_ids``.
        """
        aux_hidden, _target_logits = self.spec.target_hidden_states(input_ids, attention_mask)
        return {"aux_hidden": aux_hidden}
