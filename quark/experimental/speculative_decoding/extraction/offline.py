#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Offline extraction: dump target hidden states to disk, then train the draft.

Cheapest on GPUs (the target is only needed during the dump) at the cost of
disk (hidden states are TB-scale for big models / long context). Useful when
the same features are reused across many draft-training experiments.
"""

from __future__ import annotations

import os
from typing import Any

import torch


class OfflineExtractor:
    """Precompute-and-cache aux hidden states keyed by a batch id.

    ``dump(loader)`` writes one ``.pt`` shard per batch under
    ``<output_dir>/hidden/``; at train time ``__call__`` reads the shard back.
    """

    def __init__(self, spec_model: Any, train_cfg: Any) -> None:
        self.spec = spec_model
        self.dir = os.path.join(train_cfg.output_dir, "hidden")
        os.makedirs(self.dir, exist_ok=True)
        self._index = 0

    def _shard_path(self, i: int) -> str:
        return os.path.join(self.dir, f"batch_{i:08d}.pt")

    @torch.no_grad()
    def dump(self, loader: Any) -> int:
        n = 0
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(next(self.spec.target.parameters()).device)
            attn = batch.get("attention_mask")
            if attn is not None:
                attn = attn.to(input_ids.device)
            aux_hidden, _ = self.spec.target_hidden_states(input_ids, attn)
            torch.save(
                {
                    "aux_hidden": aux_hidden.to("cpu", torch.float16),
                    "input_ids": batch["input_ids"],
                    "loss_mask": batch["loss_mask"],
                    "attention_mask": batch["attention_mask"],
                },
                self._shard_path(i),
            )
            n += 1
        print(f"[offline] dumped {n} shards -> {self.dir}")
        return n

    def num_shards(self) -> int:
        return len([f for f in os.listdir(self.dir) if f.startswith("batch_")])

    def load_shard(self, i: int) -> dict[str, torch.Tensor]:
        return torch.load(self._shard_path(i), map_location="cpu")

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        raise RuntimeError(
            "OfflineExtractor is shard-based: call dump(loader) first, then iterate shards "
            "with load_shard(i). The trainer handles this automatically when extraction='offline'."
        )
