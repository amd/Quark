#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Dataset loading + tokenization for EAGLE-3 training.

Input format is standardized conversation JSONL:
``{"conversations": [{"role": "user"|"assistant"|"system", "content": ...}]}``.

We apply the target's chat template so training tokens exactly match serving
(template fidelity is a silent AL cap otherwise). Only assistant tokens are
supervised (``loss_mask``); prompt/user tokens are masked out.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch.utils.data import Dataset


def load_conversations(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _normalize_roles(conv: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in conv:
        role = str(m.get("role") or m.get("from") or "")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        content = str(m.get("content") or m.get("value") or "")
        out.append({"role": role, "content": content})
    return out


class ConversationDataset(Dataset):
    """Tokenizes conversations and marks assistant spans for supervision.

    Each item yields ``input_ids``, ``attention_mask`` and ``loss_mask``
    (1 on assistant tokens only). The trainer shifts these to build the
    next-token targets for the TTT unroll.
    """

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        max_seq_length: int = 4096,
        use_chat_template: bool = True,
        limit: int | None = None,
    ) -> None:
        self.rows = load_conversations(path, limit=limit)
        self.tok = tokenizer
        self.max_len = max_seq_length
        self.use_chat_template = use_chat_template and getattr(tokenizer, "chat_template", None) is not None

    def __len__(self) -> int:
        return len(self.rows)

    def _encode_with_template(self, conv: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        input_ids: list[int] = []
        loss_mask: list[int] = []
        # Build turn-by-turn so we can mark only assistant tokens for loss.
        # Render to a string then tokenize (robust across transformers versions,
        # where apply_chat_template(tokenize=True) may return an Encoding object).
        prefix_msgs: list[dict[str, str]] = []
        for msg in conv:
            prefix_msgs.append(msg)
            text = self.tok.apply_chat_template(
                prefix_msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            new_tokens = ids[len(input_ids) :]
            supervise = 1 if msg["role"] == "assistant" else 0
            input_ids.extend(new_tokens)
            loss_mask.extend([supervise] * len(new_tokens))
        return self._finalize(input_ids, loss_mask)

    def _encode_raw(self, conv: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        # No template: concatenate user prompt (unsupervised) + assistant (supervised).
        input_ids: list[int] = []
        loss_mask: list[int] = []
        for msg in conv:
            toks = self.tok(msg["content"], add_special_tokens=False)["input_ids"]
            supervise = 1 if msg["role"] == "assistant" else 0
            input_ids.extend(toks)
            loss_mask.extend([supervise] * len(toks))
        return self._finalize(input_ids, loss_mask)

    def _finalize(self, input_ids: list[int], loss_mask: list[int]) -> dict[str, torch.Tensor]:
        input_ids = input_ids[: self.max_len]
        loss_mask = loss_mask[: self.max_len]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        conv = _normalize_roles(self.rows[idx].get("conversations", []))
        is_raw = self.rows[idx].get("raw", False)
        if self.use_chat_template and not is_raw:
            return self._encode_with_template(conv)
        return self._encode_raw(conv)


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Right-pad a batch to the longest sequence."""
    max_len = max(x["input_ids"].shape[0] for x in batch)

    def pad(x: torch.Tensor, value: int) -> torch.Tensor:
        if x.shape[0] == max_len:
            return x
        return torch.cat([x, x.new_full((max_len - x.shape[0],), value)])

    return {
        "input_ids": torch.stack([pad(x["input_ids"], pad_id) for x in batch]),
        "attention_mask": torch.stack([pad(x["attention_mask"], 0) for x in batch]),
        "loss_mask": torch.stack([pad(x["loss_mask"], 0) for x in batch]),
    }
