#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Draft-vocabulary calibration (optional lm_head compression).

A smaller draft vocab shrinks the (expensive) training-time softmax and the
draft ``lm_head``. We pick the most frequent ``draft_vocab_size`` target tokens
on the training corpus and build the two tensors in the EAGLE-3 on-disk
convention that serving engines expect:

* ``d2t``: draft-id -> target-id **offset**, i.e. ``target_id - draft_id``
  (shape ``[draft_vocab_size]``, int64)
* ``t2d``: bool mask, ``True`` where a target id is in the draft vocab
  (shape ``[vocab_size]``)

``d2t`` is an offset rather than an absolute id because vLLM's
``Eagle3LlamaForCausalLM.compute_logits`` scatters draft logits with
``arange(draft_vocab_size) + d2t``; absolute ids would double-count and index
past ``vocab_size``.

We also report coverage so multilingual/Chinese blow-ups are visible.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

import torch

from quark.experimental.speculative_decoding.data.datasets import _normalize_roles, load_conversations


def calibrate_draft_vocab(
    tokenizer: Any,
    data: str,
    draft_vocab_size: int = 32000,
    save_dir: str | None = None,
    limit: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute ``d2t``/``t2d`` from token frequency on ``data``.

    Returns a dict ``{"d2t", "t2d", "coverage"}``. If ``save_dir`` is given,
    writes ``d2t.pt`` (a serialized version of the same dict).
    """
    vocab_size = len(tokenizer)
    counter: Counter[int] = Counter()
    rows = load_conversations(data, limit=limit)
    for row in rows:
        for msg in _normalize_roles(row.get("conversations", [])):
            ids = tokenizer(msg["content"], add_special_tokens=False)["input_ids"]
            counter.update(ids)

    draft_vocab_size = min(draft_vocab_size, vocab_size)
    # Always keep special tokens so generation stays well-formed.
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    most_common = [tok for tok, _ in counter.most_common()]
    chosen: list[int] = []
    seen: set[int] = set()
    for tok in list(special) + most_common:
        if tok not in seen and tok < vocab_size:
            chosen.append(tok)
            seen.add(tok)
        if len(chosen) >= draft_vocab_size:
            break
    # Pad with unused ids if the corpus was tiny.
    if len(chosen) < draft_vocab_size:
        for tok in range(vocab_size):
            if tok not in seen:
                chosen.append(tok)
                seen.add(tok)
            if len(chosen) >= draft_vocab_size:
                break

    target_ids = torch.tensor(sorted(chosen), dtype=torch.long)
    draft_ids = torch.arange(target_ids.shape[0], dtype=torch.long)
    d2t = target_ids - draft_ids
    t2d = torch.zeros(vocab_size, dtype=torch.bool)
    t2d[target_ids] = True

    total = sum(counter.values()) or 1
    covered = sum(c for tok, c in counter.items() if bool(t2d[tok]))
    coverage = covered / total

    out = {"d2t": d2t, "t2d": t2d, "coverage": torch.tensor(coverage)}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(out, os.path.join(save_dir, "d2t.pt"))
    print(f"[calibrate_draft_vocab] draft_vocab={d2t.shape[0]} coverage={coverage:.4f}")
    return out
