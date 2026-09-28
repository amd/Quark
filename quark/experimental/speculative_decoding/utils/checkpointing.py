#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Crash-resilient checkpointing for cold-start training.

Each checkpoint is a directory ``iter_<step>/`` holding the trainer state (draft
weights, optimizer, scheduler, step) plus the draft config, so training can
auto-resume after a dead extraction engine (the watchdog's job).
"""

from __future__ import annotations

import os
import re
from typing import Any

import torch

_ITER_RE = re.compile(r"iter_(\d+)$")


def save_checkpoint(output_dir: str, step: int, draft: Any, optim: Any, scheduler: Any, eagle_config: Any) -> str:
    ckpt_dir = os.path.join(output_dir, f"iter_{step:08d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "draft": draft.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        os.path.join(ckpt_dir, "trainer_state.pt"),
    )
    eagle_config.save_pretrained(ckpt_dir)
    return ckpt_dir


def find_latest_checkpoint(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None
    best_step, best_dir = -1, None
    for name in os.listdir(output_dir):
        m = _ITER_RE.match(name)
        if m and os.path.exists(os.path.join(output_dir, name, "trainer_state.pt")):
            step = int(m.group(1))
            if step > best_step:
                best_step, best_dir = step, os.path.join(output_dir, name)
    return best_dir
