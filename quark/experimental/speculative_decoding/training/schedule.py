#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Learning-rate schedules for cold-start EAGLE-3 training.

Cold-start is sensitive to the cosine horizon: it MUST be computed against the
*actual* total step count (a mismatch means the LR never anneals and the draft
under-trains). ``build_scheduler`` derives the horizon from the resolved steps.
"""

from __future__ import annotations

import math
from typing import Any

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer: Any, total_steps: int, warmup_ratio: float, schedule: str = "cosine") -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        if schedule == "constant":
            return 1.0
        if schedule == "linear":
            return 1.0 - progress
        # cosine (default): anneal to ~0 over the true horizon.
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
