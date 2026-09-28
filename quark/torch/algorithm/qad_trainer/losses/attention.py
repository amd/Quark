#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Attention-map KL distillation loss.

Used when ``kd_mode="attention"`` in QADTrainer. Computes KL-divergence between
teacher and student attention distributions at each layer, averaged across
layers, heads, and query positions. Helps the student replicate the teacher's
attention patterns (which tokens attend to which).
"""

from collections.abc import Sequence

import torch


def kd_attention_loss(student_attns: Sequence[torch.Tensor], teacher_attns: Sequence[torch.Tensor]) -> torch.Tensor:
    """KL-divergence between attention distributions, averaged per-position.

    Attention maps are already probability distributions over the last dim (keys).
    We compute per-query-position KL and average over (batch, heads, queries).
    """
    loss = torch.tensor(0.0, device=student_attns[0].device)
    n = min(len(student_attns), len(teacher_attns))
    for i in range(n):
        sa = student_attns[i].float().clamp(min=1e-8)
        ta = teacher_attns[i].float().clamp(min=1e-8)
        kl = (ta * (ta.log() - sa.log())).sum(dim=-1).mean()
        loss = loss + kl
    return loss / max(n, 1)
