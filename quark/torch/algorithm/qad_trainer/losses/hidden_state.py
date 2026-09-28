#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Hidden-state MSE distillation loss.

Used when ``kd_mode="layer"`` or ``kd_mode="attention"`` in QADTrainer.
Computes MSE between corresponding layers of student and teacher hidden states,
averaged across layers. Forces the student's internal representations to
approximate the teacher's, not just the final output distribution.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def kd_hidden_loss(student_hiddens: Sequence[torch.Tensor], teacher_hiddens: Sequence[torch.Tensor]) -> torch.Tensor:
    """Normalized MSE between hidden states at each layer.

    Normalizes by hidden dimension so the loss doesn't scale with model width.
    """
    loss = torch.tensor(0.0, device=student_hiddens[0].device)
    n = min(len(student_hiddens), len(teacher_hiddens))
    for i in range(n):
        sh = student_hiddens[i].float()
        th = teacher_hiddens[i].float()
        loss = loss + F.mse_loss(sh, th)
    return loss / max(n, 1)
