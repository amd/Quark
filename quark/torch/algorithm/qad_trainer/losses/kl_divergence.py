#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""KL-divergence logit distillation loss.

Used when ``kd_loss_type="kl"`` in QADTrainer. Computes per-token KL-divergence
between temperature-scaled teacher and student distributions, averaged over the
batch. Scaled by T^2 so gradients are comparable across temperature values.
"""

import torch
import torch.nn.functional as F


def kd_logit_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """KL-divergence between teacher and student output distributions.

    Computes per-token KL then averages over (batch * seq_len) so the scale
    is comparable to the per-token CE loss (~2-5 range).
    """
    s = student_logits.reshape(-1, student_logits.size(-1))
    t = teacher_logits.reshape(-1, teacher_logits.size(-1))
    log_s = F.log_softmax(s / temperature, dim=-1)
    prob_t = F.softmax(t / temperature, dim=-1)
    kl = F.kl_div(log_s, prob_t, reduction="sum") / s.size(0)
    return kl * (temperature**2)
