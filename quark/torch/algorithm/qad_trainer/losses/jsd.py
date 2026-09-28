#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Jensen-Shannon Divergence logit distillation loss.

Used when ``kd_loss_type="jsd"`` in QADTrainer. JSD is a symmetric, bounded
alternative to KL-divergence. For large-vocabulary models (e.g. 150k+ tokens),
the computation is chunked across the vocab dimension to avoid holding multiple
full-vocab softmax tensors in memory simultaneously.
"""

import torch
import torch.nn.functional as F


def kd_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    beta: float = 0.5,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Jensen-Shannon Divergence between teacher and student distributions.

    JSD is symmetric and interpolates between forward and reverse KL.
    Chunks across the vocabulary dimension to avoid holding multiple
    full-vocab tensors simultaneously (prevents OOM on large-vocab models).
    """
    s = student_logits.reshape(-1, student_logits.size(-1)) / temperature
    t = teacher_logits.reshape(-1, teacher_logits.size(-1)) / temperature
    n_tokens = s.size(0)
    vocab = s.size(-1)

    if vocab <= chunk_size:
        p_t = F.softmax(t, dim=-1)
        p_s = F.softmax(s, dim=-1)
        p_m = beta * p_t + (1 - beta) * p_s
        log_m = torch.log(p_m + 1e-8)
        jsd = beta * F.kl_div(log_m, p_t, reduction="sum") + (1 - beta) * F.kl_div(log_m, p_s, reduction="sum")
        return (jsd / n_tokens) * (temperature**2)

    p_t = F.softmax(t, dim=-1)
    p_s = F.softmax(s, dim=-1)
    del s, t

    jsd = torch.tensor(0.0, device=student_logits.device)
    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        pt_c = p_t[:, start:end]
        ps_c = p_s[:, start:end]
        pm_c = beta * pt_c + (1 - beta) * ps_c
        log_m_c = torch.log(pm_c + 1e-8)
        jsd = (
            jsd
            + beta * F.kl_div(log_m_c, pt_c, reduction="sum")
            + (1 - beta) * F.kl_div(log_m_c, ps_c, reduction="sum")
        )
        del pt_c, ps_c, pm_c, log_m_c

    del p_t, p_s
    return (jsd / n_tokens) * (temperature**2)
