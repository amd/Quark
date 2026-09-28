#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Training-time-test (TTT) loss for EAGLE-3.

The draft is trained to predict the next ``ttt_length`` tokens. Later positions
in the unroll are harder and noisier, so their contribution is down-weighted by
``position_decay`` (position-decay weighting). We also report a simulated
acceptance length (``sim_acc_len``) as a cheap in-training proxy -- but note it
systematically *overestimates* real served AL, which is exactly why serve-eval
in the loop (not this metric) gates checkpoint selection.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ttt_loss(
    step_logits: list[torch.Tensor],
    step_targets: list[torch.Tensor],
    loss_mask: torch.Tensor,
    position_decay: float = 0.9,
) -> dict[str, torch.Tensor]:
    """Weighted cross-entropy over a TTT unroll.

    Args:
        step_logits:  list (len = unroll depth) of draft logits ``[B, T, Vdraft]``.
        step_targets: list of target token ids ``[B, T]`` (already mapped to the
                      draft vocab; ``-100`` marks ignored/OOV positions).
        loss_mask:    ``[B, T]`` float mask (1 = supervised position).
        position_decay: per-step weight decay across the unroll.

    Returns a dict with ``loss`` (scalar), ``sim_acc_len`` and per-step
    ``acc@k`` top-1 accuracies for logging.
    """
    total_loss = step_logits[0].new_zeros(())
    weight_sum = 0.0
    per_step_acc: list[torch.Tensor] = []

    mask = loss_mask.bool()
    for k, (logits, targets) in enumerate(zip(step_logits, step_targets, strict=False)):
        w = position_decay**k
        v = logits.shape[-1]
        ce = F.cross_entropy(
            logits.reshape(-1, v),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(targets.shape)
        step_mask = mask & (targets != -100)
        denom = step_mask.sum().clamp_min(1)
        step_loss = (ce * step_mask).sum() / denom
        total_loss = total_loss + w * step_loss
        weight_sum += w

        with torch.no_grad():
            pred = logits.argmax(-1)
            correct = ((pred == targets) & step_mask).sum().float()
            per_step_acc.append(correct / denom)

    total_loss = total_loss / max(weight_sum, 1e-8)

    # sim_acc_len: expected run length if we accept while consecutive top-1 hits.
    with torch.no_grad():
        acc = torch.stack(per_step_acc) if per_step_acc else step_logits[0].new_zeros(1)
        sim_acc_len = torch.ones((), device=acc.device)
        running = torch.ones((), device=acc.device)
        for a in acc:
            running = running * a
            sim_acc_len = sim_acc_len + running

    out: dict[str, torch.Tensor] = {"loss": total_loss, "sim_acc_len": sim_acc_len}
    for k, a in enumerate(per_step_acc):
        out[f"acc@{k}"] = a
    return out
