#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Completion-masked KD and CE losses.

Used when ``completion_loss_only=True`` in QADTrainer. These functions restrict
the loss computation to only the "answer" portion of each sequence (tokens after
a marker like "Answer:"). This is useful for task-formatted data where the prompt
portion is identical between teacher and student and loss on it is wasted.
"""

from typing import Any

import torch
import torch.nn.functional as F


def build_completion_mask(input_ids: torch.Tensor, tokenizer: Any, marker_text: str = "Answer:") -> torch.Tensor:
    """Build a binary mask that is 1 for completion tokens and 0 for prompt tokens.

    For each sequence in the batch, finds the LAST occurrence of marker_text
    and sets the mask to 1 for all tokens after it. If the marker is not found,
    the entire sequence is unmasked (all 1s) to avoid zero gradients.
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.ones(batch_size, seq_len, device=input_ids.device, dtype=torch.float32)

    marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)
    if not marker_ids:
        return mask

    marker_len = len(marker_ids)
    marker_tensor = torch.tensor(marker_ids, device=input_ids.device, dtype=input_ids.dtype)

    n_windows = seq_len - marker_len + 1
    if n_windows <= 0:
        return mask

    # Vectorized last-occurrence search: compare every length-marker_len window
    # against the marker in one shot, instead of a Python B x T double loop with a
    # per-position torch.equal (each of which forces a device sync).
    windows = input_ids.unfold(dimension=1, size=marker_len, step=1)  # [B, n_windows, marker_len]
    matches = (windows == marker_tensor).all(dim=-1)  # [B, n_windows], True where marker starts
    win_idx = torch.arange(n_windows, device=input_ids.device)
    # last matching window index per row (-1 if the marker never occurs)
    last_match = torch.where(matches, win_idx, torch.full_like(win_idx, -1)).amax(dim=1)  # [B]
    # Tokens strictly before (last_match + marker_len) are masked out (0.0). If the
    # marker is absent (last_match == -1) cutoff is 0 -> whole sequence stays 1s,
    # matching the original loop's "unmask everything" fallback.
    cutoff = torch.where(last_match >= 0, last_match + marker_len, torch.zeros_like(last_match))  # [B]
    positions = torch.arange(seq_len, device=input_ids.device)
    return (positions.unsqueeze(0) >= cutoff.unsqueeze(1)).to(mask.dtype)


def masked_kd_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor, temperature: float, loss_fn: str
) -> torch.Tensor:
    """Compute KD loss only on masked (completion) tokens.

    Args:
        student_logits: (B, T, V)
        teacher_logits: (B, T, V)
        mask: (B, T) binary mask, 1 = compute loss here
        temperature: KD temperature
        loss_fn: 'jsd' or 'kl'
    """
    B, T, V = student_logits.shape
    flat_mask = mask[:, :T].reshape(-1)
    active_idx = flat_mask.nonzero(as_tuple=True)[0]

    if active_idx.numel() == 0:
        return (student_logits * 0.0).sum()

    s_flat = student_logits.reshape(-1, V)[active_idx]
    t_flat = teacher_logits.reshape(-1, V)[active_idx]

    s_scaled = s_flat / temperature
    t_scaled = t_flat / temperature

    if loss_fn == "jsd":
        p_t = F.softmax(t_scaled, dim=-1)
        p_s = F.softmax(s_scaled, dim=-1)
        p_m = 0.5 * p_t + 0.5 * p_s
        log_m = torch.log(p_m + 1e-8)
        jsd = 0.5 * F.kl_div(log_m, p_t, reduction="sum") + 0.5 * F.kl_div(log_m, p_s, reduction="sum")
        return (jsd / active_idx.numel()) * (temperature**2)
    else:
        log_s = F.log_softmax(s_scaled, dim=-1)
        prob_t = F.softmax(t_scaled, dim=-1)
        kl = F.kl_div(log_s, prob_t, reduction="sum") / active_idx.numel()
        return kl * (temperature**2)


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss only on masked (completion) tokens."""
    B, T, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = mask[:, 1:T].contiguous()

    flat_logits = shift_logits.reshape(-1, V)
    flat_labels = shift_labels.reshape(-1)
    flat_mask = shift_mask.reshape(-1)

    active_idx = flat_mask.nonzero(as_tuple=True)[0]
    if active_idx.numel() == 0:
        return (logits * 0.0).sum()

    ce = F.cross_entropy(flat_logits[active_idx], flat_labels[active_idx])
    return ce
