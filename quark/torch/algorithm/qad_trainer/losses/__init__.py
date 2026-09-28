#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""KD loss functions used by QADTrainer.

Logit distillation (used in all modes):
    kd_logit_loss  — KL-divergence on temperature-scaled logits
    kd_jsd_loss    — Jensen-Shannon divergence with vocab chunking for OOM safety

Intermediate distillation (layer / attention modes):
    kd_hidden_loss    — MSE between student and teacher hidden states
    kd_attention_loss — KL between student and teacher attention maps

Completion masking (optional, restricts loss to answer tokens only):
    build_completion_mask — locates the marker (e.g. "Answer:") in each sequence
    masked_kd_loss        — KD loss computed only on unmasked (completion) tokens
    masked_ce_loss        — CE loss computed only on unmasked (completion) tokens

On-policy KD (standalone utility, not yet wired into QADTrainer.training_step):
    collect_on_policy_prompts — extract prompt prefixes from training data
    on_policy_kd_step         — student generates, teacher supervises
"""

from .attention import kd_attention_loss
from .hidden_state import kd_hidden_loss
from .jsd import kd_jsd_loss
from .kl_divergence import kd_logit_loss
from .masked import build_completion_mask, masked_ce_loss, masked_kd_loss
from .on_policy import collect_on_policy_prompts, on_policy_kd_step

__all__ = [
    "kd_logit_loss",
    "kd_jsd_loss",
    "kd_hidden_loss",
    "kd_attention_loss",
    "build_completion_mask",
    "masked_kd_loss",
    "masked_ce_loss",
    "collect_on_policy_prompts",
    "on_policy_kd_step",
]
