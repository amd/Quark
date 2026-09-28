#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""QAD (Quantization-Aware Distillation) Trainer package.

Core trainers:
    QADTrainer     — extends HuggingFace Trainer with multi-mode KD
    QADSFTTrainer  — extends TRL SFTTrainer with the same KD logic

Subpackages:
    losses   — individual KD loss functions (KL, JSD, hidden, attention, masked, on-policy)
    datasets — ready-to-use LM datasets (WikiText, Pile, TaskMix with 25+ task formatters)
"""

from .datasets import PileLMDataset, TaskMixDataset, WikiTextLMDataset
from .losses import (
    build_completion_mask,
    collect_on_policy_prompts,
    kd_attention_loss,
    kd_hidden_loss,
    kd_jsd_loss,
    kd_logit_loss,
    masked_ce_loss,
    masked_kd_loss,
    on_policy_kd_step,
)
from .qad_trainer import QADSFTTrainer, QADTrainer

__all__ = [
    # Trainers
    "QADTrainer",
    "QADSFTTrainer",
    # Loss functions
    "kd_logit_loss",
    "kd_jsd_loss",
    "kd_hidden_loss",
    "kd_attention_loss",
    "build_completion_mask",
    "masked_kd_loss",
    "masked_ce_loss",
    "collect_on_policy_prompts",
    "on_policy_kd_step",
    # Datasets
    "WikiTextLMDataset",
    "PileLMDataset",
    "TaskMixDataset",
]
