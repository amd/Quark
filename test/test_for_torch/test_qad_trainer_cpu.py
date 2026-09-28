#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""CPU tests for QADTrainer.training_step across all KD modes/branches.

Drives ``training_step`` directly with a tiny CPU model (student + teacher) so
every KD path (kl/jsd, output/layer/attention, completion masking, on-policy,
loss-breakdown logging) is exercised without a GPU or a quantized model.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from quark.torch.algorithm.qad_trainer import QADTrainer

VOCAB = 32


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = VOCAB, d: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
        self.config = SimpleNamespace(vocab_size=vocab)

    def forward(self, input_ids=None, labels=None, output_hidden_states=False, output_attentions=False, **kw):
        h = self.emb(input_ids)
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        hs = (h, h) if output_hidden_states else None
        at = None
        if output_attentions:
            b, t = input_ids.shape
            at = (torch.softmax(torch.randn(b, 2, t, t), dim=-1),)
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=hs, attentions=at)

    def generate(self, input_ids, max_new_tokens, **kw):
        b = input_ids.shape[0]
        extra = torch.randint(0, VOCAB, (b, max_new_tokens))
        return torch.cat([input_ids, extra], dim=1)


class _DS(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        ids = torch.randint(0, VOCAB, (8,))
        return {"input_ids": ids, "labels": ids.clone()}


class _Tok:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [5, 6]


def _make_trainer(**kd_kwargs) -> QADTrainer:
    student, teacher = _TinyLM(), _TinyLM()
    student.teacher = teacher
    args = TrainingArguments(
        output_dir="/tmp/qad_cpu_test",
        max_steps=10,
        per_device_train_batch_size=2,
        logging_steps=1,
        report_to="none",
        use_cpu=True,
    )
    tr = QADTrainer(model=student, args=args, train_dataset=_DS(), **kd_kwargs)
    tr.create_optimizer()
    # Normally set inside Trainer.train(); required by the backward path.
    tr.current_gradient_accumulation_steps = 1
    return tr


def _inputs():
    return {
        "input_ids": torch.randint(0, VOCAB, (2, 8)),
        "labels": torch.randint(0, VOCAB, (2, 8)),
    }


def test_training_step_output_kl():
    tr = _make_trainer(kd_loss_type="kl")
    # Exercise the schedule-free-style optimizer.train() hook if present.
    tr.optimizer.train = lambda: None
    loss = tr.training_step(tr.model, _inputs())
    assert torch.isfinite(loss)


def test_training_step_output_jsd():
    tr = _make_trainer(kd_loss_type="jsd", kd_alpha=0.5)
    loss = tr.training_step(tr.model, _inputs())
    assert torch.isfinite(loss)


def test_training_step_completion_masking():
    tr = _make_trainer(completion_loss_only=True, kd_tokenizer=_Tok())
    loss = tr.training_step(tr.model, _inputs())
    assert torch.isfinite(loss)


def test_training_step_layer_mode():
    tr = _make_trainer(kd_mode="layer", kd_hidden_weight=0.5)
    loss = tr.training_step(tr.model, _inputs())
    assert torch.isfinite(loss)


def test_training_step_attention_mode_with_breakdown():
    tr = _make_trainer(kd_mode="attention", log_loss_breakdown=True)
    loss = tr.training_step(tr.model, _inputs())
    assert torch.isfinite(loss)


def test_training_step_on_policy():
    tr = _make_trainer(
        on_policy_kd=True,
        on_policy_every=5,
        on_policy_max_gen_len=4,
        on_policy_prompt_len=4,
        on_policy_alpha=0.3,
        log_loss_breakdown=True,
        kd_tokenizer=_Tok(),
    )
    tr.state.global_step = 5  # multiple of on_policy_every -> runs on-policy step
    tr.state.max_steps = 10
    loss = tr.training_step(tr.model, _inputs())
    assert torch.isfinite(loss)


class _TokEosOnly:
    pad_token_id = None
    eos_token_id = 2


class _TokNoPad:
    pad_token_id = None
    eos_token_id = None


def test_training_step_on_policy_eos_fallback():
    tr = _make_trainer(
        on_policy_kd=True,
        on_policy_every=5,
        on_policy_max_gen_len=4,
        on_policy_prompt_len=4,
        on_policy_alpha=0.3,
        kd_tokenizer=_TokEosOnly(),
    )
    tr.state.global_step = 5
    tr.state.max_steps = 10
    assert torch.isfinite(tr.training_step(tr.model, _inputs()))


def test_training_step_on_policy_no_pad_warns():
    tr = _make_trainer(
        on_policy_kd=True,
        on_policy_every=5,
        on_policy_max_gen_len=4,
        on_policy_prompt_len=4,
        on_policy_alpha=0.3,
        kd_tokenizer=_TokNoPad(),
    )
    tr.state.global_step = 5
    tr.state.max_steps = 10
    assert torch.isfinite(tr.training_step(tr.model, _inputs()))


def test_on_policy_helpers():
    tr = _make_trainer(on_policy_kd=True, on_policy_every=5, on_policy_alpha=0.3)
    # ramp: early progress scales linearly; later plateaus at alpha
    tr.state.global_step = 1
    tr.state.max_steps = 100
    assert 0.0 <= tr._get_on_policy_weight() < 0.3
    tr.state.global_step = 90
    assert tr._get_on_policy_weight() == 0.3
    # should_run_on_policy: only once per qualifying step
    tr.state.global_step = 5
    tr._on_policy_done_for_step = -1
    assert tr._should_run_on_policy() is True
    assert tr._should_run_on_policy() is False  # already done for this step
    tr.state.global_step = 7  # not a multiple of 5
    assert tr._should_run_on_policy() is False


def test_kl_div_loss_helper():
    tr = _make_trainer()
    s = torch.randn(2, 4, VOCAB)
    t = torch.randn(2, 4, VOCAB)
    assert torch.isfinite(tr.kl_div_loss(s, t))
    assert torch.isfinite(tr.kl_div_loss(s, t, temperature=2.0))
