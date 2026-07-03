#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Smoke tests for QADTrainer and QADSFTTrainer: MXFP4 quantized opt-125m as student, fp opt-125m as teacher."""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from quark.common.utils.testing_utils import torch_device
from quark.torch import LLMTemplate, ModelQuantizer
from quark.torch.algorithm.qad_trainer import QADSFTTrainer, QADTrainer


def _force_single_gpu_no_dataparallel(args: TrainingArguments) -> None:
    """Skip ``Trainer._wrap_model`` path ``model = nn.DataParallel(model)``.

    In ``transformers.trainer.Trainer._wrap_model``, DataParallel is applied when
    ``self.args.n_gpu > 1`` (multi-GPU, non-distributed). That wrapper does not expose
    custom attrs like ``.teacher`` on the outer module. Forcing ``_n_gpu == 1`` after
    ``Trainer`` construction avoids the wrap; must run before ``train()`` calls ``_wrap_model``.
    """
    args._n_gpu = 1


class _FakeLMDataset(Dataset[dict[str, torch.Tensor]]):
    """Minimal in-memory dataset for Trainer: input_ids, attention_mask, labels."""

    # SFTTrainer (QADSFTTrainer) expects dataset.column_names
    column_names = ["input_ids", "attention_mask", "labels"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_samples: int = 4,
        seq_len: int = 16,
    ):
        self.seq_len = seq_len
        vocab_size = getattr(tokenizer, "vocab_size", 50257)
        # Fake token ids in valid range; keep on CPU, Trainer moves to device
        self._input_ids = torch.randint(0, min(vocab_size, 50257), (num_samples, seq_len))
        self._attention_mask = torch.ones_like(self._input_ids)
        self._labels = self._input_ids.clone()

    def __len__(self) -> int:
        return self._input_ids.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._input_ids[idx].clone(),
            "attention_mask": self._attention_mask[idx].clone(),
            "labels": self._labels[idx].clone(),
        }

    def map(self, function=None, *args: object, **kwargs: object) -> "_FakeLMDataset":
        """Compatibility for SFTTrainer: dataset.map() is called; return self (data already in expected format)."""
        return self


def _get_calib_dataloader(model_name: str, device: torch.device) -> DataLoader:
    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenizer(text, return_tensors="pt")
    return DataLoader(tokenized["input_ids"].to(device), batch_size=1)


def _build_student_and_teacher(
    model_name: str = "facebook/opt-125m",
    num_layers: int = 2,
):
    """Build quantized student (2-layer opt-125m, MXFP4) and fp teacher (same structure)."""
    config = LLMTemplate.get("opt").get_config(scheme="mxfp4")

    quantizer = ModelQuantizer(config)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    model.eval()
    # model = model.to(torch_device)
    model.model.decoder.layers = model.model.decoder.layers[:num_layers]
    model.config.num_hidden_layers = num_layers
    model = model.to(torch_device)

    device = next(model.parameters()).device
    calib = _get_calib_dataloader(model_name, device)
    student = quantizer.quantize_model(model, calib)

    for p in student.parameters():  # [x[0] for x in student.named_parameters()]
        p.requires_grad = True

    # init the teacher model
    teacher = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    teacher.eval()
    teacher.model.decoder.layers = teacher.model.decoder.layers[:num_layers]
    teacher.config.num_hidden_layers = num_layers
    teacher = teacher.to(torch_device)
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher


def test_smoke_qad_trainer():
    """Run a few QAD steps: student=MXFP4 opt-125m (2 layers), teacher=fp opt-125m. No QLoRA."""
    model_name = "facebook/opt-125m"
    student_model, teacher_model = _build_student_and_teacher(model_name, num_layers=2)
    student_model.teacher = teacher_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train_dataset = _FakeLMDataset(tokenizer, num_samples=4, seq_len=16)

    training_args = TrainingArguments(
        output_dir="/tmp/quark_qad_smoke",
        max_steps=2,
        per_device_train_batch_size=2,
        logging_steps=1,
        report_to="none",
    )
    student_model.enable_input_require_grads()
    if getattr(student_model, "supports_gradient_checkpointing", False):
        student_model.config.use_cache = False
        student_model.gradient_checkpointing_enable()

    trainer = QADTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    _force_single_gpu_no_dataparallel(trainer.args)
    trainer.train()


def test_smoke_qad_sft_trainer():
    """Same as test_smoke_qad_trainer but uses QADSFTTrainer (SFT + QAD). Same model and dataset."""
    model_name = "facebook/opt-125m"
    student_model, teacher_model = _build_student_and_teacher(model_name, num_layers=2)
    student_model.teacher = teacher_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train_dataset = _FakeLMDataset(tokenizer, num_samples=4, seq_len=16)

    training_args = TrainingArguments(
        output_dir="/tmp/quark_qad_sft_smoke",
        max_steps=2,
        per_device_train_batch_size=2,
        logging_steps=1,
        report_to="none",
    )
    student_model.enable_input_require_grads()
    if getattr(student_model, "supports_gradient_checkpointing", False):
        student_model.config.use_cache = False
        student_model.gradient_checkpointing_enable()

    trainer = QADSFTTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    _force_single_gpu_no_dataparallel(trainer.args)
    trainer.train()


if __name__ == "__main__":
    test_smoke_qad_trainer()
    test_smoke_qad_sft_trainer()
