#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""On-policy KD: student generates, teacher supervises.

Standalone utility — not yet wired into QADTrainer.training_step.
Can be called directly for custom training loops that need on-policy KD.

On-policy KD addresses exposure bias: during standard (off-policy) KD, the
student only sees teacher-forced ground-truth inputs. At inference time the
student feeds its own predictions back, causing error compounding. On-policy
KD lets the student generate sequences and then trains it on those sequences
with teacher supervision.
"""

from collections.abc import Callable, Iterable, Iterator

import torch

Batch = dict[str, torch.Tensor]


@torch.no_grad()
def collect_on_policy_prompts(
    train_loader: Iterable[Batch],
    data_iter: Iterator[Batch],
    prompt_len: int,
    num_prompts: int,
    multi_gpu: bool = False,
) -> tuple[torch.Tensor, Iterator[Batch]]:
    """Extract prompt prefixes from training data for on-policy generation."""
    prompts = []
    for _ in range(num_prompts):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        ids = batch["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        prompts.append(ids[:, :prompt_len])
    return torch.cat(prompts, dim=0), data_iter


def on_policy_kd_step(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_gen_len: int,
    gen_temperature: float,
    kd_temperature: float,
    kd_loss_fn: Callable[..., torch.Tensor],
    kd_alpha: float,
    multi_gpu: bool = False,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor | None, float | None]:
    """On-policy KD: student generates a sequence, teacher provides supervision.

    1. Student generates tokens autoregressively (no gradient).
    2. Teacher computes logits on the student-generated sequence.
    3. Student re-computes logits with gradient on its own generated sequence.
    4. KD + CE loss is computed on the generated (non-prompt) portion only.

    This directly addresses error compounding: the student learns to handle
    its own distributional drift, not just teacher-forced ground truth.
    """
    student.eval()
    with torch.no_grad():
        generated = student.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_gen_len,
            do_sample=True,
            temperature=gen_temperature,
            top_p=0.9,
            pad_token_id=pad_token_id,
        )
    student.train()

    prompt_len = prompt_ids.shape[1]
    if generated.shape[1] <= prompt_len + 1:
        return None, None

    with torch.inference_mode():
        teacher_out = teacher(input_ids=generated)

    labels = generated.clone()
    labels[:, :prompt_len] = -100
    student_out = student(input_ids=generated, labels=labels)

    loss_device = student_out.logits.device
    teacher_logits = teacher_out.logits.detach().to(loss_device)

    gen_student_logits = student_out.logits[:, prompt_len - 1 : -1, :]
    gen_teacher_logits = teacher_logits[:, prompt_len - 1 : -1, :]

    kd_loss = kd_loss_fn(gen_student_logits, gen_teacher_logits, kd_temperature)
    ce_loss = student_out.loss if student_out.loss is not None else torch.tensor(0.0, device=loss_device)

    total_loss = kd_alpha * kd_loss + (1.0 - kd_alpha) * ce_loss

    del teacher_out, teacher_logits, student_out, generated, labels
    torch.cuda.empty_cache()

    return total_loss, kd_loss.item()
