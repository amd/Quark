#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""QAD Trainer module."""

import sys
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Trainer
from transformers.training_args import OptimizerNames
from transformers.utils import (  # type: ignore[attr-defined]
    is_accelerate_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

try:
    from trl import SFTTrainer  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    print("trl is required for QADSFTTrainer. Install it with: pip install trl", file=sys.stderr)  # pragma: no cover
    sys.exit(1)  # pragma: no cover

if is_accelerate_available():
    from accelerate.utils import DistributedType

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class QADTrainer(Trainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Step1: Prepare a PTQ model as the student model(student_model).
        Step2: Init a full precision model as the teacher model(teacher_model).
        Step3: let the student_model.teacher = teacher_model
        Then use the QADTrainer as a normal Trainer.
        """
        if not hasattr(self, "temperature"):
            self.temperature: float = float(kwargs.pop("temperature", 1.0))
        super().__init__(*args, **kwargs)
        assert hasattr(self.model, "teacher")

    def kl_div_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute KD loss on student and teacher logits.

        Logits distillation loss.

        Args:
            student_logits: The student model output.
            teacher_logits: The teacher model output.
        """
        model_output_log_prob = F.log_softmax(student_logits / self.temperature, dim=2)
        real_output_soft = F.softmax(teacher_logits / self.temperature, dim=2)

        loss = F.kl_div(model_output_log_prob, real_output_soft, reduction="batchmean")
        loss *= self.temperature**2
        return loss

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform a training step on a batch of inputs.

        By default, will use KL_divergence loss to measure the difference
        between teacher's output and student's output.

        Args:
            model (`nn.Module`): The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`): The inputs and
                targets of the model. The dictionary will be unpacked before
                being fed to the model. Most models expect the targets under
                the argument `labels`. Check your model's documentation for
                all accepted arguments.

        Returns:
            `torch.Tensor`: The tensor with training loss on this batch.

        NOTE: inherent version: based on transformers version 4.57.
        """
        # Prepare buffers for context parallelism
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if self.optimizer is not None and hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            # KD distillation needs raw `logits`, not the model's internal loss. Drop `labels`
            # so the forward materializes `logits`; TRL's loss path (e.g. chunked NLL) otherwise
            # returns logits=None when labels are present.
            inputs.pop("labels", None)
            with self.compute_loss_context_manager():  # type: ignore[no-untyped-call]
                with torch.no_grad():
                    teacher_outputs = model.teacher(**inputs)
                    teacher_logits = teacher_outputs.get("logits")
                    del teacher_outputs

                # forward pass
                student_outputs = model(**inputs)
                student_logits = student_outputs.get("logits")
                del student_outputs

                loss = self.kl_div_loss(student_logits, teacher_logits)
                del teacher_logits
                del student_logits

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):  # pragma: no cover
                if is_torch_xpu_available():  # pragma: no cover
                    torch.xpu.empty_cache()  # pragma: no cover
                elif is_torch_mlu_available():  # pragma: no cover
                    torch.mlu.empty_cache()  # pragma: no cover
                elif is_torch_musa_available():  # pragma: no cover
                    torch.musa.empty_cache()  # pragma: no cover
                elif is_torch_npu_available():  # pragma: no cover
                    torch.npu.empty_cache()  # pragma: no cover
                elif is_torch_mps_available():  # pragma: no cover
                    torch.mps.empty_cache()  # pragma: no cover
                elif is_torch_hpu_available():  # pragma: no cover
                    logger.warning(
                        "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                    )  # pragma: no cover
                else:  # pragma: no cover
                    torch.cuda.empty_cache()  # pragma: no cover

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learning rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()  # pragma: no cover

        if self.args.n_gpu > 1:
            loss = loss.mean()  # pragma: no cover

        if getattr(self, "use_apex", False):
            from apex import amp  # type: ignore[import-not-found]  # pragma: no cover

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:  # pragma: no cover
                scaled_loss.backward()  # pragma: no cover
        else:
            # Normalize loss for reporting if GA loss bug not fixed in compute loss
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                loss = loss / self.current_gradient_accumulation_steps  # pragma: no cover

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False  # pragma: no cover

            self.accelerator.backward(loss, **kwargs)

        return loss.detach()


class QADSFTTrainer(SFTTrainer, QADTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the QADSFTtrainer."""
        self.temperature: float = float(kwargs.pop("temperature", 1.0))
        SFTTrainer.__init__(self, *args, **kwargs)
        assert hasattr(self.model, "teacher")

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inherit from SFT trainer.

        This training_step uses QADTrainer.training_step to perform the step.
        """
        with self.maybe_activation_offload_context:
            return QADTrainer.training_step(self, model, inputs, num_items_in_batch)
