#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""QAD Trainer module — Knowledge Distillation trainer extending HuggingFace Trainer.

Supports multiple KD modes (output/layer/attention), loss types (KL/JSD),
on-policy KD, and completion masking.
"""

import sys
from collections.abc import Callable
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
    SFTTrainer = None  # pragma: no cover

if is_accelerate_available():
    from accelerate.utils import DistributedType

from quark.common.utils.log import ScreenLogger

from .losses import (
    build_completion_mask,
    kd_attention_loss,
    kd_hidden_loss,
    kd_jsd_loss,
    masked_ce_loss,
    masked_kd_loss,
)

logger = ScreenLogger(__name__)


class QADTrainer(Trainer):
    """Knowledge Distillation trainer that extends HuggingFace Trainer.

    Usage::

        student.teacher = teacher          # attach frozen teacher
        trainer = QADTrainer(
            model=student,
            args=training_args,
            train_dataset=dataset,
            temperature=1.2,               # KD softmax temperature
            kd_mode="output",              # "output" | "layer" | "attention"
            kd_loss_type="jsd",            # "kl" | "jsd"
            kd_alpha=0.85,                 # blend: alpha*KD + (1-alpha)*CE
        )
        trainer.train()

    KD modes:
        ``output``    — logit distillation only (default, used by our Qwen3 config)
        ``layer``     — logit + hidden-state MSE distillation
        ``attention`` — logit + hidden-state + attention-map KL distillation

    Loss types:
        ``kl``  — standard KL-divergence on softened logits
        ``jsd`` — Jensen-Shannon divergence with vocab chunking for OOM safety

    Optional features:
        - **Completion masking**: restrict loss to answer tokens only
          (``completion_loss_only=True``, ``completion_marker="Answer:"``)
        - **On-policy KD**: student generates sequences and teacher supervises,
          addressing exposure bias. Enable with ``on_policy_kd=True``.
          Runs every ``on_policy_every`` optimizer steps with a linearly ramped
          weight (0 → ``on_policy_alpha`` over first 30% of training).
    """

    def __init__(
        self,
        *args: Any,
        # ── Core KD parameters ──────────────────────────────────
        temperature: float = 1.0,  # softmax temperature for logit KD
        kd_mode: str = "output",  # distillation scope: output | layer | attention
        kd_loss_type: str = "kl",  # logit loss function: kl | jsd
        kd_alpha: float = 1.0,  # loss blend: alpha*KD + (1-alpha)*CE (default 1.0 = pure KD)
        kd_hidden_weight: float = 1.0,  # weight for hidden-state MSE (layer/attention modes)
        kd_attn_weight: float = 1.0,  # weight for attention KL (attention mode only)
        # ── On-policy KD ─────────────────────────────────────────
        on_policy_kd: bool = False,
        on_policy_every: int = 5,
        on_policy_max_gen_len: int = 256,
        on_policy_prompt_len: int = 64,
        on_policy_temperature: float = 0.7,
        on_policy_alpha: float = 0.3,
        # ── Completion masking ──────────────────────────────────
        completion_loss_only: bool = False,  # if True, only compute loss on answer tokens
        completion_marker: str = "Answer:",  # text boundary between prompt and completion
        kd_tokenizer: Any | None = None,  # tokenizer for finding the completion marker
        # ── Verbose loss logging (total / KD / CE) ─────────────────
        log_loss_breakdown: bool = False,
        **kwargs: Any,
    ) -> None:
        # --- Core KD config ---
        if not hasattr(self, "temperature"):
            self.temperature: float = temperature
        self.kd_mode = kd_mode
        self.kd_loss_type = kd_loss_type
        self.kd_alpha = kd_alpha
        self.kd_hidden_weight = kd_hidden_weight
        self.kd_attn_weight = kd_attn_weight

        # --- On-policy KD config ---
        self.on_policy_kd = on_policy_kd
        self.on_policy_every = on_policy_every
        self.on_policy_max_gen_len = on_policy_max_gen_len
        self.on_policy_prompt_len = on_policy_prompt_len
        self.on_policy_temperature = on_policy_temperature
        self.on_policy_alpha = on_policy_alpha
        self._on_policy_done_for_step: int = -1

        # --- Completion masking config ---
        self.completion_loss_only = completion_loss_only
        self.completion_marker = completion_marker
        self.kd_tokenizer = kd_tokenizer
        self.log_loss_breakdown = log_loss_breakdown
        # Micro-step counter for loss-breakdown logging (initialized here, not
        # lazily inside the training hot loop).
        self._qad_micro_step = 0

        # --- Derived state ---
        self._need_hidden = kd_mode in ("layer", "attention")
        self._need_attn = kd_mode == "attention"
        # Default "kl" path delegates to the original batchmean kl_div_loss so that
        # existing consumers (QADTrainer with defaults) keep identical behavior.
        # "jsd" uses the per-token kd_jsd_loss helper.
        if kd_loss_type == "jsd":
            self._kd_loss_fn: Callable[..., torch.Tensor] = kd_jsd_loss
        else:
            self._kd_loss_fn = lambda s, t, _T=None: self.kl_div_loss(s, t, _T)

        super().__init__(*args, **kwargs)
        assert hasattr(self.model, "teacher"), (
            "QADTrainer expects model.teacher to be set. "
            "Attach the frozen teacher model before creating the trainer: student.teacher = teacher"
        )

    def kl_div_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float | None = None
    ) -> torch.Tensor:
        """Legacy KL-div loss kept for backward compatibility with subclasses.

        Not called by QADTrainer.training_step (which uses self._kd_loss_fn from
        the losses module instead). Retained so external code that overrides or
        calls ``trainer.kl_div_loss(...)`` directly still works.

        ``temperature`` defaults to ``self.temperature`` (the current call sites
        always pass ``self.temperature``, so this is behavior-preserving) but is
        honored when provided, matching the ``jsd`` branch which forwards it.
        """
        t = self.temperature if temperature is None else temperature
        model_output_log_prob = F.log_softmax(student_logits / t, dim=2)
        real_output_soft = F.softmax(teacher_logits / t, dim=2)
        loss = F.kl_div(model_output_log_prob, real_output_soft, reduction="batchmean")
        loss *= t**2
        return loss

    def _get_on_policy_weight(self) -> float:
        """Linearly ramp on-policy weight from 0 → on_policy_alpha over first 30% of training."""
        alpha = self.on_policy_alpha
        max_steps = getattr(self.state, "max_steps", 0) or getattr(self.args, "max_steps", 1)
        progress = self.state.global_step / max(max_steps, 1)
        ramp_frac = 0.3
        if progress < ramp_frac:
            return alpha * (progress / ramp_frac)
        return alpha

    def _should_run_on_policy(self) -> bool:
        """True once per optimizer step at the on_policy_every interval."""
        step = self.state.global_step
        if step <= 0 or step % self.on_policy_every != 0:
            return False
        if self._on_policy_done_for_step == step:
            return False
        self._on_policy_done_for_step = step
        return True

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform a training step with multi-mode knowledge distillation.

        Flow:
            1. (Optional) Build completion mask to restrict loss to answer tokens
            2. Forward teacher (no grad) and student through the same inputs
            3. Compute logit KD loss (KL or JSD, full-sequence or masked)
            4. (layer/attention modes) Compute hidden-state MSE and/or attention KL
            5. Blend: ``loss = alpha * KD_total + (1 - alpha) * CE``
            6. (Optional) On-policy KD: student generates, teacher supervises
            7. Backward pass via accelerator
        """
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        with cp_context():
            model.train()
            if self.optimizer is not None and hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)

            # Save prompt prefixes for on-policy KD before inputs are consumed
            _op_prompt_ids = None
            if self.on_policy_kd:
                _input_ids = inputs.get("input_ids")
                if _input_ids is not None:
                    _op_prompt_ids = _input_ids[:, : self.on_policy_prompt_len].clone()

            with self.compute_loss_context_manager():  # type: ignore[no-untyped-call]
                input_ids = inputs.get("input_ids")
                labels = inputs.get("labels")

                # Step 1: Optional completion masking — only compute loss after the marker
                comp_mask = None
                if self.completion_loss_only and self.kd_tokenizer is not None and input_ids is not None:
                    comp_mask = build_completion_mask(input_ids, self.kd_tokenizer, self.completion_marker)

                # Request hidden states / attentions if needed for layer/attention KD
                fwd_kwargs: dict[str, Any] = dict(inputs)
                if self._need_hidden:
                    fwd_kwargs["output_hidden_states"] = True
                if self._need_attn:
                    fwd_kwargs["output_attentions"] = True

                # Step 2: Teacher forward (frozen, no grad) then student forward
                with torch.no_grad():
                    teacher_out = model.teacher(**fwd_kwargs)

                student_out = model(**fwd_kwargs)

                loss_device = student_out.logits.device
                teacher_logits = teacher_out.logits.detach().to(loss_device)

                # Step 3: Logit KD + CE loss (masked or full-sequence)
                if comp_mask is not None and labels is not None:
                    ce_loss = masked_ce_loss(student_out.logits, labels, comp_mask)
                    logit_kd = masked_kd_loss(
                        student_out.logits, teacher_logits, comp_mask, self.temperature, self.kd_loss_type
                    )
                else:
                    ce_loss = (
                        student_out.loss if student_out.loss is not None else torch.tensor(0.0, device=loss_device)
                    )
                    logit_kd = self._kd_loss_fn(student_out.logits, teacher_logits, self.temperature)

                # Step 4: Hidden-state and attention losses (only for layer/attention modes)
                hidden_kd = torch.tensor(0.0, device=loss_device)
                attn_kd = torch.tensor(0.0, device=loss_device)

                if self._need_hidden:
                    teacher_hiddens = [h.to(loss_device) for h in teacher_out.hidden_states]
                    hidden_kd = kd_hidden_loss(student_out.hidden_states, teacher_hiddens)
                    del teacher_hiddens

                if self._need_attn:
                    teacher_attns = [a.to(loss_device) for a in teacher_out.attentions]
                    attn_kd = kd_attention_loss(student_out.attentions, teacher_attns)
                    del teacher_attns

                # Step 5: Final loss = alpha * (logit_kd + hidden + attn) + (1 - alpha) * CE
                kd_total = logit_kd + self.kd_hidden_weight * hidden_kd + self.kd_attn_weight * attn_kd
                loss = self.kd_alpha * kd_total + (1.0 - self.kd_alpha) * ce_loss

                if getattr(self, "log_loss_breakdown", False):
                    self._qad_micro_step += 1
                    interval = max(1, int(self.args.logging_steps)) * max(
                        1, int(getattr(self.args, "gradient_accumulation_steps", 1))
                    )
                    if self._qad_micro_step % interval == 0:
                        lk = float(logit_kd.detach().float().mean().cpu())
                        ce = float(ce_loss.detach().float().mean().cpu())
                        kdt = float(kd_total.detach().float().mean().cpu())
                        lt = float(loss.detach().float().mean().cpu())
                        hid = float(hidden_kd.detach().float().mean().cpu()) if self._need_hidden else 0.0
                        attn_v = float(attn_kd.detach().float().mean().cpu()) if self._need_attn else 0.0
                        a = float(self.kd_alpha)
                        logger.info(
                            "[QAD loss] "
                            f"micro_step={self._qad_micro_step} (log every ~{self.args.logging_steps} "
                            f"optimizer steps, GAS={getattr(self.args, 'gradient_accumulation_steps', 1)}) | "
                            f"total={lt:.5f} | logit_kd={lk:.5f} kd_total={kdt:.5f} "
                            f"hidden_kd={hid:.5f} attn_kd={attn_v:.5f} | ce={ce:.5f} | "
                            f"blend={a:.2f}*kd_total + {1.0 - a:.2f}*ce | "
                            f"kd_mode={self.kd_mode} kd_loss={self.kd_loss_type} T={self.temperature}"
                        )

                del teacher_out, teacher_logits, student_out
                del logit_kd, hidden_kd, attn_kd, kd_total, ce_loss
                if comp_mask is not None:
                    del comp_mask

            # Step 6: On-policy KD — student generates, teacher supervises
            if _op_prompt_ids is not None and self._should_run_on_policy():
                op_weight = self._get_on_policy_weight()
                if op_weight > 0:
                    pad_id = 0
                    tok = getattr(self, "kd_tokenizer", None) or getattr(self, "processing_class", None)
                    if tok is not None and hasattr(tok, "pad_token_id") and tok.pad_token_id is not None:
                        pad_id = tok.pad_token_id
                    elif tok is not None and hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
                        pad_id = tok.eos_token_id
                    else:
                        logger.warning(
                            "[QAD on-policy] Could not resolve a pad/eos token id from the tokenizer; "
                            "falling back to pad_token_id=0. For most Llama/Qwen tokenizers id 0 is <unk>, "
                            "not a padding token, which can corrupt generated sequences. Pass kd_tokenizer "
                            "(or set processing_class) with a valid pad/eos token."
                        )

                    from .losses.on_policy import on_policy_kd_step

                    op_loss, op_kd_val = on_policy_kd_step(
                        student=model,
                        teacher=model.teacher,
                        prompt_ids=_op_prompt_ids,
                        max_gen_len=self.on_policy_max_gen_len,
                        gen_temperature=self.on_policy_temperature,
                        kd_temperature=self.temperature,
                        kd_loss_fn=self._kd_loss_fn,
                        kd_alpha=self.kd_alpha,
                        pad_token_id=pad_id,
                    )
                    if op_loss is not None:
                        loss = loss + op_weight * op_loss
                        if getattr(self, "log_loss_breakdown", False):
                            logger.info(
                                f"[QAD on-policy] step={self.state.global_step} "
                                f"weight={op_weight:.3f} kd={op_kd_val:.5f}"
                            )
                    del op_loss
            del _op_prompt_ids

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

        kwargs_bwd: dict[str, Any] = {}

        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs_bwd["learning_rate"] = self._get_learning_rate()  # pragma: no cover

        if self.args.n_gpu > 1:
            loss = loss.mean()  # pragma: no cover

        if getattr(self, "use_apex", False):
            from apex import amp  # type: ignore[import-not-found]  # pragma: no cover

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:  # pragma: no cover
                scaled_loss.backward()  # pragma: no cover
        else:
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                loss = loss / self.current_gradient_accumulation_steps  # pragma: no cover

            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs_bwd["scale_wrt_gas"] = False  # pragma: no cover

            self.accelerator.backward(loss, **kwargs_bwd)

        return loss.detach()


if SFTTrainer is not None:

    class QADSFTTrainer(SFTTrainer, QADTrainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize the QADSFTTrainer."""
            self.temperature: float = float(kwargs.pop("temperature", 1.0))
            SFTTrainer.__init__(self, *args, **kwargs)
            assert hasattr(self.model, "teacher")

        def training_step(
            self,
            model: nn.Module,
            inputs: dict[str, torch.Tensor | Any],
            num_items_in_batch: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Inherit from SFT trainer, uses QADTrainer.training_step."""
            with self.maybe_activation_offload_context:
                return QADTrainer.training_step(self, model, inputs, num_items_in_batch)
else:

    class QADSFTTrainer:  # type: ignore[no-redef]  # pragma: no cover
        """Placeholder used when ``trl`` is not installed.

        Importing the name still works, but constructing it raises a clear error
        instead of silently handing back ``None`` (which would fail later with an
        opaque ``TypeError: 'NoneType' object is not callable``).
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "QADSFTTrainer requires the `trl` package, which is not installed. "
                "Install it with `pip install trl`. (Use QADTrainer instead if you "
                "don't need the TRL SFT integration.)"
            )
