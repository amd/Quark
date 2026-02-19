#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quark.shares.utils.import_utils import is_transformers_available
from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.rotation.cayley import SGDG

if is_transformers_available():
    from transformers import Trainer, TrainerCallback  # type: ignore[attr-defined]
else:
    Trainer = object  # type: ignore[assignment]
    TrainerCallback = object  # type: ignore[assignment]

if TYPE_CHECKING:
    from transformers.integrations.integration_utils import TensorBoardCallback

    from quark.torch.quantization.config.config import RotationConfig


logger = ScreenLogger(__name__)


class VerboseAdam(torch.optim.Adam):
    def step(self, closure=None):  # type: ignore
        for group in self.param_groups:
            logger.debug(f"LR in Adam: {group['lr']}")
        super().step(closure)


class AdamAndSGDGOptimizer(torch.optim.Optimizer):
    """
    A PyTorch optimizer combining SGDG optimizer and Adam optimizer for two groups of parameters, each with a different learning rate.
    """

    def __init__(
        self,
        adam_params: list[torch.Tensor],
        sgdg_params: list[torch.Tensor],
        learning_rate: float,
        smooth_learning_rate: float,
    ):
        sgdg_lr = learning_rate
        params = [{"params": adam_params, "lr": smooth_learning_rate}, {"params": sgdg_params, "lr": sgdg_lr}]
        super().__init__(params, defaults={"lr": learning_rate})

        self.adam_optimizer = VerboseAdam(adam_params, lr=smooth_learning_rate)
        self.sgdg_optimizer = SGDG(sgdg_params, lr=sgdg_lr, stiefel=True)

    def step(self, closure: None | Any = None) -> None:
        for group in self.adam_optimizer.param_groups:
            group["lr"] = self.param_groups[0]["lr"]

        for group in self.sgdg_optimizer.param_groups:
            group["lr"] = self.param_groups[1]["lr"]

        self.adam_optimizer.step()  # type: ignore
        self.sgdg_optimizer.step()  # type: ignore


class RotationTrainer(Trainer):
    def __init__(self, *args: Any, **kwargs: Any):
        if not is_transformers_available():
            raise ImportError("The library `transformers` is required to use OrthogonalTrainingCallback.")

        if "original_model" not in kwargs:
            raise ValueError("Expected original_model")

        if "loss_type" not in kwargs:
            raise ValueError("Expected loss_type")

        self.loss_type = kwargs["loss_type"]
        kwargs.pop("loss_type")

        self.original_model = kwargs["original_model"]
        self.original_model = self.original_model.eval()

        kwargs.pop("original_model")

        super().__init__(*args, **kwargs)  # type: ignore[no-untyped-call]

    def compute_loss(  # type: ignore
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: None | Any = None,
    ):
        if self.loss_type == "origin":
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        elif "r_kl_top" in self.loss_type:
            _ = inputs.pop("labels", None)
            if self.loss_type == "k_top":
                k = 1000
            else:
                k = int(self.loss_type.split("_")[-1])

            with torch.no_grad():
                ori_logits = self.original_model(**inputs).logits

            outputs = model(**inputs)
            logits = outputs.logits
            top_logits, indices = logits.topk(k, dim=-1, sorted=False)
            top_ori_logits = ori_logits.gather(-1, indices)
            loss = F.kl_div(
                F.log_softmax(top_ori_logits.flatten(0, -2), dim=-1),
                F.softmax(top_logits.flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
            return (loss, outputs) if return_outputs else loss

        elif "kl_top" in self.loss_type:
            _ = inputs.pop("labels", None)
            if self.loss_type == "kl_top":
                k = 1000
            else:
                k = int(self.loss_type.split("_")[-1])

            with torch.no_grad():
                ori_logits = self.original_model(**inputs).logits

            outputs = model(**inputs)
            logits = outputs.logits
            top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)

            # TODO: add post_attn case from OSTQuant repo.
            top_logits = logits.gather(-1, indices)
            loss = F.kl_div(
                F.log_softmax(top_logits, dim=-1).flatten(0, -2),
                F.softmax(top_ori_logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )

            return (loss, outputs) if return_outputs else loss
        else:
            raise ValueError("wrong loss")


class OrthogonalTrainingCallback(TrainerCallback):
    def __init__(
        self, tensorboard_callback: "TensorBoardCallback", model: nn.Module, rotation_config: "RotationConfig"
    ):
        if not is_transformers_available():
            raise ImportError("The library `transformers` is required to use OrthogonalTrainingCallback.")

        self.model = model
        self.tensorboard_callback = tensorboard_callback

        start_params = {}
        start_params_ptr = set()

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    start_params[name] = param.data.clone()
                    start_params_ptr.add(param.data_ptr())

        self.start_params = start_params

    def on_step_begin(self, args, state, control, **kwargs):  # type: ignore
        print("", flush=True)  # For some reason HF's trainer does not always flush.

        tb_writer = self.tensorboard_callback.tb_writer
        assert tb_writer is not None

        global_maxdiff_from_start = 0
        mean_absdiffs_from_start = []
        params_dict = dict(self.model.named_parameters())
        for name, start_param in self.start_params.items():
            param = params_dict[name]
            absdiff = (param - start_param).abs()

            maxabsdiff = absdiff.max().item()
            # print(f"{name} max absdiff from start: {maxabsdiff:.2e}", flush=True)

            global_maxdiff_from_start = max(global_maxdiff_from_start, maxabsdiff)

            meanabsdiff = absdiff.mean().item()
            mean_absdiffs_from_start.append(meanabsdiff)

            tb_writer.add_scalar(f"{name}_maxdiff_from_start", maxabsdiff, state.global_step)

        global_mean_absdiff_from_start = np.mean(mean_absdiffs_from_start)

        tb_writer.add_scalar("global_maxdiff_from_start", global_maxdiff_from_start, state.global_step)
        tb_writer.add_scalar("global_mean_absdiff_from_start", global_mean_absdiff_from_start, state.global_step)

        global_maxdiff_from_eye = 0
        mean_absdiffs_from_eye = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and "smooth_values" not in name:
                with torch.no_grad():
                    reference_eye = torch.eye(param.shape[-1], device=param.device, dtype=param.dtype)
                    absdiff = (param @ param.T - reference_eye).abs()

                    absdiff_mean = absdiff.mean().item()
                    mean_absdiffs_from_eye.append(absdiff_mean)
                    absdiff_max = absdiff.max().item()

                    global_maxdiff_from_eye = max(global_maxdiff_from_eye, absdiff_max)

        global_mean_absdiff_from_eye = np.mean(mean_absdiffs_from_eye)

        tb_writer.add_scalar("global_maxdiff_from_eye", global_maxdiff_from_eye, state.global_step)
        tb_writer.add_scalar("global_mean_absdiff_from_eye", global_mean_absdiff_from_eye, state.global_step)

        tb_writer.flush()
