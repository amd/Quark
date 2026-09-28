#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ._serialization import StrictSchema
from .data import TokenDataset, TokenPurpose
from .errors import PlanSelectionError, SchemaValidationError


@dataclass(frozen=True, slots=True)
class PplResult(StrictSchema):
    nll_sum: float
    token_count: int
    ppl: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.nll_sum) or self.nll_sum < 0 or self.token_count <= 0:
            raise SchemaValidationError("PPL NLL must be finite and token_count must be positive.")
        if not math.isfinite(self.ppl) or self.ppl <= 0:
            raise SchemaValidationError("PPL must be finite and positive.")
        expected = math.exp(self.nll_sum / self.token_count)
        if not math.isclose(self.ppl, expected, rel_tol=1e-12, abs_tol=0.0):
            raise SchemaValidationError("PPL does not match exp(nll_sum / token_count).")


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, Mapping):
        logits = output.get("logits")
    else:
        logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise PlanSelectionError("Model output does not contain a logits tensor.")
    return logits


@torch.inference_mode()
def evaluate_ppl(
    model: nn.Module,
    token_dataset: TokenDataset,
    device: str | torch.device,
) -> PplResult:
    """Evaluate token-level PPL with each next-token target counted exactly once."""
    if token_dataset.purpose is not TokenPurpose.PPL:
        raise PlanSelectionError("PPL evaluation requires a PPL token dataset.")

    model.eval()
    target_device = torch.device(device)
    nll_sum_tensor = torch.zeros((), dtype=torch.float64, device=target_device)
    token_count = 0
    forward_parameters = inspect.signature(model.forward).parameters.values()
    supports_use_cache = any(
        parameter.name == "use_cache" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in forward_parameters
    )
    for sequence in token_dataset.sequences:
        input_ids = torch.tensor(sequence, dtype=torch.long, device=target_device).unsqueeze(0)
        model_kwargs = {"input_ids": input_ids}
        if supports_use_cache:
            model_kwargs["use_cache"] = False
        logits = _extract_logits(model(**model_kwargs))
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < input_ids.shape[1]:
            raise PlanSelectionError(
                f"Unexpected logits shape {tuple(logits.shape)} for input shape {tuple(input_ids.shape)}."
            )
        shift_labels = input_ids[:, 1:].contiguous()
        for start in range(0, shift_labels.shape[1], 256):
            end = min(start + 256, shift_labels.shape[1])
            shift_logits = logits[:, start:end, :].float().contiguous()
            label_chunk = shift_labels[:, start:end]
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                label_chunk.reshape(-1),
                reduction="sum",
            )
            nll_sum_tensor += loss
        token_count += shift_labels.numel()

    nll_sum = float(nll_sum_tensor)
    try:
        ppl = math.exp(nll_sum / token_count)
    except OverflowError as exc:
        raise PlanSelectionError("PPL overflowed to a non-finite value.") from exc
    if not math.isfinite(ppl):
        raise PlanSelectionError("PPL is non-finite.")
    return PplResult(nll_sum=nll_sum, token_count=token_count, ppl=ppl)


__all__ = ["PplResult", "evaluate_ppl"]
