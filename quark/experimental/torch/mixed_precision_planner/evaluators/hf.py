#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import gc
import weakref

import torch
from torch import nn

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig

from ..candidates import Candidate
from ..data import TokenDataset
from ..decision_space import DecisionSpace
from ..errors import CandidateEvaluationError, CandidateQuantizationError, PlanSelectionError
from ..evaluator import (
    EvaluatorCandidateResult,
    EvaluatorDescriptor,
    ModelFactory,
    RuntimeQConfigAudit,
)
from ..ppl import PplResult, evaluate_ppl
from ..qconfig_builder import _validate_model_binding, audit_qconfig
from ..sensitivity_profile import model_behavior_fingerprint, model_state_fingerprint


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class FreshModelEvaluator:
    """Reference evaluator that isolates every measurement in a fresh HF model."""

    def __init__(
        self,
        model_factory: ModelFactory,
        decision_space: DecisionSpace,
        expected_model_state_fingerprint: str,
        expected_model_behavior_fingerprint: str,
        *,
        device: str | torch.device,
    ) -> None:
        self._model_factory = model_factory
        self._decision_space = decision_space
        self._expected_model_state_fingerprint = expected_model_state_fingerprint
        self._expected_model_behavior_fingerprint = expected_model_behavior_fingerprint
        self._device = device
        self._model_refs: list[weakref.ReferenceType[nn.Module]] = []
        self._descriptor = EvaluatorDescriptor(
            backend="hf",
            protocol="token_ppl_v1",
            implementation_version=1,
            runtime_version=torch.__version__,
            options={"device": str(device), "model_isolation": "fresh"},
        )

    @property
    def descriptor(self) -> EvaluatorDescriptor:
        return self._descriptor

    def _fresh_model(self) -> nn.Module:
        model = self._model_factory()
        if not isinstance(model, nn.Module):
            raise PlanSelectionError("model_factory must return torch.nn.Module.")
        if any(reference() is model for reference in self._model_refs):
            raise PlanSelectionError("model_factory reused a model instance.")
        if model_state_fingerprint(model) != self._expected_model_state_fingerprint:
            raise PlanSelectionError("model_factory returned weights that differ from the profiled checkpoint.")
        if model_behavior_fingerprint(model) != self._expected_model_behavior_fingerprint:
            raise PlanSelectionError("model_factory returned behavior that differs from the profiled checkpoint.")
        model.eval()
        self._model_refs.append(weakref.ref(model))
        return model

    def evaluate_baseline(self, ppl_tokens: TokenDataset) -> PplResult:
        model: nn.Module | None = None
        try:
            model = self._fresh_model()
            _validate_model_binding(model, self._decision_space)
            return evaluate_ppl(model, ppl_tokens, self._device)
        finally:
            if model is not None:
                del model
            _clear_memory()

    def evaluate_candidate(
        self,
        decision_space: DecisionSpace,
        candidate: Candidate,
        qconfig: QConfig,
        calibration_tokens: TokenDataset,
        ppl_tokens: TokenDataset,
    ) -> EvaluatorCandidateResult:
        model: nn.Module | None = None
        try:
            model = self._fresh_model()
            audit = audit_qconfig(model, decision_space, candidate, qconfig)
            if audit.resolved_quantized:
                try:
                    model = ModelQuantizer(qconfig).quantize_model(
                        model,
                        calibration_tokens.to_dataloader(self._device),
                    )
                except Exception as exc:
                    raise CandidateQuantizationError(str(exc)) from exc
            try:
                ppl = evaluate_ppl(model, ppl_tokens, self._device)
            except Exception as exc:
                raise CandidateEvaluationError(str(exc)) from exc

            runtime_audit = RuntimeQConfigAudit(
                backend="hf",
                assignment_hash=audit.assignment_hash,
                runtime_qconfig_hash=audit.qconfig_hash,
                resolved_quantized=audit.resolved_quantized,
                resolved_native=audit.resolved_native,
                calibration_token_hash=calibration_tokens.token_hash,
                calibrated_sequences=len(calibration_tokens.sequences),
                calibrated_tokens=sum(len(sequence) for sequence in calibration_tokens.sequences),
            )
            return EvaluatorCandidateResult(
                ppl=ppl,
                qconfig_hash=audit.qconfig_hash,
                runtime_audit=runtime_audit,
            )
        finally:
            if model is not None:
                del model
            _clear_memory()

    def close(self) -> None:
        _clear_memory()


__all__ = ["FreshModelEvaluator"]
