#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from quark.torch.utils.llm.model_preparation import get_model

from .candidates import Candidates
from .decision_space import DecisionSpace
from .errors import ArtifactCompatibilityError, EvaluatorUnavailableError, SchemaValidationError
from .evaluator import CandidateEvaluator, ModelFactory
from .evaluators import FreshModelEvaluator, VllmEvaluator, VllmRuntimeConfig, validate_vllm_decision_space
from .mixed_precision_strategy import EvaluatorBackend, MixedPrecisionStrategy
from .sensitivity_profile import model_behavior_fingerprint, model_state_fingerprint


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeContext:
    model_source: str
    device: str
    model_revision: str | None = None
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 4096
    model_factory: ModelFactory | None = None

    def __post_init__(self) -> None:
        if not self.model_source:
            raise SchemaValidationError("Evaluation model_source must not be empty.")
        if self.model_revision == "":
            raise SchemaValidationError("Evaluation model_revision must not be empty.")


def _validate_vllm_model_source(
    context: EvaluationRuntimeContext,
    expected_state_fingerprint: str,
    expected_behavior_fingerprint: str,
) -> None:
    """Hash a canonical HF load from the exact path that vLLM will consume."""
    model: nn.Module | None = None
    try:
        model, _ = get_model(
            ckpt_path=context.model_source,
            data_type="auto",
            device=context.device,
            multi_gpu=False,
            multi_device=False,
            attn_implementation="eager",
            trust_remote_code=context.trust_remote_code,
        )
        if not isinstance(model, nn.Module):
            raise EvaluatorUnavailableError("Quark model loading must return torch.nn.Module.")
        if model_state_fingerprint(model) != expected_state_fingerprint:
            raise ArtifactCompatibilityError(
                "vLLM model source weights differ from the checkpoint used for sensitivity profiling."
            )
        if model_behavior_fingerprint(model) != expected_behavior_fingerprint:
            raise ArtifactCompatibilityError(
                "vLLM model source behavior differs from the checkpoint used for sensitivity profiling."
            )
    except (ArtifactCompatibilityError, EvaluatorUnavailableError):
        raise
    except Exception as exc:
        raise EvaluatorUnavailableError(f"Could not load vLLM model source for fingerprint validation: {exc}") from exc
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_evaluator(
    strategy: MixedPrecisionStrategy,
    decision_space: DecisionSpace,
    candidates: Candidates,
    context: EvaluationRuntimeContext,
) -> CandidateEvaluator:
    """Construct the explicitly selected evaluator without automatic fallback."""
    backend = strategy.plan_selection.evaluator.backend
    if backend is EvaluatorBackend.VLLM:
        validate_vllm_decision_space(decision_space)
        runtime_config = VllmRuntimeConfig(
            model=context.model_source,
            expected_model_state_fingerprint=candidates.payload.model_state_fingerprint,
            expected_model_behavior_fingerprint=candidates.payload.model_behavior_fingerprint,
            model_revision=context.model_revision,
            tensor_parallel_size=context.tensor_parallel_size,
            gpu_memory_utilization=context.gpu_memory_utilization,
            max_model_len=context.max_model_len,
            trust_remote_code=context.trust_remote_code,
        )
        if not Path(context.model_source).is_absolute():
            raise SchemaValidationError(
                "vLLM model_source must be a resolved absolute checkpoint path; use resolve_model_source()."
            )
        if context.model_revision is not None and Path(context.model_source).name != context.model_revision:
            raise SchemaValidationError("vLLM model_revision does not match the resolved snapshot path.")
        required_length = max(
            strategy.calibration.max_length,
            strategy.plan_selection.quality_gate.max_length,
        )
        if context.max_model_len <= required_length:
            raise SchemaValidationError(
                f"vLLM max_model_len={context.max_model_len} must exceed required token length {required_length}."
            )
        _validate_vllm_model_source(
            context,
            candidates.payload.model_state_fingerprint,
            candidates.payload.model_behavior_fingerprint,
        )
        return VllmEvaluator(runtime_config, decision_space)
    if backend is EvaluatorBackend.HF:
        if context.model_factory is None:
            raise EvaluatorUnavailableError("HF evaluator requires an explicit model_factory.")
        return FreshModelEvaluator(
            context.model_factory,
            decision_space,
            candidates.payload.model_state_fingerprint,
            candidates.payload.model_behavior_fingerprint,
            device=context.device,
        )
    raise EvaluatorUnavailableError(f"Unsupported evaluator backend: {backend!r}.")


__all__ = ["EvaluationRuntimeContext", "create_evaluator"]
