#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from torch import nn

from quark.torch.quantization.config.config import QConfig

from .candidates import Candidates
from .data import TokenDataset
from .decision_space import DecisionSpace, build_decision_space
from .errors import ArtifactCompatibilityError
from .evaluator import CandidateEvaluator
from .linear_programming import search_candidates
from .mixed_precision_plan import MixedPrecisionPlan
from .mixed_precision_strategy import MixedPrecisionStrategy
from .plan_selection import (
    ModelFactory,
    select_best_plan,
    select_best_plan_with_evaluator,
    validate_plan_selection_inputs,
)
from .qconfig_builder import (
    build_qconfig as build_candidate_qconfig,
)
from .qconfig_builder import (
    compile_qconfig,
    qconfig_semantic_hash,
)
from .runtime import EvaluationRuntimeContext, create_evaluator
from .sensitivity_profile import (
    SensitivityProfile,
    build_sensitivity_profile,
    model_behavior_fingerprint,
    model_state_fingerprint,
)


class MixedPrecisionPlanner:
    """Compose the five stages of the experimental Torch mixed-precision planner.

    :param MixedPrecisionStrategy strategy: Strict MVP strategy.
    """

    def __init__(self, strategy: MixedPrecisionStrategy) -> None:
        self.strategy = strategy

    def decision_space(self, model: nn.Module) -> DecisionSpace:
        """Build the post-preprocess fine-grained Decision Space."""
        return build_decision_space(model, self.strategy)

    def sensitivity_profile(self, model: nn.Module, decision_space: DecisionSpace) -> SensitivityProfile:
        """Measure Weight-MSE and storage cost for every unit-scheme pair."""
        return build_sensitivity_profile(model, decision_space, self.strategy)

    def search(self, decision_space: DecisionSpace, profile: SensitivityProfile) -> Candidates:
        """Run fixed-budget PuLP/CBC Top-K search without accessing a model."""
        return search_candidates(decision_space, profile, self.strategy)

    def select_best_plan(
        self,
        model_factory: ModelFactory,
        decision_space: DecisionSpace,
        candidates: Candidates,
        calibration_tokens: TokenDataset,
        ppl_tokens: TokenDataset,
        *,
        device: str,
    ) -> MixedPrecisionPlan:
        """Quantize/evaluate fresh candidate models and select the winner."""
        return select_best_plan(
            model_factory,
            decision_space,
            candidates,
            self.strategy,
            calibration_tokens,
            ppl_tokens,
            device=device,
        )

    def build_qconfig(
        self,
        model: nn.Module,
        decision_space: DecisionSpace,
        plan: MixedPrecisionPlan,
    ) -> QConfig:
        """Compile and audit the selected Plan against a supplied model."""
        expected_qconfig_hash = self._validate_plan(decision_space, plan)
        if model_state_fingerprint(model) != plan.payload.model_state_fingerprint:
            raise ArtifactCompatibilityError("Model weights differ from the checkpoint used to build the Plan.")
        if model_behavior_fingerprint(model) != plan.payload.model_behavior_fingerprint:
            raise ArtifactCompatibilityError("Model behavior differs from the checkpoint used to build the Plan.")
        qconfig, audit = compile_qconfig(model, decision_space, plan.payload.selected_candidate)
        if expected_qconfig_hash != audit.qconfig_hash:
            raise ArtifactCompatibilityError("Current Quark QConfig recipe differs from the evaluated Plan.")
        return qconfig

    def select_best_plan_with_evaluator(
        self,
        evaluator: CandidateEvaluator,
        decision_space: DecisionSpace,
        candidates: Candidates,
        calibration_tokens: TokenDataset,
        ppl_tokens: TokenDataset,
    ) -> MixedPrecisionPlan:
        """Select a plan using an injected evaluation backend."""
        return select_best_plan_with_evaluator(
            evaluator,
            decision_space,
            candidates,
            self.strategy,
            calibration_tokens,
            ppl_tokens,
        )

    def select_best_plan_with_runtime(
        self,
        runtime: EvaluationRuntimeContext,
        decision_space: DecisionSpace,
        candidates: Candidates,
        calibration_tokens: TokenDataset,
        ppl_tokens: TokenDataset,
    ) -> MixedPrecisionPlan:
        """Construct the Strategy-selected evaluator and select a plan."""
        validate_plan_selection_inputs(
            decision_space,
            candidates,
            self.strategy,
            calibration_tokens,
            ppl_tokens,
        )
        evaluator = create_evaluator(self.strategy, decision_space, candidates, runtime)
        return self.select_best_plan_with_evaluator(
            evaluator,
            decision_space,
            candidates,
            calibration_tokens,
            ppl_tokens,
        )

    def build_qconfig_from_plan(
        self,
        decision_space: DecisionSpace,
        plan: MixedPrecisionPlan,
    ) -> QConfig:
        """Rebuild an already-audited Plan's QConfig without loading its model."""
        expected_qconfig_hash = self._validate_plan(decision_space, plan)
        qconfig = build_candidate_qconfig(decision_space, plan.payload.selected_candidate)
        if expected_qconfig_hash != qconfig_semantic_hash(qconfig):
            raise ArtifactCompatibilityError("Current Quark QConfig recipe differs from the evaluated Plan.")
        return qconfig

    def _validate_plan(self, decision_space: DecisionSpace, plan: MixedPrecisionPlan) -> str:
        decision_space.validate_integrity()
        plan.validate_integrity()
        if plan.artifact.upstream.get("decision_space") != decision_space.artifact_id:
            raise ArtifactCompatibilityError("Plan does not belong to this Decision Space.")
        plan.artifact.validate_compatibility(
            model_fingerprint=decision_space.model_fingerprint,
            inputs={
                "plan_selection": self.strategy.plan_selection,
                "evaluator": plan.payload.evaluator,
                "calibration_token_hash": plan.payload.calibration_token_hash,
                "ppl_token_hash": plan.payload.ppl_token_hash,
            },
            upstream={
                "decision_space": decision_space.artifact_id,
                "candidates": plan.payload.candidates_id,
            },
        )
        selected_evaluation = next(
            evaluation
            for evaluation in plan.payload.evaluations
            if evaluation.candidate_id == plan.payload.selected_candidate.candidate_id
        )
        assert selected_evaluation.qconfig_hash is not None
        return selected_evaluation.qconfig_hash


__all__ = ["MixedPrecisionPlanner"]
