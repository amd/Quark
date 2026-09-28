#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import torch
from torch import nn

from .candidates import Candidate, Candidates
from .data import TokenDataset, validate_token_datasets
from .decision_space import DecisionSpace
from .errors import (
    ArtifactCompatibilityError,
    CandidateEvaluationError,
    CandidateQuantizationError,
    NoCandidatePassedError,
    PlanSelectionError,
    SchemaValidationError,
)
from .evaluator import CandidateEvaluator, ModelFactory
from .evaluators.hf import FreshModelEvaluator
from .mixed_precision_plan import (
    CandidateEvaluation,
    CandidateEvaluationStatus,
    MixedPrecisionPlan,
    MixedPrecisionPlanPayload,
)
from .mixed_precision_strategy import MixedPrecisionStrategy
from .plan_checks import validate_assignment
from .qconfig_builder import build_qconfig, qconfig_semantic_hash


def prepend_model(model: nn.Module, model_factory: ModelFactory) -> ModelFactory:
    """Return a factory that yields ``model`` once, then delegates to ``model_factory``.

    This lets an integrated workflow transfer its profiling model to baseline
    evaluation without keeping a second reference alive while candidates are
    evaluated.
    """
    preloaded_model: nn.Module | None = model

    def factory() -> nn.Module:
        nonlocal preloaded_model
        if preloaded_model is None:
            return model_factory()
        result = preloaded_model
        preloaded_model = None
        return result

    return factory


def _validate_candidate_semantics(decision_space: DecisionSpace, candidates: Candidates) -> None:
    units_by_id = {unit.unit_id: unit for unit in decision_space.payload.decision_units}
    unknown_diversity_units = sorted(set(candidates.payload.diversity_unit_ids) - set(units_by_id))
    if unknown_diversity_units:
        raise ArtifactCompatibilityError(f"Candidates reference unknown diversity units: {unknown_diversity_units}.")

    skeleton_assignments: list[tuple[str, ...]] = []
    for candidate in candidates.payload.candidates:
        try:
            validate_assignment(decision_space, candidate)
        except SchemaValidationError as exc:
            raise ArtifactCompatibilityError("Candidate assignment does not match Decision Space.") from exc
        module_assignment = {entry.module_name: entry.scheme for entry in candidate.assignment}
        skeleton_assignments.append(
            tuple(
                module_assignment[units_by_id[unit_id].members[0]] for unit_id in candidates.payload.diversity_unit_ids
            )
        )
    for index, skeleton_assignment in enumerate(skeleton_assignments):
        for previous in skeleton_assignments[:index]:
            differences = sum(current != prior for current, prior in zip(skeleton_assignment, previous, strict=True))
            if differences < candidates.payload.min_diversity_differences:
                raise ArtifactCompatibilityError("Candidates violate their recorded high-impact diversity constraint.")


def validate_plan_selection_inputs(
    decision_space: DecisionSpace,
    candidates: Candidates,
    strategy: MixedPrecisionStrategy,
    calibration_tokens: TokenDataset,
    ppl_tokens: TokenDataset,
) -> None:
    decision_space.validate_integrity()
    candidates.validate_integrity()
    if candidates.payload.decision_space_id != decision_space.artifact_id:
        raise ArtifactCompatibilityError("Candidates do not belong to this Decision Space.")
    if candidates.artifact.model_fingerprint != decision_space.model_fingerprint:
        raise ArtifactCompatibilityError("Candidates model does not match Decision Space.")
    decision_space.validate_compatibility(strategy)
    candidates.artifact.validate_compatibility(
        model_fingerprint=decision_space.model_fingerprint,
        inputs={"search": strategy.search, "seed": strategy.seed},
        upstream={
            "decision_space": decision_space.artifact_id,
            "sensitivity_profile": candidates.payload.sensitivity_profile_id,
        },
    )
    _validate_candidate_semantics(decision_space, candidates)
    current_budget_limit = strategy.search.budget.value * decision_space.payload.total_quantizable_params
    if any(candidate.total_bits > current_budget_limit + 1e-6 for candidate in candidates.payload.candidates):
        raise ArtifactCompatibilityError("Candidates exceed the current Strategy budget.")

    validate_token_datasets(calibration_tokens, ppl_tokens, strategy)


def select_best_plan(
    model_factory: ModelFactory,
    decision_space: DecisionSpace,
    candidates: Candidates,
    strategy: MixedPrecisionStrategy,
    calibration_tokens: TokenDataset,
    ppl_tokens: TokenDataset,
    *,
    device: str | torch.device,
) -> MixedPrecisionPlan:
    """Backward-compatible entry point using isolated fresh HF models."""
    evaluator = FreshModelEvaluator(
        model_factory,
        decision_space,
        candidates.payload.model_state_fingerprint,
        candidates.payload.model_behavior_fingerprint,
        device=device,
    )
    return select_best_plan_with_evaluator(
        evaluator,
        decision_space,
        candidates,
        strategy,
        calibration_tokens,
        ppl_tokens,
    )


def select_best_plan_with_evaluator(
    evaluator: CandidateEvaluator,
    decision_space: DecisionSpace,
    candidates: Candidates,
    strategy: MixedPrecisionStrategy,
    calibration_tokens: TokenDataset,
    ppl_tokens: TokenDataset,
) -> MixedPrecisionPlan:
    """Evaluate candidates through a backend-neutral runtime and select the winner."""
    descriptor = evaluator.descriptor
    evaluations: list[CandidateEvaluation] = []
    try:
        validate_plan_selection_inputs(decision_space, candidates, strategy, calibration_tokens, ppl_tokens)
        if (
            descriptor.backend != strategy.plan_selection.evaluator.backend.value
            or descriptor.protocol != strategy.plan_selection.evaluator.protocol
        ):
            raise PlanSelectionError("Evaluator descriptor does not match Strategy.")
        try:
            baseline = evaluator.evaluate_baseline(ppl_tokens)
        except Exception as exc:
            if isinstance(exc, PlanSelectionError):
                raise
            raise PlanSelectionError(f"Baseline evaluation failed: {exc}") from exc
        expected_ppl_tokens = sum(len(sequence) - 1 for sequence in ppl_tokens.sequences)
        if baseline.token_count != expected_ppl_tokens:
            raise PlanSelectionError(
                f"Evaluator baseline scored {baseline.token_count} tokens, expected {expected_ppl_tokens}."
            )

        for candidate in candidates.payload.candidates:
            try:
                qconfig = build_qconfig(decision_space, candidate)
                result = evaluator.evaluate_candidate(
                    decision_space,
                    candidate,
                    qconfig,
                    calibration_tokens,
                    ppl_tokens,
                )
                if result.ppl.token_count != expected_ppl_tokens:
                    raise PlanSelectionError(
                        f"Evaluator candidate scored {result.ppl.token_count} tokens, expected {expected_ppl_tokens}."
                    )
                expected_qconfig_hash = qconfig_semantic_hash(qconfig)
                if result.qconfig_hash != expected_qconfig_hash:
                    raise PlanSelectionError("Evaluator QConfig hash differs from the canonical candidate recipe.")
                if (
                    result.runtime_audit.backend != descriptor.backend
                    or result.runtime_audit.assignment_hash != candidate.candidate_id
                    or result.runtime_audit.calibration_token_hash != calibration_tokens.token_hash
                    or result.runtime_audit.calibrated_sequences != len(calibration_tokens.sequences)
                    or result.runtime_audit.calibrated_tokens
                    != sum(len(sequence) for sequence in calibration_tokens.sequences)
                ):
                    raise PlanSelectionError("Evaluator runtime audit does not match the candidate inputs.")
                degradation = result.ppl.ppl / baseline.ppl - 1.0
                evaluations.append(
                    CandidateEvaluation(
                        candidate_id=candidate.candidate_id,
                        status=CandidateEvaluationStatus.SUCCEEDED,
                        ppl=result.ppl,
                        degradation=degradation,
                        qconfig_hash=result.qconfig_hash,
                        runtime_audit=result.runtime_audit,
                        error=None,
                    )
                )
            except CandidateEvaluationError as exc:
                evaluations.append(
                    CandidateEvaluation(
                        candidate_id=candidate.candidate_id,
                        status=CandidateEvaluationStatus.EVAL_ERROR,
                        ppl=None,
                        degradation=None,
                        qconfig_hash=None,
                        runtime_audit=None,
                        error=str(exc),
                    )
                )
            except CandidateQuantizationError as exc:
                evaluations.append(
                    CandidateEvaluation(
                        candidate_id=candidate.candidate_id,
                        status=CandidateEvaluationStatus.QUANTIZE_ERROR,
                        ppl=None,
                        degradation=None,
                        qconfig_hash=None,
                        runtime_audit=None,
                        error=str(exc),
                    )
                )
            except PlanSelectionError:
                raise
            except Exception as exc:
                raise PlanSelectionError(
                    f"Unexpected evaluator failure for candidate {candidate.candidate_id}: {exc}"
                ) from exc
    finally:
        evaluator.close()

    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates.payload.candidates}
    gate = strategy.plan_selection.quality_gate.max_degradation
    eligible: list[tuple[Candidate, CandidateEvaluation]] = []
    for evaluation in evaluations:
        if (
            evaluation.status is CandidateEvaluationStatus.SUCCEEDED
            and evaluation.degradation is not None
            and evaluation.degradation <= gate
        ):
            eligible.append((candidate_by_id[evaluation.candidate_id], evaluation))
    if not eligible:
        summary = ", ".join(f"{evaluation.candidate_id}:{evaluation.status.value}" for evaluation in evaluations)
        raise NoCandidatePassedError(f"No candidate passed max_degradation={gate}; {summary}.")

    selected_candidate, _ = min(
        eligible,
        key=lambda item: (
            item[0].total_bits,
            item[1].ppl.ppl if item[1].ppl is not None else float("inf"),
            item[0].predicted_loss,
            item[0].candidate_id,
        ),
    )
    payload = MixedPrecisionPlanPayload(
        candidates_id=candidates.artifact_id,
        baseline=baseline,
        evaluations=tuple(evaluations),
        selected_candidate=selected_candidate,
        quality_gate_max_degradation=gate,
        selection_policy=strategy.plan_selection.selection_policy,
        evaluator=descriptor,
        calibration_token_hash=calibration_tokens.token_hash,
        ppl_token_hash=ppl_tokens.token_hash,
        model_state_fingerprint=candidates.payload.model_state_fingerprint,
        model_behavior_fingerprint=candidates.payload.model_behavior_fingerprint,
    )
    return MixedPrecisionPlan.create(
        payload,
        decision_space=decision_space,
        candidates=candidates,
        strategy=strategy,
    )


__all__ = [
    "ModelFactory",
    "prepend_model",
    "select_best_plan",
    "select_best_plan_with_evaluator",
]
