#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import copy
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import quark.experimental.torch.mixed_precision_planner.data as planner_data
import quark.experimental.torch.mixed_precision_planner.evaluators.hf as hf_evaluator
import quark.experimental.torch.mixed_precision_planner.plan_selection as plan_selection
from quark.experimental.torch.mixed_precision_planner import (
    Candidates,
    MixedPrecisionPlan,
    MixedPrecisionStrategy,
    TokenDataset,
)
from quark.experimental.torch.mixed_precision_planner._artifact import Artifact
from quark.experimental.torch.mixed_precision_planner.data import (
    TokenPurpose,
    materialize_calibration_tokens,
    materialize_ppl_tokens,
)
from quark.experimental.torch.mixed_precision_planner.decision_space import build_decision_space
from quark.experimental.torch.mixed_precision_planner.errors import (
    ArtifactCompatibilityError,
    NoCandidatePassedError,
    PlanSelectionError,
    SchemaValidationError,
)
from quark.experimental.torch.mixed_precision_planner.evaluator import (
    EvaluatorCandidateResult,
    EvaluatorDescriptor,
    RuntimeQConfigAudit,
)
from quark.experimental.torch.mixed_precision_planner.linear_programming import search_candidates
from quark.experimental.torch.mixed_precision_planner.mixed_precision_plan import (
    CandidateEvaluation,
    CandidateEvaluationStatus,
)
from quark.experimental.torch.mixed_precision_planner.plan_selection import (
    prepend_model,
    select_best_plan,
    select_best_plan_with_evaluator,
)
from quark.experimental.torch.mixed_precision_planner.ppl import PplResult, evaluate_ppl
from quark.experimental.torch.mixed_precision_planner.qconfig_builder import qconfig_semantic_hash
from quark.experimental.torch.mixed_precision_planner.sensitivity_profile import build_sensitivity_profile

_TEST_TOKENIZER_FINGERPRINT = "sha256:test-tokenizer"
_PPL_TOKEN_COUNT = 2047


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)


class TinyCausalQwen3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 8)
        self.model.layers = nn.ModuleList([TinyLayer()])
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.to(dtype=torch.bfloat16)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.model.embed_tokens(input_ids)
        layer = self.model.layers[0]
        hidden = torch.tanh(layer.q_proj(hidden) + layer.k_proj(hidden))
        return {"logits": self.lm_head(hidden)}


class UniformModel(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": torch.zeros((*input_ids.shape, 4), device=input_ids.device)}


def make_strategy(
    *,
    budget: float = 8.5,
    num_candidates: int = 1,
    max_degradation: float = 10.0,
) -> MixedPrecisionStrategy:
    return MixedPrecisionStrategy.from_dict(
        {
            "schema_version": 1,
            "hardware": {"target": "mi355"},
            "deployment_backend": {"name": "no_fusion"},
            "decision_space": {"exclude_patterns": []},
            "sensitivity_profile": {"methods": ["weight_mse"]},
            "calibration": {
                "dataset": "pileval",
                "num_samples": 2,
                "max_length": 8,
                "batch_size": 1,
            },
            "search": {
                "granularity": "fine",
                "algorithm": "linear_programming",
                "budget": {"metric": "effective_bits", "value": budget, "scope": "quantizable"},
                "num_candidates": num_candidates,
            },
            "plan_selection": {
                "evaluator": {"backend": "hf", "protocol": "token_ppl_v1"},
                "quality_gate": {
                    "metric": "ppl",
                    "dataset": "wikitext2",
                    "max_degradation": max_degradation,
                    "num_chunks": 1,
                    "max_length": 2048,
                },
                "selection_policy": "lowest_effective_bits",
            },
            "output": {"dir": "./run"},
            "seed": 42,
        }
    )


def make_tokens() -> tuple[TokenDataset, TokenDataset]:
    calibration = TokenDataset.create(
        purpose=TokenPurpose.CALIBRATION,
        dataset="mit-han-lab/pile-val-backup",
        split="validation",
        revision=None,
        tokenizer_id="tiny",
        tokenizer_fingerprint=_TEST_TOKENIZER_FINGERPRINT,
        sampling_seed=42,
        sequences=(tuple(range(8)), tuple(range(8, 16))),
    )
    ppl = TokenDataset.create(
        purpose=TokenPurpose.PPL,
        dataset="Salesforce/wikitext/wikitext-2-raw-v1",
        split="test",
        revision=None,
        tokenizer_id="tiny",
        tokenizer_fingerprint=_TEST_TOKENIZER_FINGERPRINT,
        sequences=(tuple(index % 16 for index in range(2048)),),
    )
    return calibration, ppl


def build_pipeline(
    strategy: MixedPrecisionStrategy | None = None,
) -> tuple[dict[str, torch.Tensor], object, object, TokenDataset, TokenDataset, MixedPrecisionStrategy]:
    torch.manual_seed(31)
    resolved_strategy = strategy or make_strategy()
    source = TinyCausalQwen3()
    state = copy.deepcopy(source.state_dict())
    space = build_decision_space(source, resolved_strategy)
    profile = build_sensitivity_profile(source, space, resolved_strategy)
    candidates = search_candidates(space, profile, resolved_strategy)
    calibration, ppl = make_tokens()
    return state, space, candidates, calibration, ppl, resolved_strategy


def test_token_dataset_round_trip_and_dataloader(tmp_path: Path) -> None:
    calibration, _ = make_tokens()
    path = tmp_path / "tokens.json"
    calibration.save(path)
    loaded = TokenDataset.load(path)
    assert loaded == calibration
    batch = next(iter(loaded.to_dataloader("cpu")))
    assert batch["input_ids"].shape == (1, 8)
    assert batch["input_ids"].dtype is torch.long

    with pytest.raises(SchemaValidationError, match="hash"):
        replace(calibration, token_hash="sha256:wrong")
    with pytest.raises(SchemaValidationError, match="hash"):
        replace(calibration, sampling_seed=43)
    with pytest.raises(SchemaValidationError, match="common length"):
        TokenDataset.create(
            purpose=TokenPurpose.CALIBRATION,
            dataset="dataset",
            split="split",
            revision=None,
            tokenizer_id="tokenizer",
            tokenizer_fingerprint=_TEST_TOKENIZER_FINGERPRINT,
            sequences=((1, 2), (1, 2, 3)),
        )


def test_token_dataset_and_materialization_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SchemaValidationError, match="metadata"):
        TokenDataset.create(
            purpose=TokenPurpose.CALIBRATION,
            dataset="",
            split="validation",
            revision=None,
            tokenizer_id="tiny",
            tokenizer_fingerprint=_TEST_TOKENIZER_FINGERPRINT,
            sequences=((1, 2),),
        )
    with pytest.raises(SchemaValidationError, match="non-negative"):
        TokenDataset.create(
            purpose=TokenPurpose.CALIBRATION,
            dataset="dataset",
            split="validation",
            revision=None,
            tokenizer_id="tiny",
            tokenizer_fingerprint=_TEST_TOKENIZER_FINGERPRINT,
            sequences=((1, -1),),
        )
    with pytest.raises(SchemaValidationError, match="sampling seed"):
        TokenDataset.create(
            purpose=TokenPurpose.CALIBRATION,
            dataset="dataset",
            split="validation",
            revision=None,
            tokenizer_id="tiny",
            tokenizer_fingerprint=_TEST_TOKENIZER_FINGERPRINT,
            sequences=((1, 2),),
        )
    with pytest.raises(SchemaValidationError, match="unused sampling seed"):
        TokenDataset.create(
            purpose=TokenPurpose.PPL,
            dataset="dataset",
            split="test",
            revision=None,
            tokenizer_id="tiny",
            tokenizer_fingerprint=_TEST_TOKENIZER_FINGERPRINT,
            sampling_seed=42,
            sequences=((1, 2),),
        )
    with pytest.raises(SchemaValidationError, match="non-negative integers"):
        TokenDataset.create(
            purpose=TokenPurpose.PPL,
            dataset="dataset",
            split="test",
            revision=None,
            tokenizer_id="tiny",
            tokenizer_fingerprint=_TEST_TOKENIZER_FINGERPRINT,
            sequences=((1.5, 2),),  # type: ignore[arg-type]
        )
    with pytest.raises(SchemaValidationError, match="sampling seed"):
        TokenDataset.create(
            purpose=TokenPurpose.CALIBRATION,
            dataset="dataset",
            split="validation",
            revision=None,
            tokenizer_id="tiny",
            tokenizer_fingerprint=_TEST_TOKENIZER_FINGERPRINT,
            sampling_seed=True,  # type: ignore[arg-type]
            sequences=((1, 2),),
        )

    def missing_datasets(_name: str) -> object:
        raise ImportError

    monkeypatch.setattr(planner_data.importlib, "import_module", missing_datasets)
    with pytest.raises(PlanSelectionError, match="datasets"):
        planner_data._load_hf_dataset("dataset", None, "test", None)

    tokenizer = SimpleNamespace(eos_token_id=None, encode=lambda text, add_special_tokens=False: [1] * len(text))
    with pytest.raises(PlanSelectionError, match="required"):
        planner_data._materialize_sequences([{"text": ""}, {"text": "x"}], tokenizer, count=2, length=4)


def test_materialize_tokens_uses_fixed_dataset_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDataset(list[dict[str, str]]):
        def shuffle(self, seed: int) -> FakeDataset:
            assert seed == 42
            return self

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load_dataset(*args: object, **kwargs: object) -> FakeDataset:
        calls.append((args, kwargs))
        return FakeDataset([{"text": "x" * 3000}])

    tokenizer = SimpleNamespace(eos_token_id=2, encode=lambda text, add_special_tokens=False: [1] * len(text))
    monkeypatch.setattr(
        planner_data.importlib, "import_module", lambda _name: SimpleNamespace(load_dataset=load_dataset)
    )
    strategy = make_strategy()
    strategy = replace(
        strategy,
        calibration=replace(strategy.calibration, revision="revision"),
        plan_selection=replace(
            strategy.plan_selection,
            quality_gate=replace(strategy.plan_selection.quality_gate, revision="revision"),
        ),
    )

    calibration = materialize_calibration_tokens(tokenizer, "tiny", strategy)
    ppl = materialize_ppl_tokens(tokenizer, "tiny", strategy)
    assert len(calibration.sequences) == 2 and len(calibration.sequences[0]) == 8
    assert calibration.sampling_seed == strategy.seed
    assert calibration.revision == "revision"
    assert len(ppl.sequences) == 1 and len(ppl.sequences[0]) == 2048
    assert ppl.sampling_seed is None
    assert ppl.revision == "revision"
    assert calls[0][1]["split"] == "validation"
    assert calls[1][1]["split"] == "test"
    assert calls[0][1]["streaming"] is True
    assert calls[1][1]["streaming"] is True


def test_token_level_ppl_math() -> None:
    tokens = TokenDataset.create(
        purpose=TokenPurpose.PPL,
        dataset="test",
        split="test",
        revision=None,
        tokenizer_id="test",
        tokenizer_fingerprint=_TEST_TOKENIZER_FINGERPRINT,
        sequences=((0, 1, 2, 3, 0),),
    )
    result = evaluate_ppl(UniformModel(), tokens, "cpu")
    assert result.token_count == 4
    assert result.nll_sum == pytest.approx(4 * math.log(4))
    assert result.ppl == pytest.approx(4.0)


def test_ppl_validation_and_error_paths() -> None:
    with pytest.raises(SchemaValidationError, match="NLL"):
        PplResult(-1.0, 1, 1.0)
    with pytest.raises(SchemaValidationError, match="finite and positive"):
        PplResult(0.0, 1, 0.0)
    with pytest.raises(SchemaValidationError, match="does not match"):
        PplResult(1.0, 1, 1.0)

    calibration, _ = make_tokens()
    with pytest.raises(PlanSelectionError, match="PPL token"):
        evaluate_ppl(UniformModel(), calibration, "cpu")

    tokens = TokenDataset.create(
        purpose=TokenPurpose.PPL,
        dataset="test",
        split="test",
        revision=None,
        tokenizer_id="test",
        tokenizer_fingerprint=_TEST_TOKENIZER_FINGERPRINT,
        sequences=((0, 1),),
    )

    class AttrModel(nn.Module):
        def forward(self, input_ids: torch.Tensor) -> object:
            return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 2)))

    assert evaluate_ppl(AttrModel(), tokens, "cpu").ppl == pytest.approx(2.0)

    class MissingLogits(nn.Module):
        def forward(self, input_ids: torch.Tensor) -> object:
            return object()

    with pytest.raises(PlanSelectionError, match="logits"):
        evaluate_ppl(MissingLogits(), tokens, "cpu")

    class BadShape(nn.Module):
        def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": torch.zeros((1, 2))}

    with pytest.raises(PlanSelectionError, match="shape"):
        evaluate_ppl(BadShape(), tokens, "cpu")

    class OverflowModel(nn.Module):
        def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
            logits = torch.tensor([[[0.0, -1000.0], [0.0, -1000.0]]])
            return {"logits": logits}

    with pytest.raises(PlanSelectionError, match="overflowed"):
        evaluate_ppl(OverflowModel(), tokens, "cpu")

    class NanModel(nn.Module):
        def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": torch.full((*input_ids.shape, 2), torch.nan)}

    with pytest.raises(PlanSelectionError, match="non-finite"):
        evaluate_ppl(NanModel(), tokens, "cpu")


def test_full_candidate_evaluation_and_plan_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(23)
    source = TinyCausalQwen3()
    source_state = copy.deepcopy(source.state_dict())
    strategy = make_strategy()
    space = build_decision_space(source, strategy)
    profile = build_sensitivity_profile(source, space, strategy)
    candidates = search_candidates(space, profile, strategy)
    calibration, ppl = make_tokens()
    created: list[int] = []
    created_models: list[nn.Module] = []

    def model_factory() -> nn.Module:
        model = TinyCausalQwen3()
        model.load_state_dict(source_state)
        created.append(id(model))
        created_models.append(model)
        return model

    plan = select_best_plan(
        model_factory,
        space,
        candidates,
        strategy,
        calibration,
        ppl,
        device="cpu",
    )
    assert len(created) == 2 and len(set(created)) == 2
    assert plan.payload.selected_candidate.candidate_id == candidates.payload.candidates[0].candidate_id
    assert plan.payload.evaluations[0].status is CandidateEvaluationStatus.SUCCEEDED
    assert plan.payload.calibration_token_hash == calibration.token_hash
    assert plan.payload.ppl_token_hash == ppl.token_hash
    assert plan.artifact_id == plan.artifact.artifact_id

    path = tmp_path / "mixed_precision_plan.json"
    plan.save(path)
    assert MixedPrecisionPlan.load(path) == plan

    invalid_payloads = [
        ({"candidates_id": "bad"}, "candidates_id"),
        ({"quality_gate_max_degradation": -1.0}, "quality gate"),
        ({"calibration_token_hash": "bad"}, "token hashes"),
        ({"evaluations": plan.payload.evaluations + (plan.payload.evaluations[0],)}, "duplicate"),
        (
            {"selected_candidate": replace(plan.payload.selected_candidate, candidate_id="sha256:missing")},
            "successful evaluation",
        ),
    ]
    for changes, message in invalid_payloads:
        with pytest.raises(SchemaValidationError, match=message):
            replace(plan.payload, **changes)

    evaluation = plan.payload.evaluations[0]
    assert evaluation.ppl is not None and evaluation.runtime_audit is not None
    wrong_audit = replace(evaluation.runtime_audit, assignment_hash="sha256:wrong")
    with pytest.raises(SchemaValidationError, match="runtime audit"):
        replace(plan.payload, evaluations=(replace(evaluation, runtime_audit=wrong_audit),))
    wrong_count_ppl = PplResult(
        math.log(evaluation.ppl.ppl) * (evaluation.ppl.token_count - 1),
        evaluation.ppl.token_count - 1,
        evaluation.ppl.ppl,
    )
    with pytest.raises(SchemaValidationError, match="PPL or degradation"):
        replace(plan.payload, evaluations=(replace(evaluation, ppl=wrong_count_ppl),))
    failed_gate_ppl = PplResult(
        math.log(plan.payload.baseline.ppl * 2.0) * evaluation.ppl.token_count,
        evaluation.ppl.token_count,
        plan.payload.baseline.ppl * 2.0,
    )
    failed_gate_evaluation = replace(evaluation, ppl=failed_gate_ppl, degradation=1.0)
    with pytest.raises(SchemaValidationError, match="quality gate"):
        replace(
            plan.payload,
            evaluations=(failed_gate_evaluation,),
            quality_gate_max_degradation=0.5,
        )


def test_prepend_model_reuses_profile_model_for_baseline() -> None:
    state, space, candidates, calibration, ppl, strategy = build_pipeline()
    profile_model = TinyCausalQwen3()
    profile_model.load_state_dict(state)
    loaded_models = 0

    def model_factory() -> nn.Module:
        nonlocal loaded_models
        loaded_models += 1
        model = TinyCausalQwen3()
        model.load_state_dict(state)
        return model

    selection_model_factory = prepend_model(profile_model, model_factory)
    del profile_model
    plan = select_best_plan(
        selection_model_factory,
        space,
        candidates,
        strategy,
        calibration,
        ppl,
        device="cpu",
    )

    assert loaded_models == len(candidates.payload.candidates)
    assert plan.payload.evaluations[0].status is CandidateEvaluationStatus.SUCCEEDED


def test_no_candidate_passes_quality_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(29)
    source = TinyCausalQwen3()
    state = copy.deepcopy(source.state_dict())
    strategy = make_strategy(max_degradation=0.0)
    space = build_decision_space(source, strategy)
    profile = build_sensitivity_profile(source, space, strategy)
    candidates = search_candidates(space, profile, strategy)
    calibration, ppl = make_tokens()
    results = iter(
        [
            PplResult(0.0, _PPL_TOKEN_COUNT, 1.0),
            PplResult(math.log(2) * _PPL_TOKEN_COUNT, _PPL_TOKEN_COUNT, 2.0),
        ]
    )

    monkeypatch.setattr(hf_evaluator, "evaluate_ppl", lambda *_args, **_kwargs: next(results))
    monkeypatch.setattr(
        hf_evaluator,
        "audit_qconfig",
        lambda _model, _space, candidate, qconfig: SimpleNamespace(
            resolved_quantized=(),
            resolved_native=(),
            assignment_hash=candidate.candidate_id,
            qconfig_hash=qconfig_semantic_hash(qconfig),
        ),
    )

    def factory() -> nn.Module:
        model = TinyCausalQwen3()
        model.load_state_dict(state)
        return model

    with pytest.raises(NoCandidatePassedError):
        select_best_plan(factory, space, candidates, strategy, calibration, ppl, device="cpu")


def test_plan_selection_input_validation() -> None:
    _, space, candidates, calibration, ppl, strategy = build_pipeline()

    with pytest.raises(PlanSelectionError, match="Calibration"):
        plan_selection.validate_plan_selection_inputs(
            space,
            candidates,
            strategy,
            TokenDataset.create(
                purpose=TokenPurpose.PPL,
                dataset=calibration.dataset,
                split=calibration.split,
                revision=None,
                tokenizer_id="tiny",
                tokenizer_fingerprint=_TEST_TOKENIZER_FINGERPRINT,
                sequences=calibration.sequences,
            ),
            ppl,
        )
    with pytest.raises(PlanSelectionError, match="PPL token"):
        plan_selection.validate_plan_selection_inputs(space, candidates, strategy, calibration, calibration)
    with pytest.raises(PlanSelectionError, match="same tokenizer"):
        other_tokenizer = TokenDataset.create(
            purpose=TokenPurpose.PPL,
            dataset=ppl.dataset,
            split=ppl.split,
            revision=None,
            tokenizer_id="other",
            tokenizer_fingerprint="sha256:other-tokenizer",
            sequences=ppl.sequences,
        )
        plan_selection.validate_plan_selection_inputs(space, candidates, strategy, calibration, other_tokenizer)
    wrong_dataset = TokenDataset.create(
        purpose=TokenPurpose.CALIBRATION,
        dataset="other",
        split="validation",
        revision=None,
        tokenizer_id=calibration.tokenizer_id,
        tokenizer_fingerprint=calibration.tokenizer_fingerprint,
        sampling_seed=calibration.sampling_seed,
        sequences=calibration.sequences,
    )
    with pytest.raises(PlanSelectionError, match="Calibration"):
        plan_selection.validate_plan_selection_inputs(space, candidates, strategy, wrong_dataset, ppl)
    wrong_revision = TokenDataset.create(
        purpose=calibration.purpose,
        dataset=calibration.dataset,
        split=calibration.split,
        revision="other",
        tokenizer_id=calibration.tokenizer_id,
        tokenizer_fingerprint=calibration.tokenizer_fingerprint,
        sampling_seed=calibration.sampling_seed,
        sequences=calibration.sequences,
    )
    with pytest.raises(PlanSelectionError, match="Calibration"):
        planner_data.validate_token_datasets(wrong_revision, ppl, strategy)
    with pytest.raises(PlanSelectionError, match="Calibration"):
        planner_data.validate_token_datasets(calibration, ppl, replace(strategy, seed=43))
    changed_seed_calibration = TokenDataset.create(
        purpose=calibration.purpose,
        dataset=calibration.dataset,
        split=calibration.split,
        revision=calibration.revision,
        tokenizer_id=calibration.tokenizer_id,
        tokenizer_fingerprint=calibration.tokenizer_fingerprint,
        sampling_seed=43,
        sequences=calibration.sequences,
    )
    planner_data.validate_token_datasets(changed_seed_calibration, ppl, replace(strategy, seed=43))
    with pytest.raises(ArtifactCompatibilityError, match="inputs"):
        plan_selection.validate_plan_selection_inputs(
            space,
            candidates,
            make_strategy(budget=9.0),
            calibration,
            ppl,
        )
    with pytest.raises(ArtifactCompatibilityError, match="inputs"):
        plan_selection.validate_plan_selection_inputs(
            space,
            candidates,
            replace(strategy, seed=43),
            calibration,
            ppl,
        )

    with pytest.raises(SchemaValidationError, match="Artifact envelope"):
        replace(
            candidates,
            payload=replace(candidates.payload, decision_space_id="sha256:other"),
        )
    other_artifact = Artifact.create(
        "candidates",
        candidates.payload.to_dict(),
        model_fingerprint="other",
    )
    with pytest.raises(ArtifactCompatibilityError, match="model"):
        plan_selection.validate_plan_selection_inputs(
            space,
            replace(candidates, artifact=other_artifact),
            strategy,
            calibration,
            ppl,
        )


def test_plan_selection_rechecks_recorded_candidate_diversity() -> None:
    source = TinyCausalQwen3()
    strategy = make_strategy(budget=16.0, num_candidates=3)
    space = build_decision_space(source, strategy)
    profile = build_sensitivity_profile(source, space, strategy)
    candidates = search_candidates(space, profile, strategy)
    calibration, ppl = make_tokens()
    tampered_payload = replace(
        candidates.payload,
        diversity_unit_ids=tuple(unit.unit_id for unit in space.payload.decision_units),
        min_diversity_differences=2,
    )

    with pytest.raises(ArtifactCompatibilityError, match="high-impact diversity"):
        plan_selection.validate_plan_selection_inputs(
            space,
            Candidates.create(
                tampered_payload,
                decision_space=space,
                sensitivity_profile=profile,
                strategy=strategy,
            ),
            strategy,
            calibration,
            ppl,
        )


def test_plan_selection_baseline_and_factory_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    state, space, candidates, calibration, ppl, strategy = build_pipeline()

    with pytest.raises(PlanSelectionError, match="torch.nn.Module"):
        select_best_plan(lambda: object(), space, candidates, strategy, calibration, ppl, device="cpu")  # type: ignore[arg-type]

    def factory() -> nn.Module:
        model = TinyCausalQwen3()
        model.load_state_dict(state)
        return model

    monkeypatch.setattr(
        hf_evaluator, "evaluate_ppl", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(PlanSelectionError, match="Baseline evaluation failed"):
        select_best_plan(factory, space, candidates, strategy, calibration, ppl, device="cpu")

    monkeypatch.undo()
    with pytest.raises(PlanSelectionError, match="profiled checkpoint"):
        select_best_plan(TinyCausalQwen3, space, candidates, strategy, calibration, ppl, device="cpu")


def test_plan_selection_records_quantize_and_eval_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    state, space, candidates, calibration, ppl, strategy = build_pipeline()

    def factory() -> nn.Module:
        model = TinyCausalQwen3()
        model.load_state_dict(state)
        return model

    class FailingQuantizer:
        def __init__(self, _qconfig: object) -> None:
            pass

        def quantize_model(self, *_args: object, **_kwargs: object) -> nn.Module:
            raise RuntimeError("quantize failed")

    monkeypatch.setattr(hf_evaluator, "ModelQuantizer", FailingQuantizer)
    monkeypatch.setattr(
        hf_evaluator,
        "evaluate_ppl",
        lambda *_args, **_kwargs: PplResult(0.0, _PPL_TOKEN_COUNT, 1.0),
    )
    with pytest.raises(NoCandidatePassedError, match="quantize_error"):
        select_best_plan(factory, space, candidates, strategy, calibration, ppl, device="cpu")

    monkeypatch.undo()
    results = iter([PplResult(0.0, _PPL_TOKEN_COUNT, 1.0), RuntimeError("eval failed")])

    def evaluate(*_args: object, **_kwargs: object) -> PplResult:
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(hf_evaluator, "evaluate_ppl", evaluate)
    monkeypatch.setattr(
        hf_evaluator,
        "audit_qconfig",
        lambda _model, _space, candidate, qconfig: SimpleNamespace(
            resolved_quantized=(),
            resolved_native=(),
            assignment_hash=candidate.candidate_id,
            qconfig_hash=qconfig_semantic_hash(qconfig),
        ),
    )
    with pytest.raises(NoCandidatePassedError, match="eval_error"):
        select_best_plan(factory, space, candidates, strategy, calibration, ppl, device="cpu")


def test_injected_evaluator_rejects_bad_audit_and_closes() -> None:
    _, space, candidates, calibration, ppl, strategy = build_pipeline()
    close_calls = 0

    class BadAuditEvaluator:
        descriptor = EvaluatorDescriptor("hf", "token_ppl_v1", 1, "test", {})

        def evaluate_baseline(self, _ppl_tokens: TokenDataset) -> PplResult:
            return PplResult(0.0, _PPL_TOKEN_COUNT, 1.0)

        def evaluate_candidate(
            self,
            _space: Any,
            _candidate: Any,
            qconfig: Any,
            calibration_tokens: TokenDataset,
            _ppl_tokens: TokenDataset,
        ) -> EvaluatorCandidateResult:
            qconfig_hash = qconfig_semantic_hash(qconfig)
            return EvaluatorCandidateResult(
                ppl=PplResult(0.0, _PPL_TOKEN_COUNT, 1.0),
                qconfig_hash=qconfig_hash,
                runtime_audit=RuntimeQConfigAudit(
                    backend="hf",
                    assignment_hash="sha256:wrong-candidate",
                    runtime_qconfig_hash=qconfig_hash,
                    resolved_quantized=(),
                    resolved_native=(),
                    calibration_token_hash=calibration_tokens.token_hash,
                    calibrated_sequences=len(calibration_tokens.sequences),
                    calibrated_tokens=sum(len(sequence) for sequence in calibration_tokens.sequences),
                ),
            )

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    with pytest.raises(PlanSelectionError, match="runtime audit"):
        select_best_plan_with_evaluator(
            BadAuditEvaluator(),
            space,
            candidates,
            strategy,
            calibration,
            ppl,
        )
    assert close_calls == 1


def test_plan_selection_rejects_partial_ppl_measurements() -> None:
    _, space, candidates, calibration, ppl, strategy = build_pipeline()

    class PartialPplEvaluator:
        descriptor = EvaluatorDescriptor("hf", "token_ppl_v1", 1, "test", {})

        def evaluate_baseline(self, _ppl_tokens: TokenDataset) -> PplResult:
            return PplResult(0.0, 1, 1.0)

        def evaluate_candidate(self, *_args: object, **_kwargs: object) -> EvaluatorCandidateResult:
            raise AssertionError("candidate evaluation must not start")

        def close(self) -> None:
            pass

    with pytest.raises(PlanSelectionError, match="baseline scored 1 tokens"):
        select_best_plan_with_evaluator(
            PartialPplEvaluator(),
            space,
            candidates,
            strategy,
            calibration,
            ppl,
        )


def test_plan_selection_rejects_reused_model_and_clears_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    state, space, candidates, calibration, ppl, strategy = build_pipeline()
    shared = TinyCausalQwen3()
    shared.load_state_dict(state)
    monkeypatch.setattr(
        hf_evaluator,
        "evaluate_ppl",
        lambda *_args, **_kwargs: PplResult(0.0, _PPL_TOKEN_COUNT, 1.0),
    )
    collected: list[bool] = []
    cleared: list[bool] = []
    monkeypatch.setattr(hf_evaluator.gc, "collect", lambda: collected.append(True))
    monkeypatch.setattr(hf_evaluator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hf_evaluator.torch.cuda, "empty_cache", lambda: cleared.append(True))

    with pytest.raises(PlanSelectionError, match="reused a model instance"):
        select_best_plan(lambda: shared, space, candidates, strategy, calibration, ppl, device="cpu")

    expected_cleanup_calls = len(candidates.payload.candidates) + 2
    assert len(collected) == expected_cleanup_calls
    assert len(cleared) == expected_cleanup_calls


def test_candidate_evaluation_schema_validation() -> None:
    with pytest.raises(SchemaValidationError, match="incomplete"):
        CandidateEvaluation(
            "sha256:candidate",
            CandidateEvaluationStatus.SUCCEEDED,
            None,
            None,
            None,
            None,
            None,
        )
    with pytest.raises(SchemaValidationError, match="only an error"):
        CandidateEvaluation(
            "sha256:candidate",
            CandidateEvaluationStatus.EVAL_ERROR,
            PplResult(0.0, 1, 1.0),
            0.0,
            None,
            None,
            "error",
        )
