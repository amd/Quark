#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import argparse
import copy
import gc
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import quark.experimental.torch.mixed_precision_planner.cli as planner_cli
import quark.experimental.torch.mixed_precision_planner.export as planner_export
import quark.experimental.torch.mixed_precision_planner.model_source as planner_model_source
import quark.experimental.torch.mixed_precision_planner.reload_validation as reload_validation
from quark.common.utils.testing_utils import require_torch_cuda, torch_device
from quark.experimental.torch.mixed_precision_planner import (
    EvaluationRuntimeContext,
    MixedPrecisionPlan,
    MixedPrecisionPlanner,
    MixedPrecisionStrategy,
    ReloadValidationResult,
    TokenDataset,
    evaluate_ppl,
)
from quark.experimental.torch.mixed_precision_planner._artifact import Artifact
from quark.experimental.torch.mixed_precision_planner._serialization import atomic_write_json
from quark.experimental.torch.mixed_precision_planner.cli import MixedPrecisionPlannerCLI
from quark.experimental.torch.mixed_precision_planner.data import TokenPurpose
from quark.experimental.torch.mixed_precision_planner.errors import (
    ArtifactCompatibilityError,
    ModelSourceError,
    PlanSelectionError,
    SchemaValidationError,
)
from quark.experimental.torch.mixed_precision_planner.plan_selection import prepend_model
from quark.experimental.torch.mixed_precision_planner.ppl import PplResult
from quark.torch import ModelQuantizer, export_safetensors

_TEST_TOKENIZER_FINGERPRINT = "sha256:test-tokenizer"


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


def make_strategy(output_dir: str = "./run") -> MixedPrecisionStrategy:
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
                "budget": {"metric": "effective_bits", "value": 8.5, "scope": "quantizable"},
                "num_candidates": 1,
            },
            "plan_selection": {
                "evaluator": {"backend": "hf", "protocol": "token_ppl_v1"},
                "quality_gate": {
                    "metric": "ppl",
                    "dataset": "wikitext2",
                    "max_degradation": 10.0,
                    "num_chunks": 1,
                    "max_length": 2048,
                },
                "selection_policy": "lowest_effective_bits",
            },
            "output": {"dir": output_dir},
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


def run_cli(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser()
    MixedPrecisionPlannerCLI.register_subcommand(parser)
    args = parser.parse_args(arguments)
    MixedPrecisionPlannerCLI(parser, args, []).run()


@pytest.fixture(autouse=True)
def resolve_cli_model_source_without_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner_cli,
        "resolve_model_source",
        lambda source, revision=None: SimpleNamespace(
            requested=source,
            revision=revision,
            resolved_path=f"/resolved/{source}",
            commit_hash="resolved-commit",
        ),
    )


@pytest.fixture(scope="module")
def workflow() -> SimpleNamespace:
    torch.manual_seed(37)
    source = TinyCausalQwen3()
    state = copy.deepcopy(source.state_dict())
    strategy = make_strategy()
    planner = MixedPrecisionPlanner(strategy)
    space = planner.decision_space(source)
    profile = planner.sensitivity_profile(source, space)
    candidates = planner.search(space, profile)
    calibration, ppl = make_tokens()

    def factory() -> nn.Module:
        model = TinyCausalQwen3()
        model.load_state_dict(state)
        return model

    plan = planner.select_best_plan(factory, space, candidates, calibration, ppl, device="cpu")
    qconfig = planner.build_qconfig_from_plan(space, plan)
    return SimpleNamespace(
        state=state,
        strategy=strategy,
        planner=planner,
        space=space,
        profile=profile,
        candidates=candidates,
        calibration=calibration,
        ppl=ppl,
        factory=factory,
        plan=plan,
        qconfig=qconfig,
    )


def test_planner_facade_builds_audited_qconfig(workflow: SimpleNamespace) -> None:
    qconfig = workflow.planner.build_qconfig(workflow.factory(), workflow.space, workflow.plan)
    rebuilt_qconfig = workflow.planner.build_qconfig_from_plan(workflow.space, workflow.plan)
    assert qconfig.layer_quant_config
    assert qconfig.global_quant_config.weight is None
    assert rebuilt_qconfig.to_dict() == qconfig.to_dict()

    other_artifact = Artifact.create(
        "mixed_precision_plan",
        workflow.plan.payload.to_dict(),
        model_fingerprint=workflow.space.model_fingerprint,
        upstream={"decision_space": "sha256:other"},
    )
    wrong_plan = MixedPrecisionPlan(other_artifact, workflow.plan.payload)
    with pytest.raises(ArtifactCompatibilityError, match="belong"):
        workflow.planner.build_qconfig(workflow.factory(), workflow.space, wrong_plan)
    with pytest.raises(ArtifactCompatibilityError, match="belong"):
        workflow.planner.build_qconfig_from_plan(workflow.space, wrong_plan)

    evaluations = list(workflow.plan.payload.evaluations)
    evaluations[0] = replace(evaluations[0], qconfig_hash="sha256:old-recipe")
    drifted_plan = MixedPrecisionPlan.create(
        replace(workflow.plan.payload, evaluations=tuple(evaluations)),
        decision_space=workflow.space,
        candidates=workflow.candidates,
        strategy=workflow.strategy,
    )
    with pytest.raises(ArtifactCompatibilityError, match="recipe"):
        workflow.planner.build_qconfig(workflow.factory(), workflow.space, drifted_plan)
    with pytest.raises(ArtifactCompatibilityError, match="recipe"):
        workflow.planner.build_qconfig_from_plan(workflow.space, drifted_plan)

    with pytest.raises(SchemaValidationError, match="Artifact envelope"):
        replace(
            workflow.plan,
            payload=replace(
                workflow.plan.payload,
                evaluator=replace(workflow.plan.payload.evaluator, runtime_version="changed"),
            ),
        )

    changed_model = workflow.factory()
    with torch.no_grad():
        next(changed_model.parameters()).add_(1)
    with pytest.raises(ArtifactCompatibilityError, match="checkpoint"):
        workflow.planner.build_qconfig(changed_model, workflow.space, workflow.plan)


def test_quark_cli_search_stage(tmp_path: Path, workflow: SimpleNamespace) -> None:
    strategy_path = tmp_path / "strategy.json"
    space_path = tmp_path / "decision_space.json"
    profile_path = tmp_path / "sensitivity_profile.json"
    candidates_path = tmp_path / "candidates.json"
    workflow.strategy.save(strategy_path)
    workflow.space.save(space_path)
    workflow.profile.save(profile_path)

    run_cli(
        [
            "search",
            "--strategy",
            str(strategy_path),
            "--decision-space",
            str(space_path),
            "--sensitivity-profile",
            str(profile_path),
            "--candidates",
            str(candidates_path),
        ]
    )
    assert candidates_path.exists()


def test_model_source_resolution_pins_hub_revision_and_canonicalizes_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    local = planner_model_source.resolve_model_source(str(local_model))
    assert local.resolved_path == str(local_model.resolve())
    assert local.commit_hash is None
    planner_model_source.validate_writable_paths(local, (tmp_path / "planner-output",))
    with pytest.raises(ModelSourceError, match="outside the source checkpoint"):
        planner_model_source.validate_writable_paths(local, (local_model / "exported",))
    with pytest.raises(ModelSourceError, match="local checkpoint"):
        planner_model_source.resolve_model_source(str(local_model), "main")
    with pytest.raises(ModelSourceError, match="does not exist"):
        planner_model_source.resolve_model_source(str(tmp_path / "missing"))

    snapshot = tmp_path / "cache" / "snapshots" / "0123456789abcdef"
    snapshot.mkdir(parents=True)
    calls: list[tuple[str, str | None]] = []

    def snapshot_download(*, repo_id: str, revision: str | None) -> str:
        calls.append((repo_id, revision))
        return str(snapshot)

    monkeypatch.setattr(
        planner_model_source.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(snapshot_download=snapshot_download) if name == "huggingface_hub" else __import__(name)
        ),
    )
    hub = planner_model_source.resolve_model_source("Qwen/Qwen3-0.6B-Base", "revision")
    assert calls == [("Qwen/Qwen3-0.6B-Base", "revision")]
    assert hub.resolved_path == str(snapshot.resolve())
    assert hub.commit_hash == "0123456789abcdef"


def test_cli_rejects_output_inside_model_source_before_creating_it(
    tmp_path: Path,
    workflow: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy_path = tmp_path / "strategy.json"
    workflow.strategy.save(strategy_path)
    shared_path = tmp_path / "model-and-output"

    def resolve(source: str, revision: str | None) -> planner_model_source.ResolvedModelSource:
        assert source == str(shared_path)
        assert revision is None
        assert not shared_path.exists()
        return planner_model_source.ResolvedModelSource(source, None, str(shared_path.resolve()), None)

    monkeypatch.setattr(planner_cli, "resolve_model_source", resolve)
    with pytest.raises(ModelSourceError, match="outside the source checkpoint"):
        run_cli(
            [
                "decision-space",
                "--strategy",
                str(strategy_path),
                "--model-dir",
                str(shared_path),
                "--output-dir",
                str(shared_path),
            ]
        )
    assert not shared_path.exists()


def test_cli_model_loading_and_required_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = argparse.ArgumentParser()
    MixedPrecisionPlannerCLI.register_subcommand(parser)
    missing_args = parser.parse_args(["decision-space", "--strategy", "strategy.json"])
    command = MixedPrecisionPlannerCLI(parser, missing_args, [])
    with pytest.raises(ValueError, match="model-dir"):
        command._require_model_dir()

    args = parser.parse_args(
        [
            "decision-space",
            "--strategy",
            "strategy.json",
            "--model-dir",
            "model",
            "--model-revision",
            "revision",
            "--device",
            "cpu",
        ]
    )
    command = MixedPrecisionPlannerCLI(parser, args, [])
    model = TinyCausalQwen3()
    resolution_calls: list[tuple[str, str | None]] = []
    model_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        planner_cli,
        "resolve_model_source",
        lambda source, revision: (
            resolution_calls.append((source, revision))
            or SimpleNamespace(
                requested=source,
                revision=revision,
                resolved_path="/cache/snapshots/commit",
                commit_hash="commit",
            )
        ),
    )
    monkeypatch.setattr(
        planner_cli,
        "get_model",
        lambda **kwargs: model_calls.append(kwargs) or (model, None),
    )
    assert command._load_model() is model
    assert command._load_model() is model
    assert resolution_calls == [("model", "revision")]
    assert [call["ckpt_path"] for call in model_calls] == [
        "/cache/snapshots/commit",
        "/cache/snapshots/commit",
    ]


def test_cli_model_stages(tmp_path: Path, workflow: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy_path = tmp_path / "strategy.json"
    workflow.strategy.save(strategy_path)
    monkeypatch.setattr(MixedPrecisionPlannerCLI, "_load_model", lambda _self: workflow.factory())

    run_cli(
        [
            "decision-space",
            "--strategy",
            str(strategy_path),
            "--model-dir",
            "model",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert (tmp_path / "decision_space.json").exists()

    run_cli(
        [
            "sensitivity-profile",
            "--strategy",
            str(strategy_path),
            "--model-dir",
            "model",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert (tmp_path / "sensitivity_profile.json").exists()


def test_cli_plan_and_qconfig_stages(
    tmp_path: Path, workflow: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    strategy_path = tmp_path / "strategy.json"
    workflow.strategy.save(strategy_path)
    workflow.space.save(tmp_path / "decision_space.json")
    workflow.candidates.save(tmp_path / "candidates.json")
    workflow.calibration.save(tmp_path / "calibration_tokens.json")
    workflow.ppl.save(tmp_path / "ppl_tokens.json")
    monkeypatch.setattr(planner_cli, "get_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        planner_cli,
        "fingerprint_tokenizer",
        lambda _tokenizer: workflow.calibration.tokenizer_fingerprint,
    )
    monkeypatch.setattr(MixedPrecisionPlannerCLI, "_load_model", lambda _self: workflow.factory())
    runtimes: list[EvaluationRuntimeContext] = []
    monkeypatch.setattr(
        planner_cli.MixedPrecisionPlanner,
        "select_best_plan_with_runtime",
        lambda _self, runtime, *_args, **_kwargs: runtimes.append(runtime) or workflow.plan,
    )

    common = [
        "--strategy",
        str(strategy_path),
        "--model-dir",
        "model",
        "--device",
        "cpu",
        "--output-dir",
        str(tmp_path),
    ]
    run_cli(["select-best-plan", *common])
    assert (tmp_path / "mixed_precision_plan.json").exists()
    assert runtimes[0].model_source == "/resolved/model"
    assert runtimes[0].model_revision == "resolved-commit"
    monkeypatch.setattr(
        planner_cli,
        "get_tokenizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    run_cli(["build-qconfig", *common])
    assert (tmp_path / "qconfig.json").exists()


def test_cli_rejects_cached_calibration_from_a_different_seed(
    tmp_path: Path,
    workflow: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = replace(workflow.strategy, seed=43)
    strategy.save(tmp_path / "strategy.json")
    workflow.space.save(tmp_path / "decision_space.json")
    workflow.candidates.save(tmp_path / "candidates.json")
    workflow.calibration.save(tmp_path / "calibration_tokens.json")
    workflow.ppl.save(tmp_path / "ppl_tokens.json")
    monkeypatch.setattr(planner_cli, "get_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        planner_cli,
        "fingerprint_tokenizer",
        lambda _tokenizer: workflow.calibration.tokenizer_fingerprint,
    )
    monkeypatch.setattr(
        planner_cli.MixedPrecisionPlanner,
        "select_best_plan_with_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("evaluation must not start")),
    )

    with pytest.raises(PlanSelectionError, match="sampling inputs"):
        run_cli(
            [
                "select-best-plan",
                "--strategy",
                str(tmp_path / "strategy.json"),
                "--model-dir",
                "model",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_cli_run_stage(tmp_path: Path, workflow: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = make_strategy(str(tmp_path))
    strategy_path = tmp_path / "strategy.json"
    strategy.save(strategy_path)
    finalized: list[bool] = []

    model_loads = 0

    def load_model(_self: MixedPrecisionPlannerCLI) -> nn.Module:
        nonlocal model_loads
        model_loads += 1
        return workflow.factory()

    monkeypatch.setattr(MixedPrecisionPlannerCLI, "_load_model", load_model)
    monkeypatch.setattr(planner_cli, "get_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        planner_cli,
        "fingerprint_tokenizer",
        lambda _tokenizer: workflow.calibration.tokenizer_fingerprint,
    )
    monkeypatch.setattr(planner_cli, "materialize_calibration_tokens", lambda *_args, **_kwargs: workflow.calibration)
    monkeypatch.setattr(planner_cli, "materialize_ppl_tokens", lambda *_args, **_kwargs: workflow.ppl)
    monkeypatch.setattr(planner_cli.MixedPrecisionPlanner, "decision_space", lambda _self, _model: workflow.space)
    monkeypatch.setattr(
        planner_cli.MixedPrecisionPlanner,
        "sensitivity_profile",
        lambda _self, _model, _space: workflow.profile,
    )
    monkeypatch.setattr(
        planner_cli.MixedPrecisionPlanner,
        "search",
        lambda _self, _space, _profile: workflow.candidates,
    )
    monkeypatch.setattr(
        planner_cli.MixedPrecisionPlanner,
        "select_best_plan_with_runtime",
        lambda _self, *_args, **_kwargs: workflow.plan,
    )
    qconfig = workflow.planner.build_qconfig(workflow.factory(), workflow.space, workflow.plan)
    monkeypatch.setattr(
        planner_cli.MixedPrecisionPlanner,
        "build_qconfig_from_plan",
        lambda _self, _space, _plan: qconfig,
    )
    monkeypatch.setattr(
        planner_cli,
        "_export_selected_plan",
        lambda *_args, **_kwargs: finalized.append(True),
    )

    run_cli(
        [
            "run",
            "--strategy",
            str(strategy_path),
            "--model-dir",
            "model",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert model_loads == 1
    assert finalized == [True]
    for name in (
        "decision_space.json",
        "sensitivity_profile.json",
        "candidates.json",
        "calibration_tokens.json",
        "ppl_tokens.json",
        "mixed_precision_plan.json",
        "qconfig.json",
    ):
        assert (tmp_path / name).exists()


def test_cli_run_reuses_profile_model_and_skips_qconfig_reload(
    tmp_path: Path,
    workflow: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = make_strategy(str(tmp_path))
    strategy_path = tmp_path / "strategy.json"
    strategy.save(strategy_path)
    model_loads = 0

    def load_model(_self: MixedPrecisionPlannerCLI) -> nn.Module:
        nonlocal model_loads
        model_loads += 1
        return workflow.factory()

    monkeypatch.setattr(MixedPrecisionPlannerCLI, "_load_model", load_model)
    monkeypatch.setattr(planner_cli, "get_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        planner_cli,
        "fingerprint_tokenizer",
        lambda _tokenizer: workflow.calibration.tokenizer_fingerprint,
    )
    monkeypatch.setattr(planner_cli, "materialize_calibration_tokens", lambda *_args, **_kwargs: workflow.calibration)
    monkeypatch.setattr(planner_cli, "materialize_ppl_tokens", lambda *_args, **_kwargs: workflow.ppl)
    monkeypatch.setattr(planner_cli, "_export_selected_plan", lambda *_args, **_kwargs: None)

    run_cli(
        [
            "run",
            "--strategy",
            str(strategy_path),
            "--model-dir",
            "model",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert model_loads == 1 + strategy.search.num_candidates


def test_split_export_validation_invokes_fresh_reload(
    tmp_path: Path, workflow: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_baseline = PplResult(
        nll_sum=0.0,
        token_count=workflow.plan.payload.baseline.token_count,
        ppl=1.0,
    )
    qconfig_audit = planner_export.audit_export_binding(
        workflow.factory(),
        workflow.qconfig,
        workflow.space,
        workflow.plan,
        workflow.calibration,
        workflow.ppl,
    )

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--output") + 1])
        planner_export.ReloadMeasurement(
            reloaded_hf_ppl=export_baseline,
            ppl_token_hash=workflow.ppl.token_hash,
            qconfig_hash=workflow.plan.payload.evaluations[0].qconfig_hash,
            meta_parameters=0,
            meta_buffers=0,
        ).save(output_path)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(planner_export.subprocess, "run", run)
    result = planner_export.verify_export_roundtrip(
        model_dir=tmp_path,
        ppl_tokens=workflow.ppl,
        source_hf_ppl=export_baseline,
        expected_qconfig_hash=qconfig_audit.qconfig_hash,
        max_degradation=workflow.plan.payload.quality_gate_max_degradation,
        device="cpu",
    )
    assert result.meta_parameters == result.meta_buffers == 0
    assert result.source_hf_ppl == export_baseline
    assert result.source_hf_ppl != workflow.plan.payload.baseline
    assert result.degradation == 0.0
    assert result.max_degradation == workflow.plan.payload.quality_gate_max_degradation
    assert result.ppl_token_hash == workflow.ppl.token_hash
    assert (tmp_path / "reload_validation.json").exists()
    assert not (tmp_path / "reload_measurement.json").exists()


def test_split_export_validation_reports_subprocess_and_quality_failures(
    tmp_path: Path,
    workflow: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_baseline = workflow.plan.payload.baseline
    qconfig_audit = planner_export.audit_export_binding(
        workflow.factory(),
        workflow.qconfig,
        workflow.space,
        workflow.plan,
        workflow.calibration,
        workflow.ppl,
    )

    invalid_qconfig = copy.deepcopy(workflow.qconfig)
    invalid_qconfig.layer_quant_config = {}
    with pytest.raises(SchemaValidationError):
        planner_export.audit_export_binding(
            workflow.factory(),
            invalid_qconfig,
            workflow.space,
            workflow.plan,
            workflow.calibration,
            workflow.ppl,
        )
    other_calibration = TokenDataset.create(
        purpose=workflow.calibration.purpose,
        dataset=workflow.calibration.dataset,
        split=workflow.calibration.split,
        revision=workflow.calibration.revision,
        tokenizer_id=workflow.calibration.tokenizer_id,
        tokenizer_fingerprint=workflow.calibration.tokenizer_fingerprint,
        sampling_seed=workflow.calibration.sampling_seed,
        sequences=tuple(tuple(reversed(sequence)) for sequence in workflow.calibration.sequences),
    )
    with pytest.raises(PlanSelectionError, match="token manifests"):
        planner_export.audit_export_binding(
            workflow.factory(),
            workflow.qconfig,
            workflow.space,
            workflow.plan,
            other_calibration,
            workflow.ppl,
        )

    for directory_name in ("failed", "wrong-tokens", "wrong-qconfig", "unresolved-meta", "degraded"):
        directory = tmp_path / directory_name
        directory.mkdir()
        (directory / "reload_validation.json").write_text("stale")

    monkeypatch.setattr(
        planner_export.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 1, "", "reload failed"),
    )
    with pytest.raises(PlanSelectionError, match="reload failed"):
        planner_export.verify_export_roundtrip(
            model_dir=tmp_path / "failed",
            ppl_tokens=workflow.ppl,
            source_hf_ppl=export_baseline,
            expected_qconfig_hash=qconfig_audit.qconfig_hash,
            max_degradation=workflow.plan.payload.quality_gate_max_degradation,
            device="cpu",
        )

    def wrong_tokens(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--output") + 1])
        planner_export.ReloadMeasurement(
            reloaded_hf_ppl=export_baseline,
            ppl_token_hash="sha256:wrong",
            qconfig_hash=qconfig_audit.qconfig_hash,
            meta_parameters=0,
            meta_buffers=0,
        ).save(output_path)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(planner_export.subprocess, "run", wrong_tokens)
    with pytest.raises(PlanSelectionError, match="token manifest"):
        planner_export.verify_export_roundtrip(
            model_dir=tmp_path / "wrong-tokens",
            ppl_tokens=workflow.ppl,
            source_hf_ppl=export_baseline,
            expected_qconfig_hash=qconfig_audit.qconfig_hash,
            max_degradation=workflow.plan.payload.quality_gate_max_degradation,
            device="cpu",
        )

    def wrong_qconfig(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--output") + 1])
        planner_export.ReloadMeasurement(
            reloaded_hf_ppl=export_baseline,
            ppl_token_hash=workflow.ppl.token_hash,
            qconfig_hash="sha256:wrong",
            meta_parameters=0,
            meta_buffers=0,
        ).save(output_path)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(planner_export.subprocess, "run", wrong_qconfig)
    with pytest.raises(PlanSelectionError, match="QConfig differs"):
        planner_export.verify_export_roundtrip(
            model_dir=tmp_path / "wrong-qconfig",
            ppl_tokens=workflow.ppl,
            source_hf_ppl=export_baseline,
            expected_qconfig_hash=qconfig_audit.qconfig_hash,
            max_degradation=workflow.plan.payload.quality_gate_max_degradation,
            device="cpu",
        )

    def unresolved_meta(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--output") + 1])
        planner_export.ReloadMeasurement(
            reloaded_hf_ppl=export_baseline,
            ppl_token_hash=workflow.ppl.token_hash,
            qconfig_hash=qconfig_audit.qconfig_hash,
            meta_parameters=1,
            meta_buffers=0,
        ).save(output_path)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(planner_export.subprocess, "run", unresolved_meta)
    with pytest.raises(PlanSelectionError, match="unresolved meta"):
        planner_export.verify_export_roundtrip(
            model_dir=tmp_path / "unresolved-meta",
            ppl_tokens=workflow.ppl,
            source_hf_ppl=export_baseline,
            expected_qconfig_hash=qconfig_audit.qconfig_hash,
            max_degradation=workflow.plan.payload.quality_gate_max_degradation,
            device="cpu",
        )

    def degraded(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--output") + 1])
        high_nll = math.log(export_baseline.ppl * 100) * export_baseline.token_count
        ppl = PplResult(high_nll, export_baseline.token_count, export_baseline.ppl * 100)
        planner_export.ReloadMeasurement(
            reloaded_hf_ppl=ppl,
            ppl_token_hash=workflow.ppl.token_hash,
            qconfig_hash=workflow.plan.payload.evaluations[0].qconfig_hash,
            meta_parameters=0,
            meta_buffers=0,
        ).save(output_path)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(planner_export.subprocess, "run", degraded)
    with pytest.raises(PlanSelectionError, match="exceeds"):
        planner_export.verify_export_roundtrip(
            model_dir=tmp_path / "degraded",
            ppl_tokens=workflow.ppl,
            source_hf_ppl=export_baseline,
            expected_qconfig_hash=qconfig_audit.qconfig_hash,
            max_degradation=workflow.plan.payload.quality_gate_max_degradation,
            device="cpu",
        )
    for directory_name in ("failed", "wrong-tokens", "wrong-qconfig", "unresolved-meta", "degraded"):
        directory = tmp_path / directory_name
        assert not (directory / "reload_validation.json").exists()
        assert not (directory / "reload_measurement.json").exists()


def test_reload_validation_entrypoint(
    tmp_path: Path, workflow: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens_path = tmp_path / "tokens.json"
    result_path = tmp_path / "result.json"
    workflow.ppl.save(tokens_path)

    def imported_model(**_kwargs: object) -> nn.Module:
        model = workflow.factory()
        model.quant_config = object()
        return model

    monkeypatch.setattr(reload_validation, "import_model_from_safetensors", imported_model)
    monkeypatch.setattr(reload_validation, "evaluate_ppl", lambda *_args: workflow.plan.payload.baseline)
    monkeypatch.setattr(
        reload_validation,
        "qconfig_semantic_hash",
        lambda _qconfig: workflow.plan.payload.evaluations[0].qconfig_hash,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reload-validation",
            "--model-dir",
            str(tmp_path),
            "--tokens",
            str(tokens_path),
            "--device",
            "cpu",
            "--output",
            str(result_path),
        ],
    )
    reload_validation.main()
    result = planner_export.ReloadMeasurement.load(result_path)
    assert result.reloaded_hf_ppl == workflow.plan.payload.baseline
    assert result.ppl_token_hash == workflow.ppl.token_hash
    assert result.qconfig_hash == workflow.plan.payload.evaluations[0].qconfig_hash


def test_reload_validation_result_rejects_meta_tensors() -> None:
    ppl = PplResult(0.0, 1, 1.0)
    with pytest.raises(SchemaValidationError, match="meta"):
        ReloadValidationResult(ppl, ppl, 0.0, 0.1, "sha256:tokens", "sha256:qconfig", 1, 0)


def test_reload_validation_result_rejects_mismatched_hf_metrics() -> None:
    ppl = PplResult(0.0, 1, 1.0)
    with pytest.raises(SchemaValidationError, match="degradation"):
        ReloadValidationResult(ppl, ppl, 0.1, 0.2, "sha256:tokens", "sha256:qconfig", 0, 0)
    with pytest.raises(SchemaValidationError, match="token count"):
        ReloadValidationResult(
            ppl,
            PplResult(0.0, 2, 1.0),
            0.0,
            0.1,
            "sha256:tokens",
            "sha256:qconfig",
            0,
            0,
        )
    with pytest.raises(SchemaValidationError, match="quality gate"):
        ReloadValidationResult(ppl, PplResult(math.log(2), 1, 2.0), 1.0, 0.1, "sha256:tokens", "sha256:qconfig", 0, 0)


def _run_tiny_qwen3_e2e(tmp_path: Path, device: str | torch.device) -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(41)
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    )
    source = Qwen3ForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()
    source_state = copy.deepcopy(source.state_dict())
    strategy = make_strategy(str(tmp_path))
    planner = MixedPrecisionPlanner(strategy)
    space = planner.decision_space(source)
    profile = planner.sensitivity_profile(source, space)
    candidates = planner.search(space, profile)
    calibration, ppl = make_tokens()

    def factory() -> nn.Module:
        model = Qwen3ForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()
        model.load_state_dict(source_state)
        return model

    device_name = str(device)
    selection_model_factory = prepend_model(source, factory)
    del source
    plan = planner.select_best_plan(selection_model_factory, space, candidates, calibration, ppl, device=device_name)
    qconfig = planner.build_qconfig_from_plan(space, plan)
    assert qconfig.layer_quant_config
    export_dir = tmp_path / "exported"
    export_model = factory()
    try:
        qconfig_audit = planner_export.audit_export_binding(
            export_model,
            qconfig,
            space,
            plan,
            calibration,
            ppl,
        )
        source_hf_ppl = evaluate_ppl(export_model, ppl, device_name)
        quantizer = ModelQuantizer(qconfig)
        export_model = quantizer.quantize_model(export_model, calibration.to_dataloader(device_name))
        export_model = quantizer.freeze(export_model)
        export_safetensors(
            model=export_model,
            output_dir=export_dir,
            custom_mode="quark",
            weight_format="real_quantized",
            pack_method="reorder",
        )
    finally:
        del export_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_write_json(export_dir / "qconfig.json", qconfig.to_dict())
    atomic_write_json(export_dir / "qconfig_audit.json", qconfig_audit.to_dict())
    result = planner_export.verify_export_roundtrip(
        model_dir=export_dir,
        ppl_tokens=ppl,
        source_hf_ppl=source_hf_ppl,
        expected_qconfig_hash=qconfig_audit.qconfig_hash,
        max_degradation=plan.payload.quality_gate_max_degradation,
        device=device_name,
    )
    assert result.reloaded_hf_ppl.ppl > 0
    assert (tmp_path / "exported" / "config.json").exists()
    assert (tmp_path / "exported" / "reload_validation.json").exists()


def test_tiny_qwen3_quantize_export_and_fresh_reload(tmp_path: Path) -> None:
    _run_tiny_qwen3_e2e(tmp_path, "cpu")


@require_torch_cuda
def test_tiny_qwen3_gpu_end_to_end(tmp_path: Path) -> None:
    _run_tiny_qwen3_e2e(tmp_path, torch_device)
