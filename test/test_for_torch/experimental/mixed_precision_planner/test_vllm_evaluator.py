#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import quark.experimental.torch.mixed_precision_planner.evaluators.vllm as vllm_evaluator
from quark.experimental.torch.mixed_precision_planner import (
    MixedPrecisionPlanner,
    MixedPrecisionStrategy,
    TokenDataset,
)
from quark.experimental.torch.mixed_precision_planner.data import TokenPurpose
from quark.experimental.torch.mixed_precision_planner.errors import (
    ArtifactCompatibilityError,
    CandidateEvaluationError,
    EvaluatorStateError,
    EvaluatorUnavailableError,
    PlanSelectionError,
    SchemaValidationError,
)
from quark.experimental.torch.mixed_precision_planner.evaluators.vllm import (
    VllmEvaluator,
    VllmRuntimeConfig,
    evaluate_manifest_ppl,
)
from quark.experimental.torch.mixed_precision_planner.runtime import EvaluationRuntimeContext, create_evaluator

_TEST_MODEL_FINGERPRINT = "sha256:test-model"
_TEST_BEHAVIOR_FINGERPRINT = "sha256:test-behavior"
_TEST_TOKENIZER_FINGERPRINT = "sha256:test-tokenizer"


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)


class TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)


class TinyQwen3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        layer = nn.Module()
        layer.self_attn = TinyAttention()
        layer.mlp = TinyMlp()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.to(torch.bfloat16)


def make_strategy(*, deployment_backend: str = "vllm") -> MixedPrecisionStrategy:
    return MixedPrecisionStrategy.from_dict(
        {
            "schema_version": 1,
            "hardware": {"target": "mi355"},
            "deployment_backend": {"name": deployment_backend},
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
                "budget": {"metric": "effective_bits", "value": 12.0, "scope": "quantizable"},
                "num_candidates": 2,
            },
            "plan_selection": {
                "evaluator": {"backend": "vllm", "protocol": "token_ppl_v1"},
                "quality_gate": {
                    "metric": "ppl",
                    "dataset": "wikitext2",
                    "max_degradation": 0.02,
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


class FakeSamplingParams:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeLogprob:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


class FakeLlm:
    factory_calls = 0
    rpc_calls: list[str] = []
    cache_resets = 0
    runtime_configs: list[dict[str, Any]] = []
    runtime_names = (
        "model_runner.model.layers.0.mlp.down_proj",
        "model_runner.model.layers.0.mlp.gate_up_proj",
        "model_runner.model.layers.0.self_attn.o_proj",
        "model_runner.model.layers.0.self_attn.qkv_proj",
    )

    def __init__(self, **kwargs: object) -> None:
        type(self).factory_calls += 1
        self.kwargs = kwargs

    def collective_rpc(self, method: str, args: tuple[object, ...] = ()) -> list[Any]:
        type(self).rpc_calls.append(method)
        if method == "quark_quantizable_module_names":
            return [self.runtime_names]
        if method == "reset_to_original":
            return [True]
        assert method == "requantize_with_config"
        config_path = Path(str(args[0]))
        config = json.loads(config_path.read_text())
        from quark.experimental.torch.plugin.fakequant_worker import (
            _normalize_calibration_token_ids,
            _runtime_qconfig_payload_hash,
        )

        sequences = _normalize_calibration_token_ids(args[4])
        runtime_qconfig_hash = _runtime_qconfig_payload_hash(str(config_path))
        assert runtime_qconfig_hash == args[6]
        type(self).runtime_configs.append(config)
        return [
            {
                "calibration_token_hash": args[5],
                "runtime_qconfig_hash": runtime_qconfig_hash,
                "quantized_modules": tuple(sorted(config["layer_quant_config"])),
                "calibrated_sequences": len(sequences),
                "calibrated_tokens": sum(len(sequence) for sequence in sequences),
            }
        ]

    def generate(self, prompts: list[dict[str, list[int]]], **_kwargs: object) -> list[SimpleNamespace]:
        outputs = []
        for prompt in prompts:
            token_ids = prompt["prompt_token_ids"]
            logprobs = [None] + [{token: FakeLogprob(-math.log(2.0))} for token in token_ids[1:]]
            outputs.append(SimpleNamespace(prompt_logprobs=logprobs))
        return outputs

    def reset_prefix_cache(self) -> None:
        type(self).cache_resets += 1


def test_single_vllm_engine_evaluates_baseline_and_all_candidates() -> None:
    FakeLlm.factory_calls = 0
    FakeLlm.rpc_calls = []
    FakeLlm.cache_resets = 0
    FakeLlm.runtime_configs = []
    strategy = make_strategy()
    planner = MixedPrecisionPlanner(strategy)
    model = TinyQwen3()
    space = planner.decision_space(model)
    profile = planner.sensitivity_profile(model, space)
    candidates = planner.search(space, profile)
    calibration, ppl = make_tokens()
    evaluator = VllmEvaluator(
        VllmRuntimeConfig(
            model="tiny",
            expected_model_state_fingerprint=candidates.payload.model_state_fingerprint,
            expected_model_behavior_fingerprint=candidates.payload.model_behavior_fingerprint,
            max_model_len=2048,
        ),
        space,
        llm_factory=FakeLlm,
        sampling_params_cls=FakeSamplingParams,
        runtime_version="test",
    )

    plan = planner.select_best_plan_with_evaluator(
        evaluator,
        space,
        candidates,
        calibration,
        ppl,
    )

    assert FakeLlm.factory_calls == 1
    assert FakeLlm.rpc_calls.count("requantize_with_config") == len(candidates.payload.candidates)
    assert all(
        evaluation.ppl is not None and evaluation.ppl.ppl == pytest.approx(2.0)
        for evaluation in plan.payload.evaluations
    )
    assert all(
        evaluation.runtime_audit is not None
        and evaluation.runtime_audit.backend == "vllm"
        and evaluation.runtime_audit.calibration_token_hash == calibration.token_hash
        for evaluation in plan.payload.evaluations
    )
    assert plan.payload.evaluator.backend == "vllm"
    assert (
        plan.payload.evaluator.options["expected_model_state_fingerprint"] == candidates.payload.model_state_fingerprint
    )
    assert (
        plan.payload.evaluator.options["expected_model_behavior_fingerprint"]
        == candidates.payload.model_behavior_fingerprint
    )
    assert plan.payload.evaluator.options["model"] == "tiny"
    assert FakeLlm.cache_resets > len(candidates.payload.candidates)
    assert len(FakeLlm.runtime_configs) == len(candidates.payload.candidates)
    runtime_targets = set(FakeLlm.runtime_names)
    for config in FakeLlm.runtime_configs:
        quantized_targets = set(config["layer_quant_config"])
        native_targets = {name.removesuffix(".*") for name in config["exclude"]}
        assert quantized_targets | native_targets == runtime_targets
        assert quantized_targets.isdisjoint(native_targets)
    assert evaluator._closed


def test_vllm_reset_failure_poisoning_aborts_selection() -> None:
    class ResetFailureLlm(FakeLlm):
        reset_calls = 0

        def collective_rpc(self, method: str, args: tuple[object, ...] = ()) -> list[Any]:
            if method == "reset_to_original":
                type(self).reset_calls += 1
                if type(self).reset_calls == 3:
                    type(self).rpc_calls.append(method)
                    return [False]
            return super().collective_rpc(method, args)

    ResetFailureLlm.rpc_calls = []
    strategy = make_strategy()
    planner = MixedPrecisionPlanner(strategy)
    model = TinyQwen3()
    space = planner.decision_space(model)
    profile = planner.sensitivity_profile(model, space)
    candidates = planner.search(space, profile)
    calibration, ppl = make_tokens()
    evaluator = VllmEvaluator(
        VllmRuntimeConfig(
            model="tiny",
            expected_model_state_fingerprint=candidates.payload.model_state_fingerprint,
            expected_model_behavior_fingerprint=candidates.payload.model_behavior_fingerprint,
            max_model_len=2048,
        ),
        space,
        llm_factory=ResetFailureLlm,
        sampling_params_cls=FakeSamplingParams,
        runtime_version="test",
    )

    with pytest.raises(EvaluatorStateError, match="model reset failed"):
        planner.select_best_plan_with_evaluator(evaluator, space, candidates, calibration, ppl)
    assert ResetFailureLlm.rpc_calls.count("requantize_with_config") == 1
    assert evaluator._closed


def test_vllm_evaluator_rejects_no_fusion_before_engine_start() -> None:
    FakeLlm.factory_calls = 0
    strategy = make_strategy(deployment_backend="no_fusion")
    space = MixedPrecisionPlanner(strategy).decision_space(TinyQwen3())
    with pytest.raises(EvaluatorUnavailableError, match="cannot faithfully bind"):
        VllmEvaluator(
            VllmRuntimeConfig(
                model="tiny",
                expected_model_state_fingerprint=_TEST_MODEL_FINGERPRINT,
                expected_model_behavior_fingerprint=_TEST_BEHAVIOR_FINGERPRINT,
            ),
            space,
            llm_factory=FakeLlm,
            sampling_params_cls=FakeSamplingParams,
            runtime_version="test",
        )
    assert FakeLlm.factory_calls == 0


def test_manifest_ppl_rejects_missing_target_logprob() -> None:
    _, ppl = make_tokens()

    class MissingTargetLlm:
        def generate(self, prompts: list[dict[str, list[int]]], **_kwargs: object) -> list[SimpleNamespace]:
            return [SimpleNamespace(prompt_logprobs=[None] + [{999: FakeLogprob(-1.0)}] * (len(ppl.sequences[0]) - 1))]

    with pytest.raises(CandidateEvaluationError, match="omit"):
        evaluate_manifest_ppl(MissingTargetLlm(), ppl, FakeSamplingParams)


def test_vllm_evaluator_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = make_strategy()
    planner = MixedPrecisionPlanner(strategy)
    model = TinyQwen3()
    space = planner.decision_space(model)

    def missing_vllm(name: str) -> object:
        if name == "vllm":
            raise ImportError
        return __import__(name)

    monkeypatch.setattr(vllm_evaluator.importlib, "import_module", missing_vllm)
    with pytest.raises(EvaluatorUnavailableError, match="not installed"):
        VllmEvaluator(
            VllmRuntimeConfig(
                model="tiny",
                expected_model_state_fingerprint=_TEST_MODEL_FINGERPRINT,
                expected_model_behavior_fingerprint=_TEST_BEHAVIOR_FINGERPRINT,
            ),
            space,
        )


def test_vllm_runtime_config_validation() -> None:
    with pytest.raises(SchemaValidationError, match="model source"):
        VllmRuntimeConfig(
            model="",
            expected_model_state_fingerprint=_TEST_MODEL_FINGERPRINT,
            expected_model_behavior_fingerprint=_TEST_BEHAVIOR_FINGERPRINT,
        )
    with pytest.raises(SchemaValidationError, match="model fingerprint"):
        VllmRuntimeConfig(
            model="tiny",
            expected_model_state_fingerprint="invalid",
            expected_model_behavior_fingerprint=_TEST_BEHAVIOR_FINGERPRINT,
        )
    with pytest.raises(SchemaValidationError, match="behavior fingerprint"):
        VllmRuntimeConfig(
            model="tiny",
            expected_model_state_fingerprint=_TEST_MODEL_FINGERPRINT,
            expected_model_behavior_fingerprint="invalid",
        )
    with pytest.raises(SchemaValidationError, match="model revision"):
        VllmRuntimeConfig(
            model="tiny",
            expected_model_state_fingerprint=_TEST_MODEL_FINGERPRINT,
            expected_model_behavior_fingerprint=_TEST_BEHAVIOR_FINGERPRINT,
            model_revision="",
        )
    with pytest.raises(SchemaValidationError, match="tensor_parallel_size"):
        VllmRuntimeConfig(
            model="tiny",
            expected_model_state_fingerprint=_TEST_MODEL_FINGERPRINT,
            expected_model_behavior_fingerprint=_TEST_BEHAVIOR_FINGERPRINT,
            tensor_parallel_size=0,
        )
    with pytest.raises(SchemaValidationError, match="gpu_memory_utilization"):
        VllmRuntimeConfig(
            model="tiny",
            expected_model_state_fingerprint=_TEST_MODEL_FINGERPRINT,
            expected_model_behavior_fingerprint=_TEST_BEHAVIOR_FINGERPRINT,
            gpu_memory_utilization=0.0,
        )
    with pytest.raises(SchemaValidationError, match="gpu_memory_utilization"):
        VllmRuntimeConfig(
            model="tiny",
            expected_model_state_fingerprint=_TEST_MODEL_FINGERPRINT,
            expected_model_behavior_fingerprint=_TEST_BEHAVIOR_FINGERPRINT,
            gpu_memory_utilization=1.01,
        )
    with pytest.raises(SchemaValidationError, match="max_model_len"):
        VllmRuntimeConfig(
            model="tiny",
            expected_model_state_fingerprint=_TEST_MODEL_FINGERPRINT,
            expected_model_behavior_fingerprint=_TEST_BEHAVIOR_FINGERPRINT,
            max_model_len=1,
        )


def test_runtime_rejects_no_fusion_before_constructing_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quark.experimental.torch.mixed_precision_planner.runtime as planner_runtime

    strategy = make_strategy(deployment_backend="no_fusion")
    planner = MixedPrecisionPlanner(strategy)
    model = TinyQwen3()
    space = planner.decision_space(model)
    profile = planner.sensitivity_profile(model, space)
    candidates = planner.search(space, profile)
    monkeypatch.setattr(
        planner_runtime,
        "VllmEvaluator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("vLLM construction must not run")),
    )

    with pytest.raises(EvaluatorUnavailableError, match="cannot faithfully bind"):
        create_evaluator(
            strategy,
            space,
            candidates,
            EvaluationRuntimeContext(model_source="/tiny", device="cuda", max_model_len=4096),
        )


def test_runtime_rejects_different_checkpoint_before_constructing_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quark.experimental.torch.mixed_precision_planner.runtime as planner_runtime

    strategy = make_strategy()
    planner = MixedPrecisionPlanner(strategy)
    profiled_model = TinyQwen3()
    space = planner.decision_space(profiled_model)
    profile = planner.sensitivity_profile(profiled_model, space)
    candidates = planner.search(space, profile)
    different_model = TinyQwen3()
    with torch.no_grad():
        next(different_model.parameters()).add_(1)
    monkeypatch.setattr(
        planner_runtime,
        "VllmEvaluator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("vLLM construction must not run")),
    )
    model_loads: list[dict[str, object]] = []
    monkeypatch.setattr(
        planner_runtime,
        "get_model",
        lambda **kwargs: model_loads.append(kwargs) or (different_model, None),
    )

    with pytest.raises(ArtifactCompatibilityError, match="model source weights"):
        create_evaluator(
            strategy,
            space,
            candidates,
            EvaluationRuntimeContext(
                model_source="/tiny",
                device="cuda",
                max_model_len=4096,
            ),
        )
    assert model_loads[0]["ckpt_path"] == "/tiny"

    same_weights_different_behavior = TinyQwen3()
    same_weights_different_behavior.load_state_dict(profiled_model.state_dict())
    same_weights_different_behavior.config.rope_theta = 1_000_000.0
    monkeypatch.setattr(
        planner_runtime,
        "get_model",
        lambda **_kwargs: (same_weights_different_behavior, None),
    )
    with pytest.raises(ArtifactCompatibilityError, match="model source behavior"):
        create_evaluator(
            strategy,
            space,
            candidates,
            EvaluationRuntimeContext(model_source="/tiny", device="cuda", max_model_len=4096),
        )


def test_planner_validates_token_provenance_before_vllm_model_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quark.experimental.torch.mixed_precision_planner.runtime as planner_runtime

    strategy = make_strategy()
    planner = MixedPrecisionPlanner(strategy)
    profiled_model = TinyQwen3()
    space = planner.decision_space(profiled_model)
    profile = planner.sensitivity_profile(profiled_model, space)
    candidates = planner.search(space, profile)
    calibration, ppl = make_tokens()
    wrong_seed_calibration = TokenDataset.create(
        purpose=calibration.purpose,
        dataset=calibration.dataset,
        split=calibration.split,
        revision=calibration.revision,
        tokenizer_id=calibration.tokenizer_id,
        tokenizer_fingerprint=calibration.tokenizer_fingerprint,
        sampling_seed=43,
        sequences=calibration.sequences,
    )
    monkeypatch.setattr(
        planner_runtime,
        "get_model",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("model preflight must not start")),
    )

    with pytest.raises(PlanSelectionError, match="Calibration"):
        planner.select_best_plan_with_runtime(
            EvaluationRuntimeContext(
                model_source="/tiny",
                device="cuda",
                max_model_len=4096,
            ),
            space,
            candidates,
            wrong_seed_calibration,
            ppl,
        )


def test_runtime_validates_length_and_propagates_vllm_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import quark.experimental.torch.mixed_precision_planner.runtime as planner_runtime

    strategy = make_strategy()
    planner = MixedPrecisionPlanner(strategy)
    model = TinyQwen3()
    space = planner.decision_space(model)
    profile = planner.sensitivity_profile(model, space)
    candidates = planner.search(space, profile)
    with pytest.raises(SchemaValidationError, match="resolved absolute"):
        create_evaluator(
            strategy,
            space,
            candidates,
            EvaluationRuntimeContext(model_source="tiny", device="cuda", max_model_len=4096),
        )
    with pytest.raises(SchemaValidationError, match="model_revision"):
        create_evaluator(
            strategy,
            space,
            candidates,
            EvaluationRuntimeContext(
                model_source="/tiny",
                model_revision="commit",
                device="cuda",
                max_model_len=4096,
            ),
        )
    with pytest.raises(SchemaValidationError, match="tensor_parallel_size"):
        create_evaluator(
            strategy,
            space,
            candidates,
            EvaluationRuntimeContext(
                model_source="/tiny",
                device="cuda",
                tensor_parallel_size=2,
                max_model_len=4096,
            ),
        )
    context = EvaluationRuntimeContext(model_source="/tiny", device="cuda", max_model_len=2048)

    with pytest.raises(SchemaValidationError, match="must exceed"):
        create_evaluator(strategy, space, candidates, context)
    monkeypatch.setattr(
        planner_runtime,
        "get_model",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    with pytest.raises(EvaluatorUnavailableError, match="fingerprint validation"):
        create_evaluator(
            strategy,
            space,
            candidates,
            EvaluationRuntimeContext(model_source="/tiny", device="cuda", max_model_len=4096),
        )

    created: list[tuple[VllmRuntimeConfig, object]] = []
    model_loads: list[dict[str, object]] = []
    sentinel = object()

    def create_vllm(runtime: VllmRuntimeConfig, actual_space: object) -> object:
        created.append((runtime, actual_space))
        return sentinel

    monkeypatch.setattr(planner_runtime, "VllmEvaluator", create_vllm)
    monkeypatch.setattr(
        planner_runtime,
        "get_model",
        lambda **kwargs: model_loads.append(kwargs) or (model, None),
    )
    context = EvaluationRuntimeContext(
        model_source="/cache/snapshots/commit",
        device="cuda",
        model_revision="commit",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=4096,
    )
    assert create_evaluator(strategy, space, candidates, context) is sentinel
    assert model_loads == [
        {
            "ckpt_path": "/cache/snapshots/commit",
            "data_type": "auto",
            "device": "cuda",
            "multi_gpu": False,
            "multi_device": False,
            "attn_implementation": "eager",
            "trust_remote_code": True,
        }
    ]
    assert created == [
        (
            VllmRuntimeConfig(
                model="/cache/snapshots/commit",
                expected_model_state_fingerprint=candidates.payload.model_state_fingerprint,
                expected_model_behavior_fingerprint=candidates.payload.model_behavior_fingerprint,
                model_revision="commit",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.5,
                max_model_len=4096,
                trust_remote_code=True,
            ),
            space,
        )
    ]


def test_worker_requantize_rejects_invalid_rpc_inputs() -> None:
    from quark.experimental.torch.plugin.fakequant_worker import QuarkFakeQuantWorker

    requantize = QuarkFakeQuantWorker.requantize_with_config
    worker: Any = object()
    with pytest.raises(ValueError, match="non-empty"):
        requantize(worker, {}, calibration_token_ids=[])
    with pytest.raises(ValueError, match="count"):
        requantize(worker, {}, calib_size=2, calibration_token_ids=((1, 2),))
    with pytest.raises(ValueError, match="SHA-256"):
        requantize(worker, {}, calibration_token_hash="invalid")
    with pytest.raises(ValueError, match="does not match"):
        requantize(
            worker,
            {},
            calibration_token_ids=((1, 2),),
            runtime_qconfig_hash="sha256:not-the-payload-hash",
        )
