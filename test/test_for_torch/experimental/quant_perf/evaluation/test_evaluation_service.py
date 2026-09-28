#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Managed accuracy-only evaluation service tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from quark.experimental.torch.quant_perf.evaluation import service as service_module
from quark.experimental.torch.quant_perf.session.spec import EvalProfile, Spec


def _spec(tmp_path: Path) -> Spec:
    return Spec(
        model_dir="/models/quant",
        base_model="/models/base",
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="MI355X",
        isl=128,
        osl=128,
        quant_strategy=None,
        accuracy_gap=0.05,
        session_dir=str(tmp_path),
        arch_fingerprint="eval-only",
        eval_profile=EvalProfile(
            profile_id="gsm8k-chat-default-v2",
            profile_hash="",
            model_mode="chat",
            apply_chat_template=True,
            enable_thinking=None,
            detection_reason="chat_template",
            schema_version=2,
            policy_version="quark-quant-perf-gsm8k-profile-v2",
            task="gsm8k",
            num_fewshot=5,
            prompting_strategy="cot",
            settings_source="quark_policy_default",
        ).with_computed_hash(),
        gsm8k_num_samples=20,
    )


def test_evaluation_service_runs_managed_gate_and_writes_reports(tmp_path):
    with patch(
        "quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline",
        side_effect=[0.80, 0.78],
    ):
        result = service_module.EvaluationService().run(
            _spec(tmp_path),
            "/models/quant",
        )

    assert result["status"] == "success"
    assert result["baseline"] == 0.80
    assert result["quantized"] == 0.78
    assert result["gap"] == pytest.approx(0.025)
    assert result["passed"] is True
    state = json.loads((tmp_path / "eval_state.json").read_text())
    assert state["stage"] == "done"
    assert state["profile_hash"] == _spec(tmp_path).eval_profile.profile_hash
    report = json.loads((tmp_path / "reports" / "eval.json").read_text())
    assert report["accuracy"]["quantized"] == 0.78
    assert (tmp_path / "reports" / "eval.md").is_file()


def test_evaluation_service_reuses_matching_baseline_on_resume(tmp_path):
    spec = _spec(tmp_path)
    baseline_fingerprint, quant_fingerprint = service_module._runtime_fingerprints(spec, "/models/quant")
    (tmp_path / "eval_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "quantized",
                "profile_hash": spec.eval_profile.profile_hash,
                "base_model": spec.base_model,
                "quant_model": "/models/quant",
                "gsm8k_num_samples": spec.gsm8k_num_samples,
                "baseline": 0.80,
                "baseline_artifact": "/prior/baseline",
                "baseline_fingerprint": baseline_fingerprint,
                "quant_fingerprint": quant_fingerprint,
            }
        )
    )

    with patch(
        "quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline",
        return_value=0.79,
    ) as mock_eval:
        result = service_module.EvaluationService().run(
            spec,
            "/models/quant",
        )

    assert mock_eval.call_count == 1
    assert mock_eval.call_args.kwargs["model_dir"] == "/models/quant"
    assert result["baseline"] == 0.80
    assert result["quantized"] == 0.79


def test_evaluation_service_remeasures_baseline_when_runtime_changes(tmp_path):
    spec = _spec(tmp_path)
    (tmp_path / "eval_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "quantized",
                "profile_hash": spec.eval_profile.profile_hash,
                "base_model": spec.base_model,
                "quant_model": "/models/quant",
                "gsm8k_num_samples": spec.gsm8k_num_samples,
                "baseline": 0.80,
                "baseline_artifact": "/prior/baseline",
                "baseline_fingerprint": "different-runtime",
            }
        )
    )

    with patch(
        "quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline",
        side_effect=[0.81, 0.79],
    ) as mock_eval:
        result = service_module.EvaluationService().run(
            spec,
            "/models/quant",
        )

    assert mock_eval.call_count == 2
    assert result["baseline"] == 0.81


def test_evaluation_service_resolves_cli_profile_overrides(
    monkeypatch,
    tmp_path,
):
    spec = replace(
        _spec(tmp_path),
        eval_profile=None,
        eval_discovery="off",
        eval_allow_llm=False,
        eval_task="gsm8k_cot_zeroshot",
        eval_num_fewshot=0,
        eval_prompting_strategy="cot",
        eval_thinking_mode="disabled",
        eval_max_gen_toks=768,
    )
    calls = []
    resolved = _spec(tmp_path).eval_profile

    def resolve(model_ref, **kwargs):
        calls.append((model_ref, kwargs))
        return resolved

    monkeypatch.setattr(service_module, "resolve_eval_profile", resolve)
    with patch(
        "quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline",
        side_effect=[0.80, 0.79],
    ):
        service_module.EvaluationService().run(spec, "/models/quant")

    assert calls == [
        (
            spec.base_model,
            {
                "discovery": "off",
                "allow_llm": False,
                "overrides": {
                    "task": "gsm8k_cot_zeroshot",
                    "num_fewshot": 0,
                    "prompting_strategy": "cot",
                    "enable_thinking": False,
                    "max_gen_toks": 768,
                },
                "artifact_dir": tmp_path / "evaluation" / "profile",
            },
        )
    ]


def test_evaluation_service_persists_baseline_before_quantized_failure(
    tmp_path,
):
    spec = _spec(tmp_path)

    with (
        patch(
            "quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline",
            side_effect=[0.80, RuntimeError("quant load failed")],
        ),
        pytest.raises(RuntimeError, match="quant load failed"),
    ):
        service_module.EvaluationService().run(
            spec,
            "/models/quant",
        )

    state = json.loads((tmp_path / "eval_state.json").read_text())
    assert state["stage"] == "failed"
    assert state["baseline"] == 0.80
    assert state["baseline_artifact"].endswith("evaluation/baseline/attempt-1")
    assert "quant load failed" in state["error"]

    with patch(
        "quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline",
        return_value=0.79,
    ) as mock_eval:
        result = service_module.EvaluationService().run(
            spec,
            "/models/quant",
        )

    assert mock_eval.call_count == 1
    assert result["baseline"] == 0.80
    assert result["quantized"] == 0.79


def test_evaluation_service_preserves_attempt_ledger_across_retry(
    tmp_path,
):
    spec = _spec(tmp_path)
    runtime_origins = {
        "vllm": {
            "origin": "/runtime/vllm/__init__.py",
            "matched": True,
        }
    }

    with (
        patch(
            "quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline",
            side_effect=[0.80, RuntimeError("quant load failed")],
        ),
        pytest.raises(RuntimeError, match="quant load failed"),
    ):
        service_module.EvaluationService().run(
            spec,
            "/models/quant",
            runtime_origin_evidence=runtime_origins,
        )

    with patch(
        "quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline",
        return_value=0.79,
    ):
        service_module.EvaluationService().run(
            spec,
            "/models/quant",
            runtime_origin_evidence=runtime_origins,
        )

    state = json.loads((tmp_path / "eval_state.json").read_text())
    assert [(attempt["role"], attempt["status"]) for attempt in state["attempts"]] == [
        ("baseline", "passed"),
        ("quantized", "failed"),
        ("baseline", "reused"),
        ("quantized", "passed"),
    ]
    assert state["attempts"][1]["artifact"].endswith("evaluation/quantized/attempt-1")
    assert state["attempts"][1]["error"] == ("RuntimeError: quant load failed")
    assert state["attempts"][-1]["runtime_origins"] == runtime_origins


def test_evaluation_report_records_runtime_origins(tmp_path):
    runtime_origins = {
        "quark": {
            "origin": "/runtime/quark/__init__.py",
            "matched": True,
        }
    }
    with patch(
        "quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline",
        side_effect=[0.80, 0.78],
    ):
        service_module.EvaluationService().run(
            _spec(tmp_path),
            "/models/quant",
            runtime_origin_evidence=runtime_origins,
        )

    report = json.loads((tmp_path / "reports" / "eval.json").read_text())
    assert report["runtime_origins"] == runtime_origins
