#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Regression tests for accuracy evaluation, baseline caching, and remote-code propagation."""

import ast
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quark.experimental.torch.quant_perf.evaluation.gate import AccuracyGate
from quark.experimental.torch.quant_perf.evaluation.throughput import throughput_benchmark
from quark.experimental.torch.quant_perf.pipeline.candidate_validation import (
    evaluate_candidate_accuracy,
    measure_candidate_screen,
    measure_quantized_candidate,
)
from quark.experimental.torch.quant_perf.repair.llm_repair import attempt_accuracy_repair
from quark.experimental.torch.quant_perf.repair.request import build_repair_request
from quark.experimental.torch.quant_perf.repair.service import RepairService
from quark.experimental.torch.quant_perf.session.spec import EvalProfile, Spec


def make_spec(tmp_path: Path, accuracy_gap: float = 0.02) -> Spec:
    return Spec(
        model_dir="m",
        base_model="m",
        framework="vllm",
        gpu_type="mi300x",
        gpu_arch="MI300X",
        isl=128,
        osl=128,
        quant_strategy="fp8",
        accuracy_gap=accuracy_gap,
        session_dir=str(tmp_path),
        arch_fingerprint="abc123",
        eval_profile=EvalProfile(
            profile_id="gsm8k-chat-nothink-v1",
            profile_hash="profile-hash",
            model_mode="chat",
            apply_chat_template=True,
            enable_thinking=False,
            detection_reason="chat_template",
        ),
    )


@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline", return_value=0.80)
def test_warm_baseline_caches_score_and_runs_once(mock_eval, tmp_path):
    gate = AccuracyGate(make_spec(tmp_path))
    gate.warm_baseline()
    gate.warm_baseline()

    assert gate._source_cache == 0.80
    mock_eval.assert_called_once()


@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline")
def test_eval_quantized_computes_gap_and_passes(mock_eval, tmp_path):
    gate = AccuracyGate(make_spec(tmp_path, accuracy_gap=0.05))
    gate._source_cache = 0.80
    mock_eval.return_value = 0.78
    result = gate.eval_quantized("/fake/quant_ckpt")
    assert result.gap == pytest.approx(0.025)
    assert result.passed


@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline")
def test_eval_quantized_fails_when_gap_exceeds_threshold(mock_eval, tmp_path):
    gate = AccuracyGate(make_spec(tmp_path, accuracy_gap=0.01))
    gate._source_cache = 0.80
    mock_eval.return_value = 0.60
    result = gate.eval_quantized("/fake/quant_ckpt")
    assert not result.passed
    assert result.gap == pytest.approx(0.25)


@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline")
def test_eval_quantized_warms_baseline_if_not_cached(mock_eval, tmp_path):
    gate = AccuracyGate(make_spec(tmp_path, accuracy_gap=0.05))
    mock_eval.side_effect = [0.80, 0.78]
    result = gate.eval_quantized("/fake/quant_ckpt")
    assert mock_eval.call_count == 2
    assert result.gap == pytest.approx(0.025)
    assert result.passed


@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline")
def test_eval_quantized_surfaces_load_failure_for_orchestrator(mock_eval, tmp_path):
    spec = replace(
        make_spec(tmp_path, accuracy_gap=0.05),
        vllm_extra_args=["--tensor-parallel-size 2"],
        framework_repo="/repo",
    )
    gate = AccuracyGate(spec)
    gate._source_cache = 0.80
    mock_eval.side_effect = RuntimeError("load failed")

    with pytest.raises(RuntimeError, match="load failed"):
        gate.eval_quantized("/fake/quant_ckpt")


@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline", return_value=0.78)
def test_eval_quantized_preserves_vllm_gpu_memory_utilization(
    mock_eval,
    tmp_path,
):
    spec = replace(
        make_spec(tmp_path, accuracy_gap=0.05),
        vllm_extra_args=["--gpu-memory-utilization=0.75"],
    )
    gate = AccuracyGate(spec)
    gate._source_cache = 0.80

    result = gate.eval_quantized("/fake/quant_ckpt")

    assert result.passed
    assert mock_eval.call_args.kwargs["gpu_memory_utilization"] == 0.75
    assert mock_eval.call_args.kwargs["moe_backend"] == ""
    assert mock_eval.call_args.kwargs["profile"] is spec.eval_profile


@patch(
    "quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline",
    return_value=0.78,
)
def test_eval_quantized_passes_vllm_moe_backend(mock_eval, tmp_path):
    spec = replace(
        make_spec(tmp_path, accuracy_gap=0.05),
        vllm_extra_args=["--moe-backend=triton"],
    )
    gate = AccuracyGate(spec)
    gate._source_cache = 0.80

    gate.eval_quantized("/fake/quant_ckpt")

    assert mock_eval.call_args.kwargs["moe_backend"] == "triton"


@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline", return_value=0.78)
def test_eval_quantized_passes_session_runtime(mock_eval, tmp_path):
    spec = make_spec(tmp_path, accuracy_gap=0.05)
    spec.runtime.runtime_python = "/session/venv/bin/python"
    spec.runtime.runtime_env = {"PYTHONPATH": "/session/overlay"}
    gate = AccuracyGate(spec)
    gate._source_cache = 0.80

    gate.eval_quantized("/fake/quant_ckpt")

    assert mock_eval.call_args.kwargs["runtime_python"] == ("/session/venv/bin/python")
    assert mock_eval.call_args.kwargs["runtime_env"] == {"PYTHONPATH": "/session/overlay"}


@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline")
def test_accuracy_gate_records_persistent_eval_artifacts(
    mock_eval,
    tmp_path,
):
    spec = make_spec(tmp_path, accuracy_gap=0.05)
    gate = AccuracyGate(spec)
    mock_eval.side_effect = [0.80, 0.78]
    prior_baseline = tmp_path / "evaluation" / "baseline" / "attempt-1"
    prior_quantized = tmp_path / "evaluation" / "quantized" / "attempt-1"
    prior_baseline.mkdir(parents=True)
    prior_quantized.mkdir(parents=True)
    (prior_baseline / "results-old.json").write_text("{}")
    (prior_quantized / "results-old.json").write_text("{}")

    result = gate.eval_quantized("/fake/quant_ckpt")

    assert mock_eval.call_args_list[0].kwargs["output_dir"] == (tmp_path / "evaluation" / "baseline" / "attempt-2")
    assert mock_eval.call_args_list[1].kwargs["output_dir"] == (tmp_path / "evaluation" / "quantized" / "attempt-2")
    assert result.artifacts == {
        "baseline": str(tmp_path / "evaluation" / "baseline" / "attempt-2"),
        "quantized": str(tmp_path / "evaluation" / "quantized" / "attempt-2"),
    }
    assert (prior_baseline / "results-old.json").is_file()
    assert (prior_quantized / "results-old.json").is_file()


# -- baseline health gate --------------------------------------------------


@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline", return_value=0.80)
def test_check_baseline_health_healthy(mock_eval, tmp_path):
    gate = AccuracyGate(make_spec(tmp_path))
    healthy, diag, base = gate.check_baseline_health()
    assert healthy is True and diag == "" and base == 0.80
    gate.check_baseline_health()
    mock_eval.assert_called_once()


@patch(
    "quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline", side_effect=RuntimeError("worker failed")
)
def test_check_baseline_health_evaluation_failure(mock_eval, tmp_path):
    gate = AccuracyGate(make_spec(tmp_path))
    healthy, diag, base = gate.check_baseline_health()
    assert healthy is False and "worker failed" in diag and base == 0.0
    mock_eval.assert_called_once()


@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline", return_value=0.0)
def test_check_baseline_health_below_floor(mock_eval, tmp_path):
    # Loads/generates but scores ~0 -> broken in the framework (would otherwise
    # read as gap=0, a false pass). Default floor 0.03 catches it.
    gate = AccuracyGate(make_spec(tmp_path))
    healthy, diag, base = gate.check_baseline_health()
    assert healthy is False and "floor" in diag and base == 0.0


@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline", return_value=0.0)
def test_check_baseline_health_floor_disabled(mock_eval, tmp_path):
    spec = replace(make_spec(tmp_path), baseline_floor=0.0)
    gate = AccuracyGate(spec)
    healthy, _diag, base = gate.check_baseline_health()
    assert healthy is True and base == 0.0


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], False),
        (["--trust-remote-code"], True),
        (["--trust_remote_code"], True),
        (["--no-trust-remote-code"], False),
        (["--trust-remote-code --no-trust-remote-code"], False),
        (["--no-trust-remote-code", "--trust-remote-code"], True),
        (["--trust-remote-code", "--no_trust_remote_code"], False),
    ],
)
@pytest.mark.parametrize("max_num_seqs", [None, 64])
def test_runtime_options_reach_all_accuracy_commands(tmp_path, extra_args, expected, max_num_seqs):
    seq_args = [] if max_num_seqs is None else [f"--max-num-seqs {max_num_seqs}"]
    spec = Spec(
        model_dir=str(tmp_path / "model"),
        base_model=str(tmp_path / "model"),
        framework="vllm",
        gpu_type="mi350x",
        gpu_arch="MI350X",
        isl=1024,
        osl=1024,
        quant_strategy=None,
        session_dir=str(tmp_path / "session"),
        vllm_extra_args=["--tensor-parallel-size 8", *extra_args, *seq_args],
        eval_profile=EvalProfile(
            profile_id="gsm8k-base-default-v1",
            profile_hash="test",
            model_mode="base",
            apply_chat_template=False,
            enable_thinking=None,
            detection_reason="test",
        ),
    )
    # Saved sessions retain the user's choice without a new schema field.
    spec = Spec.from_dict(spec.to_dict())
    quant_model = str(tmp_path / "quant_model")

    def verify_only(_repo, _make_prompt, verify, **_kwargs):
        passed, failure = verify()
        assert passed, failure
        return {"success": passed}

    with (
        patch(
            "quark.experimental.torch.quant_perf.evaluation.gsm8k.run_isolated_subprocess",
            return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ) as run,
        patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.read_gsm8k_score", return_value=0.8),
        patch("quark.experimental.torch.quant_perf.repair.llm_repair._run_agent_rounds", side_effect=verify_only),
        patch("quark.experimental.torch.quant_perf.repair.llm_repair._recent_managed_repair_context", return_value=""),
    ):
        gate = AccuracyGate(spec)
        assert gate.eval_quantized(quant_model).passed  # baseline, then quantized
        assert gate.eval_quantized(quant_model).passed  # a fresh quantized attempt
        assert evaluate_candidate_accuracy(spec, quant_model) == ("passed", 0.8, "")
        request = build_repair_request(
            spec,
            failure_class="accuracy_gap",
            error="accuracy regression",
            quant_ckpt_dir=quant_model,
            verifier_profile="accuracy",
            metrics={"source_gsm8k": 0.8, "quantized_gsm8k": 0.4, "gap": 0.5},
        )
        verification = RepairService()._run_verification(
            str(tmp_path / "vllm"), request, timeout_s=60, diagnostic_dir=None
        )
        assert verification.passed
        assert attempt_accuracy_repair(
            source_gsm8k=0.8,
            quantized_gsm8k=0.4,
            gap=0.5,
            framework=spec.framework,
            framework_repo=str(tmp_path / "vllm"),
            quant_ckpt_dir=quant_model,
            gpu_id=spec.gpu_id,
            tp=spec.tp,
            eval_profile=spec.eval_profile,
            trust_remote_code=spec.vllm_trust_remote_code,
            max_num_seqs=spec.vllm_max_num_seqs,
        )

    assert run.call_count == 6
    for call in run.call_args_list:
        command = call.args[0]
        model_args = command[command.index("--model_args") + 1].split(",")
        assert ("trust_remote_code=True" in model_args) is expected
        if not expected:
            assert not any(arg.startswith("trust_remote_code=") for arg in model_args)
        if max_num_seqs is None:
            assert not any(arg.startswith("max_num_seqs=") for arg in model_args)
        else:
            assert f"max_num_seqs={max_num_seqs}" in model_args
        assert "tensor_parallel_size=8" in model_args
        assert (
            "distributed_executor_backend="
            "quark.experimental.torch.quant_perf.evaluation.vllm_executor.PreparedMultiprocExecutor"
        ) in model_args

    saved_command = json.loads((tmp_path / "session/evaluation/baseline/attempt-1/command.json").read_text())
    assert saved_command["argv"] == run.call_args_list[0].args[0]


@pytest.mark.parametrize(
    ("extra_args", "expected_trust"),
    [
        ([], False),
        (["--trust-remote-code"], True),
        (["--trust-remote-code", "--no-trust-remote-code"], False),
    ],
)
@pytest.mark.parametrize("max_num_seqs", [None, 32, 64])
def test_runtime_options_reach_throughput_commands(tmp_path, extra_args, expected_trust, max_num_seqs):
    seq_args = [] if max_num_seqs is None else [f"--max-num-seqs {max_num_seqs}"]
    spec = replace(make_spec(tmp_path), vllm_extra_args=["--tensor-parallel-size 8", *extra_args, *seq_args])
    output = "THROUGHPUT_RESULT=" + json.dumps({"samples_tps": [100.0] * 3, "stable": True})
    with patch(
        "quark.experimental.torch.quant_perf.evaluation.throughput.run_isolated_subprocess",
        return_value=subprocess.CompletedProcess([], 0, stdout=output, stderr=""),
    ) as run:
        throughput_benchmark(
            spec.base_model,
            tp=spec.tp,
            trust_remote_code=spec.vllm_trust_remote_code,
            max_num_seqs=spec.vllm_max_num_seqs,
        )
        measure_quantized_candidate(spec, "quant_model")
        measure_candidate_screen(spec, "quant_model", {})

    assert run.call_count == 3
    for call in run.call_args_list:
        script = ast.parse(call.args[0][2])
        engine = next(
            node
            for node in ast.walk(script)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LLM"
        )
        options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in engine.keywords}
        assert options.get("trust_remote_code", False) is expected_trust
        assert options["tensor_parallel_size"] == 8
        assert options["max_num_seqs"] == (64 if max_num_seqs is None else max_num_seqs)


@pytest.mark.parametrize("max_num_seqs", [None, 64])
def test_runtime_options_reach_both_load_repair_commands(tmp_path, max_num_seqs):
    seq_args = [] if max_num_seqs is None else [f"--max-num-seqs={max_num_seqs}"]
    spec = replace(
        make_spec(tmp_path),
        vllm_extra_args=["--tensor-parallel-size 8", "--trust-remote-code", *seq_args],
    )
    request = build_repair_request(
        spec,
        failure_class="load_run",
        error="model initialization failed",
        quant_ckpt_dir="quant_model",
        verifier_profile="load_inference",
    )
    with patch(
        "quark.experimental.torch.quant_perf.repair.verifiers.run_isolated_subprocess",
        return_value=subprocess.CompletedProcess([], 0, stdout="ok", stderr=""),
    ) as run:
        verification = RepairService()._run_verification(
            str(tmp_path / "vllm"), request, timeout_s=60, diagnostic_dir=None
        )

    assert verification.passed
    assert run.call_count == 2
    for call, eager in zip(run.call_args_list, [True, False], strict=True):
        script = ast.parse(call.args[0][2])
        engine = next(
            node
            for node in ast.walk(script)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LLM"
        )
        options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in engine.keywords}
        assert options["tensor_parallel_size"] == 8
        assert options["enforce_eager"] is eager
        assert options["trust_remote_code"] is True
        assert options.get("max_num_seqs") == max_num_seqs
