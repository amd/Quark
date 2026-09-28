#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for LLM-backed repair and knowledge integration: signature
normalization (error + quant) that keys the fix-experience store. Pure
functions — no GPU, no subprocess, no git.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from quark.experimental.torch.quant_perf.repair import agent_loop
from quark.experimental.torch.quant_perf.repair.agent_loop import CandidateSnapshot
from quark.experimental.torch.quant_perf.repair.evidence import extract_failure_evidence
from quark.experimental.torch.quant_perf.repair.llm_repair import (
    _LARGE_GAP_FRAMEWORK_THRESHOLD,
    _recent_managed_repair_context,
    _run_agent_rounds,
    _verify_load_and_inference,
    attempt_accuracy_repair,
    attempt_benchmark_repair,
    classify_accuracy_failure,
)
from quark.experimental.torch.quant_perf.repair.signatures import (
    normalize_quant_signature,
)


@pytest.fixture
def agent_runtime(monkeypatch):
    runtime = SimpleNamespace(
        run=MagicMock(
            return_value=SimpleNamespace(
                returncode=0, stdout='{"changed_files":["loader.py"],"summary":"candidate","success":true}', stderr=""
            )
        ),
        dirty=MagicMock(return_value=False),
        commit=MagicMock(return_value=True),
        reset=MagicMock(),
        snapshot=CandidateSnapshot("", []),
    )
    monkeypatch.setattr(agent_loop.subprocess, "run", runtime.run)
    monkeypatch.setattr(agent_loop.config, "build_subprocess_env", lambda: {})
    monkeypatch.setattr(agent_loop, "commit_baseline", lambda *_args: "base-sha")
    empty = CandidateSnapshot("", [])
    monkeypatch.setattr(
        agent_loop, "_capture_candidate", lambda _repo, revision: empty if revision == "HEAD" else runtime.snapshot
    )
    monkeypatch.setattr(
        agent_loop,
        "_read_candidate",
        lambda _repo, revision: empty if revision == "HEAD" and not runtime.dirty() else runtime.snapshot,
    )
    monkeypatch.setattr(agent_loop, "commit_selected_changes", runtime.commit)
    monkeypatch.setattr(agent_loop, "reset_hard_to", runtime.reset)
    return runtime


# -- classify_accuracy_failure: large gap short-circuits to framework ---------


def test_large_gap_classified_framework_without_llm(tmp_path):
    # gap >= threshold must return 'framework' immediately (no claude subprocess),
    # so obviously-broken quantized models always get a repair attempt. If this tried
    # to shell out to `claude` it would hang/fail in the test env -- the fast
    # return is what we're asserting.
    got = classify_accuracy_failure(
        source_gsm8k=0.86,
        quantized_gsm8k=0.0,
        gap=1.0,
        quant_ckpt_dir=str(tmp_path),
        quant_strategy="fp8",
    )
    assert got == "framework"
    assert _LARGE_GAP_FRAMEWORK_THRESHOLD <= 1.0


@patch("quark.experimental.torch.quant_perf.repair.llm_repair.direct_api_call")
def test_accuracy_classifier_uses_decision_model(
    mock_call,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("QUARK_QUANT_PERF_LLM_DECISION_MODEL", "test-decision-model")
    mock_call.return_value = '{"root_cause":"quantization","confidence":0.9}\n'

    got = classify_accuracy_failure(
        source_gsm8k=0.8,
        quantized_gsm8k=0.72,
        gap=0.1,
        quant_ckpt_dir=str(tmp_path),
        quant_strategy="fp8",
    )

    assert got == "quantization"
    assert mock_call.call_args.kwargs["model"] == "test-decision-model"


@patch("quark.experimental.torch.quant_perf.repair.llm_repair.subprocess.run")
def test_recent_managed_repair_context_returns_latest_managed_patch(mock_run):
    mock_run.side_effect = [
        MagicMock(
            returncode=0,
            stdout=(
                "newer-sha\x00unrelated user commit\n"
                "repair-sha\x00Quark Quant-Perf load_run repair\n"
                "base-sha\x00Quark Quant-Perf installed package baseline\n"
            ),
        ),
        MagicMock(
            returncode=0,
            stdout=(
                "commit repair-sha\n"
                "diff --git a/vllm/model_executor/models/deepseek_v2.py "
                "b/vllm/model_executor/models/deepseek_v2.py\n"
            ),
        ),
    ]

    context = _recent_managed_repair_context("/repo")

    assert "repair-sha" in context
    assert "deepseek_v2.py" in context
    assert "--all" in mock_run.call_args_list[0].args[0]
    assert mock_run.call_args_list[1].args[0][-1] == "repair-sha"


# -- canonical failure signature --------------------------------------------


def test_error_signature_empty():
    assert extract_failure_evidence("").signature == ""
    assert extract_failure_evidence("   \n  ").signature == ""


def test_error_signature_picks_exception_line_not_traceback_frames():
    err = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/x/foo.py", line 42, in bar\n'
        "    do_it()\n"
        "ValueError: quant_method=quark not found\n"
    )
    sig = extract_failure_evidence(err).signature
    assert "quant_method=quark not found" in sig
    assert "Traceback" not in sig


def test_error_signature_stable_across_volatile_bits():
    # Same bug, different run: paths, line numbers, hex addrs, PIDs differ.
    a = extract_failure_evidence(
        'File "/home/u/a/vllm/worker.py", line 283, in init\nRuntimeError: CUDA error at 0x7f3a1c00 pid 33127'
    ).signature
    b = extract_failure_evidence(
        'File "/opt/other/vllm/worker.py", line 991, in init\nRuntimeError: CUDA error at 0x91ab pid 88'
    ).signature
    assert a == b
    assert a  # non-empty


def test_error_signature_falls_back_to_last_line():
    sig = extract_failure_evidence("just a plain message with no exception marker").signature
    assert sig == "v2|UnknownFailure|||just a plain message with no exception marker"


def test_error_signature_truncates():
    long = "ValueError: " + "x" * 1000
    assert len(extract_failure_evidence(long).signature.split("|", 4)[-1]) == 300


# -- normalize_quant_signature -----------------------------------------------


def test_quant_signature_from_dict_is_key_order_independent():
    a = normalize_quant_signature(layer_config={"mlp_mode": "fp8", "self_attn_mode": "fp8"})
    b = normalize_quant_signature(layer_config={"self_attn_mode": "fp8", "mlp_mode": "fp8"})
    assert a == b
    assert a == "mlp_mode=fp8;self_attn_mode=fp8"


def test_quant_signature_from_str_repr_of_dict():
    # mix_precision_search records str(candidate); the signature must match a dict.
    d = {"linear_attn_mode": "fp8", "mlp_mode": "fp8"}
    assert normalize_quant_signature(layer_config=str(d)) == normalize_quant_signature(layer_config=d)


def test_quant_signature_strategy_fallback():
    assert normalize_quant_signature(quant_strategy="FP8") == "strategy:fp8"
    assert normalize_quant_signature(quant_strategy="  mxfp4 ") == "strategy:mxfp4"


def test_quant_signature_auto_when_empty():
    assert normalize_quant_signature() == "auto"
    assert normalize_quant_signature(quant_strategy="", layer_config=None) == "auto"


def test_quant_signature_layer_config_takes_precedence_over_strategy():
    sig = normalize_quant_signature(quant_strategy="fp8", layer_config={"mlp_mode": "fp8"})
    assert sig == "mlp_mode=fp8"


def test_quant_signature_unparseable_str_is_kept_raw():
    assert normalize_quant_signature(layer_config="not-a-dict").startswith("raw:")


@patch("quark.experimental.torch.quant_perf.repair.verifiers.run_isolated_subprocess")
@pytest.mark.parametrize(
    "tp,overrides,timeout",
    [
        (1, {}, 900),
        (2, {}, 900),
        (
            2,
            {
                "timeout_s": 777,
                "runtime_python": "/session/venv/bin/python",
                "runtime_env": {"PYTHONPATH": "/session/overlay"},
                "gpu_memory_utilization": 0.72,
            },
            777,
        ),
    ],
    ids=["single-default", "tp-default", "tp-explicit-runtime"],
)
def test_verify_load_and_inference_preserves_runtime_settings(mock_run, tp, overrides, timeout):
    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

    ok, _ = _verify_load_and_inference("/models/q", gpu_id=3, tp=tp, **overrides)

    assert ok
    cmd = mock_run.call_args.args[0]
    assert cmd[:2] == [overrides.get("runtime_python", "python3"), "-c"]
    script = cmd[2]
    assert f"tensor_parallel_size={tp}" in script
    assert f"gpu_memory_utilization={overrides.get('gpu_memory_utilization', 0.85)}" in script
    executor = "PreparedMultiprocExecutor" if tp > 1 else "PreparedUniProcExecutor"
    assert (
        f"distributed_executor_backend='quark.experimental.torch.quant_perf.evaluation.vllm_executor.{executor}'"
        in script
    )
    assert mock_run.call_args.kwargs["preparation_timeout"] == mock_run.call_args.kwargs["timeout"] == timeout
    assert mock_run.call_args.kwargs["preparation_status_path"].name == "preparation.json"
    env = mock_run.call_args.kwargs["env"]
    assert env["ROCR_VISIBLE_DEVICES"] == ",".join(str(i) for i in range(3, 3 + tp))
    if "runtime_env" in overrides:
        assert env["PYTHONPATH"] == "/session/overlay"


@patch("quark.experimental.torch.quant_perf.repair.verifiers.run_isolated_subprocess")
def test_verify_load_and_inference_returns_timeout_failure(mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(
        cmd=["python3"],
        timeout=900,
        output="partial stdout",
        stderr="partial stderr",
    )

    ok, failure = _verify_load_and_inference("/models/q", gpu_id=0)

    assert ok is False
    assert "timed out after 900s" in failure
    assert "partial stderr" in failure


@pytest.mark.parametrize("knowledge_mode", ["static", "refreshed", "empty"])
def test_repair_agent_round_uses_configured_codegen_model(
    agent_runtime,
    monkeypatch,
    tmp_path,
    knowledge_mode,
):
    from quark.experimental.torch.quant_perf.repair.evidence import extract_failure_evidence

    monkeypatch.setenv("QUARK_QUANT_PERF_LLM_CODEGEN_MODEL", "test-codegen-model")
    agent_runtime.run.return_value = MagicMock(
        returncode=0,
        stdout='{"changed_files":[],"summary":"ok","success":true}\n',
        stderr="",
    )

    result = _run_agent_rounds(
        "/repo",
        lambda _round, _context, _evidence: "fix it",
        lambda: (True, ""),
        "test commit",
        session_dir=str(tmp_path),
        initial_evidence=extract_failure_evidence("RuntimeError: failed"),
        knowledge_text="static guidance",
        knowledge_ids=["static"],
        knowledge_for_round=(
            None
            if knowledge_mode == "static"
            else lambda _round, _evidence: (
                ("refreshed guidance", ["refreshed"]) if knowledge_mode == "refreshed" else ("", [])
            )
        ),
    )

    assert result["success"]
    assert len(result["attempts"]) == 1
    attempt = result["attempts"][0]
    assert attempt["round"] == 1
    assert attempt["changed_files"] == []
    assert attempt["tried"] == "ok"
    assert attempt["failed_because"] == ""
    assert attempt["outcome"] == "verified"
    assert attempt["baseline_sha"] == "base-sha"
    assert attempt["rolled_back"] is False
    cmd = next(call.args[0] for call in agent_runtime.run.call_args_list if call.args[0][0] == "claude")
    assert cmd[cmd.index("--model") + 1] == "test-codegen-model"
    audit = tmp_path / "llm_calls.jsonl"
    assert '"call_type": "repair"' in audit.read_text()
    expected_ids = [] if knowledge_mode == "empty" else [knowledge_mode]
    assert json.loads(audit.read_text())["knowledge_ids"] == expected_ids
    assert attempt["knowledge_ids"] == expected_ids
    assert cmd[-1] == ("fix it" if knowledge_mode == "empty" else f"fix it\n\n{knowledge_mode} guidance")


def test_repair_agent_timeout_carries_partial_progress_into_next_round(
    tmp_path,
    monkeypatch,
    agent_runtime,
):
    timeout = subprocess.TimeoutExpired(
        cmd=["claude"],
        timeout=900,
        output="inspected qwen3_5.py and found the wrapper mapping\n",
        stderr="focused test not run yet\n",
    )
    completed = MagicMock(
        returncode=0,
        stdout=(
            '{"changed_files":["vllm/model_executor/models/qwen3_5.py"],'
            '"summary":"map fused experts on the top-level wrapper",'
            '"success":true}\n'
        ),
        stderr="",
    )
    contexts = []

    agent_results = iter([timeout, completed])

    def run(command, **kwargs):
        if command[0] == "git":
            return MagicMock(returncode=0, stdout="candidate diff")
        result = next(agent_results)
        if isinstance(result, BaseException):
            raise result
        return result

    def make_prompt(_round, context, _evidence):
        contexts.append(context)
        return "fix it"

    agent_runtime.run.side_effect = run
    agent_runtime.snapshot = CandidateSnapshot("candidate diff", ["vllm/model_executor/models/qwen3_5.py"])
    result = _run_agent_rounds(
        "/repo",
        make_prompt,
        lambda: (True, ""),
        "test commit",
        timeout_s=900,
        agent_round_start_budget_s=2700,
        session_dir=str(tmp_path),
    )

    assert result["success"] is True
    assert len(contexts) == 2
    assert "inspected qwen3_5.py" in contexts[1]
    assert "focused test not run yet" in contexts[1]
    assert "vllm/model_executor/models/qwen3_5.py" in contexts[1]
    assert result["attempts"][0]["changed_files"] == [
        "vllm/model_executor/models/qwen3_5.py",
    ]


@pytest.mark.parametrize(
    "verifications,outcomes",
    [
        (
            [(False, "RuntimeError: original failure")] * 2,
            ["no_progress", "no_progress"],
        ),
        (
            [(False, "RuntimeError: original failure"), (False, "ValueError: new failure"), (True, "")],
            ["no_progress", "new_failure", "verified"],
        ),
        (
            [
                (False, "RuntimeError: original failure"),
                (False, "ValueError: new failure"),
                (False, "RuntimeError: original failure"),
            ],
            ["no_progress", "new_failure", "no_progress"],
        ),
    ],
    ids=["repeated-failure", "new-failure-then-success", "return-to-original-failure"],
)
def test_repair_agent_rounds_follow_verifier_progress(agent_runtime, verifications, outcomes):
    agent_runtime.dirty.return_value = len(verifications) == 3
    agent_runtime.snapshot = CandidateSnapshot("candidate diff", ["loader.py"])
    verifier_results = iter(verifications)
    result = _run_agent_rounds("/repo", lambda *_args: "fix it", lambda: next(verifier_results), "test commit")

    assert result["success"] is verifications[-1][0]
    assert result["rounds"] == len(verifications)
    assert sum(call.args[0][0] == "claude" for call in agent_runtime.run.call_args_list) == len(verifications)
    assert [attempt["outcome"] for attempt in result["attempts"]] == outcomes
    assert agent_runtime.commit.call_count == int(result["success"])


@pytest.mark.parametrize("trust_options", [{}, {"trust_remote_code": False}, {"trust_remote_code": True}])
@patch(
    "quark.experimental.torch.quant_perf.repair.llm_repair._recent_managed_repair_context",
    return_value="commit repair-sha\ndiff --git a/loader.py b/loader.py",
)
@patch("quark.experimental.torch.quant_perf.repair.llm_repair._run_agent_rounds")
@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline", return_value=0.78)
def test_accuracy_fix_verification_preserves_tp_and_remote_code(
    mock_eval,
    mock_rounds,
    _mock_recent_repair,
    trust_options,
):
    def _run(_repo, make_prompt, verify, **_kwargs):
        prompt = make_prompt(1, "")
        assert "Recent managed repair context" in prompt
        assert "diff --git a/loader.py b/loader.py" in prompt
        ok, _ = verify()
        return {
            "success": ok,
            "rounds": 1,
            "summary": "ok",
            "attempts": [],
            "baseline_sha": "base",
        }

    mock_rounds.side_effect = _run

    assert attempt_accuracy_repair(
        source_gsm8k=0.8,
        quantized_gsm8k=0.4,
        gap=0.5,
        framework="vllm",
        framework_repo="/repo",
        quant_ckpt_dir="/model",
        gpu_id=2,
        tp=2,
        **trust_options,
    )
    assert mock_eval.call_args.kwargs["tp"] == 2
    assert mock_eval.call_args.kwargs["trust_remote_code"] is trust_options.get("trust_remote_code", False)


@patch("quark.experimental.torch.quant_perf.repair.llm_repair._run_agent_rounds")
def test_performance_fix_uses_supplied_verifier_and_immutable_workload(
    mock_rounds,
):
    verifier_calls = []
    error = "EARLY_FAILURE\n" + ("x" * 5000) + "\nLATEST_ROOT_CAUSE"

    def verify():
        verifier_calls.append(True)
        return True, ""

    def run(_repo, make_prompt, supplied_verify, **kwargs):
        prompt = make_prompt(1, "")
        assert "TP=4" in prompt
        assert "ISL=1024" in prompt
        assert "OSL=1024" in prompt
        assert "concurrency=64" in prompt
        assert "must not change" in prompt.lower()
        assert "repair.vllm.systematic-debugging.v1" in kwargs["knowledge_text"]
        assert "LATEST_ROOT_CAUSE" in prompt
        assert "EARLY_FAILURE" not in prompt
        ok, _ = supplied_verify()
        return {
            "success": ok,
            "rounds": 1,
            "summary": "fixed",
            "attempts": [],
            "baseline_sha": "base",
        }

    mock_rounds.side_effect = run

    assert attempt_benchmark_repair(
        error=error,
        framework="vllm",
        framework_repo="/repo",
        quant_ckpt_dir="/quant",
        tp=4,
        isl=1024,
        osl=1024,
        concurrency=64,
        verify=verify,
        knowledge_text=(
            "## Retrieved knowledge\n- [repair.vllm.systematic-debugging.v1] trace the first invalid state"
        ),
        knowledge_ids=["repair.vllm.systematic-debugging.v1"],
    )
    assert verifier_calls == [True]


@patch("quark.experimental.torch.quant_perf.repair.llm_repair.get_current_branch", return_value="branch")
@patch("quark.experimental.torch.quant_perf.repair.llm_repair._run_agent_rounds")
def test_load_repair_prompt_and_audit_use_injected_knowledge(
    mock_rounds,
    _mock_branch,
    tmp_path,
):
    def run(_repo, make_prompt, _verify, **kwargs):
        assert "Current failure" in make_prompt(1, "")
        assert "repair.quark.packed-modules-mapping.v1" in kwargs["knowledge_text"]
        assert "verifiers remain authoritative" in kwargs["knowledge_text"]
        assert kwargs["timeout_s"] == 1800
        assert kwargs["agent_round_start_budget_s"] == 5400
        assert kwargs["knowledge_ids"] == ["repair.quark.packed-modules-mapping.v1"]
        return {
            "success": False,
            "rounds": 1,
            "summary": "not fixed",
            "attempts": [],
            "baseline_sha": "base",
        }

    mock_rounds.side_effect = run

    assert not __import__(
        "quark.experimental.torch.quant_perf.repair.llm_repair",
        fromlist=["attempt_load_repair"],
    ).attempt_load_repair(
        error="RuntimeError: tensor size mismatch on gate_up_proj",
        framework="vllm",
        framework_repo="/repo",
        quant_ckpt_dir=str(tmp_path),
        session_dir=str(tmp_path),
        knowledge_text=(
            "## Retrieved knowledge\n"
            "- [repair.quark.packed-modules-mapping.v1] repair mapping\n"
            "  verifiers remain authoritative"
        ),
        knowledge_ids=["repair.quark.packed-modules-mapping.v1"],
    )


@patch("quark.experimental.torch.quant_perf.repair.llm_repair.get_current_branch", return_value="branch")
@patch("quark.experimental.torch.quant_perf.repair.llm_repair._run_agent_rounds")
def test_load_repair_prompt_keeps_current_error_authoritative_over_prior_cache(
    mock_rounds,
    _mock_branch,
    tmp_path,
):
    def run(_repo, make_prompt, _verify, **kwargs):
        prompt = make_prompt(1, kwargs["initial_context"]) + "\n\n" + kwargs["knowledge_text"]
        assert "AttributeError: 'NoneType' object has no attribute 'to'" in prompt
        assert "w13_bias = layer.w13_bias.to(torch.float32)" in prompt
        assert "timed out after 300 seconds" in prompt
        assert prompt.index("Current failure") < prompt.index("Prior repair context")
        return {
            "success": False,
            "rounds": 1,
            "summary": "not fixed",
            "attempts": [],
            "baseline_sha": "base",
        }

    mock_rounds.side_effect = run

    assert not __import__(
        "quark.experimental.torch.quant_perf.repair.llm_repair",
        fromlist=["attempt_load_repair"],
    ).attempt_load_repair(
        error=("w13_bias = layer.w13_bias.to(torch.float32)\nAttributeError: 'NoneType' object has no attribute 'to'"),
        framework="vllm",
        framework_repo="/repo",
        quant_ckpt_dir=str(tmp_path),
        knowledge_text=("Prior repair context:\nTP=8 reproduction timed out after 300 seconds"),
    )
