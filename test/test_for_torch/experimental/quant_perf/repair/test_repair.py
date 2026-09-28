#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ..testing import init_git_repo
from ..testing import run_git as _git


def _request(**overrides):
    from quark.experimental.torch.quant_perf.repair import RepairRequest

    values = {
        "failure_class": "load_run",
        "error": "RuntimeError: failed",
        "model_dir": "/model",
        "quant_ckpt_dir": "/quant",
        "framework": "vllm",
        "framework_repo": "/repos/vllm",
        "kernel_repo": "/repos/aiter",
        "stack_fingerprint": {},
        "quant_signature": "mlp=mxfp4",
        "workload": {"tp": 2, "gpu_id": 4, "isl": 1024, "osl": 1024, "concurrency": 64},
        "immutable_constraints": {},
        "verifier_profile": "load_inference",
        "session_dir": "",
    }
    values.update(overrides)
    return RepairRequest(**values)


@patch(
    "quark.experimental.torch.quant_perf.repair.service.RepairService._dispatch",
    return_value=(False, None),
)
def test_repair_refuses_unmanaged_source_checkout(mock_dispatch, tmp_path):
    from quark.experimental.torch.quant_perf.repair import RepairService

    source_repo = tmp_path / "vllm"
    source_repo.mkdir()
    untracked = source_repo / "user-notes.txt"
    untracked.write_text("keep me")

    with pytest.raises(RuntimeError, match="managed workspace"):
        RepairService().repair(
            _request(
                framework_repo=str(source_repo),
                kernel_repo="",
            )
        )

    assert untracked.read_text() == "keep me"
    mock_dispatch.assert_not_called()


@pytest.mark.parametrize("generation_failure", [None, "failed", "timeout"])
def test_repair_refreshes_knowledge_from_each_verifier_failure(tmp_path, monkeypatch, generation_failure):
    import hashlib
    import subprocess

    from quark.experimental.torch.quant_perf.repair import RepairService

    repo = init_git_repo(tmp_path / "repo", {"loader.py": "original\n", "scheme.py": "original\n"})
    session = tmp_path / "session"
    prompts = []
    actual_run = subprocess.run

    def run(command, **kwargs):
        if command[0] != "claude":
            return actual_run(command, **kwargs)
        prompts.append(command[-1])
        assert (repo / "loader.py").read_text() == "original\n"
        if len(prompts) == 1:
            (repo / "scheme.py").write_text("block fp8 support\n")
        else:
            assert (repo / "scheme.py").read_text() == "block fp8 support\n"
            (repo / "loader.py").write_text("indexer scale mapping\n")
        if generation_failure and len(prompts) == 2:
            if generation_failure == "timeout":
                raise subprocess.TimeoutExpired(command, timeout=1)
            return SimpleNamespace(returncode=1, stdout="", stderr="generation failed")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"success": True, "changed_files": ["scheme.py"] if len(prompts) == 1 else ["loader.py"]}
            ),
            stderr="",
        )

    failures = iter([(False, "KeyError: 'layers.0.self_attn.indexer.wk_weights_proj.weight_scale'"), (True, "")])
    monkeypatch.setattr("quark.experimental.torch.quant_perf.repair.agent_loop.config.build_subprocess_env", lambda: {})
    monkeypatch.setattr("quark.experimental.torch.quant_perf.repair.agent_loop.subprocess.run", run)
    result = RepairService(verifier=lambda *_args: next(failures), allow_in_place_repair=True).repair(
        _request(
            error=_interleaved_worker_failure(), framework_repo=str(repo), kernel_repo="", session_dir=str(session)
        )
    )

    assert result.status == "fixed"
    assert result.attempts[0]["outcome"] == "new_failure"
    assert "restored" in prompts[1]
    queries = [json.loads(line) for line in (session / "knowledge/query_audit.jsonl").read_text().splitlines()]
    calls = [json.loads(line) for line in (session / "llm_calls.jsonl").read_text().splitlines()]
    prompt_files = sorted(session.glob("repair/RuntimeRepair-*/round-*/agent.prompt.txt"))
    expected_ids = ["repair.vllm.quark.fp8-per-block-runtime.v1", "repair.vllm.quark.fused-indexer-scale-loading.v1"]
    if generation_failure:
        expected_ids.append(expected_ids[-1])
    assert len(queries) == len(calls) == len(prompts) == len(prompt_files) == len(expected_ids)
    for round_num, (query, call, prompt, prompt_file, expected) in enumerate(
        zip(queries, calls, prompts, prompt_files, expected_ids, strict=True), 1
    ):
        assert query["round"] == call["round"] == round_num
        assert query["knowledge_ids"] == call["knowledge_ids"]
        assert expected in query["knowledge_ids"]
        assert expected in prompt
        assert prompt_file.read_text() == prompt
        assert hashlib.sha256(prompt.encode()).hexdigest() == call["prompt_hash"]
        if round_num > 1:
            assert "wk_weights_proj.weight_scale" in query["error_signature"]
            assert "KeyError" in prompt
    assert set(result.knowledge_ids) == {item for query in queries for item in query["knowledge_ids"]}
    assert (repo / "loader.py").read_text() == "indexer scale mapping\n"
    assert _git(repo, "show", "HEAD:scheme.py") == "block fp8 support"


@pytest.mark.parametrize("interrupted_phase", ["generating", "verifying"])
def test_repair_interrupt_is_not_a_verifier_rejection(tmp_path, monkeypatch, interrupted_phase):
    import subprocess

    from quark.experimental.torch.quant_perf.repair.agent_loop import run_agent_rounds

    repo = init_git_repo(tmp_path / "repo", {"loader.py": "original\n"})
    session = tmp_path / "session"
    actual_run = subprocess.run
    phases = []

    def run(command, **kwargs):
        if command[0] != "claude":
            return actual_run(command, **kwargs)
        (repo / "loader.py").write_text("unverified candidate\n")
        if interrupted_phase == "generating":
            raise KeyboardInterrupt
        return SimpleNamespace(returncode=0, stdout='{"success":true,"changed_files":["loader.py"]}', stderr="")

    def verify():
        raise KeyboardInterrupt

    monkeypatch.setattr("quark.experimental.torch.quant_perf.repair.agent_loop.config.build_subprocess_env", lambda: {})
    monkeypatch.setattr("quark.experimental.torch.quant_perf.repair.agent_loop.subprocess.run", run)
    with pytest.raises(KeyboardInterrupt):
        run_agent_rounds(
            str(repo), lambda *_args: "repair", verify, "test", session_dir=str(session), phase_callback=phases.append
        )
    assert _git(repo, "show", "HEAD:loader.py") == "original"
    assert phases[-1]["status"] == "interrupted"
    assert any(event["phase"] == interrupted_phase and event["status"] == "interrupted" for event in phases)


def test_source_router_selects_kernel_repo_from_aiter_traceback():
    from quark.experimental.torch.quant_perf.repair.source_router import resolve_repair_target

    target = resolve_repair_target(_request(error='File "/workspace/aiter/aiter/ops/flydsl/gemm.py", line 7'))

    assert target.role == "kernel"
    assert target.repo == "/repos/aiter"
    assert target.reason == "error_path"


def test_source_router_defaults_to_framework_repo():
    from quark.experimental.torch.quant_perf.repair.source_router import resolve_repair_target

    target = resolve_repair_target(_request(error="ValueError: quant method missing"))

    assert target.role == "framework"
    assert target.repo == "/repos/vllm"


def test_source_router_uses_root_frame_when_trace_contains_aiter_imports():
    from quark.experimental.torch.quant_perf.repair.source_router import resolve_repair_target

    error = """
[aiter] import /repos/aiter/aiter/jit/module_aiter_core.so
Traceback (most recent call last):
  File "/repos/vllm/vllm/model_executor/model_loader/utils.py", line 107, in process_weights_after_loading
    quant_method.process_weights_after_loading(module)
  File "/repos/vllm/vllm/model_executor/layers/quantization/quark/quark_moe.py", line 1077, in process_weights_after_loading
    w13_bias = layer.w13_bias.to(torch.float32)
AttributeError: 'NoneType' object has no attribute 'to'
"""

    target = resolve_repair_target(_request(error=error))

    assert target.role == "framework"
    assert target.repo == "/repos/vllm"
    assert target.reason == "root_file"


def test_source_router_uses_kernel_repo_for_missing_flydsl_backend():
    from quark.experimental.torch.quant_perf.repair.source_router import resolve_repair_target

    target = resolve_repair_target(
        _request(
            error=(
                'File "/repos/vllm/vllm/quark_w4a8.py", line 99\n'
                "RuntimeError: W4A8 FlyDSL backend is unavailable in the "
                "installed AITER build.\n"
                "UserWarning: leaked shared_memory"
            )
        )
    )

    assert target.role == "kernel"
    assert target.repo == "/repos/aiter"
    assert target.reason == "backend_capability"


def test_failure_evidence_extracts_distinct_root_cause_signatures():
    from quark.experimental.torch.quant_perf.repair.evidence import extract_failure_evidence

    memory = extract_failure_evidence(
        """
  File "/repos/vllm/vllm/v1/worker/utils.py", line 413, in request_memory
    raise ValueError(...)
ValueError: Free memory on device cuda:2 is less than desired GPU memory utilization
"""
    )
    missing_api = extract_failure_evidence(
        """
  File "/repos/aiter/aiter/ops/gemm_op_a4w4.py", line 20, in <module>
    from aiter.ops.flydsl import gemm_a4w4
ImportError: cannot import name 'gemm_a4w4' from 'aiter.ops.flydsl'
"""
    )
    biasless = extract_failure_evidence(
        """
  File "/repos/vllm/vllm/model_executor/layers/quantization/quark/quark_moe.py", line 1077, in process_weights_after_loading
    w13_bias = layer.w13_bias.to(torch.float32)
AttributeError: 'NoneType' object has no attribute 'to'
"""
    )

    assert memory.root_file.endswith("vllm/v1/worker/utils.py")
    assert missing_api.root_file.endswith("aiter/ops/gemm_op_a4w4.py")
    assert biasless.root_source == "w13_bias = layer.w13_bias.to(torch.float32)"
    assert len({memory.signature, missing_api.signature, biasless.signature}) == 3


def _interleaved_worker_failure():
    return (
        "(Worker_TP0 pid=10) ERROR 09-14 08:00:00 [worker.py:1] Traceback (most recent call last):\n"
        '(Worker_TP0 pid=10) ERROR 09-14 08:00:00 [worker.py:1]   File "/repos/vllm/vllm/quark.py", line 42, in get_scheme\n'
        "(Worker_TP1 pid=11) ERROR 09-14 08:00:00 [worker.py:1] Traceback (most recent call last):\n"
        '(Worker_TP1 pid=11) ERROR 09-14 08:00:00 [worker.py:1]   File "/repos/aiter/aiter/wait.py", line 3, in wait\n'
        "(Worker_TP0 pid=10) ERROR 09-14 08:00:00 [worker.py:1]     raise NotImplementedError(message)\n"
        "(Worker_TP0 pid=10) ERROR 09-14 08:00:00 [worker.py:1] NotImplementedError: No quark compatible scheme was found.\n"
        "(Worker_TP1 pid=11) ERROR 09-14 08:00:00 [worker.py:1] RuntimeError: See root cause above\n"
        + "shutdown progress\n" * 2000
        + "/usr/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: leaked shared_memory objects\n"
    )


def test_long_worker_failure_reaches_repair_knowledge_without_display_roundtrip(tmp_path):
    from quark.experimental.torch.quant_perf.evaluation.gsm8k import EvaluationFailure
    from quark.experimental.torch.quant_perf.repair.evidence import extract_failure_evidence
    from quark.experimental.torch.quant_perf.repair.service import RepairService
    from quark.experimental.torch.quant_perf.repair.source_router import resolve_repair_target
    from quark.experimental.torch.quant_perf.runtime.recovery import classify_failure

    error = EvaluationFailure(
        "/quant",
        stdout="Optional probe: ImportError: unused backend\n",
        stderr=_interleaved_worker_failure(),
        returncode=1,
    )
    assert "No quark compatible scheme" not in str(error)
    evidence = extract_failure_evidence(error)
    assert evidence.exception_type == "NotImplementedError"
    assert evidence.root_file == "/repos/vllm/vllm/quark.py"
    assert evidence.root_function == "get_scheme"
    assert evidence.root_source == "raise NotImplementedError(message)"
    assert error.stderr in evidence.full_error
    diagnosis = classify_failure(error, framework_repo="/repos/vllm", kernel_repo="/repos/aiter")
    assert diagnosis.target_role == "framework"
    request = _request(error=str(error), evidence=diagnosis.evidence, session_dir=str(tmp_path))
    assert resolve_repair_target(request).role == "framework"
    _, ids = RepairService()._knowledge_for_round(request, round_num=1, evidence=evidence)
    assert "repair.vllm.quark.fp8-per-block-runtime.v1" in ids


@pytest.mark.parametrize(
    ("error", "exception_type", "root_file"),
    [
        (
            'Traceback (most recent call last):\n  File "/work/check.py", line 2, in check\n'
            "UserWarning: warnings are errors here",
            "UserWarning",
            "/work/check.py",
        ),
        ("/work/check.py:2: UserWarning: cleanup warning", "", ""),
        (
            'Traceback (most recent call last):\n  File "/work/check.py", line 2, in check\nWarning: fatal warning',
            "Warning",
            "/work/check.py",
        ),
        ("Exception: plain exception", "Exception", ""),
        (
            'Traceback (most recent call last):\n  File "/work/old.py", line 2, in old\n'
            "ValueError: initial failure\n\nDuring handling of the above exception, another exception occurred:\n\n"
            'Traceback (most recent call last):\n  File "/work/new.py", line 4, in new\n'
            "RuntimeError: See root cause above",
            "ValueError",
            "/work/old.py",
        ),
        ('File "/work/unknown.py", line 2, in unknown\nworker exited with signal 9', "", ""),
    ],
)
def test_failure_evidence_does_not_attach_unrelated_frames(error, exception_type, root_file):
    from quark.experimental.torch.quant_perf.repair.evidence import extract_failure_evidence

    evidence = extract_failure_evidence(error)
    assert evidence.exception_type == exception_type
    assert evidence.root_file == root_file


def test_verifier_preserves_middle_exception_and_timeout_output(tmp_path):
    import subprocess

    from quark.experimental.torch.quant_perf.repair.evidence import extract_failure_evidence
    from quark.experimental.torch.quant_perf.repair.verifiers import verify_load_and_inference

    stderr = _interleaved_worker_failure()
    with patch("quark.experimental.torch.quant_perf.repair.verifiers.run_isolated_subprocess") as run:
        run.side_effect = subprocess.TimeoutExpired("worker", 10, output=b"started", stderr=stderr.encode())
        passed, diagnostic = verify_load_and_inference("/quant", 0, timeout_s=10, diagnostic_dir=tmp_path)
    assert not passed
    assert "timed out" in diagnostic
    assert extract_failure_evidence(diagnostic).exception_type == "NotImplementedError"
    assert (tmp_path / "stderr.log").read_text() == stderr
    assert (tmp_path / "stdout.log").read_text() == "started"


@pytest.mark.parametrize("saved_evidence", [False, True])
def test_repair_round_records_phase_and_candidate_before_rollback(tmp_path, monkeypatch, saved_evidence):
    import subprocess

    from quark.experimental.torch.quant_perf.repair.agent_loop import run_agent_rounds
    from quark.experimental.torch.quant_perf.repair.evidence import extract_failure_evidence, save_failure_evidence

    repo = init_git_repo(tmp_path / "repo", {"loader.py": "original\n"})
    session = tmp_path / "session"
    evidence = extract_failure_evidence(_interleaved_worker_failure())
    if saved_evidence:
        evidence = save_failure_evidence(evidence, session / "verification")
    clock = [100.0]
    monkeypatch.setattr("quark.experimental.torch.quant_perf.repair.agent_loop.config.build_subprocess_env", lambda: {})
    monkeypatch.setattr("quark.experimental.torch.quant_perf.repair.agent_loop.time.monotonic", lambda: clock[0])
    phases = []
    contexts = []
    actual_run = subprocess.run

    def run(command, **kwargs):
        if command[0] != "claude":
            return actual_run(command, **kwargs)
        progress = json.loads((session / "progress.json").read_text())
        assert progress["repair"]["phase"] == "generating"
        assert progress["repair"]["status"] == "running"
        assert progress["repair"]["max_rounds"] == 3
        clock[0] += 2
        (repo / "loader.py").write_text("candidate\n")
        (repo / "helper.py").write_text("new helper\n")
        return SimpleNamespace(returncode=0, stdout='{"success":true,"changed_files":["loader.py"]}', stderr="")

    def verify():
        progress = json.loads((session / "progress.json").read_text())
        assert progress["repair"]["phase"] == "verifying"
        assert progress["repair"]["status"] == "running"
        clock[0] += 3
        return False, evidence

    def prompt(round_num, context, evidence):
        contexts.append(context)
        return "repair"

    with patch("quark.experimental.torch.quant_perf.repair.agent_loop.subprocess.run", side_effect=run):
        result = run_agent_rounds(
            str(repo), prompt, verify, "test", session_dir=str(session), phase_callback=phases.append
        )
    assert not result["success"]
    assert result["rounds"] == 2
    assert (repo / "loader.py").read_text() == "original\n"
    assert not (repo / "helper.py").exists()
    assert "restored" in contexts[1]
    assert "No quark compatible scheme" in contexts[1]
    for attempt in result["attempts"]:
        assert attempt["baseline_sha"] == result["baseline_sha"]
        assert attempt["rolled_back"] is True
        assert "NotImplementedError" in attempt["evidence_signature"]
        assert Path(attempt["patch_path"]).read_text().endswith("+candidate\n")
        assert "+new helper" in Path(attempt["patch_path"]).read_text()
        assert "No quark compatible scheme" in Path(attempt["evidence_paths"][0]).read_text()
        assert attempt["generation_seconds"] == 2
        assert attempt["verification_seconds"] == 3
        if saved_evidence:
            assert attempt["evidence_paths"] == list(evidence.evidence_paths)
    assert any(event["phase"] == "generating" and event["status"] == "running" for event in phases)
    assert any(event["phase"] == "verifying" and event["status"] == "completed" for event in phases)
    assert all(event["elapsed_seconds"] >= 0 for event in phases)
    progress = json.loads((session / "progress.json").read_text())["repair"]
    assert progress["phase"] == "finished"
    assert progress["status"] == "not_fixed"
    assert progress["total_elapsed_seconds"] == 10
    assert len(list(session.rglob("failure.log"))) == (1 if saved_evidence else 2)


@pytest.mark.parametrize("limit", [0, 80, 5000])
def test_failure_display_stays_bounded_with_long_evidence_paths(limit):
    from dataclasses import replace

    from quark.experimental.torch.quant_perf.repair.evidence import extract_failure_evidence, render_failure_evidence

    evidence = replace(extract_failure_evidence("ValueError: root cause"), evidence_paths=("/long/path" * 1000,))
    display = render_failure_evidence(evidence, limit=limit)
    assert len(display) <= limit
    assert not limit or display.startswith("ValueError: root cause")


def test_service_verification_preserves_structured_evidence(tmp_path):
    from quark.experimental.torch.quant_perf.repair.service import RepairService

    service = RepairService(verifier=lambda *args: (False, _interleaved_worker_failure()))
    result = service._verification_result("/candidate", _request(session_dir=str(tmp_path)), timeout_s=10)
    assert result.evidence.exception_type == "NotImplementedError"
    assert result.evidence_paths
    assert "shutdown progress" in Path(result.evidence_paths[0]).read_text()
    assert len(result.failure) <= 5000
    assert result.failure.startswith("NotImplementedError")


def test_load_prompt_keeps_root_before_display_limit(tmp_path):
    from quark.experimental.torch.quant_perf.repair.llm_repair import attempt_load_repair

    with patch("quark.experimental.torch.quant_perf.repair.llm_repair._run_agent_rounds") as run:
        run.return_value = {"success": False}
        with patch("quark.experimental.torch.quant_perf.repair.llm_repair.get_current_branch", return_value="main"):
            attempt_load_repair(_interleaved_worker_failure(), "vllm", str(tmp_path), str(tmp_path))
        prompt = run.call_args.args[1](1, "")
    assert "NotImplementedError: No quark compatible scheme was found." in prompt


def test_signature_preserves_shape_and_block_size():
    from quark.experimental.torch.quant_perf.repair.evidence import extract_failure_evidence

    assert (
        extract_failure_evidence("ValueError: block_size=[128,128] shape=[16,32]").signature
        != extract_failure_evidence("ValueError: block_size=[32,32] shape=[16,32]").signature
    )


def test_repair_journey_can_be_retrieved_as_exact_or_similar_knowledge(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore
    from quark.experimental.torch.quant_perf.knowledge.terminal_experience import TerminalExperienceRecorder
    from quark.experimental.torch.quant_perf.repair import RepairService
    from quark.experimental.torch.quant_perf.repair.evidence import extract_failure_evidence
    from quark.experimental.torch.quant_perf.repair.types import RepairResult, VerificationResult

    error = (
        'File "/repos/vllm/linear.py", line 42, in load_weights\n'
        "    scale.reshape([])\nRuntimeError: shape '[]' is invalid for input of size 32768"
    )
    request = _request(
        error=error,
        session_dir=str(tmp_path),
        stack_fingerprint={
            "framework_version": "abc",
            "arch_fingerprint": "arch",
        },
    )
    ckpt = SimpleNamespace(state={"session_id": "session-1"}, save=lambda: None)
    with ExperienceStore(tmp_path / "experience.sqlite") as store:
        service = RepairService(store, workspace_manager=SimpleNamespace(ckpt=ckpt))
        service._record_journey(
            request=request,
            target_role="framework",
            target_repo="/repo",
            candidate_id="candidate",
            promoted=True,
            changed_files=["linear.py"],
            elapsed_seconds=1,
            result=RepairResult(
                "fixed",
                attempts=[{"tried": "preserve block scales", "failed_because": "wrong scheme"}],
                verifier_results=[VerificationResult("load_inference", True)],
            ),
        )
        spec = SimpleNamespace(framework="vllm", framework_version="abc", arch_fingerprint="arch")
        TerminalExperienceRecorder(store).capture(spec=spec, state=ckpt.state)
        text, ids = service._knowledge_for_round(request, round_num=1, evidence=request.evidence)
        assert any(item.startswith("experience.repair.") for item in ids)
        assert "preserve block scales" in text
        similar = extract_failure_evidence(error.replace("32768", "65536"))
        text, ids = service._knowledge_for_round(request, round_num=2, evidence=similar)
        assert any(item.startswith("experience.repair.") for item in ids)
        assert "Similar failure" in text
        assert "32768" in text and "65536" in text
        assert "Avoid repeating preserve block scales" not in text
        unrelated = extract_failure_evidence(error.replace("load_weights", "another_function"))
        _, ids = service._knowledge_for_round(request, round_num=3, evidence=unrelated)
        assert not any(item.startswith("experience.repair.") for item in ids)


def _write_quark_fused_moe_checkpoint(
    root: Path,
    *,
    expert_input_dtype: str = "fp4",
) -> Path:
    checkpoint = root / "quant"
    checkpoint.mkdir()
    global_config = {
        "weight": {"dtype": "fp4", "qscheme": "per_group"},
        "input_tensors": {
            "dtype": "fp8_e4m3",
            "is_dynamic": False,
            "qscheme": "per_tensor",
        },
    }
    expert_config = (
        json.loads(json.dumps(global_config))
        if expert_input_dtype == "fp8_e4m3"
        else {
            "weight": {"dtype": "fp4", "qscheme": "per_group"},
            "input_tensors": {
                "dtype": expert_input_dtype,
                "is_dynamic": True,
                "qscheme": "per_group",
            },
        }
    )
    expert_prefix = "model.language_model.layers.*.mlp.experts.*"
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "quantization_config": {
            "quant_method": "quark",
            "global_quant_config": global_config,
            "layer_quant_config": {
                f"{expert_prefix}.{projection}": expert_config for projection in ("gate_proj", "up_proj", "down_proj")
            },
        },
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    return checkpoint


@patch("quark.experimental.torch.quant_perf.repair.service.llm_repair.attempt_load_repair", return_value=True)
def test_repair_service_dispatches_load_failure_with_session_context(mock_fix, tmp_path):
    from quark.experimental.torch.quant_perf.repair import RepairService

    result = RepairService(allow_in_place_repair=True).repair(_request(session_dir=str(tmp_path)))

    assert result.status == "fixed"
    assert result.target_repos == ["/repos/vllm"]
    assert mock_fix.call_args.kwargs["session_dir"] == str(tmp_path)
    assert mock_fix.call_args.kwargs["tp"] == 2


@patch(
    "quark.experimental.torch.quant_perf.repair.service.load_inference_timeout_s",
    return_value=2400,
)
@patch("quark.experimental.torch.quant_perf.repair.service.llm_repair.attempt_load_repair")
def test_repair_service_uses_checkpoint_aware_load_timeout(
    mock_fix,
    mock_timeout,
):
    from quark.experimental.torch.quant_perf.repair import RepairService

    verifier = MagicMock(return_value=(True, ""))

    def repair(**kwargs):
        assert kwargs["verify"]() == (True, "")
        return False

    mock_fix.side_effect = repair

    RepairService(
        verifier=verifier,
        allow_in_place_repair=True,
    ).repair(_request())

    mock_timeout.assert_called_once_with("/quant")
    assert verifier.call_args.args[2] == 2400


@patch("quark.experimental.torch.quant_perf.repair.service.llm_repair.attempt_load_repair", return_value=True)
def test_repair_service_dispatches_aiter_load_failure_with_kernel_role(
    mock_fix,
):
    from quark.experimental.torch.quant_perf.repair import RepairService

    result = RepairService(allow_in_place_repair=True).repair(
        _request(error=("W4A8 FlyDSL backend is unavailable in the installed AITER build."))
    )

    assert result.status == "fixed"
    assert result.target_repos == ["/repos/aiter"]
    assert mock_fix.call_args.kwargs["framework_repo"] == "/repos/aiter"
    assert mock_fix.call_args.kwargs["role"] == "kernel"


@patch("quark.experimental.torch.quant_perf.repair.service.llm_repair.attempt_load_repair")
def test_repair_service_preserves_llm_attempt_outcomes(mock_fix):
    def repair(**kwargs):
        kwargs["attempts_out"].extend(
            [
                {
                    "round": 1,
                    "kind": "llm_repair",
                    "outcome": "no_progress",
                },
                {
                    "round": 2,
                    "kind": "llm_repair",
                    "outcome": "new_failure",
                },
            ]
        )
        return False

    mock_fix.side_effect = repair

    result = (
        __import__(
            "quark.experimental.torch.quant_perf.repair",
            fromlist=["RepairService"],
        )
        .RepairService(allow_in_place_repair=True)
        .repair(_request())
    )

    assert result.status == "not_fixed"
    assert [attempt["outcome"] for attempt in result.attempts] == [
        "no_progress",
        "new_failure",
    ]


@patch("quark.experimental.torch.quant_perf.repair.service.llm_repair.attempt_load_repair")
def test_repair_service_injects_verifier_and_knowledge_into_llm_fallback(
    mock_fix,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.repair import RepairService

    verifier_calls = []
    dispatched_knowledge_ids = []

    def verifier(candidate_repo, request, timeout_s):
        verifier_calls.append((candidate_repo, request, timeout_s))
        return True, ""

    def repair(**kwargs):
        assert kwargs["verify"]() == (True, "")
        text, ids = kwargs["knowledge_for_round"](1, kwargs["error"])
        assert "repair.quark.packed-modules-mapping.v1" in text
        dispatched_knowledge_ids.extend(ids)
        return True

    mock_fix.side_effect = repair
    request = _request(
        error=("RuntimeError: tensor size mismatch because packed_modules_mapping lacks gate_up_proj"),
        kernel_repo="",
        session_dir=str(tmp_path),
    )

    result = RepairService(
        verifier=verifier,
        allow_in_place_repair=True,
    ).repair(request)

    assert result.status == "fixed"
    assert "repair.quark.packed-modules-mapping.v1" in result.knowledge_ids
    assert result.knowledge_ids == dispatched_knowledge_ids
    assert verifier_calls
    assert verifier_calls[0][0] == "/repos/vllm"


@patch(
    "quark.experimental.torch.quant_perf.repair.service.verify_load_and_inference",
    return_value=(True, ""),
)
def test_load_verifier_preserves_requested_gpu_memory_utilization(
    mock_verify,
):
    from quark.experimental.torch.quant_perf.repair import RepairService

    request = _request(
        immutable_constraints={
            "gpu_memory_utilization": 0.60,
            "max_model_len": 8192,
            "original_enforce_eager": True,
        }
    )

    result = RepairService()._verification_result(
        "/candidate/vllm",
        request,
        timeout_s=900,
    )

    assert result.passed is True
    assert mock_verify.call_count == 1
    assert mock_verify.call_args.kwargs["gpu_memory_utilization"] == 0.60
    assert mock_verify.call_args.kwargs["max_model_len"] == 8192


def test_quant_config_summary_prioritizes_quantization_metadata(tmp_path):
    from quark.experimental.torch.quant_perf.repair.verifiers import (
        read_quant_config,
    )

    quant_ckpt = tmp_path / "quant"
    quant_ckpt.mkdir()
    (quant_ckpt / "config.json").write_text(
        json.dumps(
            {
                "large_model_metadata": "x" * 5000,
                "architectures": ["GlmMoeDsaForCausalLM"],
                "quantization_config": {
                    "quant_method": "quark",
                    "global_quant_config": {
                        "weight": {"dtype": "fp8_e4m3", "qscheme": "per_tensor"},
                        "input_tensors": {
                            "dtype": "fp8_e4m3",
                            "qscheme": "per_tensor",
                            "is_dynamic": False,
                        },
                    },
                    "layer_quant_config": {
                        "model.layers.*.self_attn.indexer.wk": {
                            "weight": {
                                "dtype": "fp8_e4m3",
                                "qscheme": "per_tensor",
                            },
                            "input_tensors": {
                                "dtype": "fp8_e4m3",
                                "qscheme": "per_tensor",
                                "is_dynamic": False,
                            },
                        }
                    },
                },
            }
        )
    )

    summary = read_quant_config(str(quant_ckpt))

    assert len(summary) <= 3000
    assert "quantization_config" in summary
    assert "model.layers.*.self_attn.indexer.wk" in summary
    assert "fp8_e4m3" in summary
    assert "large_model_metadata" not in summary


@patch("quark.experimental.torch.quant_perf.repair.verifiers.run_isolated_subprocess")
def test_load_verifier_preserves_stdout_root_cause_and_stderr_tail(
    mock_run,
):
    from quark.experimental.torch.quant_perf.repair.verifiers import (
        verify_load_and_inference,
    )

    mock_run.return_value = SimpleNamespace(
        returncode=1,
        stdout=(
            "worker startup\n"
            "KeyError: layers.0.self_attn.indexer.wk_weights_proj.input_scale\n" + ("worker shutdown\n" * 1000)
        ),
        stderr="RuntimeError: Engine core initialization failed",
    )

    passed, failure = verify_load_and_inference(
        "/quant",
        gpu_id=0,
        tp=8,
        kv_cache_dtype="fp8",
        trust_remote_code=True,
        max_num_seqs=8,
    )

    assert "kv_cache_dtype='fp8'" in mock_run.call_args.args[0][2]
    assert "trust_remote_code=True" in mock_run.call_args.args[0][2]
    assert "max_num_seqs=8" in mock_run.call_args.args[0][2]
    assert passed is False
    assert "wk_weights_proj.input_scale" in failure
    assert "Engine core initialization failed" in failure


def test_repair_request_does_not_infer_eager_from_aiter_requirement(tmp_path):
    from quark.experimental.torch.quant_perf.repair.request import (
        build_repair_request,
    )

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "indexer_types": ["full"],
            }
        )
    )
    spec = SimpleNamespace(
        model_dir=str(tmp_path),
        framework="vllm",
        session_dir=str(tmp_path / "session"),
        active_framework_repo="/repos/vllm",
        can_modify_framework=True,
        active_kernel_repo="/repos/aiter",
        can_modify_kernel=True,
        arch_fingerprint="glm",
        framework_version="vllm",
        kernel_version="aiter",
        framework_source_kind="installed_overlay",
        kernel_source_kind="installed_overlay",
        gpu_id=0,
        tp=8,
        isl=1024,
        osl=1024,
        bench_concurrency=64,
        accuracy_gap=0.05,
        target_gain=1.0,
        eval_profile=None,
        vllm_gpu_memory_utilization=0.6,
        vllm_trust_remote_code=False,
        vllm_max_num_seqs=None,
        vllm_kv_cache_dtype="fp8",
        gsm8k_num_samples=1319,
        runtime_python="/usr/bin/python",
        runtime_env={},
        expanded_vllm_args=["--tensor-parallel-size=8"],
    )

    request = build_repair_request(
        spec,
        failure_class="load_run",
        error="KeyError",
        quant_ckpt_dir="/quant",
        verifier_profile="load_inference",
    )

    assert request.immutable_constraints["original_enforce_eager"] is False
    assert request.immutable_constraints["gsm8k_num_samples"] == 1319
    assert request.immutable_constraints["accuracy_repair_num_samples"] == 50
    assert request.immutable_constraints["kv_cache_dtype"] == "fp8"


@pytest.mark.parametrize("trust_options", [{}, {"trust_remote_code": False}, {"trust_remote_code": True}])
@patch("quark.experimental.torch.quant_perf.repair.service.llm_repair.attempt_accuracy_repair", return_value=True)
def test_repair_service_dispatches_accuracy_gap_to_resolved_repo(mock_fix, trust_options):
    from quark.experimental.torch.quant_perf.repair import RepairService

    result = RepairService(allow_in_place_repair=True).repair(
        _request(
            failure_class="accuracy_gap",
            error="quantized model accuracy gap 1.0",
            verifier_profile="accuracy",
            immutable_constraints=trust_options,
            metrics={
                "source_gsm8k": 0.8,
                "quantized_gsm8k": 0.0,
                "gap": 1.0,
            },
        )
    )

    assert result.status == "fixed"
    assert mock_fix.call_args.kwargs["framework_repo"] == "/repos/vllm"
    assert mock_fix.call_args.kwargs["trust_remote_code"] is trust_options.get("trust_remote_code", False)


@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline", return_value=0.79)
def test_accuracy_verifier_uses_candidate_runtime_and_full_threshold(
    mock_eval,
):
    from quark.experimental.torch.quant_perf.repair import RepairService

    external_verifier = MagicMock(return_value=(False, "stale integration"))
    request = _request(
        failure_class="accuracy_gap",
        verifier_profile="accuracy",
        metrics={
            "source_gsm8k": 0.8,
            "quantized_gsm8k": 0.0,
            "gap": 1.0,
        },
        immutable_constraints={
            "accuracy_gap": 0.03,
            "eval_profile": None,
            "gpu_memory_utilization": 0.8,
            "gsm8k_num_samples": 1319,
            "accuracy_repair_num_samples": 50,
            "runtime_python": "/session/bin/python",
            "runtime_env": {
                "PYTHONPATH": "/integration/vllm:/integration/aiter",
            },
        },
        verifier=external_verifier,
    )

    passed, failure = RepairService()._verify(
        "/candidate/vllm",
        request,
        timeout_s=900,
    )

    assert passed is True
    assert failure == ""
    assert mock_eval.call_args.kwargs["num_questions"] == 50
    assert mock_eval.call_args.kwargs["runtime_python"] == ("/session/bin/python")
    assert mock_eval.call_args.kwargs["trust_remote_code"] is False
    assert mock_eval.call_args.kwargs["runtime_env"]["PYTHONPATH"].split(":")[0] == "/candidate/vllm"
    external_verifier.assert_not_called()


@patch(
    "quark.experimental.torch.quant_perf.repair.service.llm_repair.attempt_benchmark_repair",
    return_value=True,
)
def test_repair_service_routes_aiter_benchmark_failure_to_kernel_repo(mock_fix):
    from quark.experimental.torch.quant_perf.repair import RepairService

    request = _request(
        failure_class="benchmark_execution",
        error="/repos/aiter/aiter/ops/flydsl/gemm.py failed",
        verifier_profile="benchmark_execution",
        verifier=lambda: (True, ""),
    )
    result = RepairService(allow_in_place_repair=True).repair(request)

    assert result.status == "fixed"
    assert result.target_repos == ["/repos/aiter"]
    assert mock_fix.call_args.kwargs["framework_repo"] == "/repos/aiter"
    assert mock_fix.call_args.kwargs["role"] == "kernel"


@patch("quark.experimental.torch.quant_perf.repair.service.llm_repair.attempt_load_repair")
def test_repair_service_uses_candidate_worktree_and_promotes(mock_fix, tmp_path):
    from quark.experimental.torch.quant_perf.repair import RepairService
    from quark.experimental.torch.quant_perf.workspace.manager import RepoWorkspaceManager

    repo = init_git_repo(tmp_path / "vllm", {"loader.py": "VALUE = 1\n"})
    ckpt = SimpleNamespace(
        state={"repo_workspaces": {}, "transient_resources": [], "cleanup": {}},
        save=lambda: None,
    )
    manager = RepoWorkspaceManager(
        session_dir=tmp_path / "session",
        session_id="session-12345678",
        ckpt=ckpt,
        root_dir=tmp_path / "owned",
    )
    integration = manager.prepare("framework", repo)

    def fix(**kwargs):
        candidate = Path(kwargs["framework_repo"])
        (candidate / "loader.py").write_text("VALUE = 2\n")
        _git(candidate, "add", "loader.py")
        _git(candidate, "commit", "-m", "repair")
        kwargs["attempts_out"].append(
            {"kind": "llm_repair", "round": 1, "generation_seconds": 2.0, "verification_seconds": 3.0}
        )
        return True

    mock_fix.side_effect = fix
    request = _request(
        framework_repo=str(integration.integration_path),
        kernel_repo="",
        session_dir=str(tmp_path / "session"),
    )
    result = RepairService(
        workspace_manager=manager,
    ).repair(request)

    assert result.status == "fixed"
    assert (integration.integration_path / "loader.py").read_text() == "VALUE = 2\n"
    assert (repo / "loader.py").read_text() == "VALUE = 1\n"
    assert "candidates" in mock_fix.call_args.kwargs["framework_repo"]
    assert ckpt.state["repair_journey"][0]["status"] == "fixed"
    assert ckpt.state["repair_journey"][0]["target_role"] == "framework"
    assert ckpt.state["repair_journey"][0]["rounds"] == 1
    assert ckpt.state["repair_journey"][0]["elapsed_seconds"] >= 0
    progress = json.loads((tmp_path / "session/progress.json").read_text())["repair"]
    assert progress["phase"] == "finished"
    assert progress["status"] == "fixed"
    assert ckpt.state["change_ledger"][0]["intent"] == "repair"


@patch(
    "quark.experimental.torch.quant_perf.repair.service.verify_load_and_inference",
    return_value=(True, ""),
)
def test_load_verifier_prepends_candidate_runtime_pythonpath(
    mock_verify,
):
    from quark.experimental.torch.quant_perf.repair import RepairService

    request = _request(
        immutable_constraints={
            "runtime_python": "/session/bin/python",
            "runtime_env": {
                "PYTHONPATH": "/session/framework:/session/kernel",
            },
        }
    )

    RepairService()._verify(
        "/candidate/kernel",
        request,
        timeout_s=120,
    )

    assert mock_verify.call_args.kwargs["runtime_python"] == ("/session/bin/python")
    assert mock_verify.call_args.kwargs["runtime_env"]["PYTHONPATH"].split(":") == [
        "/candidate/kernel",
        "/session/framework",
        "/session/kernel",
    ]


@patch(
    "quark.experimental.torch.quant_perf.repair.service.verify_load_and_inference",
    side_effect=[(True, ""), (True, "")],
)
def test_load_verifier_preserves_memory_budget_in_both_runtime_modes(
    mock_verify,
):
    from quark.experimental.torch.quant_perf.repair import RepairService

    request = _request(
        immutable_constraints={
            "runtime_python": "/session/bin/python",
            "runtime_env": {},
            "gpu_memory_utilization": 0.85,
            "original_enforce_eager": False,
        }
    )

    passed, failure = RepairService()._verify(
        "/candidate/vllm",
        request,
        timeout_s=120,
    )

    assert passed is True
    assert failure == ""
    assert [call.kwargs["enforce_eager"] for call in mock_verify.call_args_list] == [True, False]
    assert [call.kwargs["gpu_memory_utilization"] for call in mock_verify.call_args_list] == [0.85, 0.85]


@patch(
    "quark.experimental.torch.quant_perf.repair.service.verify_load_and_inference",
    side_effect=[(True, ""), (True, "")],
)
def test_load_verifier_does_not_raise_low_requested_memory_for_eager_probe(
    mock_verify,
):
    from quark.experimental.torch.quant_perf.repair import RepairService

    passed, failure = RepairService()._verify(
        "/candidate/vllm",
        _request(
            immutable_constraints={
                "runtime_env": {},
                "gpu_memory_utilization": 0.2,
                "original_enforce_eager": False,
            }
        ),
        timeout_s=120,
    )

    assert passed is True
    assert failure == ""
    assert [call.kwargs["gpu_memory_utilization"] for call in mock_verify.call_args_list] == [0.2, 0.2]


@patch(
    "quark.experimental.torch.quant_perf.repair.service.verify_load_and_inference",
    side_effect=[(True, ""), (False, "compiled runtime failed")],
)
def test_load_verifier_rejects_eager_only_repair(mock_verify):
    from quark.experimental.torch.quant_perf.repair import RepairService

    passed, failure = RepairService()._verify(
        "/candidate/vllm",
        _request(
            immutable_constraints={
                "runtime_env": {},
                "gpu_memory_utilization": 0.85,
                "original_enforce_eager": False,
            }
        ),
        timeout_s=120,
    )

    assert passed is False
    assert "original runtime" in failure
    assert mock_verify.call_count == 2


@patch("quark.experimental.torch.quant_perf.repair.service.llm_repair.attempt_load_repair")
@patch(
    "quark.experimental.torch.quant_perf.repair.service.verify_load_and_inference",
    side_effect=[(True, ""), (True, "")],
)
def test_load_repair_records_each_runtime_verification(
    mock_verify,
    mock_repair,
):
    from quark.experimental.torch.quant_perf.repair import RepairService

    def repair(**kwargs):
        passed, _failure = kwargs["verify"]()
        return passed

    mock_repair.side_effect = repair
    result = RepairService(allow_in_place_repair=True).repair(
        _request(
            immutable_constraints={
                "runtime_env": {},
                "gpu_memory_utilization": 0.85,
                "original_enforce_eager": False,
            }
        )
    )

    assert result.status == "fixed"
    assert result.verifier_results[0].metrics["checks"] == [
        {"mode": "eager", "passed": True, "failure": ""},
        {"mode": "original", "passed": True, "failure": ""},
    ]


@patch(
    "quark.experimental.torch.quant_perf.repair.service.llm_repair.attempt_load_repair",
    return_value=True,
)
def test_repair_without_candidate_source_change_is_not_fixed(
    mock_fix,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.repair import RepairService

    repo = tmp_path / "vllm"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "loader.py").write_text("VALUE = 1\n")
    _git(repo, "add", "loader.py")
    _git(repo, "commit", "-m", "base")
    manager, integration, ckpt = _workspace_manager(tmp_path, repo)

    result = RepairService(
        workspace_manager=manager,
    ).repair(
        _request(
            framework_repo=str(integration.integration_path),
            kernel_repo="",
        )
    )

    assert result.status == "not_fixed"
    assert result.attempts[-1]["outcome"] == "no_source_change"
    assert not ckpt.state.get("change_ledger")
    mock_fix.assert_called_once()


@patch("quark.experimental.torch.quant_perf.repair.service.llm_repair.attempt_load_repair")
def test_installed_overlay_rejects_compiled_repair(mock_fix, tmp_path):
    from quark.experimental.torch.quant_perf.repair import RepairService
    from quark.experimental.torch.quant_perf.workspace.manager import RepoWorkspaceManager

    repo = tmp_path / "vllm-overlay"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "loader.py").write_text("VALUE = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    ckpt = SimpleNamespace(
        state={"repo_workspaces": {}, "transient_resources": [], "cleanup": {}},
        save=lambda: None,
    )
    manager = RepoWorkspaceManager(
        session_dir=tmp_path / "session",
        session_id="session-12345678",
        ckpt=ckpt,
        root_dir=tmp_path / "owned",
    )
    integration = manager.prepare("framework", repo)

    def fix(**kwargs):
        candidate = Path(kwargs["framework_repo"])
        (candidate / "kernel.cpp").write_text("int value = 2;\n")
        _git(candidate, "add", "kernel.cpp")
        _git(candidate, "commit", "-m", "compiled repair")
        return True

    mock_fix.side_effect = fix
    request = _request(
        framework_repo=str(integration.integration_path),
        kernel_repo="",
        stack_fingerprint={"framework_source_kind": "installed_overlay"},
    )

    result = RepairService(
        workspace_manager=manager,
    ).repair(request)

    assert result.status == "not_fixed"
    assert not (integration.integration_path / "kernel.cpp").exists()
    assert result.attempts[-1]["outcome"] == "source_repo_required"


def _workspace_manager(tmp_path: Path, repo: Path):
    from quark.experimental.torch.quant_perf.workspace.manager import RepoWorkspaceManager

    ckpt = SimpleNamespace(
        state={"repo_workspaces": {}, "transient_resources": [], "cleanup": {}},
        save=lambda: None,
    )
    manager = RepoWorkspaceManager(
        session_dir=tmp_path / "session",
        session_id="session-12345678",
        ckpt=ckpt,
        root_dir=tmp_path / "owned",
    )
    integration = manager.prepare("framework", repo)
    return manager, integration, ckpt


@pytest.mark.parametrize("failure", ["", "commit", "verification_mutation"])
def test_repair_promotes_exact_verified_source_despite_incomplete_agent_summary(tmp_path, monkeypatch, failure):
    import subprocess

    from quark.experimental.torch.quant_perf.repair import RepairService

    repo = init_git_repo(tmp_path / "repo", {"loader.py": "original\n", "old.py": "rename me\n"})
    manager, integration, ckpt = _workspace_manager(tmp_path, repo)
    actual_run = subprocess.run
    ckpt.record_phase_event = lambda *_args, **_kwargs: None

    def run(command, **kwargs):
        if command[0] != "claude":
            if failure == "commit" and command[:2] == ["git", "commit"] and "--allow-empty" not in command:
                return subprocess.CompletedProcess(command, 1, "", "commit rejected")
            return actual_run(command, **kwargs)
        candidate = Path(kwargs["cwd"])
        (candidate / "loader.py").write_text("from helper import value\n")
        (candidate / "helper.py").write_text("value = 1\n")
        (candidate / "old.py").rename(candidate / "renamed, helper.py")
        cache = candidate / "__pycache__"
        cache.mkdir()
        (cache / "loader.pyc").write_bytes(b"generated")
        return SimpleNamespace(returncode=0, stdout='{"success":true,"changed_files":["loader.py"]}', stderr="")

    def verify(repo_path, *_args):
        candidate = Path(repo_path)
        assert (candidate / "helper.py").read_text() == "value = 1\n"
        if failure == "verification_mutation":
            # Once tracked, even a cache file must be part of the verified diff.
            _git(candidate, "add", "__pycache__/loader.pyc")
            _git(candidate, "commit", "-m", "Unexpected verification side effect")
        return True, ""

    monkeypatch.setattr("quark.experimental.torch.quant_perf.repair.agent_loop.config.build_subprocess_env", lambda: {})
    monkeypatch.setattr("quark.experimental.torch.quant_perf.repair.agent_loop.subprocess.run", run)
    result = RepairService(workspace_manager=manager, verifier=verify).repair(
        _request(
            framework_repo=str(integration.integration_path), kernel_repo="", session_dir=str(tmp_path / "session")
        )
    )
    assert result.status == ("not_fixed" if failure else "fixed")
    if failure:
        assert (integration.integration_path / "loader.py").read_text() == "original\n"
        assert not any(attempt["outcome"] == "verified" for attempt in result.attempts)
        return
    expected = ["helper.py", "loader.py", "old.py", "renamed, helper.py"]
    assert result.artifacts[0]["changed_files"] == expected
    assert result.attempts[0]["changed_files"] == expected
    assert ckpt.state["repair_journey"][0]["changed_files"] == expected
    assert _git(integration.integration_path, "show", "HEAD:helper.py") == "value = 1"
    assert _git(integration.integration_path, "show", "HEAD:renamed, helper.py") == "rename me"
    assert not (integration.integration_path / "old.py").exists()
    assert "__pycache__" not in Path(result.artifacts[0]["patch_path"]).read_text()


def _biasless_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "vllm"
    source = repo / "vllm" / "model_executor" / "layers" / "quantization" / "quark" / "quark_moe.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def process_weights_after_loading(layer):\n"
        "    w13_bias = layer.w13_bias.to(torch.float32)\n"
        "    w2_bias = layer.w2_bias.to(torch.float32)\n"
        "    layer.w13_bias = torch.nn.Parameter(w13_bias, requires_grad=False)\n"
        "    layer.w2_bias = torch.nn.Parameter(w2_bias, requires_grad=False)\n"
    )
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    return repo


@patch("quark.experimental.torch.quant_perf.repair.service.llm_repair.attempt_load_repair")
def test_verified_recipe_guides_current_source_repair_without_patch_replay(
    mock_repair,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.repair import RepairService

    repo = _biasless_repo(tmp_path)
    manager, integration, ckpt = _workspace_manager(tmp_path, repo)

    def repair(**kwargs):
        text, _ = kwargs["knowledge_for_round"](1, kwargs["error"])
        assert "repair.vllm.quark.optional-moe-bias.v1" in text
        candidate = Path(kwargs["framework_repo"])
        source = candidate / "vllm/model_executor/layers/quantization/quark/quark_moe.py"
        source.write_text(
            source.read_text().replace(
                "    w13_bias = layer.w13_bias.to(torch.float32)\n",
                "    if layer.w13_bias is not None:\n        w13_bias = layer.w13_bias.to(torch.float32)\n",
            )
        )
        _git(candidate, "add", ".")
        _git(candidate, "commit", "-m", "repair optional bias")
        assert kwargs["verify"]() == (True, "")
        return True

    mock_repair.side_effect = repair

    def verifier(repo_path, _request, _timeout_s):
        source = Path(repo_path) / "vllm/model_executor/layers/quantization/quark/quark_moe.py"
        return "if layer.w13_bias is not None:" in source.read_text(), ""

    result = RepairService(
        workspace_manager=manager,
        verifier=verifier,
    ).repair(
        _request(
            framework_repo=str(integration.integration_path),
            kernel_repo="",
            error=(
                'File "/repos/vllm/vllm/model_executor/layers/quantization/'
                'quark/quark_moe.py", line 1077, '
                "in process_weights_after_loading\n"
                "    w13_bias = layer.w13_bias.to(torch.float32)\n"
                "AttributeError: 'NoneType' object has no attribute 'to'"
            ),
        )
    )

    assert result.status == "fixed"
    assert [artifact["kind"] for artifact in result.artifacts] == ["repair_patch"]
    assert "repair.vllm.quark.optional-moe-bias.v1" in (result.knowledge_ids)
    assert mock_repair.call_count == 1
    assert ckpt.state["repair_journey"][0]["promoted"] is True
    assert ckpt.state["repair_journey"][0]["changed_files"] == [
        "vllm/model_executor/layers/quantization/quark/quark_moe.py"
    ]
