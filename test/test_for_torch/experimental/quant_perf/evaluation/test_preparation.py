#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Preparation deadlines and handover without GPU or wall-clock waits."""

import json
import subprocess
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from quark.experimental.torch.quant_perf.evaluation import execution


@pytest.mark.parametrize("non_block", [False, True])
def test_preparation_hands_over_after_real_decode(tmp_path, monkeypatch, non_block):
    from quark.experimental.torch.quant_perf.evaluation import preparation

    class Executor:
        def __init__(self):
            self.calls = []

        def collective_rpc(self, method, timeout=None, **kwargs):
            self.calls.append((method, timeout, kwargs))
            if kwargs.get("non_block"):
                return Future()
            return [True]

    class PreparedExecutor(preparation.PreparationExecutorMixin, Executor):
        pass

    monkeypatch.setenv(preparation.DEADLINE_ENV, "1000")
    preparation_status_path = tmp_path / "ready.json"
    monkeypatch.setenv(preparation.STATUS_ENV, str(preparation_status_path))
    prefill = SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(num_output_tokens=[0]))
    decode = SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(num_output_tokens=[1]))
    with patch.object(preparation.time, "monotonic", return_value=10) as clock:
        executor = PreparedExecutor()
        result = executor.collective_rpc("execute_model", timeout=300, args=(prefill,), non_block=non_block)
        assert executor.calls[-1][1] == 990
        if non_block:
            result.set_result([True])
        assert not preparation_status_path.exists()
        clock.return_value = 400  # Cold work has already exceeded 300 seconds.
        executor.collective_rpc("sample_tokens", timeout=300)
        assert executor.calls[-1][1] == 600
        result = executor.collective_rpc("execute_model", timeout=300, args=(decode,), non_block=non_block)
        assert executor.calls[-1][2]["args"][0] is decode
        if non_block:
            assert not preparation_status_path.exists()
            result.set_result([True])
            assert result.result() == [True]
        else:
            assert result == [True]
        assert json.loads(preparation_status_path.read_text())["completed_at"] == 400
        assert json.loads(preparation_status_path.read_text())["status"] == "ready"
        executor.collective_rpc("sample_tokens", timeout=300)
        assert executor.calls[-1][1] == 300
        assert [call[0] for call in executor.calls] == [
            "execute_model",
            "sample_tokens",
            "execute_model",
            "sample_tokens",
        ]  # No extra generation, RNG, or cache RPCs.

        expired = PreparedExecutor()
        clock.return_value = 1001
        with pytest.raises(TimeoutError, match="preparation"):
            expired.collective_rpc("execute_model", timeout=300, args=(decode,))
        assert expired.calls == []


@pytest.mark.parametrize("ready", [False, True])
def test_subprocess_preparation_budget_and_cleanup(tmp_path, monkeypatch, ready):
    preparation_status_path = tmp_path / "ready.json"
    process = MagicMock(pid=1234, returncode=-15)
    now = [0.0]

    def communicate(timeout=None):
        if now[0] >= 1000:
            return "complete stdout", "complete stderr"
        now[0] += timeout
        if ready and now[0] >= 400 and not preparation_status_path.exists():
            preparation_status_path.write_text(json.dumps({"status": "ready", "completed_at": now[0]}))
        raise subprocess.TimeoutExpired(["eval"], timeout)

    process.communicate.side_effect = communicate
    with (
        patch.object(execution.subprocess, "Popen", return_value=process) as popen,
        patch("time.monotonic", side_effect=lambda: now[0]),
        patch.object(execution.os, "killpg") as killpg,
        pytest.raises(subprocess.TimeoutExpired) as exc,
    ):
        execution.run_isolated_subprocess(
            ["eval"],
            timeout=700,
            preparation_timeout=1000,
            preparation_status_path=preparation_status_path,
            env={"EXISTING": "kept"},
        )
    assert exc.value.phase == ("evaluation" if ready else "preparation")
    assert exc.value.timeout == (700 if ready else 1000)
    assert now[0] == (1100 if ready else 1000)
    assert exc.value.stdout == "complete stdout"
    assert exc.value.stderr == "complete stderr"
    killpg.assert_called_once()
    assert popen.call_args.kwargs["env"]["EXISTING"] == "kept"


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_decode_publishes_failure_and_cannot_be_overwritten(tmp_path, monkeypatch, cancelled):
    from quark.experimental.torch.quant_perf.evaluation import preparation

    monkeypatch.setenv(preparation.STATUS_ENV, str(tmp_path / "ready.json"))
    monkeypatch.setenv(preparation.DEADLINE_ENV, "1000")
    future = Future()

    class Executor:
        def collective_rpc(self, *args, **kwargs):
            return future

    class PreparedExecutor(preparation.PreparationExecutorMixin, Executor):
        pass

    decode = SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(num_output_tokens=[1]))
    with patch.object(preparation.time, "monotonic", return_value=400):
        executor = PreparedExecutor()
        assert executor.collective_rpc("execute_model", args=(decode,), non_block=True) is future
        if cancelled:
            future.cancel()
        else:
            future.set_exception(RuntimeError("worker failed"))
        notification = json.loads((tmp_path / "ready.json").read_text())
        assert notification["status"] == "failed"
        assert notification["failure_kind"] == ("cancelled" if cancelled else "rpc_error")
        later = Future()
        later.set_result([True])
        executor._decode_completed(later)
        assert json.loads((tmp_path / "ready.json").read_text()) == notification
        assert executor._preparation_deadline == 1000


@pytest.mark.parametrize("non_block", [False, True])
def test_late_decode_reports_timeout_without_callback_exception(tmp_path, monkeypatch, caplog, non_block):
    from quark.experimental.torch.quant_perf.evaluation import preparation

    monkeypatch.setenv(preparation.DEADLINE_ENV, "1000")
    monkeypatch.setenv(preparation.STATUS_ENV, str(tmp_path / "status.json"))
    clock = [400]
    monkeypatch.setattr(preparation.time, "monotonic", lambda: clock[0])

    class Executor:
        def collective_rpc(self, *args, **kwargs):
            clock[0] = 1001
            return Future() if non_block else [True]

    class PreparedExecutor(preparation.PreparationExecutorMixin, Executor):
        pass

    executor = PreparedExecutor()
    decode = SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(num_output_tokens=[1]))
    result = executor.collective_rpc("execute_model", args=(decode,))
    if non_block:
        result.set_result([True])
        assert result.result() == [True]
    notification = json.loads((tmp_path / "status.json").read_text())
    assert notification["status"] == "failed"
    assert notification["failure_kind"] == "timeout"
    assert executor._preparation_deadline == 1000
    assert "exception calling callback" not in caplog.text


@pytest.mark.parametrize("failure_kind,exits", [("timeout", False), ("rpc_error", False), ("rpc_error", True)])
def test_parent_surfaces_preparation_failure_even_when_child_exits(tmp_path, failure_kind, exits):
    status_path = tmp_path / "status.json"
    process = MagicMock(pid=1234, returncode=0 if exits else -15)

    def communicate(timeout=None):
        if status_path.exists():
            return "stdout evidence", "stderr evidence"
        status_path.write_text(
            json.dumps(
                {"status": "failed", "completed_at": 20, "failure_kind": failure_kind, "reason": "first decode failed"}
            )
        )
        if exits:
            return "stdout evidence", "stderr evidence"
        raise subprocess.TimeoutExpired(["eval"], timeout)

    process.communicate.side_effect = communicate
    with (
        patch.object(execution.subprocess, "Popen", return_value=process),
        patch.object(execution.time, "monotonic", return_value=20),
        patch.object(execution.os, "killpg") as killpg,
        pytest.raises(Exception) as error,
    ):
        execution.run_isolated_subprocess(
            ["eval"],
            timeout=700,
            preparation_timeout=1000,
            preparation_status_path=status_path,
        )
    assert error.value.phase == "preparation"
    assert isinstance(error.value, subprocess.TimeoutExpired) == (failure_kind == "timeout")
    assert error.value.stdout == "stdout evidence"
    assert error.value.stderr == "stderr evidence"
    assert "first decode failed" in str(error.value)
    killpg.assert_called_once()
