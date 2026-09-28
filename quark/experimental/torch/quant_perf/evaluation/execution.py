#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Shared subprocess and cache preparation for model evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

from .preparation import DEADLINE_ENV, STATUS_ENV, read_preparation_status

_PROCESS_GROUP_TERMINATION_GRACE_S = 10
_MAX_MODEL_STARTUP_TIMEOUT_S = 3600
_ESTIMATED_CHECKPOINT_READ_BYTES_PER_SECOND = 256 * 1024**2
_MODEL_STARTUP_BUFFER_S = 300


class EvaluationTimeout(subprocess.TimeoutExpired):
    """Subprocess timeout identifying the exhausted evaluation phase."""

    def __init__(self, command: list[str], timeout: float, phase: str, elapsed: float, reason: str = "", **kwargs: str):
        self.phase = phase
        self.elapsed = elapsed
        self.reason = reason
        super().__init__(command, timeout, **kwargs)

    def __str__(self) -> str:
        detail = f": {self.reason}" if self.reason else ""
        return f"{self.phase} timed out: elapsed={self.elapsed:.1f}s, budget={self.timeout}s{detail}"


class PreparationFailure(RuntimeError):
    """Preparation failed before evaluation, with subprocess output preserved."""

    phase = "preparation"

    def __init__(self, reason: str, *, timed_out: bool = False):
        super().__init__(reason)
        self.timed_out = timed_out
        self.elapsed = 0.0
        self.stdout = ""
        self.stderr = ""


def subprocess_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def model_startup_timeout_s(model_dir: str, *, minimum_s: int) -> int:
    """Scale model startup time for large sharded checkpoints."""
    checkpoint = Path(model_dir)
    try:
        total_bytes = sum(path.stat().st_size for path in checkpoint.glob("*.safetensors") if path.is_file())
    except OSError:
        return minimum_s
    if total_bytes <= 0:
        return minimum_s
    estimated_seconds = math.ceil(total_bytes / _ESTIMATED_CHECKPOINT_READ_BYTES_PER_SECOND) + _MODEL_STARTUP_BUFFER_S
    return min(
        _MAX_MODEL_STARTUP_TIMEOUT_S,
        max(minimum_s, estimated_seconds),
    )


def remove_stale_aiter_jit_locks(env: dict[str, str]) -> list[str]:
    """Remove session-local AITER baton files left by a terminated process group."""
    jit_dir = env.get("AITER_JIT_DIR")
    if not jit_dir:
        return []

    build_dir = Path(jit_dir) / "build"
    candidates = {
        *build_dir.glob("lock_*"),
        *build_dir.glob("*/build/lock"),
    }
    removed: list[str] = []
    for path in sorted(candidates):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            continue
        removed.append(str(path))
    return removed


def run_isolated_subprocess(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    capture_output: bool = True,
    preparation_timeout: float | None = None,
    preparation_status_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run in a process group, optionally budgeting preparation before evaluation."""
    started = time.monotonic()
    phase = "preparation" if preparation_timeout is not None else "execution"
    budget = preparation_timeout if preparation_timeout is not None else timeout
    deadline = started + budget
    if preparation_timeout is not None:
        if not math.isfinite(preparation_timeout) or preparation_timeout <= 0 or preparation_status_path is None:
            raise ValueError("preparation requires a finite positive timeout and a preparation_status_path")
        if preparation_status_path.exists():
            raise ValueError(f"preparation notification already exists: {preparation_status_path}")
        env = {
            **(os.environ if env is None else env),
            DEADLINE_ENV: str(deadline),
            STATUS_ENV: str(preparation_status_path),
            "VLLM_ENGINE_READY_TIMEOUT_S": str(math.ceil(preparation_timeout)),
        }
    remove_stale_aiter_jit_locks(env or {})
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
        start_new_session=True,
    )

    def observe_preparation() -> None:
        nonlocal phase, budget, started, deadline
        if phase != "preparation" or preparation_status_path is None:
            return
        try:
            notification = read_preparation_status(preparation_status_path)
        except (OSError, ValueError) as error:
            raise PreparationFailure(str(error)) from error
        if notification is None:
            return
        if notification["status"] == "failed":
            raise PreparationFailure(
                str(notification.get("reason") or "preparation failed"),
                timed_out=notification.get("failure_kind") == "timeout",
            )
        completed = float(notification["completed_at"])
        if not started <= completed <= time.monotonic():
            raise PreparationFailure("invalid preparation completion time")
        if completed > deadline:
            raise PreparationFailure("preparation completed after its deadline", timed_out=True)
        phase, budget, started = "evaluation", timeout, completed
        deadline = completed + timeout

    try:
        if preparation_timeout is None:
            stdout, stderr = process.communicate(timeout=timeout)
        else:
            while True:
                observe_preparation()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, budget)
                try:
                    stdout, stderr = process.communicate(
                        timeout=min(5.0, remaining) if phase == "preparation" else remaining,
                    )
                    observe_preparation()
                    break
                except subprocess.TimeoutExpired:
                    # Recheck the atomic notification before declaring timeout.
                    continue
    except BaseException as error:
        elapsed = time.monotonic() - started
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=_PROCESS_GROUP_TERMINATION_GRACE_S)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        if isinstance(error, PreparationFailure):
            error.elapsed = elapsed
            error.stdout, error.stderr = stdout or "", stderr or ""
            if not error.timed_out:
                raise
        elif not isinstance(error, subprocess.TimeoutExpired):
            raise
        raise EvaluationTimeout(
            command,
            budget,
            phase,
            elapsed,
            reason=str(error) if isinstance(error, PreparationFailure) else "",
            output=stdout or "",
            stderr=stderr or "",
        ) from error
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout or "",
        stderr or "",
    )


def configure_vllm_cache_env(
    env: dict[str, str],
    model_dir: str,
) -> str:
    """Isolate vLLM compile artifacts by effective checkpoint config."""
    config_path = Path(model_dir) / "config.json"
    try:
        config = json.loads(config_path.read_text())
        payload = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (OSError, ValueError, TypeError):
        payload = str(Path(model_dir).resolve())
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:16]
    cache_base = Path(
        os.environ.get(
            "QUARK_QUANT_PERF_VLLM_CACHE_BASE",
            "/tmp/quark_quant_perf_vllm_cache",
        )
    )
    cache_root = str(cache_base / fingerprint)
    env["VLLM_CACHE_ROOT"] = cache_root
    return cache_root
