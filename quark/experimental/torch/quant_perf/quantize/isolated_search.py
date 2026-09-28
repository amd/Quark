#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Run mixed-precision search/export in a process-owned GPU lifetime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.evaluation.execution import run_isolated_subprocess
from quark.experimental.torch.quant_perf.quantize.search import run_module_search
from quark.experimental.torch.quant_perf.runtime.backends import configure_search_runtime_env
from quark.experimental.torch.quant_perf.session.persistence import write_json_atomic
from quark.experimental.torch.quant_perf.session.spec import Checkpoint, Spec, StageError
from quark.experimental.torch.quant_perf.workspace.sources import activate_runtime

_RESULT_FILENAME = "module_search_worker_result.json"


def result_path(session_dir: str | Path) -> Path:
    return Path(session_dir) / "runtime" / _RESULT_FILENAME


def _reload_checkpoint(destination: Checkpoint) -> None:
    reloaded = Checkpoint.load(destination.session_dir)
    if reloaded is None:
        raise StageError(
            "quantize",
            "mixed-precision worker did not preserve session state",
            code="search_worker_missing_checkpoint",
        )
    destination.state = reloaded.state


def _worker_command(spec: Spec) -> list[str]:
    return [
        spec.runtime_python or sys.executable,
        "-m",
        "quark.experimental.torch.quant_perf.quantize.isolated_search",
        "--session",
        str(Path(spec.session_dir).resolve()),
    ]


def _run_worker(spec: Spec, output: Path) -> subprocess.CompletedProcess[str]:
    output.unlink(missing_ok=True)
    env = {**os.environ, **dict(spec.runtime_env)}
    configure_search_runtime_env(env, spec.effective_search_moe_backend)
    return run_isolated_subprocess(
        _worker_command(spec),
        cwd=Path.cwd(),
        env=env,
        timeout=spec.search_timeout_s,
        capture_output=False,
    )


def _prepare_timeout_salvage(ckpt: Checkpoint) -> bool:
    search_state = ckpt.state.get("mix_precision_search")
    if not isinstance(search_state, dict):
        return False
    result = search_state.get("result")
    queue = search_state.get("candidate_queue")
    if not isinstance(result, dict) or not isinstance(result.get("best_config"), dict) or not queue:
        return False
    if search_state.get("status") == "running":
        search_state["status"] = "partial_timeout"
        search_state["termination_reason"] = "search_timeout"
        ckpt.record_phase_event(
            "mix_precision_search",
            "partial_timeout",
            total_configs_evaluated=result.get("total_configs_evaluated"),
            total_configs_available=result.get("total_configs_available"),
        )
        ckpt.save()
    return search_state.get("status") in {"completed", "partial_timeout"}


def run_isolated_module_search(
    spec: Spec,
    ckpt: Checkpoint,
) -> str:
    """Run search/export in a subprocess so its GPU allocations die with it."""
    output = result_path(spec.session_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = _run_worker(spec, output)
    except subprocess.TimeoutExpired as error:
        _reload_checkpoint(ckpt)
        if not _prepare_timeout_salvage(ckpt):
            raise StageError(
                "quantize",
                f"mixed-precision search worker timed out after {spec.search_timeout_s:g}s",
                code="search_worker_timeout",
            ) from error
        try:
            completed = _run_worker(spec, output)
        except subprocess.TimeoutExpired as export_error:
            _reload_checkpoint(ckpt)
            raise StageError(
                "quantize",
                f"mixed-precision export worker timed out after {spec.search_timeout_s:g}s",
                code="search_export_worker_timeout",
            ) from export_error
    _reload_checkpoint(ckpt)
    if not output.is_file():
        raise StageError(
            "quantize",
            f"mixed-precision worker exited with returncode={completed.returncode} without a result",
            code="search_worker_missing_result",
        )
    payload = json.loads(output.read_text())
    if completed.returncode != 0 or payload.get("status") != "success":
        raise StageError(
            str(payload.get("stage") or "quantize"),
            str(payload.get("message") or "mixed-precision worker failed"),
            code=str(payload.get("code") or "search_worker_failed"),
            diagnostic=str(payload.get("diagnostic") or payload.get("message") or ""),
        )
    quant_ckpt_dir = str(payload.get("quant_ckpt_dir") or "")
    if not quant_ckpt_dir:
        raise StageError(
            "quantize",
            "mixed-precision worker returned no quantized checkpoint",
            code="search_worker_missing_artifact",
        )
    return quant_ckpt_dir


def _worker_payload(session_dir: Path) -> dict[str, Any]:
    ckpt = Checkpoint.load(session_dir)
    if ckpt is None:
        raise StageError(
            "quantize",
            f"mixed-precision worker found no state.json under {session_dir}",
            code="search_worker_missing_checkpoint",
        )
    run_spec = ckpt.state.get("run_spec")
    if not isinstance(run_spec, dict):
        raise StageError(
            "quantize",
            "mixed-precision worker found no persisted run_spec",
            code="search_worker_missing_spec",
        )
    spec = Spec.from_dict(
        run_spec,
        runtime_context=dict(ckpt.state.get("runtime_context") or {}),
    )
    activate_runtime(spec)
    os.environ.update(spec.runtime_env)
    for key, value in (ckpt.state.get("retained_runtime_env") or {}).items():
        os.environ[str(key)] = str(value)
    configure_search_runtime_env(os.environ, spec.effective_search_moe_backend)
    quant_ckpt_dir = run_module_search(spec, ckpt)
    return {
        "status": "success",
        "quant_ckpt_dir": quant_ckpt_dir,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args(argv)
    session_dir = Path(args.session).resolve()
    output = result_path(session_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = _worker_payload(session_dir)
        returncode = 0
    except StageError as error:
        payload = {
            "status": "failed",
            "stage": error.stage,
            "message": error.message,
            "code": error.code,
            "diagnostic": error.diagnostic,
        }
        returncode = 1
    except Exception as error:
        payload = {
            "status": "failed",
            "stage": "quantize",
            "message": f"{type(error).__name__}: {error}",
            "code": "search_worker_exception",
            "diagnostic": traceback.format_exc(),
        }
        returncode = 1
    write_json_atomic(output, payload)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
