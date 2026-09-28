#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""GSM8K evaluation helpers used by AccuracyGate.

Primary path: offline evaluation via lm_eval's vllm backend (vllm_causallms),
with --apply_chat_template.  This is critical for chat-tuned models (Qwen3,
Llama-3-Instruct, etc.): Quark's evaluate_gsm8k_offline() uses a bare
completion prompt format ("Question: ... Answer:") without chat template, which
produces artificially low and unstable scores on chat models -- a 12.6% gap was
observed on Qwen3-0.6B fp8 where the server-mode (chat-template) result showed
gap≈0.  Using lm_eval vllm backend ensures both baseline and quantized evals
apply the same chat template, making the comparison valid.

"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.runtime.backends import (
    configure_aiter_mxfp4_moe_ksplit,
    configure_mxfp4_runtime_env,
    is_mxfp4_moe_model,
    model_requires_aiter_runtime,
    resolve_kv_cache_dtype,
)
from quark.experimental.torch.quant_perf.session.spec import EvalProfile

from .execution import (
    PreparationFailure,
    configure_vllm_cache_env,
    model_startup_timeout_s,
    run_isolated_subprocess,
    subprocess_text,
)
from .preparation import preparation_succeeded, prepared_executor_backend


class DependencyError(RuntimeError):
    """lm_eval subprocess failed due to a missing Python package, not a model
    or accuracy problem. Repair cannot install dependencies, so callers should
    surface the error directly rather than burning code-generation rounds."""


class EvaluationFailure(RuntimeError):
    """Structured lm_eval subprocess failure with preserved diagnostics."""

    def __init__(
        self,
        model_dir: str,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
        phase: str = "",
        elapsed_s: float | None = None,
        timeout_s: float | None = None,
    ):
        self.model_dir = model_dir
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.phase = phase
        self.elapsed_s = elapsed_s
        self.timeout_s = timeout_s
        super().__init__(
            f"lm_eval failed for {model_dir} (phase={phase}, returncode={returncode}, "
            f"elapsed={elapsed_s}s, budget={timeout_s}s)\n"
            f"stdout tail:\n{stdout[-12000:]}\n"
            f"stderr tail:\n{stderr[-12000:]}"
        )


def gsm8k_eval_offline(
    model_dir: str,
    gpu_id: int = 0,
    num_questions: int = 1319,
    tp: int = 1,
    gpu_memory_utilization: float = 0.85,
    moe_backend: str = "",
    profile: EvalProfile | None = None,
    *,
    trust_remote_code: bool = False,
    max_num_seqs: int | None = None,
    runtime_python: str = "",
    runtime_env: dict[str, str] | None = None,
    output_dir: str | Path | None = None,
    kv_cache_dtype: str | None = None,
) -> float:
    """Evaluate GSM8K accuracy using lm_eval's vllm offline backend.

    Uses vllm_causallms (lm_eval --model vllm) with the session's frozen
    EvalProfile. Chat templates and thinking controls are applied only when the
    profile requests them. No HTTP server is required.

    tp: tensor parallel size for large models that don't fit on a single GPU.
    Pass num_questions < 1319 for a lightweight spot-check (maps to --limit).
    """
    if profile is None:
        raise ValueError("gsm8k_eval_offline requires a frozen EvalProfile")
    kv_cache_dtype = resolve_kv_cache_dtype(model_dir, kv_cache_dtype)
    visible = ",".join(str(gpu_id + i) for i in range(tp))
    env = {
        **os.environ,
        **dict(runtime_env or {}),
        "VLLM_PLUGINS": "",
        "ROCR_VISIBLE_DEVICES": visible,
    }
    configure_vllm_cache_env(env, model_dir)
    # Enable AITER when required by either the quantization scheme or model
    # architecture. Runtime backend selection is independent of whether vLLM
    # executes eagerly or with CUDA graphs.
    uses_mxfp4_moe = is_mxfp4_moe_model(model_dir)
    requires_aiter_runtime = model_requires_aiter_runtime(model_dir)
    backend = configure_mxfp4_runtime_env(
        env,
        enable_aiter_moe=uses_mxfp4_moe or requires_aiter_runtime,
        select_mxfp4_moe_backend=uses_mxfp4_moe,
        moe_backend=moe_backend,
    )
    configure_aiter_mxfp4_moe_ksplit(env, model_dir, moe_backend or backend)
    _moe_backend_override = f",moe_backend={moe_backend}" if moe_backend else ""
    if not moe_backend and backend == "flydsl":
        _moe_backend_override = ",moe_backend=aiter,enable_prefix_caching=False"
    tp_arg = f",tensor_parallel_size={tp}" if tp > 1 else ""
    limit_args = ["--limit", str(num_questions)] if num_questions < 1319 else []
    model_args = (
        f"pretrained={model_dir},gpu_memory_utilization={gpu_memory_utilization},"
        f"max_model_len={profile.max_model_len},"
        f"max_gen_toks={profile.max_gen_toks}"
    )
    if profile.enable_thinking is not None:
        model_args += f",enable_thinking={profile.enable_thinking}"
    if trust_remote_code:
        model_args += ",trust_remote_code=True"
    if max_num_seqs is not None:
        model_args += f",max_num_seqs={max_num_seqs}"
    model_args += f"{tp_arg}{_moe_backend_override}"
    if kv_cache_dtype != "auto":
        model_args += f",kv_cache_dtype={kv_cache_dtype}"
    model_args += f",distributed_executor_backend={prepared_executor_backend(tp)}"
    command = [
        runtime_python or sys.executable,
        "-m",
        "quark.experimental.torch.quant_perf.evaluation.lm_eval_launcher" if trust_remote_code else "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        profile.task,
        "--num_fewshot",
        str(profile.num_fewshot),
        "--gen_kwargs",
        ",".join(f"{key}={value}" for key, value in profile.gen_kwargs.items()),
    ]
    if profile.apply_chat_template:
        command.append("--apply_chat_template")
    if profile.system_instruction:
        command.extend(["--system_instruction", profile.system_instruction])

    output_context: AbstractContextManager[str]
    if output_dir is None:
        output_context = tempfile.TemporaryDirectory()
    else:
        persistent = Path(output_dir)
        persistent.mkdir(parents=True, exist_ok=True)
        output_context = nullcontext(str(persistent))

    with output_context as out_dir:
        eval_command = command + [
            "--batch_size",
            profile.batch_size,
            "--output_path",
            out_dir,
            "--log_samples",
        ]
        artifact_root = Path(out_dir).resolve()
        preparation_timeout = float(
            env.get(
                "QUARK_QUANT_PERF_PREPARATION_TIMEOUT_S",
                str(model_startup_timeout_s(model_dir, minimum_s=600) + 1800),
            )
        )
        status_path = artifact_root / "preparation.json"
        (artifact_root / "command.json").write_text(
            json.dumps(
                {
                    "argv": eval_command + limit_args,
                    "preparation_timeout_s": preparation_timeout,
                    "evaluation_timeout_s": 7200,
                },
                indent=2,
            )
        )
        (artifact_root / "profile.json").write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
        try:
            proc = run_isolated_subprocess(
                eval_command + limit_args,
                capture_output=True,
                timeout=7200,
                env=env,
                preparation_timeout=preparation_timeout,
                preparation_status_path=status_path,
            )
        except subprocess.TimeoutExpired as exc:
            (artifact_root / "stdout.log").write_text(subprocess_text(exc.stdout or exc.output))
            (artifact_root / "stderr.log").write_text(subprocess_text(exc.stderr))
            raise EvaluationFailure(
                model_dir,
                stdout=subprocess_text(exc.stdout or exc.output),
                stderr=subprocess_text(exc.stderr),
                timed_out=True,
                phase=getattr(exc, "phase", "evaluation"),
                elapsed_s=getattr(exc, "elapsed", None),
                timeout_s=exc.timeout,
            ) from exc
        except PreparationFailure as exc:
            (artifact_root / "stdout.log").write_text(exc.stdout)
            (artifact_root / "stderr.log").write_text(exc.stderr)
            raise EvaluationFailure(
                model_dir,
                stdout=exc.stdout,
                stderr=f"{exc}\n{exc.stderr}",
                phase=exc.phase,
                elapsed_s=exc.elapsed,
            ) from exc
        (artifact_root / "stdout.log").write_text(subprocess_text(proc.stdout))
        (artifact_root / "stderr.log").write_text(subprocess_text(proc.stderr))
        if proc.returncode != 0:
            _classify_lm_eval_failure(
                env,
                model_dir,
                runtime_python=runtime_python,
                returncode=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                phase="evaluation" if preparation_succeeded(status_path) else "preparation",
            )
        try:
            return read_gsm8k_score(Path(out_dir), profile=profile)
        except RuntimeError as exc:
            if "no results" not in str(exc):
                raise
            raise EvaluationFailure(
                model_dir,
                returncode=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                phase="evaluation" if preparation_succeeded(status_path) else "preparation",
            ) from exc


def _classify_lm_eval_failure(
    lm_eval_env: dict[str, Any],
    model_dir: str,
    *,
    runtime_python: str = "",
    returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
    phase: str = "",
) -> None:
    """Called when lm_eval exits non-zero.

    Runs a targeted import probe **in the same environment lm_eval used** to
    distinguish a Python dependency error (ModuleNotFoundError/ImportError at
    import time) from a real model/accuracy failure.  Raises DependencyError
    for the former so AccuracyGate skips repair and surfaces an actionable
    message; raises a plain RuntimeError for the latter so repair can attempt a
    fix.
    """
    import sys as _sys

    # Probe using the same env as lm_eval so PYTHONPATH/ROCR_VISIBLE_DEVICES
    # match -- a probe against the bare system environment could give a false
    # negative if framework_repo was injected via PYTHONPATH.
    probe = (
        "import vllm.model_executor.layers.quantization; "
        "import vllm.model_executor.layers.quantization.compressed_tensors"
    )
    probe_result = subprocess.run(
        [runtime_python or _sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
        env=lm_eval_env,
    )
    if probe_result.returncode != 0:
        probe_stderr = probe_result.stderr
        missing = None
        for line in probe_stderr.splitlines():
            if "ModuleNotFoundError: No module named" in line:
                missing = line.split("No module named")[-1].strip().strip("'\"")
                break
            if "ImportError:" in line:
                missing = line.split("ImportError:")[-1].strip()
                break
        if missing:
            _pkg_map = {"compressed_tensors": "compressed-tensors"}
            pkg = next(
                (v for k, v in _pkg_map.items() if k in missing),
                missing.split(".")[0].replace("_", "-"),
            )
            raise DependencyError(
                f"lm_eval failed because a Python dependency is missing or incompatible: "
                f"'{missing}'. "
                f"Fix: pip install --upgrade '{pkg}' then re-run quark.experimental.torch.quant_perf. "
                f"(repair cannot fix dependency installation issues.)"
            )
    raise EvaluationFailure(
        model_dir,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        phase=phase,
    )


def read_gsm8k_score(
    out_dir: Path,
    profile: EvalProfile | None = None,
) -> float:
    result_files = list(out_dir.rglob("results*.json"))
    if not result_files:
        raise RuntimeError(f"lm_eval produced no results*.json under {out_dir}")
    result_path = max(
        result_files,
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    data = json.loads(result_path.read_text())
    results = data["results"]
    if profile is not None:
        task = results.get(profile.task)
        if task is None:
            raise RuntimeError(f"task {profile.task!r} not found in results keys: {list(results)}")
        if profile.metric not in task:
            raise RuntimeError(f"metric {profile.metric!r} not found for task {profile.task!r}")
        return task[profile.metric]
    # Accept any gsm8k variant task name (gsm8k, gsm8k_cot_zeroshot, etc.)
    for key in results:
        if "gsm8k" in key:
            return results[key]["exact_match,flexible-extract"]
    raise RuntimeError(f"no gsm8k task found in results keys: {list(results)}")
