#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from quark.experimental.torch.quant_perf.evaluation.execution import (
    PreparationFailure,
    model_startup_timeout_s,
    run_isolated_subprocess,
    subprocess_text,
)
from quark.experimental.torch.quant_perf.evaluation.preparation import prepared_executor_backend
from quark.experimental.torch.quant_perf.runtime.backends import resolve_kv_cache_dtype

DEFAULT_VERIFIER_TIMEOUT_S = 900
_QUANT_TENSOR_FIELDS = (
    "dtype",
    "qscheme",
    "group_size",
    "block_size",
    "ch_axis",
    "is_dynamic",
    "scale_format",
)
_MAX_LAYER_PATTERNS_PER_SCHEME = 8


def _compact_quant_spec(config: object) -> dict[str, object]:
    if not isinstance(config, dict):
        return {}
    summary: dict[str, object] = {}
    for tensor_kind in ("weight", "input_tensors", "output_tensors"):
        tensor_config = config.get(tensor_kind)
        if not isinstance(tensor_config, dict):
            continue
        compact = {
            field: tensor_config[field]
            for field in _QUANT_TENSOR_FIELDS
            if field in tensor_config and tensor_config[field] is not None
        }
        if compact:
            summary[tensor_kind] = compact
    return summary


def _summarize_layer_quant_config(config: object) -> dict[str, object]:
    if not isinstance(config, dict):
        return {}
    groups: dict[str, dict[str, object]] = {}
    for pattern, quant_spec in config.items():
        compact_spec = _compact_quant_spec(quant_spec)
        signature = json.dumps(compact_spec, sort_keys=True)
        group = groups.setdefault(
            signature,
            {
                "entry_count": 0,
                "patterns": [],
                "quantization": compact_spec,
            },
        )
        entry_count = group["entry_count"]
        assert isinstance(entry_count, int)
        group["entry_count"] = entry_count + 1
        patterns = group["patterns"]
        if isinstance(patterns, list) and len(patterns) < _MAX_LAYER_PATTERNS_PER_SCHEME:
            patterns.append(str(pattern))
    return {
        "entry_count": len(config),
        "scheme_groups": list(groups.values()),
    }


def read_quant_config(quant_ckpt_dir: str, limit: int = 3000) -> str:
    """Return a compact quantization-first checkpoint summary for repair."""
    path = Path(quant_ckpt_dir) / "config.json"
    if not path.exists():
        return "{}"
    try:
        raw = path.read_text()
        config = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return raw[:limit] if "raw" in locals() else "{}"
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        return raw[:limit]
    summary = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "quantization_config": {
            "quant_method": quant_config.get("quant_method"),
            "quant_mode": quant_config.get("quant_mode"),
            "layer_quant_config": _summarize_layer_quant_config(quant_config.get("layer_quant_config")),
            "global_quant_config": _compact_quant_spec(quant_config.get("global_quant_config")),
            "kv_cache_quant_config": _compact_quant_spec(quant_config.get("kv_cache_quant_config")),
            "exclude": quant_config.get("exclude"),
        },
    }
    return json.dumps(
        summary,
        ensure_ascii=False,
        separators=(",", ": "),
    )[:limit]


def load_inference_timeout_s(quant_ckpt_dir: str) -> int:
    """Scale verifier time for checkpoints whose weights cannot load in 15 minutes."""
    return model_startup_timeout_s(
        quant_ckpt_dir,
        minimum_s=DEFAULT_VERIFIER_TIMEOUT_S,
    )


def _verification_failure_text(stdout: object, stderr: object) -> str:
    return f"stdout:\n{subprocess_text(stdout)}\nstderr:\n{subprocess_text(stderr)}"


def verify_load_and_inference(
    quant_ckpt_dir: str,
    gpu_id: int,
    tp: int = 1,
    timeout_s: int = DEFAULT_VERIFIER_TIMEOUT_S,
    runtime_python: str = "",
    runtime_env: dict[str, str] | None = None,
    enforce_eager: bool = True,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    diagnostic_dir: Path | None = None,
    max_num_seqs: int | None = None,
    trust_remote_code: bool = False,
    kv_cache_dtype: str | None = None,
) -> tuple[bool, str]:
    kv_cache_dtype = resolve_kv_cache_dtype(quant_ckpt_dir, kv_cache_dtype)
    kv_cache_args = f", kv_cache_dtype={kv_cache_dtype!r}" if kv_cache_dtype != "auto" else ""
    visible = ",".join(str(gpu_id + index) for index in range(tp))
    env = {
        **os.environ,
        **dict(runtime_env or {}),
        "VLLM_PLUGINS": "",
        "ROCR_VISIBLE_DEVICES": visible,
    }
    engine_overrides = ""
    if max_num_seqs is not None:
        engine_overrides += f", max_num_seqs={max_num_seqs}"
    if trust_remote_code:
        engine_overrides += ", trust_remote_code=True"
    script = (
        "from vllm import LLM, SamplingParams\n"
        f"llm = LLM(model={quant_ckpt_dir!r}, tensor_parallel_size={tp}, "
        f"enforce_eager={enforce_eager!r}, "
        f"gpu_memory_utilization={gpu_memory_utilization}, "
        f"distributed_executor_backend={prepared_executor_backend(tp)!r}, "
        f"max_model_len={max_model_len}{engine_overrides}{kv_cache_args})\n"
        "out = llm.generate(['hello'], SamplingParams(max_tokens=1))\n"
        "assert out and out[0].outputs\nprint('ok')"
    )
    try:
        with tempfile.TemporaryDirectory() as temporary_dir:
            artifact_dir = diagnostic_dir if diagnostic_dir is not None else Path(temporary_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            result = run_isolated_subprocess(
                [runtime_python or "python3", "-c", script],
                capture_output=True,
                timeout=timeout_s,
                env=env,
                preparation_timeout=timeout_s,
                preparation_status_path=artifact_dir / "preparation.json",
            )
    except subprocess.TimeoutExpired as error:
        if diagnostic_dir is not None:
            _write_output(diagnostic_dir, error.stdout, error.stderr)
        diagnostic = _verification_failure_text(
            error.stdout,
            error.stderr,
        )
        suffix = f": {diagnostic}" if diagnostic else ""
        return False, f"load/inference verification timed out after {timeout_s}s{suffix}"
    except PreparationFailure as error:
        if diagnostic_dir is not None:
            _write_output(diagnostic_dir, error.stdout, error.stderr)
        return (
            False,
            f"load/inference preparation failed: {error}\n{_verification_failure_text(error.stdout, error.stderr)}",
        )
    if diagnostic_dir is not None:
        _write_output(diagnostic_dir, result.stdout, result.stderr)
    if result.returncode == 0 and "ok" in result.stdout:
        return True, ""
    return False, _verification_failure_text(
        result.stdout,
        result.stderr,
    )


def _write_output(directory: Path, stdout: object, stderr: object) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stdout.log").write_text(subprocess_text(stdout))
    (directory / "stderr.log").write_text(subprocess_text(stderr))
