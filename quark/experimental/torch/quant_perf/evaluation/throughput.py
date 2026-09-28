#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Comparable, noise-aware throughput measurement."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from quark.experimental.torch.quant_perf.runtime.backends import (
    configure_aiter_mxfp4_moe_ksplit,
    configure_mxfp4_runtime_env,
    is_mxfp4_moe_model,
    model_requires_aiter_runtime,
    resolve_kv_cache_dtype,
)

from .execution import configure_vllm_cache_env, run_isolated_subprocess, subprocess_text


class BenchmarkFailure(RuntimeError):
    """Structured throughput subprocess failure with diagnostics."""

    def __init__(
        self,
        model_dir: str,
        *,
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
        bench_error: str = "",
    ):
        self.model_dir = model_dir
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.bench_error = bench_error
        super().__init__(
            f"throughput_benchmark failed for {model_dir}: {bench_error}\n"
            f"stdout tail:\n{stdout[-12000:]}\n"
            f"stderr tail:\n{stderr[-12000:]}"
        )


@dataclass(frozen=True)
class ThroughputMeasurement:
    """Robust warm-throughput evidence from one fresh model process."""

    samples_tps: tuple[float, ...]
    median_tps: float
    mad_tps: float
    relative_mad: float
    warmup_tps: float | None
    stable: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["samples_tps"] = list(self.samples_tps)
        return value


def num_prompts_for(concurrency: int, isl: int, osl: int) -> int:
    """Choose enough requests to amortize pipeline ramp-up and drain."""
    seq_cost = isl + osl
    if seq_cost <= 1024:
        factor = 10
    elif seq_cost <= 4096:
        factor = 5
    elif seq_cost <= 16384:
        factor = 3
    else:
        factor = 2
    return max(concurrency * factor, concurrency)


def throughput_gain(new_tps: float, base_tps: float) -> float:
    """Return a guarded new/base throughput ratio."""
    if base_tps <= 0.0 or new_tps <= 0.0:
        return 0.0
    return new_tps / base_tps


def measure_throughput(
    model_dir: str,
    gpu_id: int = 0,
    isl: int = 1024,
    osl: int = 1024,
    num_prompts: int | None = None,
    tp: int = 1,
    concurrency: int = 64,
    gpu_memory_utilization: float = 0.8,
    timed_samples: int = 3,
    max_timed_samples: int = 4,
    moe_backend: str = "",
    runtime_python: str = "",
    runtime_env: dict[str, str] | None = None,
    trust_remote_code: bool = False,
    max_num_seqs: int | None = None,
    kv_cache_dtype: str | None = None,
) -> ThroughputMeasurement:
    """Measure output throughput in a fresh, warmed subprocess."""
    kv_cache_dtype = resolve_kv_cache_dtype(model_dir, kv_cache_dtype)
    kv_cache_args = f", kv_cache_dtype={kv_cache_dtype!r}" if kv_cache_dtype != "auto" else ""
    if num_prompts is None:
        num_prompts = num_prompts_for(concurrency, isl, osl)
    visible = ",".join(str(gpu_id + i) for i in range(tp))
    subprocess_env = {
        **os.environ,
        **dict(runtime_env or {}),
        "VLLM_PLUGINS": "",
        "ROCR_VISIBLE_DEVICES": visible,
    }
    configure_vllm_cache_env(subprocess_env, model_dir)
    uses_mxfp4_moe = is_mxfp4_moe_model(model_dir)
    backend = configure_mxfp4_runtime_env(
        subprocess_env,
        enable_aiter_moe=uses_mxfp4_moe or model_requires_aiter_runtime(model_dir),
        select_mxfp4_moe_backend=uses_mxfp4_moe,
        moe_backend=moe_backend,
    )
    configure_aiter_mxfp4_moe_ksplit(subprocess_env, model_dir, moe_backend or backend)
    engine_overrides = f", moe_backend={moe_backend!r}" if moe_backend else ""
    if not moe_backend and backend == "flydsl":
        engine_overrides = ", moe_backend='aiter'"
    if trust_remote_code:
        engine_overrides += ", trust_remote_code=True"
    max_model_len = isl + osl + 256
    if max_num_seqs is None:
        max_num_seqs = min(concurrency, num_prompts)
    script = f"""
import os, time, sys, json, random
os.environ["VLLM_PLUGINS"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = "{visible}"
from vllm import LLM, SamplingParams
try:
    _cfg = json.load(open(os.path.join({model_dir!r}, "config.json")))
    _vocab = _cfg.get("vocab_size") or (_cfg.get("text_config") or {{}}).get("vocab_size") or 32000
except Exception:
    _vocab = 32000
_hi = max(100, min(int(_vocab) - 1, 30000))
random.seed(1234)
prompts = [{{"prompt_token_ids": [random.randint(5, _hi) for _ in range({isl})]}}
           for _ in range({num_prompts})]
params = SamplingParams(max_tokens={osl}, temperature=0.0, ignore_eos=True)
llm = LLM(model={model_dir!r}, gpu_memory_utilization={gpu_memory_utilization}, max_model_len={max_model_len},
          max_num_seqs={max_num_seqs}, tensor_parallel_size={tp},
          enable_prefix_caching=False{engine_overrides}{kv_cache_args})
def run_once():
    t0 = time.monotonic()
    outputs = llm.generate(prompts, params)
    elapsed = time.monotonic() - t0
    n_done = len(outputs)
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    if n_done < {num_prompts} or total_tokens <= 0 or elapsed <= 0:
        print(f"BENCH_ERROR=completed={{n_done}}/{num_prompts} tokens={{total_tokens}} elapsed={{elapsed:.3f}}")
        sys.exit(2)
    return total_tokens / elapsed

warmup_tps = run_once()
samples = [run_once() for _ in range({timed_samples})]
median = sorted(samples)[len(samples) // 2]
mad = sorted(abs(value - median) for value in samples)[len(samples) // 2]
relative_mad = mad / median if median > 0 else 1.0
if relative_mad > 0.01 and len(samples) < {max_timed_samples}:
    samples.append(run_once())
    ordered = sorted(samples)
    mid = len(ordered) // 2
    median = (
        ordered[mid]
        if len(ordered) % 2
        else (ordered[mid - 1] + ordered[mid]) / 2.0
    )
    deviations = sorted(abs(value - median) for value in samples)
    dmid = len(deviations) // 2
    mad = (
        deviations[dmid]
        if len(deviations) % 2
        else (deviations[dmid - 1] + deviations[dmid]) / 2.0
    )
    relative_mad = mad / median if median > 0 else 1.0
print("THROUGHPUT_RESULT=" + json.dumps({{
    "samples_tps": samples,
    "warmup_tps": warmup_tps,
    "stable": relative_mad <= 0.015,
}}))
"""
    try:
        result = run_isolated_subprocess(
            [runtime_python or "python3", "-c", script],
            capture_output=True,
            timeout=3600,
            env=subprocess_env,
        )
    except subprocess.TimeoutExpired as exc:
        raise BenchmarkFailure(
            model_dir,
            stdout=subprocess_text(exc.stdout or exc.output),
            stderr=subprocess_text(exc.stderr),
            timed_out=True,
            bench_error="timeout",
        ) from exc
    for line in result.stdout.splitlines():
        if line.startswith("THROUGHPUT_RESULT="):
            payload = json.loads(line.split("=", 1)[1])
            samples = tuple(float(value) for value in payload["samples_tps"])
            median_tps = float(statistics.median(samples))
            deviations = tuple(abs(value - median_tps) for value in samples)
            mad_tps = float(statistics.median(deviations))
            relative_mad = mad_tps / median_tps if median_tps > 0 else 1.0
            return ThroughputMeasurement(
                samples_tps=samples,
                median_tps=median_tps,
                mad_tps=mad_tps,
                relative_mad=relative_mad,
                warmup_tps=(float(payload["warmup_tps"]) if payload.get("warmup_tps") is not None else None),
                stable=bool(
                    payload.get(
                        "stable",
                        relative_mad <= 0.015,
                    )
                ),
            )
        if line.startswith("THROUGHPUT="):
            value = float(line.split("=", 1)[1])
            return ThroughputMeasurement(
                samples_tps=(value,),
                median_tps=value,
                mad_tps=0.0,
                relative_mad=0.0,
                warmup_tps=None,
                stable=True,
            )
    bench_error = next(
        (line for line in result.stdout.splitlines() if line.startswith("BENCH_ERROR=")),
        "",
    )
    raise BenchmarkFailure(
        model_dir,
        stdout=result.stdout,
        stderr=result.stderr,
        bench_error=bench_error,
    )


def throughput_benchmark(
    model_dir: str,
    gpu_id: int = 0,
    isl: int = 1024,
    osl: int = 1024,
    num_prompts: int | None = None,
    tp: int = 1,
    concurrency: int = 64,
    gpu_memory_utilization: float = 0.8,
    moe_backend: str = "",
    runtime_python: str = "",
    runtime_env: dict[str, str] | None = None,
    trust_remote_code: bool = False,
    max_num_seqs: int | None = None,
    kv_cache_dtype: str | None = None,
) -> float:
    return measure_throughput(
        model_dir,
        gpu_id=gpu_id,
        isl=isl,
        osl=osl,
        num_prompts=num_prompts,
        tp=tp,
        concurrency=concurrency,
        gpu_memory_utilization=gpu_memory_utilization,
        moe_backend=moe_backend,
        runtime_python=runtime_python,
        runtime_env=runtime_env,
        trust_remote_code=trust_remote_code,
        max_num_seqs=max_num_seqs,
        kv_cache_dtype=kv_cache_dtype,
    ).median_tps
