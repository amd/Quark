#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Landing adapter for vLLM, Quark Quant-Perf's default inference framework.

Chosen for the MVP after a real E2E run repeatedly hit an Atom-specific NCCL
crash (`Cuda failure 'no kernel image is available for execution on the
device'` inside `torch.distributed.barrier()`, ATOM ModelRunner subprocess)
that a cross-validation run of raw vLLM, on the same GPU with the same
quantized checkpoint, did NOT reproduce -- confirmed via
tests/spikes/README.md's 2026-07-07 "cross-validated with native vLLM" entry. vLLM is
therefore the working landing path until the Atom-specific root cause is
found; both adapters share the same OpenAI-compatible readiness probe
(landing/base.py:wait_ready), so nothing downstream (AccuracyGate, PerfOpt)
needs to know which one is running.
"""

from __future__ import annotations

import os
import subprocess
import sys

from quark.experimental.torch.quant_perf.evaluation.execution import configure_vllm_cache_env
from quark.experimental.torch.quant_perf.landing.base import ensure_port_available, wait_ready
from quark.experimental.torch.quant_perf.runtime.backends import (
    configure_aiter_mxfp4_moe_ksplit,
    configure_mxfp4_runtime_env,
    extract_kv_cache_dtype,
    is_mxfp4_moe_model,
    model_requires_aiter_runtime,
    resolve_kv_cache_dtype,
)
from quark.experimental.torch.quant_perf.session.spec import DEFAULT_SERVER_HOST, ServerHandle, Spec, StageError


def _explicit_moe_backend(args: list[str]) -> str:
    for index, token in enumerate(args):
        if token == "--moe-backend" and index + 1 < len(args):
            return args[index + 1]
        if token.startswith("--moe-backend="):
            return token.split("=", 1)[1]
    return ""


def start_vllm_server(
    model_dir: str,
    tp: int,
    port: int,
    kv_cache_scheme: str | None,
    profiler_dir: str | None = None,
    extra_args: list[str] | None = None,
    gpu_id: int = 0,
    python_exe: str = "",
    runtime_env: dict[str, str] | None = None,
    host: str = DEFAULT_SERVER_HOST,
) -> subprocess.Popen[bytes]:
    """Starts vLLM's OpenAI-compatible server. vLLM's quantization config is
    auto-detected from the checkpoint's config.json ("quantization": "quark")
    -- no extra flag needed, unlike Atom's --kv_cache_dtype requirement.

    profiler_dir enables PyTorch trace collection via vLLM's built-in
    /start_profile and /stop_profile endpoints (vllm.entrypoints.serve.profile),
    which are always mounted by register_vllm_serve_api_routers(). Passing
    profiler_dir only enables the capability -- actual capture is driven by
    collect.py's explicit POST /start_profile / POST /stop_profile calls,
    matching the same pattern used for Atom."""
    import json

    extra_args, requested_dtype = extract_kv_cache_dtype(list(extra_args or []), kv_cache_scheme)
    kv_cache_dtype = resolve_kv_cache_dtype(model_dir, requested_dtype)
    cmd = [
        python_exe or sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_dir,
        "--tensor-parallel-size",
        str(tp),
        "--host",
        host,
        "--port",
        str(port),
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--trust-remote-code",
    ]
    if profiler_dir:
        cmd += [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": profiler_dir,
                    "ignore_frontend": True,
                    "torch_profiler_record_shapes": True,
                }
            ),
        ]
    cmd += list(extra_args or [])

    env = {
        **os.environ,
        **dict(runtime_env or {}),
        # Starts at gpu_id, not always 0 -- see the matching comment in
        # atom_adapter.py; a real E2E run on an 8-GPU host with GPU0-3 busy
        # showed a hardcoded range(tp) can't target a free non-zero GPU.
        "ROCR_VISIBLE_DEVICES": ",".join(str(gpu_id + i) for i in range(tp)),
        # Atom registers itself as a global vLLM platform plugin (vLLM's
        # load_general_plugins() picks it up regardless of which landing
        # framework is chosen), and that plugin's code has been observed to
        # be incompatible with the installed vLLM version on this host
        # (ModuleNotFoundError on vllm.v1.attention.backends.mla.prefill,
        # see tests/spikes/README.md) -- disable plugin auto-discovery so
        # this adapter gets a clean vLLM, not a partially-broken Atom one.
        "VLLM_PLUGINS": "",
    }
    configure_vllm_cache_env(env, model_dir)

    moe_backend = _explicit_moe_backend(cmd)
    uses_mxfp4_moe = is_mxfp4_moe_model(model_dir)
    backend = configure_mxfp4_runtime_env(
        env,
        enable_aiter_moe=uses_mxfp4_moe or model_requires_aiter_runtime(model_dir),
        select_mxfp4_moe_backend=uses_mxfp4_moe,
        moe_backend=moe_backend,
    )
    configure_aiter_mxfp4_moe_ksplit(env, model_dir, moe_backend or backend)
    if backend == "flydsl" and moe_backend in {"", "auto"}:
        cmd += ["--moe-backend", "aiter", "--no-enable-prefix-caching"]

    return subprocess.Popen(cmd, env=env, start_new_session=True)


def serve(
    quant_ckpt_dir: str,
    spec: Spec,
    profiler_dir: str | None = None,
    timeout_s: int = 600,
) -> ServerHandle:
    """Starts vLLM on the quantized checkpoint and waits for it to become
    ready. profiler_dir wires up PyTorch trace collection via vLLM's built-in
    /start_profile and /stop_profile endpoints -- same collect.py flow as
    Atom, no code changes needed downstream."""
    ensure_port_available(spec.server_port, host=spec.server_host)
    extra_args = spec.vllm_passthrough_args
    if profiler_dir:
        # Profile the requested workload with the same limits as throughput.
        for flag, value in (
            ("--gpu-memory-utilization", spec.vllm_gpu_memory_utilization),
            ("--max-model-len", spec.isl + spec.osl + 256),
            ("--max-num-seqs", spec.bench_concurrency),
        ):
            if not any(arg.split("=", 1)[0] == flag for arg in extra_args):
                extra_args.extend([flag, str(value)])
    proc = start_vllm_server(
        model_dir=quant_ckpt_dir,
        tp=spec.tp,
        port=spec.server_port,
        kv_cache_scheme=spec.vllm_kv_cache_dtype,
        host=spec.server_host,
        profiler_dir=profiler_dir,
        extra_args=extra_args,
        gpu_id=spec.gpu_id,
        python_exe=spec.runtime_python,
        runtime_env=spec.runtime_env,
    )
    handle = ServerHandle(port=spec.server_port, proc=proc, process_group_id=proc.pid)
    if not wait_ready(spec.server_port, timeout_s=timeout_s, proc=proc, host=spec.server_host):
        returncode = proc.poll()
        handle.stop()
        raise StageError(
            "land",
            f"vLLM server on {spec.server_host}:{spec.server_port} never became ready",
            code="server_start_failed",
            diagnostic=(
                f"vLLM server exited with return code {returncode}"
                if returncode is not None
                else "vLLM server did not become ready before timeout"
            ),
        )
    return handle
