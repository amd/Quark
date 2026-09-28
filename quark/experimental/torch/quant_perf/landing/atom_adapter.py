#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Landing adapter for the experimental Atom framework.

Design ref: IMPL_SPEC §4.3.1, corrected against the real ATOM source (no
`atom` console-script; the real entry point is a module invocation) -- see
tests/spikes/spike_04_atom_serve.py / spike_05_atom_trace.py, which this
adapter's implementation is based on.
"""

from __future__ import annotations

import os
import subprocess
import sys

from quark.experimental.torch.quant_perf import config
from quark.experimental.torch.quant_perf.landing.base import ensure_port_available, wait_ready
from quark.experimental.torch.quant_perf.session.spec import DEFAULT_SERVER_HOST, ServerHandle, Spec, StageError


def start_atom_server(
    model_dir: str,
    tp: int,
    port: int,
    kv_cache_scheme: str | None,
    profiler_dir: str | None = None,
    gpu_id: int = 0,
    python_exe: str = "",
    runtime_env: dict[str, str] | None = None,
    host: str = DEFAULT_SERVER_HOST,
) -> subprocess.Popen[bytes]:
    """Starts the Atom inference server. profiler_dir not None only enables
    the profiling *capability* -- it does not automatically start capturing
    (see perfopt/collect.py, which issues the explicit /start_profile)."""
    cmd = [
        python_exe or sys.executable,
        "-m",
        "atom.entrypoints.openai_server",
        "--model",
        model_dir,
        "-tp",
        str(tp),
        "--host",
        host,
        "--port",
        str(port),
        "--kv_cache_dtype",
        kv_cache_scheme or "bf16",  # Atom only accepts bf16/fp8, no "auto"
        "--trust-remote-code",
    ]
    if profiler_dir:
        cmd += ["--torch-profiler-dir", profiler_dir]
    env = {
        **os.environ,
        **dict(runtime_env or {}),
        # Starts at gpu_id, not always 0 -- a real E2E run on an 8-GPU host
        # where GPU0-3 were busy showed the server must be able to target a
        # non-zero-indexed physical GPU (spec.gpu_id), not just the first tp
        # devices on the box.
        "ROCR_VISIBLE_DEVICES": ",".join(str(gpu_id + i) for i in range(tp)),
    }
    if profiler_dir:
        env["ATOM_PROFILER_MORE"] = "1"  # detailed shape/callstack, feeds TraceLens's roofline

    return subprocess.Popen(cmd, cwd=config.atom_root(), env=env, start_new_session=True)


def serve(
    quant_ckpt_dir: str,
    spec: Spec,
    profiler_dir: str | None = None,
    timeout_s: int = 600,
) -> ServerHandle:
    """Starts Atom on the quantized checkpoint and waits for it to become ready."""
    ensure_port_available(spec.server_port, host=spec.server_host)
    proc = start_atom_server(
        model_dir=quant_ckpt_dir,
        tp=spec.tp,
        port=spec.server_port,
        kv_cache_scheme=spec.kv_cache_scheme,
        host=spec.server_host,
        profiler_dir=profiler_dir,
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
            f"Atom server on {spec.server_host}:{spec.server_port} never became ready",
            code="server_start_failed",
            diagnostic=(
                f"Atom server exited with return code {returncode}"
                if returncode is not None
                else "Atom server did not become ready before timeout"
            ),
        )
    return handle
