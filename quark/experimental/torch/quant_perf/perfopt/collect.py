#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Trace collection: warm up a running server, capture a steady-state decode
window, and return the resulting PyTorch Chrome trace for TraceLens.

Design ref: IMPL_SPEC §4.4. The current vLLM and experimental Atom adapters
expose the same /start_profile and /stop_profile endpoints; neither starts
capturing just because a profiler directory flag was passed at launch -- the
capture window here is always driven by an explicit start/stop pair, ported
from the verified behavior in tests/spikes/spike_05_atom_trace.py.
"""

from __future__ import annotations

import concurrent.futures
import logging
import pathlib
import re
import subprocess
import sys
import time

import requests  # type: ignore[import-untyped]

from quark.experimental.torch.quant_perf.landing.base import get_model_id
from quark.experimental.torch.quant_perf.perfopt._tracelens import is_tracelens_available

logger = logging.getLogger(__name__)

_RANK0_TRACE_RE = re.compile(
    r"(?:^|[_.-])rank_?0(?:[_.-]|$)",
    re.IGNORECASE,
)
_ANY_RANK_TRACE_RE = re.compile(
    r"(?:^|[_.-])rank_?\d+(?:[_.-]|$)",
    re.IGNORECASE,
)


def select_engine_rank0_trace(
    traces: list[pathlib.Path],
) -> pathlib.Path | None:
    """Select one engine rank-0 trace without accepting frontend traces."""
    engine_traces = [pathlib.Path(trace) for trace in traces if "async_llm" not in pathlib.Path(trace).name.lower()]
    if not engine_traces:
        return None
    rank0 = [trace for trace in engine_traces if _RANK0_TRACE_RE.search(trace.name)]
    if rank0:
        candidates = rank0
    elif any(_ANY_RANK_TRACE_RE.search(trace.name) for trace in engine_traces):
        return None
    else:
        candidates = engine_traces

    def _selection_key(path: pathlib.Path) -> tuple[int, str]:
        try:
            modified = path.stat().st_mtime_ns
        except OSError:
            modified = 0
        return modified, str(path)

    return max(candidates, key=_selection_key)


def _split_steady_state(raw_trace: pathlib.Path, num_steps: int = 32) -> pathlib.Path:
    """Run TraceLens split_inference_trace_annotation to extract the
    decode-only steady-state window from a full inference trace.

    The raw trace includes CUDAGraph capture, warmup iterations, and
    prefill-decode mix steps that distort bottleneck statistics. The
    decode_only_steady_state window contains only representative decode
    steps, matching the workload the throughput benchmark measures.

    Returns the steady-state trace path if split succeeds, or the original
    raw trace as fallback (with a warning) so the caller is never broken.
    """
    if not is_tracelens_available():
        logger.warning("TraceLens is unavailable; using the raw trace.")
        return raw_trace

    split_dir = raw_trace.parent / "steady_state"
    split_dir.mkdir(exist_ok=True)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "TraceLens.TraceUtils.split_inference_trace_annotation",
                str(raw_trace),
                "-o",
                str(split_dir),
                "--find-steady-state",
                "--num-steps",
                str(num_steps),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.warning(
                "split_inference_trace_annotation failed (rc=%d): %s -- using raw trace",
                result.returncode,
                result.stderr[-500:],
            )
            return raw_trace
    except Exception as exc:
        logger.warning("split_inference_trace_annotation error (%s) -- using raw trace", exc)
        return raw_trace

    # Prefer decode_only_steady_state (pure decode, no prefill contamination).
    # Fall back to mixed_steady_state, then raw trace.
    for pattern in ("decode_only_steady_state*.gz", "mixed_steady_state*.gz"):
        candidates = sorted(split_dir.glob(pattern))
        if candidates:
            logger.info("using steady-state trace: %s", candidates[0])
            return candidates[0]

    logger.warning("no steady-state trace found after split -- using raw trace")
    return raw_trace


def _completions_request(base: str, model_id: str, prompt: str, osl: int) -> None:
    requests.post(
        f"{base}/v1/completions",
        timeout=300,
        json={"model": model_id, "prompt": prompt, "max_tokens": osl, "temperature": 0},
    )


def collect_trace(
    server_port: int,
    profiler_dir: str,
    isl: int,
    osl: int,
    warmup_steps: int = 8,
    profile_steps: int = 32,
    trace_wait_s: int = 600,
    extract_steady_state: bool = True,
) -> pathlib.Path:
    """Collects a steady-state decode trace and returns its path.

    The server must already be running with a profiler directory configured
    (landing.atom_adapter.serve(profiler_dir=...)) -- that only enables
    capturing, it does not start it.

    Warmup and profile requests are sent concurrently (pool size = step count).
    Concurrent requests form real batches on the server side, producing a
    bottleneck distribution representative of actual serving. This also
    drastically reduces trace size vs. serial requests: N concurrent requests
    complete in ~N decode steps instead of N*osl, shrinking the trace by ~N×
    and cutting stop_profile flush time accordingly.
    """
    base = f"http://localhost:{server_port}"
    model_id = get_model_id(server_port)
    prompt = "A " * isl

    from quark.experimental.torch.quant_perf.session.spec import StageError

    def _send_batch(n: int) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(_completions_request, base, model_id, prompt, osl) for _ in range(n)]
            for f in concurrent.futures.as_completed(futs):
                f.result()  # propagate any request-level exceptions

    _send_batch(warmup_steps)
    trace_dir = pathlib.Path(profiler_dir)
    existing_traces = (
        {path.resolve() for path in trace_dir.rglob("*.pt.trace.json.gz")} if trace_dir.exists() else set()
    )

    resp = requests.post(f"{base}/start_profile", timeout=30)
    if resp.status_code not in (200, 204):
        raise StageError(
            "perfopt",
            f"POST /start_profile returned {resp.status_code} -- "
            "server may not support profiling (check --profiler-config was passed at startup)",
        )

    _send_batch(profile_steps)

    # stop_profile flushes the PyTorch trace to disk before returning --
    # on large MoE models (e.g. 35B MoE) the raw trace can reach 1+ GB .gz
    # and flush takes 20-30 minutes; use 3600s to avoid ReadTimeout.
    resp = requests.post(f"{base}/stop_profile", timeout=3600)
    if resp.status_code not in (200, 204):
        raise StageError(
            "perfopt",
            f"POST /stop_profile returned {resp.status_code}",
        )

    deadline = time.time() + trace_wait_s
    observed: list[pathlib.Path] = []
    while time.time() < deadline:
        observed = sorted(trace_dir.rglob("*.pt.trace.json.gz"))
        new_traces = [trace for trace in observed if trace.resolve() not in existing_traces]
        raw = select_engine_rank0_trace(new_traces)
        if raw is not None:
            if extract_steady_state:
                return _split_steady_state(
                    raw,
                    num_steps=profile_steps,
                )
            return raw
        time.sleep(2)
    candidates = ", ".join(str(path) for path in observed) or "none"
    raise StageError(
        "perfopt",
        f"engine rank-0 trace not found in {profiler_dir} after {trace_wait_s}s; observed candidates: {candidates}",
    )
