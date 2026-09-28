#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Framework-agnostic Landing helpers: the OpenAI-compatible readiness probe
shared by the vLLM and experimental Atom adapters.

Design ref: IMPL_SPEC §4.3.1 wait_atom_ready -- the probe itself only talks
to the standard /health, /v1/models, /v1/completions endpoints, so it is not
actually Atom-specific despite the name in the design doc.
"""

from __future__ import annotations

import socket
import subprocess
import time
from typing import Any

import requests  # type: ignore[import-untyped]

from quark.experimental.torch.quant_perf.session.spec import DEFAULT_SERVER_HOST, StageError


def ensure_port_available(port: int, host: str = DEFAULT_SERVER_HOST) -> None:
    """Ensure that a server address is available for binding.

    :param port: TCP port to check.
    :param host: IPv4 address or hostname the server will bind.
    :raises StageError: If the address cannot be bound.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(port)))
    except OSError as error:
        raise StageError(
            "land",
            f"server address {host}:{port} is unavailable",
            code="address_in_use",
            diagnostic=f"{type(error).__name__}: {error}",
        ) from error
    finally:
        sock.close()


def _server_base_url(host: str, port: int) -> str:
    """Return a locally reachable URL for a server bind address."""
    request_host = DEFAULT_SERVER_HOST if host in ("", "0.0.0.0") else host
    return f"http://{request_host}:{port}"


def wait_ready(
    port: int,
    timeout_s: int = 600,
    proc: subprocess.Popen[Any] | None = None,
    host: str = DEFAULT_SERVER_HOST,
) -> bool:
    """A three-tier readiness probe (ported from inference_optimizer's
    baseline.py:1524 approach): /health -> /v1/models non-empty -> a real
    1-token completion. All three tiers are retried together until
    timeout_s -- a real E2E run showed the HTTP server (FastAPI/uvicorn) can
    already answer /health before its EngineCore subprocess has finished
    loading the model, so /v1/models or /v1/completions can transiently fail
    right after /health first succeeds; treating that as a hard failure
    (the previous behavior, no retry past tier 1) reported "never became
    ready" seconds before the server actually finished starting.

    `proc` (a subprocess.Popen), if given, is polled on every retry -- a
    server that crashes during startup (e.g. a missing dependency) must fail
    within seconds, not silently burn the full timeout_s retrying a health
    check that can never succeed (found via a real ATOM startup crash during
    E2E testing: proc.poll() was never checked here, so a dead server was
    indistinguishable from a merely-slow one)."""
    base = _server_base_url(host, port)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            # 1. HTTP health
            if requests.get(f"{base}/health", timeout=2).status_code != 200:
                time.sleep(3)
                continue

            # 2. Model loaded
            models = requests.get(f"{base}/v1/models", timeout=5).json()
            if not models.get("data"):
                time.sleep(3)
                continue
            model_id = models["data"][0]["id"]

            # 3. Real inference probe (guards against a router idling with nothing loaded)
            resp = requests.post(
                f"{base}/v1/completions",
                timeout=30,
                json={"model": model_id, "prompt": "1+1=", "max_tokens": 1},
            )
            if resp.status_code == 200:
                return True
            time.sleep(3)
        except Exception:
            time.sleep(3)
    return False


def get_model_id(port: int, host: str = DEFAULT_SERVER_HOST) -> str:
    """Return the model ID reported by an OpenAI-compatible server.

    :param port: Server TCP port.
    :param host: Server bind address.
    :return: Loaded model identifier.
    """
    return requests.get(f"{_server_base_url(host, port)}/v1/models", timeout=5).json()["data"][0]["id"]
