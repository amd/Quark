#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Validate GGUF exports against llama.cpp."""

from __future__ import annotations

import subprocess
from pathlib import Path

from quark.common.utils.import_utils import is_gguf_available_and_minimum_version
from quark.torch.export.llama_cpp_export.formats import LlamaCppExportFormat


def validate_gguf_metadata(gguf_path: Path, export_format: LlamaCppExportFormat) -> dict[str, int]:
    """Check GGUF file_type and tensor-type histogram.

    Args:
        gguf_path: Path to a GGUF shard.
        export_format: Expected export format descriptor.

    Returns:
        Summary dict with file_type and tensor counts.
    """
    if not is_gguf_available_and_minimum_version():
        raise ImportError("gguf>=0.10.0 is required to validate GGUF exports")

    from collections import Counter

    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))
    file_type = int(reader.fields["general.file_type"].parts[-1].tolist()[0])
    expected = int(export_format.llama_file_type)
    if file_type != expected:
        raise ValueError(
            f"GGUF file_type mismatch for {gguf_path.name}: "
            f"expected {expected} ({export_format.name}), got {file_type}"
        )

    hist = Counter(t.tensor_type.name for t in reader.tensors)
    return {
        "file_type": file_type,
        "tensor_count": len(reader.tensors),
        **{f"type_{k}": v for k, v in sorted(hist.items())},
    }


def run_llama_cpp_load_test(
    gguf_path: Path,
    llama_cli: Path,
    *,
    timeout_s: int = 120,
) -> str:
    """Load a GGUF in llama-server and run a one-token completion.

    Args:
        gguf_path: First GGUF shard path (multi-shard siblings auto-discovered).
        llama_cli: Path to llama-cli binary (used to locate llama-server).
        timeout_s: Subprocess timeout.

    Returns:
        Generated text from the completion API.
    """
    llama_server = llama_cli.with_name("llama-server")
    if not llama_server.exists():
        raise FileNotFoundError(f"Missing llama-server: {llama_server}")

    import json
    import socket
    import time
    import urllib.error
    import urllib.request

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    prompt = (
        "<|im_start|>user\n"
        "小米的总裁是谁？请直接回答姓名。"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    proc = subprocess.Popen(
        [
            str(llama_server),
            "-m",
            str(gguf_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-ngl",
            "99",
            "-c",
            "2048",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        health_url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("llama-server exited before becoming ready")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        else:
            raise TimeoutError(f"llama-server not ready within {timeout_s}s")

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/completion",
            data=json.dumps(
                {
                    "prompt": prompt,
                    "n_predict": 64,
                    "temperature": 0,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("content", "").strip()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
