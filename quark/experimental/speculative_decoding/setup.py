#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""One-time environment preparation for the validated EAGLE-3 CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from quark.experimental.speculative_decoding.torchspec_runner import (
    _model_name,
    find_runner_assets,
)

_DEFAULT_CACHE = "~/.cache/amd-quark/eagle3"
_DEFAULT_IMAGE = "quark-specdec-rocm:latest"
# Pin the revision used by the validated Qwen3-8B runner.  Setup checks out this
# exact commit instead of silently following TorchSpec main.
_DEFAULT_TORCHSPEC_COMMIT = "d230a3f13212e8a3eb35e04f99fbdd8fa45241f0"


def _output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(cache_dir: Path, base_model: str, image: str) -> Path:
    model_config = cache_dir / "models" / _model_name(base_model) / "config.json"
    try:
        gpu_inventory: Any = json.loads(_output(["rocm-smi", "--showproductname", "--json"]))
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        gpu_inventory = None
    manifest: dict[str, Any] = {
        "created_at_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu_inventory": gpu_inventory,
        "base_model": base_model,
        "model_config": str(model_config),
        "model_config_sha256": _sha256(model_config),
        "torchspec_revision": _output(["git", "-C", str(cache_dir / "TorchSpec"), "rev-parse", "HEAD"]),
        "image": image,
        "image_id": _output(["docker", "image", "inspect", "--format", "{{.Id}}", image]),
    }
    out = cache_dir / "manifests" / f"setup-{_model_name(base_model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


def prepare_environment(
    base_model: str,
    cache_dir: str = _DEFAULT_CACHE,
    image: str = _DEFAULT_IMAGE,
    torchspec_commit: str = _DEFAULT_TORCHSPEC_COMMIT,
    force_build: bool = False,
    base_image: str | None = None,
) -> Path:
    """Prepare and record the external dependencies used by the full pipeline."""
    cache = Path(os.environ.get("QUARK_EAGLE3_CACHE", cache_dir)).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    assets = find_runner_assets()
    env = os.environ.copy()
    env.update(
        {
            "BASE_MODEL": base_model,
            "CACHE_DIR": str(cache),
            "TORCHSPEC_DIR": str(cache / "TorchSpec"),
            "MODELS": str(cache / "models"),
            "IMG": image,
            "TORCHSPEC_COMMIT": torchspec_commit,
            "FORCE_BUILD": "1" if force_build else "0",
            "RUNNER_PROFILE": (
                "minimax_m3_best_recipe" if "minimax-m3" in base_model.lower() else "qwen3_8b_quick_start"
            ),
        }
    )
    # Unset leaves the Dockerfile's own pin in charge.
    if base_image:
        env["VLLM_ROCM_IMAGE"] = base_image
    subprocess.run(
        ["bash", str(assets / "eagle3" / "common" / "scripts" / "00_setup.sh"), "--base_model", base_model],
        check=True,
        env=env,
    )
    manifest = _write_manifest(cache, base_model, image)
    print(f"Environment manifest: {manifest}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the validated 8-GPU EAGLE-3 environment")
    parser.add_argument("--base_model", default="Qwen/Qwen3-8B", help="target Hugging Face ID")
    parser.add_argument("--cache-dir", default=_DEFAULT_CACHE, help="persistent dependency/model cache")
    parser.add_argument("--image", default=_DEFAULT_IMAGE, help="ROCm runner image tag")
    parser.add_argument("--torchspec-commit", default=_DEFAULT_TORCHSPEC_COMMIT)
    parser.add_argument("--force-build", action="store_true", help="rebuild the Docker image even when present")
    parser.add_argument(
        "--base-image",
        default=os.environ.get("VLLM_ROCM_IMAGE"),
        help="vLLM ROCm base image to build on (default: the pin in eagle3/common/docker/Dockerfile.rocm)",
    )
    args = parser.parse_args()
    prepare_environment(
        base_model=args.base_model,
        cache_dir=args.cache_dir,
        image=args.image,
        torchspec_commit=args.torchspec_commit,
        force_build=args.force_build,
        base_image=args.base_image,
    )


if __name__ == "__main__":
    main()
