#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Capture the software and accelerator identities used by a Quant-Perf run."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from quark.experimental.torch.quant_perf.workspace.git import find_git_root

_PACKAGES = {
    "quark": ("amd-quark", "quark"),
    "torch": ("torch", "torch"),
    "transformers": ("transformers", "transformers"),
    "vllm": ("vllm", "vllm"),
    "aiter": ("amd-aiter", "aiter"),
    "TraceLens": ("TraceLens", "TraceLens"),
    "safetensors": ("safetensors", "safetensors"),
    "datasets": ("datasets", "datasets"),
    "lm-eval": ("lm-eval", "lm_eval"),
    "anthropic": ("anthropic", "anthropic"),
    "claude-agent-sdk": ("claude-agent-sdk", "claude_agent_sdk"),
    "huggingface-hub": ("huggingface-hub", "huggingface_hub"),
    "anyio": ("anyio", "anyio"),
    "python-dotenv": ("python-dotenv", "dotenv"),
    "PyYAML": ("PyYAML", "yaml"),
    "requests": ("requests", "requests"),
}


def _git(repo: Path | None, *args: str) -> str:
    if repo is None:
        return ""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _package_identity(distribution: str, module: str) -> dict[str, str]:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = ""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    origin = ""
    if spec is not None:
        origin = spec.origin or next(iter(spec.submodule_search_locations or ()), "")
    if not version and origin:
        version_file = Path(origin).parent / "_version.py"
        try:
            match = re.search(
                r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
                version_file.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        except OSError:
            match = None
        if match is not None:
            version = match.group(1)
    git_root = find_git_root(Path(origin)) if origin else None
    return {
        "version": version,
        "origin": str(origin),
        "git_sha": _git(git_root, "rev-parse", "HEAD"),
        "git_description": _git(git_root, "describe", "--tags", "--always", "--dirty"),
    }


def _command_identity(command: str) -> dict[str, str]:
    path = shutil.which(command)
    if not path:
        return {"path": "", "version": ""}
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return {"path": path, "version": ""}
    output = (result.stdout or result.stderr).strip().splitlines()
    return {
        "path": path,
        "version": output[0] if result.returncode == 0 and output else "",
    }


def _rocm_smi_gpu_models() -> list[str]:
    command = shutil.which("rocm-smi")
    if not command:
        return []
    try:
        result = subprocess.run(
            [command, "--showproductname", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(result.stdout) if result.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    models = []
    for card in payload.values():
        if not isinstance(card, dict):
            continue
        model = card.get("Card Series")
        if not model or model == "N/A":
            model = card.get("GFX Version") or card.get("Card Model")
        if model and model not in models:
            models.append(str(model))
    return models


def _nvidia_smi_gpu_models() -> list[str]:
    command = shutil.which("nvidia-smi")
    if not command:
        return []
    try:
        result = subprocess.run(
            [command, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return list(dict.fromkeys(line.strip() for line in result.stdout.splitlines() if line.strip()))


def collect_runtime_inventory() -> dict[str, object]:
    """Return a reproducible snapshot without exposing environment secrets."""
    package_inventory = {
        label: _package_identity(distribution, module) for label, (distribution, module) in _PACKAGES.items()
    }
    accelerator = {
        "rocm": "",
        "cuda": "",
        "gpu_models": [],
    }
    try:
        import torch

        accelerator["rocm"] = str(torch.version.hip or "")
        accelerator["cuda"] = str(torch.version.cuda or "")
    except Exception:
        pass
    accelerator["gpu_models"] = _rocm_smi_gpu_models() or _nvidia_smi_gpu_models()

    geak_root = Path(os.environ["GEAK_ROOT"]).resolve() if os.environ.get("GEAK_ROOT") else None
    geak_git_root = find_git_root(geak_root)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "accelerator": accelerator,
        "packages": package_inventory,
        "tools": {
            "claude": _command_identity("claude"),
            "forge-gemm-tune": _command_identity("forge-gemm-tune"),
        },
        "geak": {
            "root": str(geak_root or ""),
            "git_sha": _git(geak_git_root, "rev-parse", "HEAD"),
            "git_description": _git(
                geak_git_root,
                "describe",
                "--tags",
                "--always",
                "--dirty",
            ),
        },
    }
