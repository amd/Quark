#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Collect non-mutating facts needed to plan an AMD Quark installation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def command_output(command: list[str]) -> str | None:
    if not shutil.which(command[0]):
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (result.stdout or result.stderr).strip()
    return text[:2000] if text else None


def collect() -> dict[str, object]:
    packages = {
        "amd-quark": distribution_version("amd-quark"),
        "torch": distribution_version("torch"),
        "onnx": distribution_version("onnx"),
        "onnxruntime": distribution_version("onnxruntime"),
        "onnxruntime-gpu": distribution_version("onnxruntime-gpu"),
        "onnxruntime-rocm": distribution_version("onnxruntime-rocm", "onnxruntime_rocm"),
    }
    tools = {name: shutil.which(name) for name in ("g++", "hipcc", "nvcc", "rocm-smi", "nvidia-smi")}
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
        },
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "environment": {
            key: os.environ.get(key)
            for key in ("ROCM_PATH", "HIP_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_VISIBLE_DEVICES")
        },
        "packages": packages,
        "tools": tools,
        "accelerator_evidence": {
            "rocm_smi": command_output(["rocm-smi", "--showproductname"]),
            "nvidia_smi": command_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON to this path instead of stdout")
    args = parser.parse_args()
    payload = json.dumps(collect(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
