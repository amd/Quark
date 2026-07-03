#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Environment collection utilities for Quark.

This module provides classes and functions to collect hardware and software
environment information. It is shipped with the quark wheel package and can
be imported directly.

For CLI usage, see tools/ci/collect_env.py.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

import torch

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class HardwareInfoCollector:
    def __init__(self) -> None:
        self.accelerator_type: str | None = self._get_accelerator_type()
        self.gpu_models: str | None = self._get_gpu_models()
        self.cpu_model: str | None = self._get_cpu_model()
        self.cuda_version: str | None = self._get_cuda_version()
        self.rocm_version: str | None = self._get_rocm_version()

    def _get_accelerator_type(self) -> str | None:
        if shutil.which("rocminfo"):
            try:
                out = subprocess.check_output(["rocminfo"], stderr=subprocess.DEVNULL, text=True)
                for line in out.splitlines():
                    if "Marketing Name" in line and "Instinct" in line:
                        return line.split("Marketing Name:")[1].strip()
            except Exception:
                pass
        if shutil.which("nvidia-smi"):
            return "NVIDIA GPU"
        return None

    def _get_gpu_models(self) -> str | None:
        if shutil.which("rocminfo"):
            try:
                out = subprocess.check_output(["rocminfo"], stderr=subprocess.DEVNULL, text=True)
                models = []
                for line in out.splitlines():
                    if "Name:" in line and "gfx" in line:
                        models.append(line.split("Name:")[1].strip())
                return "/".join(models) if models else None
            except Exception:
                pass
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
                return "/".join([line.strip() for line in out.splitlines()]) if out else None  # pragma: no cover
            except Exception:  # pragma: no cover
                pass
        return None

    def _get_cpu_model(self) -> str | None:
        try:
            if shutil.which("lscpu"):
                result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if "Model name:" in line:
                            return " ".join(line.split("Model name:")[1].split())
        except Exception:
            return platform.processor()
        return platform.processor()

    def _get_cuda_version(self) -> str | None:
        if hasattr(torch.version, "cuda") and torch.version.cuda:
            return torch.version.cuda
        return None

    def _get_rocm_version(self) -> str | None:
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        version_files = [
            os.path.join(rocm_path, ".info", "version"),
            os.path.join(rocm_path, "version.txt"),
            os.path.join(rocm_path, ".info", "version-dev"),
        ]
        for f in version_files:
            if os.path.exists(f):
                try:  # pragma: no cover
                    with open(f) as fh:
                        version = fh.read().strip()
                    if version:
                        return version
                except Exception:
                    continue
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accelerator_type": self.accelerator_type,
            "gpu_models": self.gpu_models,
            "cpu_model": self.cpu_model,
            "cuda_version": self.cuda_version,
            "rocm_version": self.rocm_version,
        }


class SoftwareInfoCollector:
    def __init__(self) -> None:
        self.os: str = self._get_os()
        self.python_version: str = self._get_python_version()
        self.cuda_driver_version: str | None = self._get_cuda_driver_version()

    def _get_os(self) -> str:
        try:
            if shutil.which("lsb_release"):
                out = subprocess.run(["lsb_release", "-ds"], capture_output=True, text=True)
                if out.returncode == 0:
                    return out.stdout.strip().strip('"')
        except Exception:
            pass
        try:
            with open("/etc/os-release") as f:
                info = {k: v.strip('"') for k, v in (line.strip().split("=", 1) for line in f if "=" in line)}
                if "PRETTY_NAME" in info:
                    return info["PRETTY_NAME"]
        except Exception:
            pass
        return platform.platform()

    def _get_python_version(self) -> str:
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _get_cuda_driver_version(self) -> str | None:
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                if out:
                    return out.splitlines()[0]
            except Exception:
                return None
        return None

    def parse_pip_freeze(self) -> dict[str, dict[str, str]]:
        if not shutil.which("pip"):
            return {}
        lines = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
        ).stdout.splitlines()
        packages: dict[str, dict[str, str]] = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "==" in line:
                name, version = line.split("==", 1)
                packages[name] = {"version": version}
        return packages

    def parse_conda_list(self) -> dict[str, dict[str, str]]:
        if not shutil.which("conda"):
            return {}
        lines = subprocess.run(["conda", "list"], capture_output=True, text=True).stdout.splitlines()
        packages: dict[str, dict[str, str]] = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                packages[parts[0]] = {"version": parts[1]}
        return packages

    def collect_packages(self) -> dict[str, dict[str, str]]:
        packages = self.parse_pip_freeze()
        packages.update(self.parse_conda_list())
        return packages

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "os": self.os,
            "python": self.python_version,
            "cuda_driver_version": self.cuda_driver_version,
        }

        packages = self.collect_packages()
        for name, info in packages.items():
            if "version" in info:
                data[name] = info["version"]

        return data


def collect_environment() -> dict[str, Any]:
    """Collect both hardware and software environment information."""
    hardware = HardwareInfoCollector()
    software = SoftwareInfoCollector()

    env_dict = {
        "hardware": hardware.to_dict(),
        "software": software.to_dict(),
    }
    return env_dict


def save_environment_to_json(output_path: str, env_dict: dict[str, Any], indent: int = 2) -> None:
    """
    Save the collected environment information to a JSON file.

    Args:
        output_path (str): Path to the output JSON file.
        env_dict (dict): The environment dictionary, e.g., from collect_environment().
        indent (int): JSON indentation for readability.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(env_dict, f, indent=indent)
        logger.info(f"Environment saved to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save environment to JSON: {e}") from e


def print_environment(env_dict: dict[str, Any]) -> None:
    """Print environment information in human-readable format."""
    print("Collecting environment information...\n")

    print("=" * 60)
    print("Hardware Information")
    print("=" * 60)
    for key, value in env_dict.get("hardware", {}).items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Software Information")
    print("=" * 60)
    software = env_dict.get("software", {})

    # Print key software versions first
    priority_keys = ["os", "python", "torch", "amd-quark", "cuda_driver_version"]
    for key in priority_keys:
        if key in software:
            print(f"  {key}: {software[key]}")

    # Print other packages
    print("\n  Installed packages:")
    for key, value in sorted(software.items()):
        if key not in priority_keys and key != "cuda_driver_version":
            print(f"    {key}: {value}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect hardware and software environment information")
    parser.add_argument(
        "--json",
        type=str,
        metavar="FILE",
        help="Save collected environment info to a JSON file",
    )
    # Keep old argument for backward compatibility
    parser.add_argument(
        "--save_environment_to_json",
        type=str,
        metavar="FILE",
        help=argparse.SUPPRESS,  # Hidden, for backward compatibility
    )
    args = parser.parse_args()

    env = collect_environment()

    # Handle both --json and --save_environment_to_json for backward compatibility
    json_output = args.json or args.save_environment_to_json

    if json_output:
        save_environment_to_json(json_output, env)
    else:
        print_environment(env)
