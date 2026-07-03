#!/usr/bin/env python3
#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Script to detect extension_subdir based on Python version and PyTorch GPU support
# Returns extension_subdir in the format: py<major><minor>_cu<cuda_major><cuda_minor> or py<major><minor>_cpu

import sys

import torch

from quark.common.utils.import_utils import is_package_lower_or_equal


def detect_pytorch_extension_dir() -> str:
    """Detect PyTorch extension subdirectory based on Python version and GPU support.

    Mirrors the naming logic of ``torch.utils.cpp_extension._get_build_directory``.

    For torch <= 2.10: only CUDA is recognized, HIP falls back to CPU.
    For torch >= 2.11: HIP takes priority and uses ``rocm{version_no_dots}``.

    Returns:
        Extension subdirectory string in format:
        - py<major><minor>_cu<cuda_major><cuda_minor> for CUDA
        - py<major><minor>_rocm<version_no_dots> for HIP (torch >= 2.11)
        - py<major><minor>_cpu for CPU-only or HIP on torch <= 2.10

    Examples:
        >>> detect_pytorch_extension_dir()  # Python 3.11 with CUDA 12.1
        'py311_cu121'
        >>> detect_pytorch_extension_dir()  # Python 3.11 with HIP 6.3.0, torch >= 2.11
        'py311_rocm630'
        >>> detect_pytorch_extension_dir()  # Python 3.10 CPU-only
        'py310_cpu'
    """

    py_ver = sys.version_info
    py_str = f"{py_ver.major}{py_ver.minor}"

    cuda_ver = None
    if hasattr(torch.version, "cuda") and torch.version.cuda and torch.version.cuda != "":
        cuda_ver = torch.version.cuda

    hip_ver = None
    if hasattr(torch.version, "hip") and torch.version.hip and torch.version.hip != "":
        hip_ver = torch.version.hip

    old_torch = is_package_lower_or_equal("torch", "2.10.99")

    if old_torch:
        if cuda_ver is not None:
            parts = cuda_ver.split(".")
            if len(parts) >= 2:
                ext_subdir = f"py{py_str}_cu{parts[0]}{parts[1]}"
            else:
                ext_subdir = f"py{py_str}_cpu"
        else:
            ext_subdir = f"py{py_str}_cpu"
    else:
        if hip_ver is not None:
            ext_subdir = f"py{py_str}_rocm{hip_ver.replace('.', '')}"
        elif cuda_ver is not None:
            parts = cuda_ver.split(".")
            if len(parts) >= 2:
                ext_subdir = f"py{py_str}_cu{parts[0]}{parts[1]}"
            else:
                ext_subdir = f"py{py_str}_cpu"
        else:
            ext_subdir = f"py{py_str}_cpu"

    return ext_subdir


def main() -> None:
    """Main entry point when script is run directly."""
    ext_subdir = detect_pytorch_extension_dir()
    print(ext_subdir)


if __name__ == "__main__":
    main()
