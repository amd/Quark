#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Shared utilities for vLLM plugin scripts."""

from pathlib import Path


def find_repo_root() -> str:
    """Locate Quark repo root (pyproject.toml + quark/) by walking up from this file."""
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "pyproject.toml").exists() and (p / "quark").is_dir():
            return str(p)
        p = p.parent
    raise RuntimeError("Could not find Quark repo root (pyproject.toml + quark/)")
