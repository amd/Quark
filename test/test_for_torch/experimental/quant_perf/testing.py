#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Shared helpers for Quant-Perf unit tests."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path


def run_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def init_git_repo(
    repo: Path,
    files: Mapping[str, str],
    *,
    message: str = "base",
) -> Path:
    repo.mkdir(parents=True)
    run_git(repo, "init", "-b", "main")
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "user.name", "Test User")
    for relative_path, content in files.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-m", message)
    return repo
