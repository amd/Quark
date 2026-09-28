#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Shared classification rules for managed source trees."""

from __future__ import annotations

import os
from pathlib import Path

_GENERATED_DIRECTORY_NAMES = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "CMakeFiles",
        "__pycache__",
        "_skbuild",
        "build",
    }
)

_GENERATED_FILE_SUFFIXES = frozenset(
    {
        ".d",
        ".log",
        ".ninja",
        ".o",
        ".obj",
        ".pyc",
        ".pyo",
        ".tmp",
    }
)

_COMPILED_ASSET_SUFFIXES = frozenset(
    {
        ".a",
        ".co",
        ".hsaco",
        ".so",
    }
)


def source_search_excluded_directory_names() -> tuple[str, ...]:
    return tuple(sorted(_GENERATED_DIRECTORY_NAMES))


def is_generated_workspace_path(path: str | Path) -> bool:
    candidate = Path(path)
    return any(part in _GENERATED_DIRECTORY_NAMES for part in candidate.parts) or (
        candidate.suffix.lower() in _GENERATED_FILE_SUFFIXES
    )


def is_runtime_generated_change(path: str | Path) -> bool:
    candidate = Path(path)
    parts = candidate.parts
    return is_generated_workspace_path(candidate) or (
        len(parts) == 3 and parts[:2] == ("aiter", "jit") and _has_shared_library_name(candidate.name)
    )


def is_searchable_source(path: str | Path, repo: str | Path) -> bool:
    candidate = Path(path).absolute()
    root = Path(repo).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return False
    return not is_generated_workspace_path(relative)


def _has_shared_library_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".so") or ".so." in lower


def is_compiled_runtime_asset(path: str | Path) -> bool:
    path = Path(path)
    if path.suffix.lower() in _COMPILED_ASSET_SUFFIXES or _has_shared_library_name(path.name):
        return True
    if path.suffix or not os.access(path, os.X_OK):
        return False
    try:
        with path.open("rb") as stream:
            return stream.read(4) == b"\x7fELF"
    except OSError:
        return False
