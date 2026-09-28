#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Ownership-marker based garbage collection for Quark Quant-Perf worktrees."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MARKER_NAME = ".quark-quant-perf-owned.json"
_MANAGED_BRANCH_PREFIX = "quark-quant-perf-"
_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40,64}$")


@dataclass(frozen=True)
class _WorkspaceMarker:
    """Validated filesystem fields loaded from an ownership marker."""

    path: Path
    worktree: Path
    source_repo: Path | None
    resource_type: str
    work_branch: str
    base_sha: str
    owner_pid: int


@dataclass(frozen=True)
class _RegisteredWorktree:
    """Git-owned metadata for one registered worktree."""

    branch: str | None


@dataclass(frozen=True)
class _ManagedBranch:
    """Git-verified managed branch eligible for empty-branch cleanup."""

    name: str
    base_sha: str


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _prune_empty(path: Path, root: Path) -> None:
    current = path
    while _within(current, root):
        try:
            current.rmdir()
        except OSError:
            return
        if current.resolve() == root.resolve():
            return
        current = current.parent


def _load_marker(marker: Path, root: Path) -> _WorkspaceMarker:
    """Load and validate marker-controlled filesystem paths.

    :param marker: Ownership marker path.
    :param root: Allowed garbage-collection root.
    :return: Validated marker values.
    :raises ValueError: If the marker is untrusted or inconsistent.
    """
    if marker.is_symlink():
        raise ValueError("refusing symlink ownership marker")
    if hasattr(os, "geteuid") and marker.lstat().st_uid != os.geteuid():
        raise ValueError("refusing ownership marker owned by another user")

    data = json.loads(marker.read_text())
    if not isinstance(data, dict):
        raise ValueError("ownership marker must contain a JSON object")

    raw_worktree = Path(str(data["worktree_path"]))
    if not raw_worktree.is_absolute():
        raise ValueError("worktree path must be absolute")
    worktree = raw_worktree.resolve()
    if not _within(worktree, root):
        raise ValueError(f"refusing path outside GC root: {worktree}")
    if marker.parent.resolve() != worktree.parent:
        raise ValueError(f"ownership marker does not describe its sibling worktree: {worktree}")

    source_value = str(data.get("source_repo") or "").strip()
    source_repo = None
    if source_value:
        raw_source = Path(source_value)
        if not raw_source.is_absolute():
            raise ValueError("source repository path must be absolute")
        source_repo = raw_source.resolve()

    return _WorkspaceMarker(
        path=marker,
        worktree=worktree,
        source_repo=source_repo,
        resource_type=str(data.get("resource_type") or ""),
        work_branch=str(data.get("work_branch") or ""),
        base_sha=str(data.get("base_sha") or ""),
        owner_pid=int(data.get("owner_pid") or 0),
    )


def _registered_worktree(source_repo: Path, worktree: Path) -> _RegisteredWorktree | None:
    """Return Git metadata when a repository registers the exact worktree.

    :param source_repo: Repository claimed by the ownership marker.
    :param worktree: Worktree path to find.
    :return: Registered worktree metadata, or ``None`` when unverified.
    """
    if not source_repo.is_dir() or not (source_repo / ".git").exists():
        return None
    listed = _git(source_repo, "worktree", "list", "--porcelain")
    if listed.returncode != 0:
        return None

    record: dict[str, str] = {}
    for line in [*listed.stdout.splitlines(), ""]:
        if line:
            key, _, value = line.partition(" ")
            record[key] = value
            continue
        registered_path = record.get("worktree")
        if registered_path and Path(registered_path).resolve() == worktree:
            branch_ref = record.get("branch", "")
            prefix = "refs/heads/"
            branch = branch_ref[len(prefix) :] if branch_ref.startswith(prefix) else None
            return _RegisteredWorktree(branch=branch)
        record = {}
    return None


def _managed_branch(
    source_repo: Path,
    marker: _WorkspaceMarker,
    registered: _RegisteredWorktree,
) -> _ManagedBranch | None:
    """Validate marker branch metadata against Git-owned state.

    :param source_repo: Repository that registers the worktree.
    :param marker: Validated ownership marker.
    :param registered: Git metadata for the marker worktree.
    :return: Verified managed branch, or ``None`` when not applicable.
    :raises ValueError: If marker branch metadata conflicts with Git.
    """
    if marker.resource_type != "integration_worktree" or not marker.work_branch or not marker.base_sha:
        return None
    if registered.branch != marker.work_branch:
        raise ValueError(
            f"branch mismatch for {marker.worktree}: marker={marker.work_branch!r}, registered={registered.branch!r}"
        )
    if not marker.work_branch.startswith(_MANAGED_BRANCH_PREFIX):
        raise ValueError(f"refusing unmanaged branch: {marker.work_branch}")
    if not _COMMIT_SHA.fullmatch(marker.base_sha):
        raise ValueError(f"invalid base SHA for {marker.worktree}")

    reflog = _git(
        source_repo,
        "reflog",
        "show",
        "--format=%H",
        f"refs/heads/{marker.work_branch}",
    )
    reflog_entries = [line.strip() for line in reflog.stdout.splitlines() if line.strip()]
    creation_sha = reflog_entries[-1] if reflog_entries else ""
    if reflog.returncode != 0 or not creation_sha:
        raise ValueError(f"cannot verify branch creation point: {marker.work_branch}")
    if marker.base_sha != creation_sha:
        raise ValueError(f"base SHA mismatch for {marker.worktree}: marker={marker.base_sha!r}, git={creation_sha!r}")
    return _ManagedBranch(name=marker.work_branch, base_sha=creation_sha)


def _delete_empty_branch(source_repo: Path, branch: _ManagedBranch) -> bool:
    """Delete a managed branch only when its tree still matches its base.

    :param source_repo: Repository containing the branch.
    :param branch: Git-verified managed branch.
    :return: Whether the branch was deleted.
    """
    branch_tree = _git(
        source_repo,
        "rev-parse",
        "--verify",
        f"refs/heads/{branch.name}^{{tree}}",
    )
    base_tree = _git(
        source_repo,
        "rev-parse",
        "--verify",
        f"{branch.base_sha}^{{tree}}",
    )
    if (
        branch_tree.returncode != 0
        or base_tree.returncode != 0
        or branch_tree.stdout.strip() != base_tree.stdout.strip()
    ):
        return False
    deleted = _git(
        source_repo,
        "branch",
        "-D",
        "--",
        branch.name,
    )
    return deleted.returncode == 0


def gc_workspaces(
    roots: list[str | Path],
    *,
    dry_run: bool = True,
    remove_empty_branches: bool = False,
    max_age_hours: float = 24.0,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "roots": [str(Path(root).resolve()) for root in roots],
        "scanned": 0,
        "candidates": 0,
        "removed": [],
        "branches_removed": [],
        "skipped_live": [],
        "skipped_fresh": [],
        "errors": [],
    }
    cutoff = time.time() - max_age_hours * 3600.0
    seen: set[str] = set()
    for raw_root in roots:
        root = Path(raw_root).resolve()
        if not root.is_dir():
            continue
        for marker_path in root.rglob(_MARKER_NAME):
            result["scanned"] += 1
            try:
                marker = _load_marker(marker_path, root)
            except (OSError, ValueError, KeyError, TypeError) as error:
                result["errors"].append(f"{marker_path}: {error}")
                continue
            if str(marker.worktree) in seen:
                continue
            seen.add(str(marker.worktree))
            if _pid_alive(marker.owner_pid):
                result["skipped_live"].append(str(marker.worktree))
                continue
            try:
                fresh = marker.path.stat().st_mtime > cutoff
            except OSError:
                fresh = False
            if fresh:
                result["skipped_fresh"].append(str(marker.worktree))
                continue
            result["candidates"] += 1
            if dry_run:
                continue
            try:
                registered = (
                    _registered_worktree(marker.source_repo, marker.worktree)
                    if marker.source_repo is not None
                    else None
                )
                managed_branch = None
                if marker.source_repo is not None and marker.source_repo.is_dir() and registered is None:
                    result["errors"].append(
                        f"{marker.path}: worktree is not registered by source repo: {marker.source_repo}"
                    )
                if remove_empty_branches and registered is not None and marker.source_repo is not None:
                    try:
                        managed_branch = _managed_branch(marker.source_repo, marker, registered)
                    except ValueError as error:
                        result["errors"].append(f"{marker.path}: {error}")

                if registered is not None and marker.source_repo is not None:
                    removed = _git(
                        marker.source_repo,
                        "worktree",
                        "remove",
                        "--force",
                        str(marker.worktree),
                    )
                    if removed.returncode != 0 and marker.worktree.exists():
                        shutil.rmtree(marker.worktree)
                    _git(marker.source_repo, "worktree", "prune")
                elif marker.worktree.exists():
                    shutil.rmtree(marker.worktree)
                marker.path.unlink(missing_ok=True)
                result["removed"].append(str(marker.worktree))

                if (
                    managed_branch is not None
                    and marker.source_repo is not None
                    and _delete_empty_branch(marker.source_repo, managed_branch)
                ):
                    result["branches_removed"].append(managed_branch.name)
                _prune_empty(marker.path.parent, root)
            except Exception as error:
                result["errors"].append(f"{marker.worktree}: {type(error).__name__}: {error}")
    return result
