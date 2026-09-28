#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..testing import run_git as _git


def _marker(
    root: Path,
    *,
    source_repo: Path,
    worktree: Path,
    resource_type: str = "integration_worktree",
    owner_pid: int = 0,
    work_branch: str = "",
    base_sha: str = "",
) -> Path:
    worktree.mkdir(parents=True, exist_ok=True)
    marker = worktree.parent / ".quark-quant-perf-owned.json"
    marker.write_text(
        json.dumps(
            {
                "session_id": "session",
                "resource_type": resource_type,
                "worktree_path": str(worktree),
                "source_repo": str(source_repo),
                "owner_pid": owner_pid,
                "work_branch": work_branch,
                "base_sha": base_sha,
            }
        )
    )
    old = time.time() - 48 * 3600
    os.utime(marker, (old, old))
    return marker


def _repo_with_base(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "file.py").write_text("VALUE = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_gc_dry_run_reports_orphan_without_deleting(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.gc import gc_workspaces

    root = tmp_path / "owned"
    worktree = root / "session" / "framework" / "integration" / "worktree"
    _marker(
        root,
        source_repo=tmp_path / "missing-source",
        worktree=worktree,
    )

    result = gc_workspaces([root], dry_run=True, max_age_hours=1)

    assert result["candidates"] == 1
    assert result["removed"] == []
    assert worktree.exists()


def test_gc_removes_owned_orphan_with_missing_source(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.gc import gc_workspaces

    root = tmp_path / "owned"
    worktree = root / "session" / "framework" / "integration" / "worktree"
    _marker(
        root,
        source_repo=tmp_path / "missing-source",
        worktree=worktree,
    )

    result = gc_workspaces([root], dry_run=False, max_age_hours=1)

    assert str(worktree) in result["removed"]
    assert not worktree.exists()


def test_gc_keeps_candidate_with_live_owner(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.gc import gc_workspaces

    root = tmp_path / "owned"
    worktree = root / "session" / "kernel" / "candidates" / "k" / "worktree"
    _marker(
        root,
        source_repo=tmp_path / "missing-source",
        worktree=worktree,
        resource_type="candidate_worktree",
        owner_pid=os.getpid(),
    )

    result = gc_workspaces([root], dry_run=False, max_age_hours=1)

    assert result["skipped_live"] == [str(worktree)]
    assert worktree.exists()


def test_gc_keeps_integration_worktree_with_live_owner(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.gc import gc_workspaces

    root = tmp_path / "owned"
    worktree = root / "session" / "framework" / "integration" / "worktree"
    _marker(
        root,
        source_repo=tmp_path / "missing-source",
        worktree=worktree,
        resource_type="integration_worktree",
        owner_pid=os.getpid(),
    )

    result = gc_workspaces([root], dry_run=False, max_age_hours=1)

    assert result["skipped_live"] == [str(worktree)]
    assert worktree.exists()


def test_gc_deletes_empty_owned_branch_when_requested(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.gc import gc_workspaces

    repo, base = _repo_with_base(tmp_path)
    branch = "quark-quant-perf-empty"
    _git(repo, "branch", branch, base)
    root = tmp_path / "owned"
    worktree = root / "session" / "framework" / "integration" / "worktree"
    _git(repo, "worktree", "add", str(worktree), branch)
    _marker(
        root,
        source_repo=repo,
        worktree=worktree,
        work_branch=branch,
        base_sha=base,
    )

    result = gc_workspaces(
        [root],
        dry_run=False,
        remove_empty_branches=True,
        max_age_hours=1,
    )

    assert str(worktree) in result["removed"]
    assert branch in result["branches_removed"]
    assert (
        branch
        not in _git(
            repo,
            "branch",
            "--format=%(refname:short)",
        ).splitlines()
    )


def test_gc_refuses_branch_cleanup_for_unregistered_worktree(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.gc import gc_workspaces

    repo, base = _repo_with_base(tmp_path)
    branch = "quark-quant-perf-unrelated"
    _git(repo, "branch", branch, base)

    root = tmp_path / "owned"
    worktree = root / "session" / "framework" / "integration" / "worktree"
    _marker(
        root,
        source_repo=repo,
        worktree=worktree,
        work_branch=branch,
        base_sha=base,
    )

    result = gc_workspaces(
        [root],
        dry_run=False,
        remove_empty_branches=True,
        max_age_hours=1,
    )

    assert branch in _git(repo, "branch", "--format=%(refname:short)").splitlines()
    assert result["branches_removed"] == []
    assert any("not registered" in error for error in result["errors"])


def test_gc_refuses_marker_branch_that_differs_from_registered_branch(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.gc import gc_workspaces

    repo, base = _repo_with_base(tmp_path)
    registered_branch = "quark-quant-perf-registered"
    unrelated_branch = "quark-quant-perf-unrelated"
    _git(repo, "branch", registered_branch, base)
    _git(repo, "branch", unrelated_branch, base)

    root = tmp_path / "owned"
    worktree = root / "session" / "framework" / "integration" / "worktree"
    _git(repo, "worktree", "add", str(worktree), registered_branch)
    _marker(
        root,
        source_repo=repo,
        worktree=worktree,
        work_branch=unrelated_branch,
        base_sha=base,
    )

    result = gc_workspaces(
        [root],
        dry_run=False,
        remove_empty_branches=True,
        max_age_hours=1,
    )

    branches = _git(repo, "branch", "--format=%(refname:short)").splitlines()
    assert registered_branch in branches
    assert unrelated_branch in branches
    assert result["branches_removed"] == []
    assert any("branch mismatch" in error for error in result["errors"])


def test_gc_refuses_marker_base_that_differs_from_branch_creation_point(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.gc import gc_workspaces

    repo, base = _repo_with_base(tmp_path)
    branch = "quark-quant-perf-changed"
    _git(repo, "branch", branch, base)

    root = tmp_path / "owned"
    worktree = root / "session" / "framework" / "integration" / "worktree"
    _git(repo, "worktree", "add", str(worktree), branch)
    (worktree / "file.py").write_text("VALUE = 2\n")
    _git(worktree, "add", ".")
    _git(worktree, "commit", "-m", "change")
    forged_base = _git(worktree, "rev-parse", "HEAD")
    _marker(
        root,
        source_repo=repo,
        worktree=worktree,
        work_branch=branch,
        base_sha=forged_base,
    )

    result = gc_workspaces(
        [root],
        dry_run=False,
        remove_empty_branches=True,
        max_age_hours=1,
    )

    assert branch in _git(repo, "branch", "--format=%(refname:short)").splitlines()
    assert result["branches_removed"] == []
    assert any("base SHA mismatch" in error for error in result["errors"])
