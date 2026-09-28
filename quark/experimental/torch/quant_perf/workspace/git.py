#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Shared git helper utilities used by orchestration and repair."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def find_git_root(path: Path | None) -> Path | None:
    """Return the nearest parent Git worktree, accepting either files or directories."""
    if path is None:
        return None
    current = path.resolve()
    if current.is_file():
        current = current.parent
    return next((candidate for candidate in (current, *current.parents) if (candidate / ".git").exists()), None)


def get_current_branch(repo: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def get_head_sha(repo: str) -> str:
    """Current HEAD commit SHA (no new commit created). Used as the revert
    point for the greedy per-patch apply/validate loop."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def source_only_git_env(repo: str, source_dir: str, run_dir: str) -> dict[str, str]:
    """Keep generated files out of Git diffs in copies of an existing source tree.

    Freeze the inventory at HEAD before optimization, not at the mutable index.
    Files remain available to builds; this only controls automatic Git staging.
    Explicit additions still require the caller's normal patch validation.
    """
    source_root = Path(source_dir).resolve()
    relative_dir = source_root.relative_to(Path(repo).resolve())
    baseline = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    inventory = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "-z", baseline, "--", f":(literal){relative_dir}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    paths = [Path(name).relative_to(relative_dir).as_posix() for name in inventory.split("\0") if name]
    if not paths or any("\n" in name or "\r" in name for name in paths):
        raise ValueError("Source-only Git capture requires a nonempty, line-safe committed file inventory")
    # Ignore new files while traversing every directory to admit original files.
    patterns = ["*", "!*/", *("!/" + re.sub(r"([\\*?\[\] ])", r"\\\1", name) for name in paths)]
    excludes = Path(run_dir).resolve() / "source-only.gitignore"
    excludes.parent.mkdir(parents=True, exist_ok=True)
    excludes.write_text(f"# Source baseline: {baseline}\n" + "\n".join(patterns) + "\n")
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    env = {
        name: os.environ[name]
        for index in range(count)
        for name in (f"GIT_CONFIG_KEY_{index}", f"GIT_CONFIG_VALUE_{index}")
    }
    env.update(
        GIT_CONFIG_COUNT=str(count + 1),
        **{f"GIT_CONFIG_KEY_{count}": "core.excludesFile", f"GIT_CONFIG_VALUE_{count}": str(excludes)},
    )
    return env


def worktree_dirty(repo: str, pathspec: str = "") -> bool:
    """True if the working tree has any changes (tracked or untracked) under
    `pathspec` (whole repo if empty). Used to detect GEAK's in-place pollution."""
    cmd = ["git", "status", "--porcelain"]
    if pathspec:
        cmd.append(pathspec)
    result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    return bool(result.stdout.strip())


def clean_untracked(repo: str, pathspec: str = "") -> None:
    """Remove untracked files/dirs (git clean -fd), scoped to `pathspec` when
    given. `git reset --hard` restores tracked files but never removes untracked
    ones (e.g. hipify-generated headers), so this complements it."""
    cmd = ["git", "clean", "-fd"]
    if pathspec:
        cmd.append(pathspec)
    subprocess.run(cmd, cwd=repo, capture_output=True, text=True)


def commit_changes(repo: str, message: str) -> bool:
    """Stage all changes and commit. Returns True if commit succeeded."""
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def commit_selected_changes(
    repo: str,
    paths: list[str] | tuple[str, ...],
    message: str,
) -> bool:
    """Commit only the explicitly authorized tracked source paths."""
    selected = tuple(dict.fromkeys(path for path in paths if path))
    if not selected:
        return False
    reset = subprocess.run(
        ["git", "reset", "--quiet"],
        cwd=repo,
        capture_output=True,
    )
    if reset.returncode != 0:
        return False
    staged = subprocess.run(
        ["git", "--literal-pathspecs", "add", "--", *selected],
        cwd=repo,
        capture_output=True,
    )
    if staged.returncode != 0:
        return False
    diff = subprocess.run(
        ["git", "--literal-pathspecs", "diff", "--cached", "--quiet", "--", *selected],
        cwd=repo,
        capture_output=True,
    )
    if diff.returncode != 1:
        return False
    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def has_uncommitted_changes(repo: str) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def commit_baseline(repo: str, message: str) -> str:
    """Freeze the current working tree (all changes) into a baseline commit and
    return its SHA. Used by repair to establish a rollback point that INCLUDES any
    base fixes already applied to the repo, so per-round reset only discards the
    agent's own trial changes -- never the base fixes Quark Quant-Perf depends on.
    Uses --allow-empty so a clean tree still yields a valid baseline SHA."""
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        cwd=repo,
        capture_output=True,
    )
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


def reset_hard_to(repo: str, sha: str) -> None:
    """Hard-reset to a specific commit and remove untracked files. Unlike
    `git checkout -- .` (which reverts to repo HEAD and cannot preserve an
    intermediate baseline), this rolls the tree back to exactly `sha`,
    keeping everything committed at or before that point intact."""
    subprocess.run(["git", "reset", "--hard", sha], cwd=repo, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=repo, capture_output=True)


def apply_patch(repo: str, patch_path: str, check_only: bool = False) -> bool:
    cmd = ["git", "apply"]
    if check_only:
        cmd.append("--check")
    cmd.append(patch_path)
    result = subprocess.run(cmd, cwd=repo, capture_output=True)
    return result.returncode == 0


def cherry_pick_commits(repo: str, commits: tuple[str, ...]) -> bool:
    result = subprocess.run(
        ["git", "cherry-pick", *commits],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    subprocess.run(
        ["git", "cherry-pick", "--abort"],
        cwd=repo,
        capture_output=True,
    )
    return False
