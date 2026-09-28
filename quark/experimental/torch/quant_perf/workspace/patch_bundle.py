#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Export terminal, validated repository changes as portable patch artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.session.state import SessionState


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=False,
        timeout=60,
    )


def _text(repo: Path, *args: str) -> str:
    result = _git(repo, *args)
    return result.stdout.decode(errors="replace").strip() if result.returncode == 0 else ""


def export_patch_bundle(
    session_dir: str | Path,
    state: SessionState,
) -> dict[str, str]:
    output = Path(session_dir).resolve() / "reports" / "patches"
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "session_id": state.get("session_id"),
        "generated_at": datetime.now(UTC).isoformat(),
        "roles": {},
        "runtime_origin_evidence": dict(state.get("runtime_origin_evidence") or {}),
    }
    paths: dict[str, str] = {}
    resolved_sources = state.get("resolved_sources") or {}
    for role, workspace in (state.get("repo_workspaces") or {}).items():
        repo = Path(str(workspace.get("source_repo") or ""))
        base_sha = str(workspace.get("base_sha") or "")
        final_sha = str(workspace.get("final_sha") or base_sha)
        row = {
            "source_kind": (resolved_sources.get(role) or {}).get("kind", ""),
            "source_repo": str(repo),
            "base_sha": base_sha,
            "final_sha": final_sha,
            "work_branch": workspace.get("work_branch", ""),
            "branch_retained": bool(workspace.get("branch_retained")),
            "changed": False,
            "changed_files": [],
            "commits": [],
            "patch": "",
            "patch_sha256": "",
        }
        manifest["roles"][role] = row
        if not repo.is_dir() or not base_sha or not final_sha or base_sha == final_sha:
            continue
        diff = _git(
            repo,
            "diff",
            "--binary",
            "--full-index",
            f"{base_sha}..{final_sha}",
        )
        if diff.returncode != 0 or not diff.stdout.strip():
            row["error"] = diff.stderr.decode(errors="replace").strip() or "empty final diff"
            continue
        patch_path = output / f"{role}.diff"
        patch_path.write_bytes(diff.stdout)
        changed_files = _text(
            repo,
            "diff",
            "--name-only",
            f"{base_sha}..{final_sha}",
        ).splitlines()
        commits = _text(
            repo,
            "log",
            "--format=%H%x09%s",
            f"{base_sha}..{final_sha}",
        ).splitlines()
        row.update(
            {
                "changed": True,
                "changed_files": changed_files,
                "commits": [
                    {
                        "sha": line.split("\t", 1)[0],
                        "subject": (line.split("\t", 1)[1] if "\t" in line else ""),
                    }
                    for line in commits
                    if line
                ],
                "patch": str(patch_path),
                "patch_sha256": hashlib.sha256(diff.stdout).hexdigest(),
            }
        )
        paths[role] = str(patch_path)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    paths["manifest"] = str(manifest_path)
    return paths
