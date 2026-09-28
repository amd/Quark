#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path

from quark.experimental.torch.quant_perf import config


def patch_dir() -> Path:
    path = Path(config.experience_store_path()).parent / "repair_patches"
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_patch(
    framework_repo: str,
    baseline_sha: str,
    *,
    session_dir: str = "",
    label: str = "repair",
) -> tuple[str, list[str]]:
    try:
        diff = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "--full-index",
                "--no-renames",
                f"{baseline_sha}..HEAD",
            ],
            cwd=framework_repo,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
        files = subprocess.run(
            ["git", "diff", "--no-renames", "--name-only", "-z", f"{baseline_sha}..HEAD"],
            cwd=framework_repo,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return "", []
    if not diff.strip():
        return "", []
    if session_dir:
        safe_label = (
            re.sub(
                r"[^A-Za-z0-9._-]+",
                "-",
                label,
            ).strip("-")
            or "repair"
        )
        target_dir = Path(session_dir) / "repairs" / safe_label
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = patch_dir()
    path = target_dir / f"{uuid.uuid4().hex}.diff"
    path.write_text(diff)
    return str(path), [name for name in files.split("\0") if name]
