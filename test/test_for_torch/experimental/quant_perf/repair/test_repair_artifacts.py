#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from pathlib import Path

from ..testing import init_git_repo
from ..testing import run_git as _git


def test_export_patch_prefers_session_artifact_directory(tmp_path):
    from quark.experimental.torch.quant_perf.repair.artifacts import export_patch

    repo = init_git_repo(tmp_path / "repo", {"module.py": "VALUE = 1\n"})
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "module.py").write_text("VALUE = 2\n")
    _git(repo, "commit", "-am", "fix")

    patch, files = export_patch(
        str(repo),
        base,
        session_dir=str(tmp_path / "session"),
        label="accuracy-repair",
    )

    path = Path(patch)
    assert path.is_file()
    assert path.parent == (tmp_path / "session" / "repairs" / "accuracy-repair")
    assert files == ["module.py"]
