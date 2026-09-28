#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..testing import init_git_repo
from ..testing import run_git as _git


def _repo(tmp_path: Path, name: str) -> tuple[Path, str, str]:
    repo = init_git_repo(tmp_path / name, {"module.py": "VALUE = 1\n"})
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "module.py").write_text("VALUE = 2\n")
    _git(repo, "commit", "-am", "optimized")
    final = _git(repo, "rev-parse", "HEAD")
    return repo, base, final


def test_export_patch_bundle_writes_role_diffs_and_manifest(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.patch_bundle import export_patch_bundle

    framework, fw_base, fw_final = _repo(tmp_path, "framework")
    kernel, kernel_base, kernel_final = _repo(tmp_path, "kernel")
    state = {
        "session_id": "session-1",
        "repo_workspaces": {
            "framework": {
                "source_repo": str(framework),
                "base_sha": fw_base,
                "final_sha": fw_final,
                "work_branch": "quark-quant-perf-session",
                "branch_retained": True,
            },
            "kernel": {
                "source_repo": str(kernel),
                "base_sha": kernel_base,
                "final_sha": kernel_final,
                "work_branch": "quark-quant-perf-session",
                "branch_retained": True,
            },
        },
        "resolved_sources": {
            "framework": {"kind": "explicit_git"},
            "kernel": {"kind": "explicit_git"},
        },
        "runtime_origin_evidence": {"vllm": {"matched": True}},
    }

    result = export_patch_bundle(tmp_path / "session", state)

    manifest = json.loads(Path(result["manifest"]).read_text())
    assert Path(result["framework"]).is_file()
    assert Path(result["kernel"]).is_file()
    assert "VALUE = 2" in Path(result["framework"]).read_text()
    assert manifest["roles"]["framework"]["base_sha"] == fw_base
    assert manifest["roles"]["kernel"]["final_sha"] == kernel_final
    patch = Path(result["framework"]).read_bytes()
    assert manifest["roles"]["framework"]["patch_sha256"] == (hashlib.sha256(patch).hexdigest())


def test_export_patch_bundle_skips_unchanged_role(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.patch_bundle import export_patch_bundle

    repo, base, _final = _repo(tmp_path, "framework")
    state = {
        "session_id": "session-1",
        "repo_workspaces": {
            "framework": {
                "source_repo": str(repo),
                "base_sha": base,
                "final_sha": base,
                "work_branch": "quark-quant-perf-session",
                "branch_retained": False,
            }
        },
    }

    result = export_patch_bundle(tmp_path / "session", state)

    manifest = json.loads(Path(result["manifest"]).read_text())
    assert "framework" not in result
    assert manifest["roles"]["framework"]["changed"] is False
