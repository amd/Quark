#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from quark.experimental.torch.quant_perf.session.spec import Checkpoint, Spec
from quark.experimental.torch.quant_perf.workspace.git import commit_selected_changes
from quark.experimental.torch.quant_perf.workspace.manager import (
    RepoWorkspaceManager,
    WorkspaceError,
)

from ..testing import init_git_repo
from ..testing import run_git as _git


def _repo(tmp_path: Path) -> Path:
    return init_git_repo(tmp_path / "repo", {"kernel.py": "VALUE = 1\n"})


def _ckpt() -> SimpleNamespace:
    return SimpleNamespace(
        state={"repo_workspaces": {}, "transient_resources": [], "cleanup": {}},
        save=lambda: None,
    )


def _manager(tmp_path: Path, ckpt=None) -> RepoWorkspaceManager:
    return RepoWorkspaceManager(
        session_dir=tmp_path / "session",
        session_id="session-12345678",
        ckpt=ckpt or _ckpt(),
        root_dir=tmp_path / "owned",
    )


def test_prepare_rejects_dirty_source_repo(tmp_path):
    repo = _repo(tmp_path)
    (repo / "local.txt").write_text("uncommitted\n")

    with pytest.raises(WorkspaceError, match="dirty"):
        _manager(tmp_path).prepare("framework", repo)

    assert _git(repo, "branch", "--show-current") == "main"
    assert (repo / "local.txt").read_text() == "uncommitted\n"


def test_commit_selected_changes_excludes_unrelated_files(tmp_path):
    repo = _repo(tmp_path)
    (repo / "kernel.py").write_text("VALUE = 2\n")
    unrelated = repo / "profile.log"
    unrelated.write_text("temporary profiling output\n")

    committed = commit_selected_changes(
        str(repo),
        ["kernel.py"],
        "Retain selected kernel source",
    )

    assert committed
    assert _git(repo, "show", "--name-only", "--format=") == "kernel.py"
    assert _git(repo, "status", "--porcelain") == "?? profile.log"


def test_prepare_ignores_untracked_codex_runtime_metadata(tmp_path):
    repo = _repo(tmp_path)
    (repo / ".codex").mkdir()
    (repo / ".codex" / "state.sqlite").write_text("runtime")

    workspace = _manager(tmp_path).prepare("framework", repo)

    assert workspace.integration_path.is_dir()
    assert (repo / ".codex" / "state.sqlite").read_text() == "runtime"


def test_prepare_allows_dirty_files_inside_clean_submodule(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    child = tmp_path / "child"
    child.mkdir()
    _git(child, "init", "-b", "main")
    _git(child, "config", "user.email", "test@example.com")
    _git(child, "config", "user.name", "Test User")
    (child / "tracked.txt").write_text("tracked\n")
    _git(child, "add", "tracked.txt")
    _git(child, "commit", "-m", "child")

    repo = _repo(tmp_path)
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(child),
            "deps/child",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo, "commit", "-am", "add submodule")
    (repo / "deps" / "child" / "generated.tmp").write_text("generated\n")

    workspace = _manager(tmp_path).prepare("framework", repo)

    assert workspace.integration_path.is_dir()
    assert (repo / "deps" / "child" / "generated.tmp").is_file()


def test_prepare_failure_removes_partial_worktree_and_branch(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)
    manager._init_submodules = lambda _path: (_ for _ in ()).throw(RuntimeError("submodule setup failed"))

    with pytest.raises(RuntimeError, match="submodule setup failed"):
        manager.prepare("framework", repo)

    assert "quark-quant-perf-session" not in _git(repo, "branch", "--format=%(refname:short)")
    assert not manager.root_dir.exists()


def test_prepare_can_skip_eager_submodule_initialization(tmp_path):
    repo = _repo(tmp_path)
    (repo / ".gitmodules").write_text(
        '[submodule "deps/large"]\n\tpath = deps/large\n\turl = https://example.invalid/large.git\n'
    )
    _git(repo, "add", ".gitmodules")
    _git(repo, "commit", "-m", "declare submodule")
    manager = RepoWorkspaceManager(
        session_dir=tmp_path / "session",
        session_id="session-12345678",
        ckpt=_ckpt(),
        root_dir=tmp_path / "owned",
        init_submodules=False,
    )

    workspace = manager.prepare("kernel", repo)

    assert workspace.integration_path.is_dir()
    assert not (workspace.integration_path / "deps" / "large").exists()


def test_prepare_integration_worktree_keeps_source_repo_unchanged(tmp_path):
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    manager = _manager(tmp_path)

    workspace = manager.prepare("framework", repo)

    assert workspace.source_repo == repo.resolve()
    assert workspace.integration_path.is_dir()
    assert workspace.base_sha == base
    assert workspace.work_branch == "quark-quant-perf-session"
    assert _git(repo, "branch", "--show-current") == "main"
    assert _git(workspace.integration_path, "branch", "--show-current") == "quark-quant-perf-session"
    assert _git(workspace.integration_path, "status", "--porcelain") == ""
    marker = json.loads((workspace.integration_path.parent / ".quark-quant-perf-owned.json").read_text())
    assert marker["session_id"] == "session-12345678"
    assert marker["resource_type"] == "integration_worktree"
    assert marker["owner_pid"] == os.getpid()


def test_same_source_repo_reuses_one_integration_worktree(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)

    framework = manager.prepare("framework", repo)
    kernel = manager.prepare("kernel", repo)

    assert kernel.integration_path == framework.integration_path
    assert kernel.work_branch == framework.work_branch


def test_candidate_worktree_is_isolated_and_removed_after_context(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)
    integration = manager.prepare("kernel", repo)

    with manager.candidate("kernel", "moe-align") as candidate:
        assert candidate.path.is_dir()
        assert candidate.base_sha == _git(integration.integration_path, "rev-parse", "HEAD")
        (candidate.path / "kernel.py").write_text("VALUE = 2\n")
        assert (integration.integration_path / "kernel.py").read_text() == "VALUE = 1\n"
        candidate_path = candidate.path

    assert not candidate_path.exists()


def test_candidate_ignores_untracked_hipify_generated_headers(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)
    integration = manager.prepare("kernel", repo)
    generated = integration.integration_path / "csrc" / "include" / "kernel_hip.h"
    generated.parent.mkdir(parents=True)
    generated.write_text("// !!! This is a file automatically generated by hipify!!!\n")

    with manager.candidate("kernel", "hipify-runtime-artifact") as candidate:
        assert candidate.path.is_dir()


def test_candidate_ignores_untracked_hipify_generated_hip_source(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)
    integration = manager.prepare("kernel", repo)
    generated = integration.integration_path / "csrc" / "pybind" / "kernel.hip"
    generated.parent.mkdir(parents=True)
    generated.write_text("// !!! This is a file automatically generated by hipify!!!\n")

    with manager.candidate("kernel", "hipify-runtime-source") as candidate:
        assert candidate.path.is_dir()


def test_candidate_ignores_untracked_aiter_runtime_build_artifacts(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)
    integration = manager.prepare("kernel", repo)
    generated_files = [
        integration.integration_path / "build" / "temp.linux" / "object.o",
        integration.integration_path / "aiter" / "jit" / "build" / "module" / "kernel.cpp",
        integration.integration_path / "aiter" / "jit" / "module_runtime.so",
    ]
    for generated in generated_files:
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text("runtime generated\n")

    with manager.candidate("kernel", "aiter-runtime-artifacts") as candidate:
        assert candidate.path.is_dir()


def test_candidate_identifies_hipify_artifact_by_breadcrumb_not_filename(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)
    integration = manager.prepare("kernel", repo)
    generated = integration.integration_path / "csrc" / "hip" / "generated.source"
    generated.parent.mkdir(parents=True)
    generated.write_text("// !!! This is a file automatically generated by hipify!!!\n")

    with manager.candidate("kernel", "hipify-renamed-artifact") as candidate:
        assert candidate.path.is_dir()


def test_candidate_rejects_untracked_hip_suffix_without_generated_marker(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)
    integration = manager.prepare("kernel", repo)
    generated = integration.integration_path / "csrc" / "include" / "kernel_hip.h"
    generated.parent.mkdir(parents=True)
    generated.write_text("// manually authored source\n")

    with pytest.raises(WorkspaceError, match="dirty before GEAK"):
        with manager.candidate("kernel", "manual-hip-source"):
            pass


def test_execution_worktree_is_isolated_and_preserves_source(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)

    with manager.execution_worktree(
        "quantizer",
        repo,
        "direct-ptq",
    ) as worktree:
        worktree = Path(worktree)
        assert worktree.is_dir()
        (worktree / "generated.py").write_text("temporary\n")
        execution_path = worktree

    assert not execution_path.exists()
    assert not (repo / "generated.py").exists()
    assert _git(repo, "status", "--porcelain") == ""
    assert (repo / "kernel.py").read_text() == "VALUE = 1\n"


def test_candidate_records_repair_intent(tmp_path):
    repo = _repo(tmp_path)
    ckpt = _ckpt()
    manager = _manager(tmp_path, ckpt)
    manager.prepare("framework", repo)

    with manager.candidate(
        "framework",
        "load-fix",
        intent="repair",
    ) as candidate:
        resource = ckpt.state["transient_resources"][0]
        assert candidate.intent == "repair"
        assert resource["intent"] == "repair"
        assert resource["candidate_id"] == "load-fix"


def test_transaction_opens_multiple_candidate_worktrees(tmp_path):
    framework_root = tmp_path / "framework"
    kernel_root = tmp_path / "kernel"
    framework_root.mkdir()
    kernel_root.mkdir()
    framework_repo = _repo(framework_root)
    kernel_repo = _repo(kernel_root)
    manager = _manager(tmp_path)
    manager.prepare("framework", framework_repo)
    manager.prepare("kernel", kernel_repo)

    with manager.transaction(
        intent="repair",
        candidate_id="cross-repo-fix",
        roles=("framework", "kernel"),
    ) as transaction:
        assert set(transaction.workspaces) == {"framework", "kernel"}
        assert transaction.workspaces["framework"].path.is_dir()
        assert transaction.workspaces["kernel"].path.is_dir()


def test_promote_candidate_applies_committed_change_to_integration(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)
    integration = manager.prepare("framework", repo)

    with manager.candidate(
        "framework",
        "load-fix",
        intent="repair",
    ) as candidate:
        (candidate.path / "kernel.py").write_text("VALUE = 7\n")
        _git(candidate.path, "add", "kernel.py")
        _git(candidate.path, "commit", "-m", "candidate repair")
        promoted = manager.promote_candidate(
            candidate,
            message="accepted repair",
        )

    assert promoted
    assert (integration.integration_path / "kernel.py").read_text() == "VALUE = 7\n"
    assert (repo / "kernel.py").read_text() == "VALUE = 1\n"


def test_promote_transaction_applies_all_repo_changes(tmp_path):
    framework_root = tmp_path / "framework"
    kernel_root = tmp_path / "kernel"
    framework_root.mkdir()
    kernel_root.mkdir()
    framework_repo = _repo(framework_root)
    kernel_repo = _repo(kernel_root)
    manager = _manager(tmp_path)
    framework = manager.prepare("framework", framework_repo)
    kernel = manager.prepare("kernel", kernel_repo)

    with manager.transaction(
        intent="repair",
        candidate_id="cross-repo",
        roles=("framework", "kernel"),
    ) as transaction:
        for role, value in (("framework", 8), ("kernel", 9)):
            candidate = transaction.workspaces[role]
            (candidate.path / "kernel.py").write_text(f"VALUE = {value}\n")
            _git(candidate.path, "add", "kernel.py")
            _git(candidate.path, "commit", "-m", role)
        promoted = manager.promote_transaction(
            transaction,
            message="accepted cross-repo repair",
        )

    assert promoted == ["framework", "kernel"]
    assert (framework.integration_path / "kernel.py").read_text() == "VALUE = 8\n"
    assert (kernel.integration_path / "kernel.py").read_text() == "VALUE = 9\n"


def test_candidate_worktree_is_removed_when_geak_raises(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)
    manager.prepare("kernel", repo)
    candidate_path = None

    with pytest.raises(RuntimeError, match="GEAK failed"), manager.candidate("kernel", "moe-align") as candidate:
        candidate_path = candidate.path
        raise RuntimeError("GEAK failed")

    assert candidate_path is not None
    assert not candidate_path.exists()


def test_candidate_reclaims_owned_stale_worktree(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)
    integration = manager.prepare("kernel", repo)
    base_sha = _git(integration.integration_path, "rev-parse", "HEAD")
    path = manager.root_dir / "kernel" / "candidates" / "stale" / "worktree"
    path.parent.mkdir(parents=True)
    _git(
        integration.source_repo,
        "worktree",
        "add",
        "--detach",
        str(path),
        base_sha,
    )
    manager._marker(
        path,
        "candidate_worktree",
        source_repo=str(integration.source_repo),
        role="kernel",
        candidate_id="stale",
        base_sha=base_sha,
        owner_pid=99999999,
    )

    with manager.candidate("kernel", "stale") as candidate:
        assert candidate.path == path
        marker = json.loads((path.parent / ".quark-quant-perf-owned.json").read_text())
        assert marker["owner_pid"] == os.getpid()

    assert not path.exists()


def test_candidate_does_not_reclaim_live_owner(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)
    integration = manager.prepare("kernel", repo)
    base_sha = _git(integration.integration_path, "rev-parse", "HEAD")
    path = manager.root_dir / "kernel" / "candidates" / "active" / "worktree"
    path.parent.mkdir(parents=True)
    _git(
        integration.source_repo,
        "worktree",
        "add",
        "--detach",
        str(path),
        base_sha,
    )
    manager._marker(
        path,
        "candidate_worktree",
        source_repo=str(integration.source_repo),
        role="kernel",
        candidate_id="active",
        base_sha=base_sha,
        owner_pid=os.getpid(),
    )

    with pytest.raises(WorkspaceError, match="active owner"), manager.candidate("kernel", "active"):
        pass

    manager._remove_worktree(
        integration.source_repo,
        path,
        "candidate_worktree",
    )


def test_candidate_setup_failure_removes_partial_worktree(tmp_path):
    repo = _repo(tmp_path)
    ckpt = _ckpt()
    manager = _manager(tmp_path, ckpt)
    manager.prepare("kernel", repo)
    manager._init_submodules = lambda _path: (_ for _ in ()).throw(RuntimeError("submodule setup failed"))

    with pytest.raises(RuntimeError, match="submodule setup failed"), manager.candidate("kernel", "moe-align"):
        pass

    candidates = manager.root_dir / "kernel" / "candidates"
    assert not list(candidates.glob("*/worktree")) if candidates.exists() else True
    assert ckpt.state["transient_resources"] == []


def test_prepare_recreates_missing_integration_worktree_on_resume(tmp_path):
    repo = _repo(tmp_path)
    ckpt = _ckpt()
    manager = _manager(tmp_path, ckpt)
    workspace = manager.prepare("framework", repo)
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(workspace.integration_path)],
        cwd=repo,
        check=True,
    )
    (workspace.integration_path.parent / ".quark-quant-perf-owned.json").unlink()

    resumed = _manager(tmp_path, ckpt).prepare("framework", repo)

    assert resumed.integration_path.is_dir()
    assert _git(resumed.integration_path, "branch", "--show-current") == resumed.work_branch


def test_prepare_recreates_incomplete_owned_integration_worktree_on_resume(
    tmp_path,
):
    repo = _repo(tmp_path)
    ckpt = _ckpt()
    manager = _manager(tmp_path, ckpt)
    workspace = manager.prepare("framework", repo)
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(workspace.integration_path)],
        cwd=repo,
        check=True,
    )
    workspace.integration_path.mkdir()
    (workspace.integration_path / "runtime-cache").mkdir()

    resumed = _manager(tmp_path, ckpt).prepare("framework", repo)

    assert (resumed.integration_path / ".git").is_file()
    assert not (resumed.integration_path / "runtime-cache").exists()
    assert _git(resumed.integration_path, "branch", "--show-current") == resumed.work_branch


def test_terminal_cleanup_removes_worktree_and_empty_branch(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)
    workspace = manager.prepare("framework", repo)
    path = workspace.integration_path

    manager.cleanup_terminal()

    assert not path.exists()
    assert "quark-quant-perf-session" not in _git(repo, "branch", "--format=%(refname:short)")
    assert not manager.root_dir.exists()


def test_terminal_cleanup_ignores_python_bytecode_artifacts(tmp_path):
    repo = _repo(tmp_path)
    ckpt = _ckpt()
    manager = _manager(tmp_path, ckpt)
    workspace = manager.prepare("framework", repo)
    cache = workspace.integration_path / "__pycache__"
    cache.mkdir()
    (cache / "kernel.cpython-312.pyc").write_bytes(b"runtime bytecode")

    manager.cleanup_terminal()

    assert ckpt.state["cleanup"]["status"] == "complete"
    assert not workspace.integration_path.exists()
    assert (
        workspace.work_branch
        not in _git(
            repo,
            "branch",
            "--format=%(refname:short)",
        ).splitlines()
    )


def test_prepare_recreates_branch_after_terminal_cleanup_for_resume(tmp_path):
    repo = _repo(tmp_path)
    ckpt = _ckpt()
    manager = _manager(tmp_path, ckpt)
    workspace = manager.prepare("kernel", repo)

    manager.cleanup_terminal()
    assert "quark-quant-perf-session" not in _git(repo, "branch", "--format=%(refname:short)")

    resumed = _manager(tmp_path, ckpt).prepare("kernel", repo)

    assert resumed.integration_path.is_dir()
    assert _git(resumed.integration_path, "branch", "--show-current") == ("quark-quant-perf-session")
    assert _git(resumed.integration_path, "rev-parse", "HEAD") == (workspace.base_sha)
    assert ckpt.state["cleanup"]["status"] == "pending"


def test_terminal_cleanup_drops_branch_with_only_empty_commits(tmp_path):
    repo = _repo(tmp_path)
    manager = _manager(tmp_path)
    workspace = manager.prepare("framework", repo)
    _git(workspace.integration_path, "commit", "--allow-empty", "-m", "baseline")

    manager.cleanup_terminal()

    assert "quark-quant-perf-session" not in _git(repo, "branch", "--format=%(refname:short)")


def test_terminal_cleanup_keeps_branch_with_committed_changes(tmp_path):
    repo = _repo(tmp_path)
    ckpt = _ckpt()
    manager = _manager(tmp_path, ckpt)
    workspace = manager.prepare("kernel", repo)
    (workspace.integration_path / "kernel.py").write_text("VALUE = 3\n")
    _git(workspace.integration_path, "add", "kernel.py")
    _git(workspace.integration_path, "commit", "-m", "keep optimization")
    final_sha = _git(workspace.integration_path, "rev-parse", "HEAD")

    manager.cleanup_terminal()

    assert not workspace.integration_path.exists()
    assert "quark-quant-perf-session" in _git(repo, "branch", "--format=%(refname:short)")
    assert ckpt.state["repo_workspaces"]["kernel"]["final_sha"] == final_sha
    assert ckpt.state["cleanup"]["status"] == "complete"


def test_terminal_cleanup_checkpoints_retained_branch_before_removal(
    tmp_path,
    monkeypatch,
):
    repo = _repo(tmp_path)
    session_dir = tmp_path / "session"
    ckpt = Checkpoint.fresh(
        Spec(
            model_dir="model",
            base_model="model",
            framework="vllm",
            gpu_type="mi300x",
            gpu_arch="MI300X",
            isl=128,
            osl=128,
            quant_strategy="fp8",
            session_dir=str(session_dir),
        )
    )
    manager = _manager(tmp_path, ckpt)
    workspace = manager.prepare("kernel", repo)
    (workspace.integration_path / "kernel.py").write_text("VALUE = 5\n")
    _git(workspace.integration_path, "add", "kernel.py")
    _git(workspace.integration_path, "commit", "-m", "keep optimization")
    final_sha = _git(workspace.integration_path, "rev-parse", "HEAD")
    remove_worktree: Callable[[Path, Path, str], None] = manager._remove_worktree

    def remove_then_interrupt(
        source_repo: Path,
        path: Path,
        resource_type: str,
    ) -> None:
        remove_worktree(source_repo, path, resource_type)
        raise KeyboardInterrupt

    monkeypatch.setattr(manager, "_remove_worktree", remove_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        manager.cleanup_terminal()

    persisted = Checkpoint.load(session_dir)
    assert persisted is not None
    record = persisted.state["repo_workspaces"]["kernel"]
    assert record["final_sha"] == final_sha
    assert record["branch_retained"] is True

    _manager(tmp_path, persisted).cleanup_terminal()

    assert workspace.work_branch in _git(repo, "branch", "--format=%(refname:short)").splitlines()


def test_terminal_cleanup_updates_both_roles_for_shared_repo(tmp_path):
    repo = _repo(tmp_path)
    ckpt = _ckpt()
    manager = _manager(tmp_path, ckpt)
    framework = manager.prepare("framework", repo)
    manager.prepare("kernel", repo)
    (framework.integration_path / "kernel.py").write_text("VALUE = 4\n")
    _git(framework.integration_path, "add", "kernel.py")
    _git(framework.integration_path, "commit", "-m", "keep optimization")

    manager.cleanup_terminal()

    for role in ("framework", "kernel"):
        record = ckpt.state["repo_workspaces"][role]
        assert record["status"] == "removed"
        assert record["branch_retained"] is True
        assert record["final_sha"]


def test_terminal_cleanup_refuses_unmarked_worktree(tmp_path):
    repo = _repo(tmp_path)
    ckpt = _ckpt()
    manager = _manager(tmp_path, ckpt)
    workspace = manager.prepare("framework", repo)
    (workspace.integration_path.parent / ".quark-quant-perf-owned.json").unlink()

    manager.cleanup_terminal()

    assert workspace.integration_path.exists()
    assert ckpt.state["cleanup"]["status"] == "failed"
    assert "ownership marker missing" in ckpt.state["cleanup"]["errors"][0]


def test_terminal_cleanup_preserves_uncommitted_integration_changes(tmp_path):
    repo = _repo(tmp_path)
    ckpt = _ckpt()
    manager = _manager(tmp_path, ckpt)
    workspace = manager.prepare("framework", repo)
    (workspace.integration_path / "kernel.py").write_text("VALUE = 9\n")

    manager.cleanup_terminal()

    assert workspace.integration_path.exists()
    assert ckpt.state["cleanup"]["status"] == "failed"
    assert "dirty at terminal cleanup" in ckpt.state["cleanup"]["errors"][0]
