#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ..testing import init_git_repo
from ..testing import run_git as _git


def _repo(tmp_path: Path, package: str) -> Path:
    return init_git_repo(tmp_path / f"{package}-repo", {f"{package}/__init__.py": "VALUE = 1\n"})


def test_explicit_source_has_highest_priority(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.sources import resolve_source

    repo = _repo(tmp_path, "vllm")
    source = resolve_source(
        role="framework",
        mode="auto",
        explicit_repo=str(repo),
        package_names=("vllm",),
    )

    assert source.kind == "explicit_git"
    assert source.source_root == str(repo.resolve())
    assert source.modifiable is True


def test_auto_discovers_editable_git_source(tmp_path, monkeypatch):
    from quark.experimental.torch.quant_perf.workspace import sources as workspace_sources

    repo = _repo(tmp_path, "vllm")
    monkeypatch.setattr(
        workspace_sources,
        "_package_origin",
        lambda _name: repo / "vllm",
    )
    monkeypatch.setattr(
        workspace_sources,
        "_editable_project_root",
        lambda _name: repo,
    )

    source = workspace_sources.resolve_source(
        role="framework",
        mode="auto",
        package_names=("vllm",),
    )

    assert source.kind == "editable_git"
    assert source.source_root == str(repo.resolve())


def test_auto_materializes_installed_package_as_git_overlay(
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.workspace import sources as workspace_sources

    site = tmp_path / "site-packages"
    package = site / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    monkeypatch.setattr(
        workspace_sources,
        "_package_origin",
        lambda _name: package,
    )
    monkeypatch.setattr(
        workspace_sources,
        "_editable_project_root",
        lambda _name: None,
    )

    source = workspace_sources.resolve_source(
        role="framework",
        mode="auto",
        package_names=("vllm",),
    )
    managed = workspace_sources.materialize_source(
        source,
        session_dir=tmp_path / "session",
    )

    overlay = Path(managed.source_root)
    assert managed.kind == "installed_overlay"
    assert (overlay / "vllm" / "__init__.py").read_text() == "VALUE = 1\n"
    assert _git(overlay, "status", "--porcelain") == ""
    assert _git(overlay, "rev-parse", "HEAD")
    assert (package / "__init__.py").read_text() == "VALUE = 1\n"


def test_installed_overlay_copies_sources_links_binaries_and_omits_generated_files(
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.workspace import sources as workspace_sources

    site = tmp_path / "site-packages"
    package = site / "vllm"
    package.mkdir(parents=True)
    source_file = package / "kernel.py"
    source_file.write_text("VALUE = 1\n")
    shared_library = package / "_C.abi3.so"
    shared_library.write_bytes(b"\x7fELFcompiled extension")
    executable = package / "vllm-rs"
    executable.write_bytes(b"\x7fELFcompiled executable")
    executable.chmod(0o755)
    generated_source = package / "build" / "module" / "kernel.hpp"
    generated_source.parent.mkdir(parents=True)
    generated_source.write_text("__global__ void duplicate_kernel() {}\n")
    cache_file = package / "__pycache__" / "kernel.pyc"
    cache_file.parent.mkdir()
    cache_file.write_bytes(b"cache")
    stable_generated_source = package / "generated" / "stable_kernel.py"
    stable_generated_source.parent.mkdir()
    stable_generated_source.write_text("def stable_kernel():\n    pass\n")
    monkeypatch.setattr(
        workspace_sources,
        "_package_origin",
        lambda _name: package,
    )
    monkeypatch.setattr(
        workspace_sources,
        "_editable_project_root",
        lambda _name: None,
    )

    source = workspace_sources.resolve_source(
        role="framework",
        mode="auto",
        package_names=("vllm",),
    )
    managed = workspace_sources.materialize_source(
        source,
        session_dir=tmp_path / "session",
    )

    overlay_package = Path(managed.source_root) / "vllm"
    assert (overlay_package / "kernel.py").is_file()
    assert not (overlay_package / "kernel.py").is_symlink()
    assert (overlay_package / "_C.abi3.so").is_symlink()
    assert (overlay_package / "_C.abi3.so").resolve() == shared_library
    assert (overlay_package / "vllm-rs").is_symlink()
    assert (overlay_package / "vllm-rs").resolve() == executable
    assert not (overlay_package / "build").exists()
    assert not (overlay_package / "__pycache__").exists()
    assert (overlay_package / "generated" / "stable_kernel.py").is_file()
    assert _git(Path(managed.source_root), "ls-files", "-s", "vllm/_C.abi3.so").startswith("120000 ")


def test_reused_overlay_rejects_missing_runtime_asset(
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.workspace import sources as workspace_sources

    site = tmp_path / "site-packages"
    package = site / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    shared_library = package / "_C.abi3.so"
    shared_library.write_bytes(b"\x7fELFcompiled extension")
    monkeypatch.setattr(
        workspace_sources,
        "_package_origin",
        lambda _name: package,
    )
    monkeypatch.setattr(
        workspace_sources,
        "_editable_project_root",
        lambda _name: None,
    )
    source = workspace_sources.resolve_source(
        role="framework",
        mode="auto",
        package_names=("vllm",),
    )
    workspace_sources.materialize_source(
        source,
        session_dir=tmp_path / "session",
    )
    shared_library.unlink()

    with pytest.raises(
        workspace_sources.SourceResolutionError,
        match="runtime asset is missing",
    ):
        workspace_sources.materialize_source(
            source,
            session_dir=tmp_path / "session",
        )


def test_readonly_mode_never_materializes_a_modifiable_source(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.sources import materialize_source, resolve_source

    source = resolve_source(
        role="framework",
        mode="readonly",
        explicit_repo=str(_repo(tmp_path, "vllm")),
        package_names=("vllm",),
    )
    managed = materialize_source(source, session_dir=tmp_path / "session")

    assert managed.kind == "readonly"
    assert managed.modifiable is False


def test_aiter_overlay_includes_sibling_aiter_meta(tmp_path, monkeypatch):
    from quark.experimental.torch.quant_perf.workspace import sources as workspace_sources

    site = tmp_path / "site-packages"
    aiter = site / "aiter"
    meta = site / "aiter_meta"
    aiter.mkdir(parents=True)
    meta.mkdir()
    (aiter / "__init__.py").write_text("")
    config = aiter / "jit" / "optCompilerConfig.json"
    config.parent.mkdir()
    config.write_text("{}")
    jit_library = aiter / "jit" / "module_quant.so"
    jit_library.write_bytes(b"\x7fELFjit module")
    jit_build_source = aiter / "jit" / "build" / "module_quant" / "kernel.cpp"
    jit_build_source.parent.mkdir(parents=True)
    jit_build_source.write_text("void generated_kernel() {}\n")
    canonical_source = meta / "csrc" / "kernel.py"
    canonical_source.parent.mkdir()
    canonical_source.write_text("TILE = 64\n")
    ck_generator = meta / "3rdparty" / "composable_kernel" / "example" / "generate.py"
    ck_generator.parent.mkdir(parents=True)
    ck_generator.write_text("print('generate')\n")
    code_object = meta / "hsa" / "gfx950" / "kernel.co"
    code_object.parent.mkdir(parents=True)
    code_object.write_bytes(b"\x7fELFcode object")
    monkeypatch.setattr(
        workspace_sources,
        "_package_origin",
        lambda name: {"aiter": aiter, "aiter_meta": meta}.get(name),
    )
    monkeypatch.setattr(
        workspace_sources,
        "_editable_project_root",
        lambda _name: None,
    )

    source = workspace_sources.resolve_source(
        role="kernel",
        mode="auto",
        package_names=("aiter", "aiter_meta"),
    )
    managed = workspace_sources.materialize_source(
        source,
        session_dir=tmp_path / "session",
    )

    overlay = Path(managed.source_root)
    assert (overlay / "aiter" / "__init__.py").is_file()
    assert (overlay / "aiter" / "jit" / "optCompilerConfig.json").is_file()
    assert (overlay / "aiter" / "jit" / "module_quant.so").is_symlink()
    assert not (overlay / "aiter" / "jit" / "build").exists()
    assert (overlay / "aiter_meta" / "csrc" / "kernel.py").is_file()
    assert (overlay / "aiter_meta" / "3rdparty" / "composable_kernel" / "example" / "generate.py").is_file()
    assert (overlay / "aiter_meta" / "hsa" / "gfx950" / "kernel.co").is_symlink()


def test_overlay_rejects_symlink_that_escapes_package_root(
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.workspace import sources as workspace_sources

    site = tmp_path / "site-packages"
    package = site / "vllm"
    package.mkdir(parents=True)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    (package / "escaped.py").symlink_to(outside)
    monkeypatch.setattr(
        workspace_sources,
        "_package_origin",
        lambda _name: package,
    )
    monkeypatch.setattr(
        workspace_sources,
        "_editable_project_root",
        lambda _name: None,
    )

    source = workspace_sources.resolve_source(
        role="framework",
        mode="auto",
        package_names=("vllm",),
    )

    with pytest.raises(
        workspace_sources.SourceResolutionError,
        match="symlink escapes",
    ):
        workspace_sources.materialize_source(
            source,
            session_dir=tmp_path / "session",
        )


def test_runtime_origin_verifier_requires_managed_root(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.sources import (
        SessionRuntime,
        verify_runtime_origins,
    )

    root = tmp_path / "overlay"
    package = root / "demo_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    runtime = SessionRuntime(
        pythonpath_prefixes=(str(root),),
        expected_origins={"demo_pkg": str(root)},
    )

    evidence = verify_runtime_origins(runtime)

    assert evidence["demo_pkg"]["matched"] is True
    assert str(root) in evidence["demo_pkg"]["origin"]


def test_runtime_origin_mismatch_is_explicit(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.sources import (
        RuntimeOriginMismatch,
        SessionRuntime,
        verify_runtime_origins,
    )

    runtime = SessionRuntime(
        expected_origins={"json": str(tmp_path / "not-json")},
    )

    with pytest.raises(RuntimeOriginMismatch, match="runtime_origin_mismatch"):
        verify_runtime_origins(runtime)


def test_source_state_is_json_serializable(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.sources import resolve_source

    source = resolve_source(
        role="framework",
        mode="readonly",
        explicit_repo=str(tmp_path),
        package_names=("vllm",),
    )

    assert json.loads(json.dumps(source.to_dict()))["kind"] == "readonly"


def test_compiled_changes_are_detected_for_overlay_gate(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.sources import compiled_changes

    repo = _repo(tmp_path, "vllm")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "kernel.cpp").write_text("int value = 1;\n")
    _git(repo, "add", "kernel.cpp")
    _git(repo, "commit", "-m", "compiled change")

    assert compiled_changes(repo, base) == ["kernel.cpp"]
