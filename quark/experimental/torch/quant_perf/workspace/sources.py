#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Resolve runtime packages into Quark Quant-Perf-managed source workspaces."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from quark.experimental.torch.quant_perf.session.spec import Spec
from quark.experimental.torch.quant_perf.workspace.git import find_git_root
from quark.experimental.torch.quant_perf.workspace.source_tree import (
    is_compiled_runtime_asset,
    is_generated_workspace_path,
)


class SourceResolutionError(RuntimeError):
    pass


class RuntimeOriginMismatch(RuntimeError):
    pass


_COMPILED_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hip",
    ".hpp",
}


@dataclass(frozen=True)
class ResolvedSource:
    role: str
    kind: str
    source_root: str = ""
    origin_path: str = ""
    package_names: tuple[str, ...] = ()
    package_roots: tuple[str, ...] = ()
    version: str = ""
    git_sha: str = ""
    modifiable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionRuntime:
    python_exe: str = sys.executable
    pythonpath_prefixes: tuple[str, ...] = ()
    ld_library_path_prefixes: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    expected_origins: dict[str, str] = field(default_factory=dict)

    def subprocess_env(
        self,
        base: dict[str, str] | None = None,
    ) -> dict[str, str]:
        result = dict(os.environ if base is None else base)
        if self.pythonpath_prefixes:
            existing = [item for item in result.get("PYTHONPATH", "").split(os.pathsep) if item]
            result["PYTHONPATH"] = os.pathsep.join([*self.pythonpath_prefixes, *existing])
        if self.ld_library_path_prefixes:
            existing = [item for item in result.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item]
            result["LD_LIBRARY_PATH"] = os.pathsep.join([*self.ld_library_path_prefixes, *existing])
        result.update(self.env)
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SessionRuntime:
        data = data or {}
        return cls(
            python_exe=str(data.get("python_exe") or sys.executable),
            pythonpath_prefixes=tuple(data.get("pythonpath_prefixes") or ()),
            ld_library_path_prefixes=tuple(data.get("ld_library_path_prefixes") or ()),
            env={str(key): str(value) for key, value in (data.get("env") or {}).items()},
            expected_origins={str(key): str(value) for key, value in (data.get("expected_origins") or {}).items()},
        )


def activate_runtime(spec: Spec) -> SessionRuntime:
    preferred = [
        path
        for path in (
            spec.active_framework_repo,
            spec.active_kernel_repo,
        )
        if path
    ]
    existing = [
        path
        for path in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if path
        and path
        not in {
            spec.framework_repo,
            spec.kernel_repo,
            *preferred,
        }
    ]
    pythonpath = os.pathsep.join(preferred + existing)
    sys.path[:] = [
        *preferred,
        *[entry for entry in sys.path if entry not in preferred],
    ]
    cache_root = Path(spec.session_dir).resolve() / "runtime" / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_env = {
        "QUARK_QUANT_PERF_VLLM_CACHE_BASE": str(cache_root / "vllm"),
        "TRITON_CACHE_DIR": str(cache_root / "triton"),
        "TORCH_EXTENSIONS_DIR": str(cache_root / "torch_extensions"),
        "AITER_JIT_DIR": str(cache_root / "aiter_jit"),
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
    }
    if spec.mxfp4_moe_backend == "flydsl" or spec.mxfp4_gemm_backend == "flydsl" or spec.w4a8_gemm_backend == "flydsl":
        cache_env.update(
            {
                "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
                "FLYDSL_RUNTIME_CACHE_DIR": str(cache_root / "flydsl"),
            }
        )
    kernel_root = Path(spec.active_kernel_repo).resolve() if spec.active_kernel_repo else None
    if kernel_root is not None and (kernel_root / "aiter").is_dir():
        cache_env["AITER_ROOT_DIR"] = str(kernel_root)
        meta_root = kernel_root / "aiter_meta"
        if meta_root.is_dir():
            cache_env["AITER_META_DIR"] = str(meta_root)
    spec.runtime.runtime_python = sys.executable
    spec.runtime.runtime_env = {
        **({"PYTHONPATH": pythonpath} if pythonpath else {}),
        **cache_env,
    }
    expected_origins: dict[str, str] = {}
    packages = {
        "framework": ("vllm",) if spec.framework == "vllm" else ("atom",),
        "kernel": ("aiter", "aiter_meta"),
    }
    roots = {
        "framework": spec.active_framework_repo,
        "kernel": spec.active_kernel_repo,
    }
    for role, names in packages.items():
        root = roots[role]
        if not root:
            continue
        for name in names:
            if (Path(root) / name).exists():
                expected_origins[name] = root
    spec.runtime.runtime_origins = expected_origins
    runtime = SessionRuntime(
        python_exe=spec.runtime_python,
        pythonpath_prefixes=tuple(preferred),
        env=cache_env,
        expected_origins=dict(expected_origins),
    )
    os.environ["PYTHONPATH"] = pythonpath
    return runtime


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise SourceResolutionError(
            f"git {' '.join(args)} failed in {repo}: {(result.stderr or result.stdout).strip()}"
        )
    return result.stdout.strip()


def _package_origin(package: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin:
        return Path(spec.origin).resolve().parent
    locations = list(spec.submodule_search_locations or ())
    return Path(locations[0]).resolve() if locations else None


def _editable_project_root(package: str) -> Path | None:
    try:
        distribution = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        return None
    distribution_path = getattr(distribution, "_path", None)
    if distribution_path is None:
        return None
    direct_url = Path(distribution_path) / "direct_url.json"
    if not direct_url.is_file():
        return None
    try:
        payload = json.loads(direct_url.read_text())
    except (OSError, ValueError, TypeError):
        return None
    if not (payload.get("dir_info") or {}).get("editable"):
        return None
    parsed = urlparse(str(payload.get("url") or ""))
    if parsed.scheme != "file":
        return None
    root = Path(unquote(parsed.path)).resolve()
    return root if root.is_dir() else None


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _readonly(
    role: str,
    *,
    source_root: str = "",
    package_names: tuple[str, ...] = (),
) -> ResolvedSource:
    return ResolvedSource(
        role=role,
        kind="readonly",
        source_root=source_root,
        package_names=package_names,
        modifiable=False,
    )


def resolve_source(
    *,
    role: str,
    mode: str,
    explicit_repo: str = "",
    package_names: tuple[str, ...],
) -> ResolvedSource:
    if mode not in {"explicit", "auto", "readonly"}:
        raise SourceResolutionError(f"unsupported workspace source mode: {mode}")
    if mode == "readonly":
        return _readonly(
            role,
            source_root=str(Path(explicit_repo).resolve()) if explicit_repo else "",
            package_names=package_names,
        )
    if explicit_repo:
        root = Path(explicit_repo).resolve()
        if not root.is_dir():
            raise SourceResolutionError(f"{role} repository does not exist: {root}")
        git_root = find_git_root(root)
        if git_root is None or git_root != root:
            raise SourceResolutionError(f"{role} repository is not a Git root: {root}")
        return ResolvedSource(
            role=role,
            kind="explicit_git",
            source_root=str(root),
            origin_path=str(root),
            package_names=package_names,
            git_sha=_git(root, "rev-parse", "HEAD"),
            modifiable=True,
        )
    if mode == "explicit":
        return _readonly(role, package_names=package_names)

    roots = [(package, _package_origin(package)) for package in package_names]
    roots = [(name, root) for name, root in roots if root is not None]
    if not roots:
        return _readonly(role, package_names=package_names)

    for package, origin in roots:
        editable = _editable_project_root(package)
        git_root = find_git_root(editable or origin)
        if git_root is not None:
            return ResolvedSource(
                role=role,
                kind="editable_git",
                source_root=str(git_root),
                origin_path=str(origin),
                package_names=package_names,
                package_roots=tuple(str(root) for _, root in roots),
                version=_package_version(package),
                git_sha=_git(git_root, "rev-parse", "HEAD"),
                modifiable=True,
            )

    primary_name, primary_root = roots[0]
    return ResolvedSource(
        role=role,
        kind="installed_package",
        origin_path=str(primary_root),
        package_names=tuple(name for name, _ in roots),
        package_roots=tuple(str(root) for _, root in roots),
        version=_package_version(primary_name),
        modifiable=True,
    )


def _validate_package_tree(root: Path) -> None:
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            target = path.resolve(strict=True)
            target.relative_to(resolved_root)
        except (OSError, ValueError) as error:
            raise SourceResolutionError(f"package symlink escapes source root: {path}") from error


def _copy_package(source: Path, target: Path) -> None:
    _validate_package_tree(source)

    def _ignore_generated(directory: str, names: list[str]) -> list[str]:
        current = Path(directory)
        return [name for name in names if is_generated_workspace_path((current / name).relative_to(source))]

    def _copy_or_reference(source_file: str, target_file: str) -> str:
        if is_compiled_runtime_asset(source_file):
            Path(target_file).symlink_to(Path(source_file).resolve())
            return target_file
        return shutil.copy2(source_file, target_file)

    shutil.copytree(
        source,
        target,
        symlinks=True,
        ignore=_ignore_generated,
        copy_function=_copy_or_reference,
    )


def _validate_materialized_overlay(overlay: Path) -> None:
    for path in overlay.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            path.resolve(strict=True)
        except OSError as error:
            raise SourceResolutionError(f"overlay runtime asset is missing: {path}") from error


def materialize_source(
    source: ResolvedSource,
    *,
    session_dir: str | Path,
) -> ResolvedSource:
    if source.kind in {"explicit_git", "editable_git", "readonly"}:
        return source
    if source.kind != "installed_package":
        raise SourceResolutionError(f"unsupported source kind: {source.kind}")
    overlay = Path(session_dir).resolve() / "workspaces" / f"{source.role}-overlay" / "source"
    if (overlay / ".git").exists():
        _validate_materialized_overlay(overlay)
        return ResolvedSource(
            **{
                **source.to_dict(),
                "kind": "installed_overlay",
                "source_root": str(overlay),
                "git_sha": _git(overlay, "rev-parse", "HEAD"),
            }
        )
    if overlay.exists():
        shutil.rmtree(overlay)
    overlay.mkdir(parents=True)
    for name, raw_root in zip(
        source.package_names,
        source.package_roots,
        strict=True,
    ):
        package_root = Path(raw_root)
        _copy_package(package_root, overlay / name)
    _git(overlay, "init", "-b", "main")
    _git(overlay, "config", "user.email", "quark-quant-perf@local")
    _git(overlay, "config", "user.name", "Quark Quant-Perf")
    _git(overlay, "add", "-A")
    _git(overlay, "commit", "-m", "Quark Quant-Perf installed package baseline")
    return ResolvedSource(
        **{
            **source.to_dict(),
            "kind": "installed_overlay",
            "source_root": str(overlay),
            "git_sha": _git(overlay, "rev-parse", "HEAD"),
        }
    )


def verify_runtime_origins(
    runtime: SessionRuntime,
) -> dict[str, dict[str, object]]:
    if not runtime.expected_origins:
        return {}
    packages = list(runtime.expected_origins)
    script = (
        "import importlib.util,json\n"
        f"packages={packages!r}\n"
        "out={}\n"
        "for name in packages:\n"
        " spec=importlib.util.find_spec(name)\n"
        " origin=''\n"
        " if spec is not None:\n"
        "  if spec.origin: origin=spec.origin\n"
        "  elif spec.submodule_search_locations:\n"
        "   origin=list(spec.submodule_search_locations)[0]\n"
        " out[name]=origin\n"
        "print(json.dumps(out))\n"
    )
    result = subprocess.run(
        [runtime.python_exe, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env=runtime.subprocess_env(),
    )
    if result.returncode != 0:
        raise RuntimeOriginMismatch(
            f"runtime_origin_mismatch: origin probe failed: {(result.stderr or result.stdout)[-1000:]}"
        )
    try:
        origins = json.loads(result.stdout)
    except ValueError as error:
        raise RuntimeOriginMismatch("runtime_origin_mismatch: invalid origin probe output") from error
    evidence: dict[str, dict[str, object]] = {}
    mismatches = []
    for package, expected in runtime.expected_origins.items():
        actual = str(origins.get(package) or "")
        try:
            matched = bool(actual) and Path(actual).resolve().is_relative_to(Path(expected).resolve())
        except (OSError, RuntimeError):
            matched = False
        evidence[package] = {
            "expected": expected,
            "origin": actual,
            "matched": matched,
        }
        if not matched:
            mismatches.append(f"{package}: expected under {expected}, got {actual or 'missing'}")
    if mismatches:
        raise RuntimeOriginMismatch("runtime_origin_mismatch: " + "; ".join(mismatches))
    return evidence


def compiled_changes(
    repo: str | Path,
    base_sha: str,
) -> list[str]:
    root = Path(repo)
    result = _git(
        root,
        "diff",
        "--name-only",
        f"{base_sha}..HEAD",
        check=False,
    )
    return [path for path in result.splitlines() if Path(path).suffix.lower() in _COMPILED_SUFFIXES]
