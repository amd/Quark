#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Session-local compiler provenance for runtime GPU kernels."""

from __future__ import annotations

import hashlib
import json
import os
import pickletools
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

PROVENANCE_SCHEMA_VERSION = "quark.quant_perf.kernel_provenance.v1"
PROVENANCE_FILENAME = "kernel_provenance.json"
RESOLUTION_SCHEMA_VERSION = "quark.quant_perf.kernel_source_resolution.v1"
RESOLUTION_FILENAME = "kernel_source_resolution.json"

_GPU_FUNC_RE = re.compile(r"\bgpu\.func\s+@(?P<name>[^\s(]+).*?\bkernel\b")
_LOC_DEF_RE = re.compile(
    r"^(?P<id>#[A-Za-z0-9_.]+)\s*=\s*"
    r'loc\("(?P<path>[^"]+\.py)":(?P<line>\d+):(?P<column>\d+)\)',
    re.MULTILINE,
)
_DIRECT_LOC_RE = re.compile(r'loc\("(?P<path>[^"]+\.py)":(?P<line>\d+):(?P<column>\d+)\)')
_LOC_REF_RE = re.compile(r"loc\((?P<id>#[A-Za-z0-9_.]+)\)")
_CACHE_DIR_RE = re.compile(r"^(?P<builder>.+)_(?P<manager>[0-9a-f]{32})$")
_NUMBERED_KERNEL_RE = re.compile(r"^(?P<symbol>.+)_\d+$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return ""


@lru_cache(maxsize=64)
def _repo_revision(repo: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _pickle_text_values(data: bytes) -> list[str]:
    values: list[str] = []
    try:
        for operation, argument, _ in pickletools.genops(data):
            if operation.name in {"BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"} and isinstance(argument, str):
                values.append(argument)
    except (EOFError, TypeError, ValueError):
        return []
    return values


def _source_ir(data: bytes) -> str:
    candidates = [value for value in _pickle_text_values(data) if "gpu.func @" in value]
    return max(candidates, key=len) if candidates else ""


def _source_roots(
    source_roots: dict[str, str | Path],
) -> tuple[tuple[str, Path], ...]:
    rows = []
    for role, root in source_roots.items():
        path = Path(root).resolve()
        if path.is_dir():
            rows.append((str(role), path))
    return tuple(rows)


def _owned_source(
    path: str,
    roots: tuple[tuple[str, Path], ...],
) -> tuple[str, Path, Path] | None:
    source = Path(path).resolve()
    if not source.is_file():
        return None
    for role, root in roots:
        try:
            relative = source.relative_to(root)
        except ValueError:
            continue
        return role, root, relative
    return None


def _location_for_function(
    line: str,
    locations: dict[str, tuple[str, int]],
) -> tuple[str, int] | None:
    direct = _DIRECT_LOC_RE.search(line)
    if direct:
        return direct.group("path"), int(direct.group("line"))
    reference = _LOC_REF_RE.search(line)
    if reference:
        return locations.get(reference.group("id"))
    return None


def _kernel_rows(
    source_ir: str,
    roots: tuple[tuple[str, Path], ...],
) -> list[tuple[str, str, Path, Path, int]]:
    locations = {
        match.group("id"): (
            match.group("path"),
            int(match.group("line")),
        )
        for match in _LOC_DEF_RE.finditer(source_ir)
    }
    managed_locations = []
    for path, line in locations.values():
        owned = _owned_source(path, roots)
        if owned:
            managed_locations.append((*owned, line))

    rows = []
    for line in source_ir.splitlines():
        match = _GPU_FUNC_RE.search(line)
        if not match:
            continue
        runtime_name = match.group("name")
        location = _location_for_function(line, locations)
        owned = _owned_source(location[0], roots) if location else None
        source_line = location[1] if location else 0
        if owned is None:
            unique = {(role, root, relative, line_number) for role, root, relative, line_number in managed_locations}
            if len(unique) != 1:
                continue
            role, root, relative, source_line = next(iter(unique))
        else:
            role, root, relative = owned
        numbered = _NUMBERED_KERNEL_RE.match(runtime_name)
        source_symbol = numbered.group("symbol") if numbered else runtime_name
        rows.append(
            (
                runtime_name,
                source_symbol,
                root,
                relative,
                source_line,
            )
        )
    return rows


def collect_flydsl_cache_provenance(
    cache_root: str | Path,
    *,
    source_roots: dict[str, str | Path],
    gpu_arch: str = "",
) -> list[dict[str, Any]]:
    """Collect trusted source evidence from FlyDSL cache pickles.

    The cache is parsed with :mod:`pickletools`; objects are never unpickled.
    """
    if not os.fspath(cache_root).strip():
        return []
    cache = Path(cache_root).resolve()
    if not cache.is_dir():
        return []
    roots = _source_roots(source_roots)
    if not roots:
        return []

    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for artifact in sorted(cache.rglob("*.pkl")):
        try:
            data = artifact.read_bytes()
        except OSError:
            continue
        source_ir = _source_ir(data)
        if not source_ir:
            continue
        parent_match = _CACHE_DIR_RE.match(artifact.parent.name)
        builder = parent_match.group("builder") if parent_match else artifact.parent.name
        for (
            runtime_name,
            source_symbol,
            repo,
            relative,
            source_line,
        ) in _kernel_rows(source_ir, roots):
            source = repo / relative
            role = next(role for role, root in roots if root == repo)
            normalized_name = runtime_name if runtime_name.endswith(".kd") else f"{runtime_name}.kd"
            record = {
                "runtime_kernel_name": normalized_name,
                "compiler": "flydsl",
                "backend": "",
                "gpu_arch": str(gpu_arch or ""),
                "source_file": str(source),
                "source_repo": str(repo),
                "source_repo_role": role,
                "source_relpath": relative.as_posix(),
                "source_symbol": source_symbol,
                "source_line": int(source_line or 0),
                "builder_symbol": builder,
                "launcher_source_file": str(source),
                "cache_key_hash": artifact.stem,
                "artifact_sha256": _sha256_bytes(data),
                "source_sha256": _sha256_file(source),
                "repo_revision": _repo_revision(str(repo)),
                "artifact_file": str(artifact),
            }
            key = (
                record["runtime_kernel_name"],
                record["source_file"],
                record["cache_key_hash"],
            )
            records[key] = record
    return list(records.values())


def write_kernel_provenance(
    session_dir: str | Path,
    records: list[dict[str, Any]],
) -> Path:
    session = Path(session_dir).resolve()
    session.mkdir(parents=True, exist_ok=True)
    path = session / PROVENANCE_FILENAME
    temporary = path.with_suffix(".json.tmp")
    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "entries": list(records),
    }
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)
    return path


def load_kernel_provenance(
    session_dir: str | Path,
) -> list[dict[str, Any]]:
    path = Path(session_dir)
    if path.is_dir():
        path = path / PROVENANCE_FILENAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    if payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        return []
    entries = payload.get("entries")
    return [dict(entry) for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []


def write_kernel_source_resolution(
    session_dir: str | Path,
    journeys: list[dict[str, Any]],
) -> Path:
    session = Path(session_dir).resolve()
    session.mkdir(parents=True, exist_ok=True)
    path = session / RESOLUTION_FILENAME
    temporary = path.with_suffix(".json.tmp")
    entries = []
    for journey in journeys:
        if not isinstance(journey, dict):
            continue
        entries.append(
            {
                "kernel_id": str(journey.get("kernel_id") or ""),
                "name": str(journey.get("name") or ""),
                "outcome": str(journey.get("outcome") or ""),
                "skip_reason": str(journey.get("skip_reason") or ""),
                "source_mapping": dict(journey.get("source_mapping") or {}),
            }
        )
    payload = {
        "schema_version": RESOLUTION_SCHEMA_VERSION,
        "entries": entries,
    }
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)
    return path


def load_kernel_source_resolution(
    session_dir: str | Path,
) -> list[dict[str, Any]]:
    path = Path(session_dir)
    if path.is_dir():
        path = path / RESOLUTION_FILENAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    if payload.get("schema_version") != RESOLUTION_SCHEMA_VERSION:
        return []
    entries = payload.get("entries")
    return [dict(entry) for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []
