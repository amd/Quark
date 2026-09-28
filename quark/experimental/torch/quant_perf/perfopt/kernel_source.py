#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Kernel patchability classification and deterministic source resolution."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.perfopt.source_rules import SOURCE_RULES, SourceRule
from quark.experimental.torch.quant_perf.workspace.source_tree import (
    is_searchable_source,
    source_search_excluded_directory_names,
)

SOURCE_RESOLVER_VERSION = 8

_TORCH_COMPILE_PREFIXES = (
    "triton_poi_fused",
    "triton_red_fused",
    "triton_per_fused",
)

_NON_PATCHABLE_MARKERS: tuple[str, ...] = (
    "rocblas",
    "hipblas",
    "hipblaslt",
    "rocblaslt",
    "tensile",
    "miopen",
    "ck_kernels",
    "nccl",
    "rccl",
    "hipmemcpy",
    "__amd_rocclr_copybuffer",
    "cijk_",
)

_MANGLE_SUFFIX_RE = re.compile(r"(_[0-9a-f]{8,}|_\d+d\d+d\d+.*)$")
_GENERATED_SOURCE_LOC_RE = re.compile(r'^#loc\s*=\s*loc\("([^"]+\.py)":\d+:\d+\)')
_QUALIFIED_TEMPLATE_TYPE_RE = re.compile(r"\b((?:[A-Za-z_]\w*::)+[A-Za-z_]\w*)\s*<")
_CPP_SOURCE_INCLUDES = (
    "--include=*.cu",
    "--include=*.cuh",
    "--include=*.cpp",
    "--include=*.h",
    "--include=*.hip",
    "--include=*.hpp",
)


class ResolutionConfidence(StrEnum):
    EXACT = "exact"
    OPERATOR_RULE = "operator_rule"
    UNIQUE_DEFINITION = "unique_definition"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


class MappingKind(StrEnum):
    EDITABLE_SOURCE = "editable_source"
    DEPENDENCY_SOURCE = "dependency_source"
    PRECOMPILED_BINARY = "precompiled_binary"
    GENERATED_ARTIFACT = "generated_artifact"
    EXTERNAL_UNMANAGED_SOURCE = "external_unmanaged_source"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class SourceResolution:
    mapping_kind: MappingKind = MappingKind.UNRESOLVED
    source_file: str | None = None
    source_repo: str = ""
    source_repo_role: str = ""
    source_symbol: str = ""
    builder_symbol: str = ""
    launcher_source_file: str = ""
    launcher_symbol: str = ""
    live_call_seam: str = ""
    build_module: str = ""
    compiler: str = ""
    runtime_kernel_name: str = ""
    backend: str = ""
    gpu_arch: str = ""
    repo_revision: str = ""
    cache_key_hash: str = ""
    artifact_sha256: str = ""
    source_sha256: str = ""
    binary_file: str = ""
    config_file: str = ""
    method: str = ""
    confidence: ResolutionConfidence = ResolutionConfidence.UNRESOLVED
    alternatives: tuple[str, ...] = ()
    reason: str = ""
    resolver_version: int = SOURCE_RESOLVER_VERSION

    @property
    def retryable(self) -> bool:
        if self.mapping_kind in {
            MappingKind.DEPENDENCY_SOURCE,
            MappingKind.PRECOMPILED_BINARY,
            MappingKind.GENERATED_ARTIFACT,
            MappingKind.EXTERNAL_UNMANAGED_SOURCE,
        }:
            return False
        return self.confidence in {
            ResolutionConfidence.AMBIGUOUS,
            ResolutionConfidence.UNRESOLVED,
        }

    @property
    def patchable(self) -> bool:
        return self.mapping_kind is MappingKind.EDITABLE_SOURCE and bool(self.source_file)

    def to_dict(self) -> dict[str, Any]:
        source_relpath = ""
        if self.source_file and self.source_repo:
            try:
                source_relpath = os.path.relpath(
                    self.source_file,
                    self.source_repo,
                ).replace(os.sep, "/")
            except ValueError:
                source_relpath = ""
        repo_revision = self.repo_revision
        if not repo_revision and self.source_repo:
            repo_revision = _git_head(self.source_repo)
        source_sha256 = self.source_sha256
        if not source_sha256 and self.source_file:
            source_sha256 = _file_sha256(self.source_file)
        return {
            "mapping_kind": self.mapping_kind.value,
            "patchable": self.patchable,
            "source_file": self.source_file or "",
            "source_relpath": source_relpath,
            "source_repo": self.source_repo,
            "source_repo_role": self.source_repo_role,
            "source_symbol": self.source_symbol,
            "builder_symbol": self.builder_symbol,
            "launcher_source_file": self.launcher_source_file,
            "launcher_symbol": self.launcher_symbol,
            "live_call_seam": self.live_call_seam,
            "build_module": self.build_module,
            "compiler": self.compiler,
            "runtime_kernel_name": self.runtime_kernel_name,
            "backend": self.backend,
            "gpu_arch": self.gpu_arch,
            "repo_revision": repo_revision,
            "cache_key_hash": self.cache_key_hash,
            "artifact_sha256": self.artifact_sha256,
            "source_sha256": source_sha256,
            "binary_file": self.binary_file,
            "config_file": self.config_file,
            "method": self.method,
            "confidence": self.confidence.value,
            "alternatives": list(self.alternatives),
            "reason": self.reason,
            "resolver_version": self.resolver_version,
            "retryable": self.retryable,
        }


def _aiter_path(
    kernel_repo: str | Path,
    relative_path: str | Path,
) -> Path:
    repo_root = Path(kernel_repo).resolve()
    relative = Path(relative_path)
    if (
        relative.parts
        and relative.parts[0] in {"3rdparty", "csrc", "gradlib", "hsa"}
        and (repo_root / "aiter_meta").is_dir()
    ):
        repo_root = repo_root / "aiter_meta"
    return repo_root / relative


def _with_dependency_ownership(
    resolution: SourceResolution,
    kernel_repo: str,
) -> SourceResolution:
    if resolution.mapping_kind is not MappingKind.EDITABLE_SOURCE or not resolution.source_file or not kernel_repo:
        return resolution
    dependency_root = _aiter_path(
        kernel_repo,
        "3rdparty/composable_kernel",
    ).resolve()
    if not dependency_root.is_dir() or not Path(resolution.source_file).resolve().is_relative_to(dependency_root):
        return resolution
    return replace(
        resolution,
        mapping_kind=MappingKind.DEPENDENCY_SOURCE,
        source_repo=str(dependency_root),
        source_repo_role="composable_kernel",
        reason="composable_kernel dependency is not managed",
    )


def _non_patchable_kernel(
    kernel_name: str,
) -> tuple[MappingKind, str] | None:
    lower = kernel_name.lower()
    if any(lower.startswith(prefix) for prefix in _TORCH_COMPILE_PREFIXES):
        return (
            MappingKind.GENERATED_ARTIFACT,
            (
                "torch.compile generated kernel "
                "(generated artifact; no stable one-to-one repository source): "
                f"{kernel_name[:60]}"
            ),
        )
    for marker in _NON_PATCHABLE_MARKERS:
        if marker in lower:
            return (
                MappingKind.PRECOMPILED_BINARY,
                (
                    "vendor library kernel (precompiled binary, not rewritable): "
                    f"marker='{marker}' in '{kernel_name[:80]}'"
                ),
            )
    return None


def classify_kernel(kernel_name: str) -> tuple[bool, str]:
    """Reject runtime-generated and precompiled symbols before source lookup."""
    classification = _non_patchable_kernel(kernel_name)
    return (True, "") if classification is None else (False, classification[1])


def _runtime_classification_resolution(
    kernel_names: list[str],
) -> SourceResolution | None:
    """Convert known non-patchable runtime kernels into source resolutions."""
    for kernel_name in kernel_names:
        classification = _non_patchable_kernel(kernel_name)
        if classification is None:
            continue
        mapping_kind, reason = classification
        return SourceResolution(
            mapping_kind=mapping_kind,
            runtime_kernel_name=kernel_name,
            method="runtime_classification",
            confidence=ResolutionConfidence.EXACT,
            reason=reason,
        )
    return None


def generated_kernel_provenance(
    kernel_name: str,
    *,
    cache_roots: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Describe persistent compiler artifacts for a generated kernel."""
    stem = kernel_name.removesuffix(".kd")
    roots = sorted({str(Path(root).resolve()) for root in cache_roots if root and Path(root).is_dir()})

    def _artifacts(suffix: str) -> list[str]:
        paths: set[str] = set()
        for root in roots:
            paths.update(str(path.resolve()) for path in Path(root).rglob(f"{stem}{suffix}") if path.is_file())
        return sorted(paths)

    generated_sources = _artifacts(".source")
    generated_ttirs = _artifacts(".ttir")
    generated_origins: set[str] = set()
    for source in generated_sources:
        try:
            head = Path(source).read_text(
                encoding="utf-8",
                errors="replace",
            )[:4096]
        except OSError:
            continue
        for line in head.splitlines():
            match = _GENERATED_SOURCE_LOC_RE.match(line)
            if match:
                generated_origins.add(str(Path(match.group(1)).resolve()))
                break

    origins = sorted(generated_origins)
    reason = (
        "torch.compile generated kernel; generated source artifacts are "
        "available, but no stable one-to-one repository source exists"
    )
    return {
        "mapping_kind": "generated_artifact",
        "patchable": False,
        "source_file": "",
        "generated_source_file": (generated_sources[0] if generated_sources else ""),
        "generated_source_files": generated_sources,
        "generated_ttir_file": (generated_ttirs[0] if generated_ttirs else ""),
        "generated_ttir_files": generated_ttirs,
        "generated_origin_file": origins[0] if origins else "",
        "generated_origin_files": origins,
        "method": "torchinductor_generated",
        "confidence": "generated_artifact",
        "reason": reason,
        "resolver_version": SOURCE_RESOLVER_VERSION,
        "retryable": False,
    }


def _demangle(kernel_name: str) -> str:
    name = kernel_name.strip()
    if name.startswith("_Z"):
        candidates = []
        pos = 2
        if pos < len(name) and name[pos] == "N":
            pos += 1
        while pos < len(name):
            num_start = pos
            while pos < len(name) and name[pos].isdigit():
                pos += 1
            if pos == num_start:
                break
            length = int(name[num_start:pos])
            ident = name[pos : pos + length]
            if ident and re.match(r"^[A-Za-z_]\w*$", ident) and len(ident) > 3:
                candidates.append(ident)
            pos += length
            if pos < len(name) and name[pos] in ("I", "E", "L", "v", "b", "c"):
                break
        if candidates:
            specific = [candidate for candidate in candidates if candidate != "kentry"]
            return (specific or candidates)[-1]
        return name[:40]
    for prefix in ("void ", "__global__ ", "__device__ "):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    name = re.split(r"[<(]", name)[0].strip()
    name = name.removesuffix(".kd")
    name = name.rsplit("::", 1)[-1]
    return _MANGLE_SUFFIX_RE.sub("", name)


def _definition_name_candidates(name: str) -> list[str]:
    candidates = [name]
    for marker in (
        "_A_DTYPE_FORMAT_",
        "_BLOCK_SIZE_M_",
        "_num_experts_",
        "_numel_",
    ):
        if marker in name:
            candidates.append(name.split(marker, 1)[0])
    numbered = re.match(r"^(.+)_\d+$", name)
    if numbered:
        candidates.append(numbered.group(1))
    parts = name.split("_")
    for end in range(len(parts) - 1, 1, -1):
        candidates.append("_".join(parts[:end]))
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _grep_source_files(
    repo: str | Path,
    pattern: str,
    includes: tuple[str, ...],
) -> list[str]:
    if not pattern or not Path(repo).exists():
        return []
    exclude_directories = [f"--exclude-dir={directory}" for directory in source_search_excluded_directory_names()]
    try:
        result = subprocess.run(
            [
                "grep",
                "-rlE",
                *includes,
                *exclude_directories,
                pattern,
                str(repo),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    return sorted(line for line in result.stdout.splitlines() if line and is_searchable_source(line, repo))


def _grep_definitions(repo: str, name: str) -> list[str]:
    hits = {
        *_grep_source_files(
            repo,
            rf"\bdef\s+{re.escape(name)}\s*\(",
            ("--include=*.py",),
        ),
        *_grep_source_files(
            repo,
            rf"\b{re.escape(name)}\s*[(<]",
            _CPP_SOURCE_INCLUDES,
        ),
    }
    return sorted(hits)


def _compiler_for(path: str) -> str:
    suffix = Path(path).suffix
    if suffix in {".cc", ".cu", ".cuh", ".cpp", ".h", ".hip", ".hpp"}:
        return "hip_cpp"
    if suffix != ".py":
        return "unknown"
    try:
        head = Path(path).read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        return "python"
    if any(marker in head for marker in ("import flydsl", "from flydsl", "@flyc.kernel")):
        return "flydsl"
    if "import triton" in head or "@triton.jit" in head:
        return "triton"
    return "python"


def _file_defines_symbol(path: str | Path, symbol: str) -> bool:
    if not symbol:
        return True
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    escaped = re.escape(symbol)
    return bool(
        re.search(rf"\bdef\s+{escaped}\s*\(", text)
        or re.search(rf"__global__[^\n;{{]*\b{escaped}\b", text)
        or re.search(rf"\b{escaped}\s*[(<]", text)
    )


def _without_cpp_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _file_defines_gpu_symbol(
    path: str | Path,
    symbol: str,
) -> bool:
    if not symbol:
        return Path(path).is_file()
    try:
        text = Path(path).read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return False
    escaped = re.escape(symbol)
    suffix = Path(path).suffix
    if suffix == ".py":
        return bool(
            re.search(
                rf"@(?:triton\.jit|flyc\.kernel|[^ \n]*kernel)"
                rf"[^\n]*\n(?:\s*@[^\n]*\n)*\s*def\s+{escaped}\b",
                text,
            )
        )
    if suffix not in {".cu", ".cuh", ".cpp", ".h", ".hip", ".hpp"}:
        return False
    text = _without_cpp_comments(text)
    return bool(
        re.search(
            rf"__global__[\s\S]{{0,512}}?\b{escaped}\b"
            rf"\s*(?:<[^;{{}}]*>)?\s*\([^;{{}}]*\)"
            rf"[^;{{}}]*\{{",
            text,
        )
    )


def _file_defines_source_symbol(
    path: str | Path,
    symbol: str,
) -> bool:
    if not symbol:
        return Path(path).is_file()
    try:
        text = Path(path).read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return False
    escaped = re.escape(symbol)
    suffix = Path(path).suffix
    if suffix == ".py":
        return bool(re.search(rf"\bdef\s+{escaped}\s*\(", text))
    if suffix not in {".cu", ".cuh", ".cpp", ".h", ".hip", ".hpp"}:
        return False
    text = _without_cpp_comments(text)
    return bool(
        re.search(
            rf"\b{escaped}\b\s*(?:<[^;{{}}]*>)?"
            rf"\s*\([^;{{}}]*\)[^;{{}}]*\{{",
            text,
        )
    )


def _python_launcher_resolution(
    source: str | Path,
    symbol: str,
    repo: str | Path,
    role: str,
) -> SourceResolution:
    return SourceResolution(
        mapping_kind=MappingKind.UNRESOLVED,
        source_repo=str(Path(repo).resolve()),
        source_repo_role=role,
        launcher_source_file=str(Path(source).resolve()),
        launcher_symbol=symbol,
        compiler="python",
        method="python_launcher",
        confidence=ResolutionConfidence.UNRESOLVED,
        reason="full framework source repository required",
    )


def _qualified_type_definitions(
    repo: str,
    qualified_type: str,
) -> list[str]:
    namespace, _, symbol = qualified_type.rpartition("::")
    hits = _grep_source_files(
        repo,
        rf"\b(struct|class)\s+{re.escape(symbol)}\b",
        _CPP_SOURCE_INCLUDES,
    )
    if not namespace:
        return hits
    namespace_pattern = re.compile(rf"\bnamespace\s+{re.escape(namespace.split('::', 1)[0])}\b")
    matches = []
    for hit in hits:
        try:
            text = _without_cpp_comments(
                Path(hit).read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )
        except OSError:
            continue
        if namespace_pattern.search(text):
            matches.append(hit)
    return matches


def _file_sha256(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _git_head(repo: str | Path) -> str:
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


def _runtime_name_path_score(
    path: str | Path,
    runtime_names: list[str],
) -> int:
    runtime_text = "".join(re.findall(r"[a-z0-9]+", " ".join(runtime_names).lower()))
    path_tokens = set(re.findall(r"[a-z0-9]+", Path(path).stem.lower()))
    return sum(len(token) for token in path_tokens if len(token) >= 4 and token in runtime_text)


def _template_kernel_types(runtime_names: list[str]) -> list[str]:
    candidates = []
    for runtime_name in runtime_names:
        for qualified_type in _QUALIFIED_TEMPLATE_TYPE_RE.findall(
            runtime_name,
        ):
            symbol = qualified_type.rsplit("::", 1)[-1]
            if symbol != "kentry" and "kernel" in symbol.lower():
                candidates.append(qualified_type)
    return list(dict.fromkeys(candidates))


def _candidate_names(bottleneck: dict[str, Any]) -> list[str]:
    names = []
    for value in (
        bottleneck.get("op_name"),
        bottleneck.get("device_kernel_name"),
        *(bottleneck.get("kernel_names") or []),
    ):
        if value and value not in names:
            names.append(str(value))
    return names


def _parent_ops(bottleneck: dict[str, Any]) -> list[str]:
    values = [
        bottleneck.get("parent_op_name"),
        *(bottleneck.get("parent_op_names") or []),
    ]
    return list(dict.fromkeys(str(value) for value in values if value))


def _provenance_resolution(
    names: list[str],
    records: list[dict[str, Any]],
    repos: list[tuple[str, str]],
    gpu_arch: str,
) -> SourceResolution | None:
    normalized = {name if name.endswith(".kd") else f"{name}.kd" for name in names}
    repo_by_path = {str(Path(repo).resolve()): label for label, repo in repos}
    matches: list[SourceResolution] = []
    for record in records:
        runtime_name = str(record.get("runtime_kernel_name") or "")
        if runtime_name not in normalized:
            continue
        record_arch = str(record.get("gpu_arch") or "")
        if gpu_arch and record_arch and record_arch != gpu_arch:
            continue
        source_path = Path(str(record.get("source_file") or "")).absolute()
        if not source_path.is_file():
            continue
        source = source_path.resolve()
        source_repo = str(Path(str(record.get("source_repo") or source.parent)).resolve())
        label = repo_by_path.get(source_repo)
        if not label:
            continue
        try:
            source.relative_to(source_repo)
        except ValueError:
            continue
        if not is_searchable_source(source_path, source_repo):
            continue
        expected_hash = str(record.get("source_sha256") or "")
        if expected_hash and _file_sha256(source) != expected_hash:
            continue
        expected_revision = str(record.get("repo_revision") or "")
        if expected_revision and _git_head(source_repo) != expected_revision:
            continue
        symbol = str(record.get("source_symbol") or "")
        if symbol and not _file_defines_symbol(source, symbol):
            continue
        matches.append(
            SourceResolution(
                mapping_kind=MappingKind.EDITABLE_SOURCE,
                source_file=str(source),
                source_repo=source_repo,
                source_repo_role=str(record.get("source_repo_role") or label.removesuffix("_repo")),
                source_symbol=symbol,
                builder_symbol=str(record.get("builder_symbol") or ""),
                launcher_source_file=str(record.get("launcher_source_file") or source),
                compiler=str(record.get("compiler") or ""),
                runtime_kernel_name=runtime_name,
                backend=str(record.get("backend") or ""),
                gpu_arch=record_arch or gpu_arch,
                repo_revision=expected_revision,
                cache_key_hash=str(record.get("cache_key_hash") or ""),
                artifact_sha256=str(record.get("artifact_sha256") or ""),
                source_sha256=expected_hash,
                method="compiler_manifest",
                confidence=ResolutionConfidence.EXACT,
                reason=f"{label}: compiler provenance manifest",
            )
        )
    if not matches:
        return None
    unique = {
        (
            match.source_file,
            match.source_symbol,
            match.compiler,
        )
        for match in matches
    }
    if len(unique) > 1:
        return SourceResolution(
            mapping_kind=MappingKind.AMBIGUOUS,
            method="compiler_manifest",
            confidence=ResolutionConfidence.AMBIGUOUS,
            alternatives=tuple(sorted(match.source_file or "" for match in matches)),
            reason="multiple compiler provenance records matched",
        )
    return matches[0]


def _gpu_arch_dir(gpu_arch: str) -> str:
    value = str(gpu_arch or "").lower()
    if value.startswith("gfx"):
        return value
    if value in {"mi350x", "mi355x"} or "355" in value:
        return "gfx950"
    if value in {"mi300x", "mi308x", "mi325x"}:
        return "gfx942"
    return ""


def _aiter_code_object_resolution(
    names: list[str],
    kernel_repo: str,
    gpu_arch: str,
) -> SourceResolution | None:
    if not kernel_repo:
        return None
    symbols = {name.removesuffix(".kd") for name in names if "f4gemm_" in name or "fmoe_" in name}
    if not symbols:
        return None
    arch = _gpu_arch_dir(gpu_arch)
    hsa_root = _aiter_path(kernel_repo, "hsa")
    if arch:
        hsa_root = hsa_root / arch
    if not hsa_root.is_dir():
        return None
    matches = []
    for config in hsa_root.rglob("*.csv"):
        try:
            with config.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, csv.Error):
            continue
        for row in rows:
            kernel_name = str(row.get("knl_name") or row.get("kernelName") or row.get("kernel_name") or "")
            if not kernel_name or not any(
                symbol == kernel_name or symbol in kernel_name or kernel_name in symbol for symbol in symbols
            ):
                continue
            co_name = str(row.get("co_name") or row.get("binary") or "")
            binary = config.parent / co_name
            if not binary.is_file():
                continue
            matches.append((kernel_name, config, binary))
    if not matches:
        return None
    unique = {(str(config.resolve()), str(binary.resolve())) for _, config, binary in matches}
    if len(unique) > 1:
        return SourceResolution(
            mapping_kind=MappingKind.AMBIGUOUS,
            method="aiter_code_object_manifest",
            confidence=ResolutionConfidence.AMBIGUOUS,
            alternatives=tuple(sorted(binary for _, binary in unique)),
            reason="multiple AITER code objects matched runtime kernel",
        )
    kernel_name, config, binary = matches[0]
    launcher_name = "asm_gemm_a4w4.cu" if "f4gemm_" in kernel_name else "asm_fmoe.cu"
    launcher = _aiter_path(
        kernel_repo,
        Path("csrc") / "py_itfs_cu" / launcher_name,
    )
    return SourceResolution(
        mapping_kind=MappingKind.PRECOMPILED_BINARY,
        source_repo=str(Path(kernel_repo).resolve()),
        source_repo_role="kernel",
        launcher_source_file=(str(launcher) if launcher.is_file() else ""),
        compiler="aiter_asm",
        runtime_kernel_name=kernel_name,
        gpu_arch=gpu_arch,
        binary_file=str(binary.resolve()),
        config_file=str(config.resolve()),
        method="aiter_code_object_manifest",
        confidence=ResolutionConfidence.EXACT,
        reason=("AITER precompiled code object; implementation source is not shipped"),
    )


def _unique_gpu_definition(
    repo: str | Path,
    symbol: str,
) -> str:
    hits = [hit for hit in _grep_definitions(str(repo), symbol) if _file_defines_gpu_symbol(hit, symbol)]
    return hits[0] if len(hits) == 1 else ""


def _template_kernel_type_resolution(
    names: list[str],
    repos: list[tuple[str, str]],
) -> SourceResolution | None:
    if not any(_demangle(name) == "kentry" for name in names):
        return None
    definitions: dict[str, tuple[str, str, str, str]] = {}
    for qualified_type in _template_kernel_types(names):
        symbol = qualified_type.rsplit("::", 1)[-1]
        for label, repo in repos:
            for hit in _qualified_type_definitions(
                repo,
                qualified_type,
            ):
                definitions.setdefault(
                    hit,
                    (qualified_type, symbol, label, repo),
                )
    if not definitions:
        return None
    if len(definitions) > 1:
        return SourceResolution(
            mapping_kind=MappingKind.AMBIGUOUS,
            method="template_kernel_type",
            confidence=ResolutionConfidence.AMBIGUOUS,
            alternatives=tuple(sorted(definitions)),
            reason="multiple template kernel type definitions matched",
        )

    source, (qualified_type, symbol, label, repo) = next(iter(definitions.items()))
    launcher_symbol = "kentry" if any(_demangle(name) == "kentry" for name in names) else ""
    ck_root = _aiter_path(repo, "3rdparty/composable_kernel").resolve()
    launcher_root = (
        ck_root if ck_root.is_dir() and Path(source).resolve().is_relative_to(ck_root) else Path(repo).resolve()
    )
    launcher_source = _unique_gpu_definition(launcher_root, launcher_symbol) if launcher_symbol else ""

    return SourceResolution(
        mapping_kind=MappingKind.EDITABLE_SOURCE,
        source_file=source,
        source_repo=repo,
        source_repo_role=label.removesuffix("_repo"),
        source_symbol=symbol,
        launcher_source_file=launcher_source,
        launcher_symbol=launcher_symbol,
        compiler=_compiler_for(source),
        method="template_kernel_type",
        confidence=ResolutionConfidence.UNIQUE_DEFINITION,
        reason=f"{label}: unique template kernel type {qualified_type}",
    )


def _exact_gpu_definition_resolution(
    names: list[str],
    repos: list[tuple[str, str]],
) -> SourceResolution | None:
    definitions: dict[str, tuple[str, str, str]] = {}
    for label, repo in repos:
        for raw_name in names:
            symbol = _demangle(raw_name)
            if not symbol:
                continue
            for hit in _grep_definitions(repo, symbol):
                if _file_defines_gpu_symbol(hit, symbol):
                    definitions.setdefault(
                        hit,
                        (symbol, label, repo),
                    )
    if len(definitions) == 1:
        source, (symbol, label, repo) = next(iter(definitions.items()))
        return SourceResolution(
            mapping_kind=MappingKind.EDITABLE_SOURCE,
            source_file=source,
            source_repo=repo,
            source_repo_role=label.removesuffix("_repo"),
            source_symbol=symbol,
            compiler=_compiler_for(source),
            method="exact_gpu_definition",
            confidence=ResolutionConfidence.UNIQUE_DEFINITION,
            reason=f"{label}: exact GPU definition {symbol}",
        )
    if len(definitions) > 1:
        scores = {
            source: _runtime_name_path_score(
                source,
                names,
            )
            for source in definitions
        }
        best_score = max(scores.values())
        best_matches = [source for source, score in scores.items() if score == best_score]
        if best_score > 0 and len(best_matches) == 1:
            source = best_matches[0]
            symbol, label, repo = definitions[source]
            return SourceResolution(
                mapping_kind=MappingKind.EDITABLE_SOURCE,
                source_file=source,
                source_repo=repo,
                source_repo_role=label.removesuffix("_repo"),
                source_symbol=symbol,
                compiler=_compiler_for(source),
                method="exact_gpu_definition_path_match",
                confidence=ResolutionConfidence.UNIQUE_DEFINITION,
                reason=f"{label}: runtime name uniquely matched {Path(source).name}",
            )
        return SourceResolution(
            mapping_kind=MappingKind.AMBIGUOUS,
            method="exact_gpu_definition",
            confidence=ResolutionConfidence.AMBIGUOUS,
            alternatives=tuple(sorted(definitions)),
            reason=("multiple exact GPU definitions matched across managed repositories"),
        )
    return None


def _rule_resolution(
    rule: SourceRule,
    repo: str,
    label: str,
) -> SourceResolution | None:
    role = label.removesuffix("_repo")
    if rule.repo_role and rule.repo_role != role:
        return None
    source = _aiter_path(repo, rule.relative_source) if role == "kernel" else Path(repo) / rule.relative_source
    validator = (
        _file_defines_gpu_symbol if rule.compiler in {"flydsl", "triton", "hip_cpp"} else _file_defines_source_symbol
    )
    if not source.is_file() or not is_searchable_source(source, repo) or not validator(source, rule.source_symbol):
        return None
    launcher = (
        _aiter_path(repo, rule.relative_launcher)
        if role == "kernel" and rule.relative_launcher
        else Path(repo) / rule.relative_launcher
        if rule.relative_launcher
        else None
    )
    return SourceResolution(
        mapping_kind=MappingKind.EDITABLE_SOURCE,
        source_file=str(source),
        source_repo=str(Path(repo).resolve()),
        source_repo_role=role,
        source_symbol=rule.source_symbol,
        builder_symbol=rule.builder_symbol,
        launcher_source_file=(str(launcher) if launcher is not None and launcher.is_file() else ""),
        live_call_seam=rule.live_call_seam,
        build_module=rule.build_module,
        compiler=rule.compiler,
        method="operator_rule",
        confidence=ResolutionConfidence.OPERATOR_RULE,
        reason=f"{label}: operator rule {rule.label}",
    )


def _aiter_module_source(
    kernel_repo: str,
    raw_source: str,
) -> Path | None:
    value = str(raw_source or "")
    marker = "AITER_CSRC_DIR}/"
    if marker in value:
        relative = value.split(marker, 1)[1]
        relative = relative.split("'", 1)[0].split('"', 1)[0]
        path = _aiter_path(
            kernel_repo,
            Path("csrc") / relative,
        )
    else:
        stripped = value.strip("'\"")
        path = Path(stripped)
        if not path.is_absolute():
            path = Path(kernel_repo) / stripped
    return path.resolve() if path.is_file() and is_searchable_source(path, kernel_repo) else None


def _aiter_build_module_resolution(
    names: list[str],
    kernel_repo: str,
) -> SourceResolution | None:
    config_path = Path(kernel_repo) / "aiter" / "jit" / "optCompilerConfig.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    symbols = {_demangle(name) for name in names if _demangle(name)}
    matches: dict[str, dict[str, set[str]]] = {}
    for module_name, module in config.items():
        if not isinstance(module, dict):
            continue
        for raw_source in module.get("srcs") or []:
            source = _aiter_module_source(
                kernel_repo,
                str(raw_source),
            )
            if source is None:
                continue
            for symbol in symbols:
                if not _file_defines_gpu_symbol(source, symbol):
                    continue
                row = matches.setdefault(
                    str(source),
                    {"symbols": set(), "modules": set()},
                )
                row["symbols"].add(symbol)
                row["modules"].add(str(module_name))
    if len(matches) == 1:
        source_name, metadata = next(iter(matches.items()))
        symbol = sorted(metadata["symbols"])[0]
        modules = sorted(metadata["modules"])
        return SourceResolution(
            mapping_kind=MappingKind.EDITABLE_SOURCE,
            source_file=source_name,
            source_repo=str(Path(kernel_repo).resolve()),
            source_repo_role="kernel",
            source_symbol=symbol,
            build_module=",".join(modules),
            compiler="hip_cpp",
            method="aiter_build_module",
            confidence=ResolutionConfidence.EXACT,
            reason=(f"kernel_repo: AITER build module {','.join(modules)}"),
        )
    if len(matches) > 1:
        return SourceResolution(
            mapping_kind=MappingKind.AMBIGUOUS,
            method="aiter_build_module",
            confidence=ResolutionConfidence.AMBIGUOUS,
            alternatives=tuple(sorted(matches)),
            reason="multiple AITER build-module sources matched",
        )
    return None


def _upstream_source_resolution(
    bottleneck: dict[str, Any],
    names: list[str],
    repos: list[tuple[str, str]],
    kernel_repo: str,
) -> SourceResolution | None:
    source_file = str(bottleneck.get("source_file") or "")
    if not source_file:
        return None
    source_path = Path(source_file).absolute()
    resolved_source_path = source_path.resolve()
    for label, repo in repos:
        resolved_repo = Path(repo).resolve()
        try:
            resolved_source_path.relative_to(resolved_repo)
        except ValueError:
            continue
        if not source_path.is_file():
            continue
        role = label.removesuffix("_repo")
        if not is_searchable_source(source_path, resolved_repo):
            return SourceResolution(
                mapping_kind=MappingKind.GENERATED_ARTIFACT,
                source_repo=str(resolved_repo),
                source_repo_role=role,
                method="upstream_source_file",
                confidence=ResolutionConfidence.EXACT,
                alternatives=(str(source_path),),
                reason=f"{label}: upstream source_file is a runtime-generated build artifact",
            )
        compiler = _compiler_for(str(source_path))
        validator = _file_defines_source_symbol if compiler == "python" else _file_defines_gpu_symbol
        symbol = next(
            (
                candidate
                for name in names
                for candidate in _definition_name_candidates(_demangle(name))
                if validator(source_path, candidate)
            ),
            "",
        )
        if compiler == "python":
            return _python_launcher_resolution(source_path, symbol, resolved_repo, role)
        if not symbol:
            continue
        return _with_dependency_ownership(
            SourceResolution(
                mapping_kind=MappingKind.EDITABLE_SOURCE,
                source_file=str(resolved_source_path),
                source_repo=str(resolved_repo),
                source_repo_role=role,
                source_symbol=symbol,
                compiler=compiler,
                method="upstream_source_file",
                confidence=ResolutionConfidence.EXACT,
                reason=f"{label}: upstream source_file",
            ),
            kernel_repo,
        )
    return None


def _definition_search_resolution(
    names: list[str],
    repos: list[tuple[str, str]],
    kernel_repo: str,
) -> SourceResolution:
    alternatives: dict[str, tuple[str, str, str]] = {}
    for label, repo in repos:
        for raw_name in names:
            for symbol in _definition_name_candidates(_demangle(raw_name)):
                for hit in _grep_definitions(repo, symbol):
                    if _file_defines_source_symbol(hit, symbol):
                        alternatives.setdefault(hit, (symbol, label, repo))
    if len(alternatives) == 1:
        source, (symbol, label, repo) = next(iter(alternatives.items()))
        if _compiler_for(source) == "python":
            return _python_launcher_resolution(source, symbol, repo, label.removesuffix("_repo"))
        return _with_dependency_ownership(
            SourceResolution(
                mapping_kind=MappingKind.EDITABLE_SOURCE,
                source_file=source,
                source_repo=repo,
                source_repo_role=label.removesuffix("_repo"),
                source_symbol=symbol,
                compiler=_compiler_for(source),
                method="unique_definition",
                confidence=ResolutionConfidence.UNIQUE_DEFINITION,
                reason=f"{label}: unique definition {symbol}",
            ),
            kernel_repo,
        )
    if alternatives:
        return SourceResolution(
            mapping_kind=MappingKind.AMBIGUOUS,
            method="definition_search",
            confidence=ResolutionConfidence.AMBIGUOUS,
            alternatives=tuple(sorted(alternatives)),
            reason="multiple source definitions matched across managed repositories",
        )
    return SourceResolution(
        mapping_kind=MappingKind.UNRESOLVED,
        method="definition_search",
        confidence=ResolutionConfidence.UNRESOLVED,
        reason="no source found after deterministic resolution",
    )


def resolve_kernel_source_repo(
    bottleneck: dict[str, Any],
    framework_repo: str,
    kernel_repo: str = "",
    *,
    context_tags: tuple[str, ...] = (),
    provenance_records: list[dict[str, Any]] | None = None,
    gpu_arch: str = "",
    external_source_roots: dict[str, str] | None = None,
) -> SourceResolution:
    """Resolve one kernel to a unique editable source with auditable evidence."""
    repos = [
        (label, str(Path(repo).resolve()))
        for label, repo in (
            ("kernel_repo", kernel_repo),
            ("framework_repo", framework_repo),
        )
        if repo
    ]

    def _finalize(
        resolution: SourceResolution,
    ) -> SourceResolution:
        return _with_dependency_ownership(
            resolution,
            kernel_repo,
        )

    names = _candidate_names(bottleneck)
    runtime_classification = _runtime_classification_resolution(names)
    if runtime_classification is not None:
        return runtime_classification

    upstream = _upstream_source_resolution(
        bottleneck,
        names,
        repos,
        kernel_repo,
    )
    if upstream is not None:
        return upstream

    parents = _parent_ops(bottleneck)
    provenance = _provenance_resolution(
        names,
        list(provenance_records or []),
        repos,
        gpu_arch,
    )
    if provenance is not None:
        return _finalize(provenance)
    code_object = _aiter_code_object_resolution(
        names,
        kernel_repo,
        gpu_arch,
    )
    if code_object is not None:
        return code_object
    for rule in SOURCE_RULES:
        if not rule.matches(names, parents, context_tags):
            continue
        for label, repo in repos:
            resolution = _rule_resolution(rule, repo, label)
            if resolution:
                return _finalize(resolution)
    build_module_resolution = _aiter_build_module_resolution(
        names,
        kernel_repo,
    )
    if build_module_resolution is not None:
        return _finalize(build_module_resolution)
    template_kernel_type = _template_kernel_type_resolution(
        names,
        repos,
    )
    if template_kernel_type is not None:
        return _finalize(template_kernel_type)
    exact_definition = _exact_gpu_definition_resolution(names, repos)
    if exact_definition is not None:
        return _finalize(exact_definition)
    external_repos = [
        (f"{role}_repo", str(Path(root).resolve())) for role, root in (external_source_roots or {}).items() if root
    ]
    external_definition = _exact_gpu_definition_resolution(
        names,
        external_repos,
    )
    if external_definition is not None:
        if external_definition.source_file:
            return replace(
                external_definition,
                mapping_kind=MappingKind.EXTERNAL_UNMANAGED_SOURCE,
                method="external_exact_gpu_definition",
                reason=(f"{external_definition.source_repo_role}: source is outside managed modifiable workspaces"),
            )
        return external_definition

    return _definition_search_resolution(names, repos, kernel_repo)
