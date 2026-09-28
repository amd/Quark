#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import pickle
from pathlib import Path

from quark.experimental.torch.quant_perf.perfopt.kernel_provenance import (
    PROVENANCE_SCHEMA_VERSION,
    RESOLUTION_SCHEMA_VERSION,
    collect_flydsl_cache_provenance,
    load_kernel_provenance,
    load_kernel_source_resolution,
    write_kernel_provenance,
    write_kernel_source_resolution,
)

from ..testing import init_git_repo
from ..testing import run_git as _git


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "aiter"
    source = repo / "aiter" / "ops" / "flydsl" / "kernels" / "mxfp4_preshuffle.py"
    init_git_repo(
        repo,
        {"aiter/ops/flydsl/kernels/mxfp4_preshuffle.py": "@flyc.kernel\ndef kernel_gemm(x):\n    return x\n"},
    )
    return repo, source


def _cache_artifact(
    cache_root: Path,
    *,
    source: Path,
    runtime_name: str = "kernel_gemm_0",
) -> Path:
    artifact = cache_root / "launch_gemm_0123456789abcdef0123456789abcdef" / "abc123def4567890.pkl"
    artifact.parent.mkdir(parents=True)
    source_ir = (
        f'#loc1 = loc("{source}":2:0)\n'
        "module {\n"
        "  gpu.module @kernels {\n"
        f"    gpu.func @{runtime_name}() kernel loc(#loc1)\n"
        "  }\n"
        "}\n"
    )
    artifact.write_bytes(
        pickle.dumps(
            {
                "entry": "launch_gemm",
                "source_ir": source_ir,
            }
        )
    )
    return artifact


def test_collect_flydsl_cache_provenance_uses_mlir_without_unpickling(
    tmp_path,
):
    repo, source = _repo(tmp_path)
    cache_root = tmp_path / "flydsl"
    artifact = _cache_artifact(cache_root, source=source)

    records = collect_flydsl_cache_provenance(
        cache_root,
        source_roots={"kernel": repo},
        gpu_arch="MI355X",
    )

    assert len(records) == 1
    record = records[0]
    assert record["runtime_kernel_name"] == "kernel_gemm_0.kd"
    assert record["compiler"] == "flydsl"
    assert record["source_file"] == str(source.resolve())
    assert record["source_relpath"] == ("aiter/ops/flydsl/kernels/mxfp4_preshuffle.py")
    assert record["source_symbol"] == "kernel_gemm"
    assert record["builder_symbol"] == "launch_gemm"
    assert record["source_repo_role"] == "kernel"
    assert record["cache_key_hash"] == artifact.stem
    assert record["repo_revision"] == _git(repo, "rev-parse", "HEAD")
    assert record["source_sha256"]
    assert record["artifact_sha256"]


def test_collect_flydsl_cache_provenance_rejects_external_source(
    tmp_path,
):
    repo, _ = _repo(tmp_path)
    external = tmp_path / "external.py"
    external.write_text("def kernel_gemm():\n    pass\n")
    cache_root = tmp_path / "flydsl"
    _cache_artifact(cache_root, source=external)

    assert (
        collect_flydsl_cache_provenance(
            cache_root,
            source_roots={"kernel": repo},
            gpu_arch="MI355X",
        )
        == []
    )


def test_collect_flydsl_cache_provenance_ignores_malformed_pickle(
    tmp_path,
):
    repo, _ = _repo(tmp_path)
    artifact = tmp_path / "flydsl" / "launch_bad_deadbeef" / "bad.pkl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"not-a-pickle")

    assert (
        collect_flydsl_cache_provenance(
            artifact.parents[1],
            source_roots={"kernel": repo},
            gpu_arch="MI355X",
        )
        == []
    )


def test_collect_flydsl_cache_provenance_ignores_empty_cache_root(
    tmp_path,
    monkeypatch,
):
    repo, source = _repo(tmp_path)
    _cache_artifact(tmp_path / "unexpected-cache", source=source)
    monkeypatch.chdir(tmp_path)

    assert (
        collect_flydsl_cache_provenance(
            "",
            source_roots={"kernel": repo},
            gpu_arch="MI355X",
        )
        == []
    )


def test_kernel_provenance_artifact_round_trip(tmp_path):
    repo, source = _repo(tmp_path)
    cache_root = tmp_path / "flydsl"
    _cache_artifact(cache_root, source=source)
    records = collect_flydsl_cache_provenance(
        cache_root,
        source_roots={"kernel": repo},
        gpu_arch="MI355X",
    )

    path = write_kernel_provenance(tmp_path / "session", records)
    document = json.loads(path.read_text())

    assert document["schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert document["entries"] == records
    assert load_kernel_provenance(path.parent) == records


def test_kernel_source_resolution_artifact_projects_kernel_journey(
    tmp_path,
):
    journeys = [
        {
            "kernel_id": "kernel_gemm_0.kd",
            "name": "kernel_gemm_0.kd",
            "outcome": "selected",
            "source_mapping": {
                "mapping_kind": "editable_source",
                "method": "compiler_manifest",
                "confidence": "exact",
                "source_file": "/repo/kernel.py",
                "patchable": True,
                "retryable": False,
            },
        },
        {
            "kernel_id": "asm.kd",
            "name": "asm.kd",
            "outcome": "skipped",
            "skip_reason": "precompiled",
            "source_mapping": {
                "mapping_kind": "precompiled_binary",
                "method": "aiter_code_object_manifest",
                "confidence": "exact",
                "binary_file": "/repo/kernel.co",
                "patchable": False,
                "retryable": False,
            },
        },
    ]

    path = write_kernel_source_resolution(
        tmp_path / "session",
        journeys,
    )
    document = json.loads(path.read_text())

    assert document["schema_version"] == RESOLUTION_SCHEMA_VERSION
    assert [row["kernel_id"] for row in document["entries"]] == [
        "kernel_gemm_0.kd",
        "asm.kd",
    ]
    assert document["entries"][1]["skip_reason"] == "precompiled"
    assert load_kernel_source_resolution(path.parent) == document["entries"]
