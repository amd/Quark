#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for raw GPU-kernel aggregation fallback in PerfOpt locate."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import quark.experimental.torch.quant_perf.perfopt._tracelens as tracelens
import quark.experimental.torch.quant_perf.perfopt.locate as locate
from quark.experimental.torch.quant_perf.perfopt.locate import _raw_kernel_bottlenecks


def _write_trace(
    path: Path,
    events: list[dict],
    *,
    gzipped: bool = False,
) -> Path:
    content = json.dumps({"traceEvents": events}).encode()
    if gzipped:
        trace = path / "trace.json.gz"
        with gzip.open(trace, "wb") as stream:
            stream.write(content)
        return trace
    trace = path / "trace.json"
    trace.write_bytes(content)
    return trace


def test_raw_kernel_bottlenecks_aggregates_and_returns_schema(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            {"cat": "kernel", "name": "fmha_fwd", "dur": 1000},
            {"cat": "kernel", "name": "fmha_fwd", "dur": 2000},
            {"cat": "kernel", "name": "gemm_kernel", "dur": 500},
        ],
    )

    result = _raw_kernel_bottlenecks(trace, top_n=5)

    assert len(result) == 2
    assert result[0]["kernel_names"] == ["fmha_fwd"]
    assert result[0]["device_kernel_name"] == "fmha_fwd"
    assert result[0]["op_name"] == "fmha_fwd"
    assert result[0]["parent_op_name"] == ""
    assert result[0]["parent_op_names"] == []
    assert result[0]["call_count"] == 2
    assert result[0]["gemm_shape"] == {
        "M": None,
        "N": None,
        "K": None,
    }
    assert result[0]["kernel_time_us"] == 3000.0
    assert result[0]["roofline_bound"] == "UNKNOWN"
    assert result[0]["sol_pct"] is None
    assert result[0]["repro_ir"] is None


def test_raw_kernel_bottlenecks_respects_top_n(tmp_path):
    trace = _write_trace(
        tmp_path,
        [{"cat": "kernel", "name": f"k{i}", "dur": i * 100} for i in range(10, 0, -1)],
    )
    result = _raw_kernel_bottlenecks(trace, top_n=3)
    assert len(result) == 3
    assert result[0]["kernel_time_us"] == 1000.0


def test_raw_kernel_bottlenecks_reads_gzip(tmp_path):
    trace = _write_trace(
        tmp_path,
        [{"cat": "kernel", "name": "fp8_quant", "dur": 500}],
        gzipped=True,
    )
    assert _raw_kernel_bottlenecks(trace, top_n=5)[0]["op_name"] == "fp8_quant"


def test_raw_kernel_bottlenecks_skips_non_kernel_events(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            {"cat": "cpu_op", "name": "aten::mm", "dur": 9999},
            {"cat": "kernel", "name": "gpu_kernel", "dur": 100},
        ],
    )
    result = _raw_kernel_bottlenecks(trace, top_n=5)
    assert [row["op_name"] for row in result] == ["gpu_kernel"]


def test_raw_kernel_bottlenecks_empty_trace(tmp_path):
    trace = _write_trace(tmp_path, [])
    assert _raw_kernel_bottlenecks(trace, top_n=5) == []


def test_locate_bottlenecks_falls_back_when_tracelens_is_unavailable(monkeypatch, tmp_path):
    trace = _write_trace(
        tmp_path,
        [{"cat": "kernel", "name": "mxfp4_gemm", "dur": 500}],
    )
    monkeypatch.setattr(tracelens.importlib.util, "find_spec", lambda name: None)

    result = locate.locate_bottlenecks(trace, top_n=5)

    assert result[0]["op_name"] == "mxfp4_gemm"
    assert result[0]["roofline_bound"] == "UNKNOWN"


def test_locate_bottlenecks_falls_back_when_tracelens_lacks_gpu_arch(monkeypatch, tmp_path):
    trace = _write_trace(
        tmp_path,
        [{"cat": "kernel", "name": "mxfp4_gemm", "dur": 500}],
    )

    def unsupported_arch(*, gpu_arch_platform):
        raise KeyError(gpu_arch_platform)

    monkeypatch.setattr(
        locate,
        "_load_tracelens",
        lambda: SimpleNamespace(resolve_gpu_arch=unsupported_arch),
        raising=False,
    )

    result = locate.locate_bottlenecks(trace, gpu="MI355X", top_n=5)

    assert result[0]["op_name"] == "mxfp4_gemm"
    assert result[0]["roofline_bound"] == "UNKNOWN"


def test_locate_bottlenecks_prefers_explicit_gpu_arch_json(monkeypatch, tmp_path):
    trace = _write_trace(tmp_path, [])
    arch_json = tmp_path / "MI355X.json"
    arch_json.write_text('{"name": "MI355X"}')
    resolved = []

    def resolve_arch(**kwargs):
        resolved.append(kwargs)
        raise RuntimeError("resolved explicit architecture")

    monkeypatch.setattr(
        locate,
        "_load_tracelens",
        lambda: SimpleNamespace(resolve_gpu_arch=resolve_arch),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="resolved explicit architecture"):
        locate.locate_bottlenecks(
            trace,
            gpu="MI355X",
            gpu_arch_json_path=arch_json,
            top_n=5,
        )

    assert resolved == [{"gpu_arch_json_path": str(arch_json)}]
