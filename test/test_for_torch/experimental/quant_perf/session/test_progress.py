#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for quark.experimental.torch.quant_perf.session.progress: progress.json read-modify-write with
atomic replace and a capped rolling warnings list (IMPL_SPEC §4.8)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quark.experimental.torch.quant_perf.session.progress import read_progress, write_progress


def test_read_progress_missing_file_returns_none(tmp_path):
    assert read_progress(tmp_path) is None


def test_write_then_read_roundtrip(tmp_path):
    write_progress(tmp_path, stage="quantize")
    data = read_progress(tmp_path)
    assert data["stage"] == "quantize"
    assert "started_at" in data
    assert "updated_at" in data


def test_write_progress_preserves_untouched_fields(tmp_path):
    write_progress(tmp_path, stage="quantize")
    write_progress(tmp_path, stage_detail="round 1/20")
    data = read_progress(tmp_path)
    assert data["stage"] == "quantize"
    assert data["stage_detail"] == "round 1/20"


def test_warnings_accumulate():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        write_progress(d, warning="first warning")
        write_progress(d, warning="second warning")
        data = read_progress(d)
        assert data["warnings"] == ["first warning", "second warning"]


def test_warnings_capped_at_50(tmp_path):
    for i in range(60):
        write_progress(tmp_path, warning=f"warning {i}")
    data = read_progress(tmp_path)
    assert len(data["warnings"]) == 50
    assert data["warnings"][0] == "warning 10"
    assert data["warnings"][-1] == "warning 59"


def test_elapsed_seconds_present_and_nonnegative(tmp_path):
    write_progress(tmp_path, stage="quantize")
    data = read_progress(tmp_path)
    assert data["elapsed_seconds"] >= 0


@pytest.mark.parametrize("status", ["running", "not_fixed"])
def test_repair_elapsed_time_is_projected_without_writing(tmp_path, monkeypatch, status):
    write_progress(
        tmp_path,
        repair={
            "status": status,
            "started_at": datetime.fromtimestamp(100, UTC).isoformat(),
            "repair_started_at": datetime.fromtimestamp(80, UTC).isoformat(),
            "elapsed_seconds": 5.0,
            "total_elapsed_seconds": 25.0,
        },
    )
    stored = (tmp_path / "progress.json").read_text()
    monkeypatch.setattr("quark.experimental.torch.quant_perf.session.progress.time.time", lambda: 130.0)
    repair = read_progress(tmp_path)["repair"]
    assert repair["elapsed_seconds"] == (30.0 if status == "running" else 5.0)
    assert repair["total_elapsed_seconds"] == (50.0 if status == "running" else 25.0)
    assert (tmp_path / "progress.json").read_text() == stored


def test_concurrent_writers_use_independent_temporary_files(monkeypatch, tmp_path):
    replace_barrier = threading.Barrier(2)
    original_replace = Path.replace
    errors = []
    temporary_paths = []

    def synchronized_replace(source, target):
        if source.name.startswith("progress.tmp"):
            temporary_paths.append(source)
            replace_barrier.wait(timeout=5)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", synchronized_replace)

    def write(**fields):
        try:
            write_progress(tmp_path, **fields)
        except Exception as error:
            errors.append(error)

    writers = [
        threading.Thread(target=write, kwargs={"stage": "quantize"}),
        threading.Thread(target=write, kwargs={"stage_detail": "round 1"}),
    ]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join()

    assert errors == []
    assert len(set(temporary_paths)) == 2
    assert read_progress(tmp_path) is not None
