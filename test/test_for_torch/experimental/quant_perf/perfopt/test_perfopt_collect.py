#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from pathlib import Path
from types import SimpleNamespace

import quark.experimental.torch.quant_perf.perfopt._tracelens as tracelens
import quark.experimental.torch.quant_perf.perfopt.collect as collect
from quark.experimental.torch.quant_perf.perfopt.collect import select_engine_rank0_trace


def test_select_engine_rank0_trace_ignores_frontend_and_accepts_rank0(tmp_path):
    frontend = tmp_path / "host_123.async_llm.1.pt.trace.json.gz"
    engine = tmp_path / "dp0_pp0_tp0_dcp0_ep0_rank0.2.pt.trace.json.gz"
    frontend.touch()
    engine.touch()

    assert select_engine_rank0_trace([frontend, engine]) == engine


def test_select_engine_rank0_trace_accepts_rank_0_and_is_order_independent(
    tmp_path,
):
    rank1 = tmp_path / "worker_rank_1.pt.trace.json.gz"
    rank0 = tmp_path / "worker_rank_0.pt.trace.json.gz"
    rank1.touch()
    rank0.touch()

    assert select_engine_rank0_trace([rank1, rank0]) == rank0
    assert select_engine_rank0_trace([rank0, rank1]) == rank0


def test_select_engine_rank0_trace_waits_when_only_frontend_exists(tmp_path):
    frontend = Path(tmp_path / "host_123.async_llm.1.pt.trace.json.gz")
    frontend.touch()

    assert select_engine_rank0_trace([frontend]) is None


def test_split_steady_state_skips_subprocess_without_tracelens(monkeypatch, tmp_path):
    raw_trace = tmp_path / "rank0.pt.trace.json.gz"
    raw_trace.touch()
    subprocess_calls = []

    monkeypatch.setattr(tracelens.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(
        collect.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)) or SimpleNamespace(returncode=0),
    )

    assert collect._split_steady_state(raw_trace) == raw_trace
    assert subprocess_calls == []
