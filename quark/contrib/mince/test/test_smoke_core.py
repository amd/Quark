#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""End-to-end smoke: size -> select n* -> freeze runs without error on fixtures."""

from __future__ import annotations

from quark.contrib.mince.montecarlo import run_sizing
from quark.contrib.mince.subset import build_frozen_subset
from quark.contrib.mince.test.utils import Fixture


def test_size_then_freeze_ifeval(ifeval_data: Fixture) -> None:
    """Sizing then freezing IFEVAL runs end to end and yields an n-sized subset."""
    config, model_paths = ifeval_data
    bundle = run_sizing(config, model_paths, B=200, seed=42)

    # Every candidate N has a worst-case P95 drift.
    assert set(bundle["worst_p95_by_n"]) == set(config.candidate_ns)
    assert all(v >= 0 for v in bundle["worst_p95_by_n"].values())

    # Freeze on a candidate N directly (n* selection is covered in
    # test_selection.py; a tiny fixture is not guaranteed to converge).
    n = sorted(bundle["worst_p95_by_n"])[len(config.candidate_ns) // 2]
    art, _ = build_frozen_subset(config, model_paths, n=n, seed=42)
    assert art["n"] == n
    assert len(art["indices"]) == n


def test_sizing_is_deterministic(gsm8k_data: Fixture) -> None:
    """Two sizing runs at the same seed produce identical worst-case P95 curves."""
    config, model_paths = gsm8k_data
    a = run_sizing(config, model_paths, B=200, seed=42)
    b = run_sizing(config, model_paths, B=200, seed=42)
    assert a["worst_p95_by_n"] == b["worst_p95_by_n"]


def test_sizing_runs_for_all_simple_benchmarks(
    mmlu_data: Fixture, mmlu_pro_data: Fixture, commonsense_qa_data: Fixture, piqa_data: Fixture
) -> None:
    """Sizing completes for every remaining benchmark, covering all candidate Ns."""
    for fixture in (mmlu_data, mmlu_pro_data, commonsense_qa_data, piqa_data):
        config, model_paths = fixture
        bundle = run_sizing(config, model_paths, B=100, seed=1)
        assert set(bundle["worst_p95_by_n"]) == set(config.candidate_ns)
