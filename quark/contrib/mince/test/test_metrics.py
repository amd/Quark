#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for mince.mince_metrics + montecarlo vectorized-MC parity.

The vectorized Monte-Carlo helpers (montecarlo.py) must produce exactly the same
drift as the slow, obvious ``compute_accuracy(subset) - compute_accuracy(full)``
reference from mince_metrics.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from quark.contrib.mince.data_loader import load_benchmark_items
from quark.contrib.mince.mince_metrics import compute_accuracy, compute_drift
from quark.contrib.mince.montecarlo import (
    precompute_vectors_ifeval,
    precompute_vectors_simple,
    subsample_drift_ifeval,
    subsample_drift_simple,
)
from quark.contrib.mince.test.utils import Fixture


def test_compute_accuracy_gsm8k(gsm8k_data: Fixture) -> None:
    config, model_paths = gsm8k_data
    items = load_benchmark_items(config, model_paths)
    accs = compute_accuracy(items, "modelA", config)
    # exact_match = i % 2 for i in 0..5 -> [0,1,0,1,0,1] -> mean 0.5
    assert accs["exact_match"] == 0.5


def test_compute_drift_is_abs_subset_minus_full() -> None:
    full = {"acc": 0.80}
    subset = {"acc": 0.75}
    assert compute_drift(full, subset)["acc"] == pytest.approx(0.05)


def test_simple_vectorized_matches_reference(mmlu_data: Fixture) -> None:
    config, model_paths = mmlu_data
    items = load_benchmark_items(config, model_paths)

    vals, full_acc = precompute_vectors_simple(items, "modelA", "acc")
    assert full_acc == compute_accuracy(items, "modelA", config)["acc"]

    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = rng.choice(len(items), size=5, replace=False)
        vec_drift = subsample_drift_simple(vals, full_acc, idx)
        sub_items = [items[i] for i in idx]
        ref_drift = compute_accuracy(sub_items, "modelA", config)["acc"] - full_acc
        assert abs(vec_drift - ref_drift) < 1e-12


def test_ifeval_vectorized_matches_reference(ifeval_data: Fixture) -> None:
    config, model_paths = ifeval_data
    items = load_benchmark_items(config, model_paths)

    vectors, full_accs = precompute_vectors_ifeval(items, "modelA")
    ref_full = compute_accuracy(items, "modelA", config)
    for m in ["prompt_strict", "prompt_loose", "inst_strict", "inst_loose"]:
        assert abs(full_accs[m] - ref_full[m]) < 1e-12

    rng = np.random.default_rng(1)
    for _ in range(50):
        idx = rng.choice(len(items), size=5, replace=False)
        vec_drift = subsample_drift_ifeval(vectors, full_accs, idx)
        sub_items = [items[i] for i in idx]
        ref = compute_accuracy(sub_items, "modelA", config)
        for m in vec_drift:
            assert abs(vec_drift[m] - (ref[m] - full_accs[m])) < 1e-12
