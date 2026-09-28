#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for subset.build_frozen_subset: same seed -> identical subset; stable schema."""

from __future__ import annotations

import numpy as np

from quark.contrib.mince.montecarlo import random_subset
from quark.contrib.mince.subset import build_frozen_subset
from quark.contrib.mince.test.utils import Fixture


def test_same_seed_gives_identical_indices(gsm8k_data: Fixture) -> None:
    config, model_paths = gsm8k_data
    a, _ = build_frozen_subset(config, model_paths, n=3, seed=42)
    b, _ = build_frozen_subset(config, model_paths, n=3, seed=42)
    assert a["indices"] == b["indices"]
    assert [it["index"] for it in a["items"]] == a["indices"]


def test_different_seed_changes_draw(mmlu_data: Fixture) -> None:
    config, model_paths = mmlu_data
    a, _ = build_frozen_subset(config, model_paths, n=5, seed=42)
    b, _ = build_frozen_subset(config, model_paths, n=5, seed=7)
    # The draw is seed-dependent: both are valid size-5 subsets, and these two
    # fixed seeds produce different index sets (default_rng is deterministic).
    assert len(a["indices"]) == len(b["indices"]) == 5
    assert a["indices"] != b["indices"]


def test_indices_are_sorted_and_unique(mmlu_pro_data: Fixture) -> None:
    config, model_paths = mmlu_pro_data
    art, _ = build_frozen_subset(config, model_paths, n=5, seed=42)
    idx = art["indices"]
    assert idx == sorted(idx)
    assert len(set(idx)) == len(idx)
    assert all(0 <= i < config.total_items for i in idx)


def test_artifact_schema_is_stable(ifeval_data: Fixture) -> None:
    config, model_paths = ifeval_data
    art, _ = build_frozen_subset(config, model_paths, n=4, seed=42)
    assert set(art) == {
        "benchmark",
        "models",
        "n",
        "total_items",
        "reduction_pct",
        "seed",
        "strategy",
        "generated",
        "lm_eval_version",
        "indices",
        "items",
        "drift",
    }
    assert art["benchmark"] == "ifeval"
    assert art["n"] == 4
    assert art["strategy"] == "random"
    assert art["models"] == list(model_paths)
    # Provenance for the frozen indices: recorded so a later reader knows which
    # lm-eval enumerated the eval_docs these positions index into.
    assert isinstance(art["lm_eval_version"], str) and art["lm_eval_version"]
    # IFEval enriched items carry the doc.key identifier.
    assert "key" in art["items"][0]
    # Per-model drift table is persisted, keyed by model then metric.
    first_model = art["models"][0]
    assert set(art["drift"][first_model]["inst_strict"]) == {"full", "subset", "drift"}


def test_random_subset_matches_default_rng() -> None:
    rng = np.random.default_rng(42)
    got = random_subset(100, 10, rng, sort=True)
    expected = sorted(np.random.default_rng(42).choice(100, size=10, replace=False).tolist())
    assert got == expected


def test_random_subset_unsorted_is_raw_choice() -> None:
    # The MC hot path uses the unsorted draw; it must equal rng.choice exactly
    # (same RNG consumption) so drift curves stay byte-identical.
    rng = np.random.default_rng(7)
    got = random_subset(50, 8, rng)
    expected = np.random.default_rng(7).choice(50, size=8, replace=False)
    assert list(got) == list(expected)
