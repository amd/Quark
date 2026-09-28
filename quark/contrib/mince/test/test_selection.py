#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for mince.selection: marginal-gain n* rule + convergence guard."""

import pytest

from quark.contrib.mince.selection import (
    DEFAULT_TAU,
    compute_worst_p95_by_n,
    select_n_star,
)


def test_selects_first_n_below_threshold() -> None:
    # Worst-case P95 drift flattens: big gains early, then < 1 pp per step.
    worst = {50: 0.10, 100: 0.06, 150: 0.045, 200: 0.041, 250: 0.038}
    # 100->150 gain = 0.015 (>=0.01); 150->200 gain = 0.004 (<0.01) -> n*=200
    assert select_n_star(worst, tau=DEFAULT_TAU) == 200


def test_smaller_tau_pushes_n_star_higher() -> None:
    worst = {50: 0.10, 100: 0.06, 150: 0.045, 200: 0.041, 250: 0.038}
    # tau=0.003: 200->250 gain = 0.003 not < 0.003; never converges -> raises
    with pytest.raises(ValueError, match="not converged"):
        select_n_star(worst, tau=0.003)


def test_non_convergence_raises() -> None:
    worst = {50: 0.20, 100: 0.15, 150: 0.10, 200: 0.05}  # steady 0.05 drops
    with pytest.raises(ValueError, match="not converged"):
        select_n_star(worst, tau=0.01)


def test_needs_two_candidates() -> None:
    with pytest.raises(ValueError, match="at least two"):
        select_n_star({100: 0.05})


def test_compute_worst_p95_takes_max_across_models_and_metrics() -> None:
    all_model_results = {
        "m1": {100: {"acc": {"p95_abs": 0.04}}, 200: {"acc": {"p95_abs": 0.02}}},
        "m2": {100: {"acc": {"p95_abs": 0.05}}, 200: {"acc": {"p95_abs": 0.03}}},
    }
    worst = compute_worst_p95_by_n(all_model_results, [100, 200], ["acc"])
    assert worst == {100: 0.05, 200: 0.03}


def test_selection_deterministic() -> None:
    worst = {50: 0.10, 100: 0.06, 150: 0.045, 200: 0.041, 250: 0.038}
    assert select_n_star(worst) == select_n_star(worst)
