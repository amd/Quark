#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""n* selection rule: MINCE paper's marginal-gain method to select the subset size.

n* is the smallest candidate N at which the per-step drop in worst-case (P95)
absolute drift falls below a threshold tau (i.e. adding more items/samples yields
diminishing returns). "Worst-case" = the max P95 |drift| across the calibration
model(s) at a given N.

"""

from __future__ import annotations

# Default marginal-gain threshold: 1 percentage point improvement in worst-case
# P95 |drift| per step (the paper default).
# Users override it with size.py's--tau flag
DEFAULT_TAU = 0.01


def compute_worst_p95_by_n(
    all_model_results: dict[str, dict[int, dict[str, dict[str, float]]]],
    candidate_ns: list[int],
    metrics: list[str],
) -> dict[int, float]:
    """Collapse per-model/per-metric MC stats into worst-case P95 drift per N.

    Args:
        all_model_results: model -> N -> metric -> stats dict (with "p95_abs"),
            as produced by the Monte-Carlo sizing sweep.
        candidate_ns: the swept N values.
        metrics: the benchmark's metric names.

    Returns:
        Mapping N -> max P95 |drift| across all models and metrics.
    """
    worst: dict[int, float] = {}
    for n in candidate_ns:
        worst[n] = max(
            all_model_results[model][n][metric]["p95_abs"] for model in all_model_results for metric in metrics
        )
    return worst


def select_n_star(
    worst_p95_by_n: dict[int, float],
    tau: float = DEFAULT_TAU,
) -> int:
    """Return n* = smallest N where the per-step drop in worst-case P95 |drift| < tau.

    Args:
        worst_p95_by_n: mapping N -> worst-case P95 |drift| (see
            :func:`compute_worst_p95_by_n`).
        tau: marginal-improvement threshold (absolute, e.g. 0.01 = 1 pp).

    Raises:
        ValueError: if fewer than two candidate N values are given, or no step
            has a non-negative marginal below ``tau`` across the sweep
            (the sweep has not converged). MINCE does not select an n* in that
            case.
    """
    sorted_ns = sorted(worst_p95_by_n)
    if len(sorted_ns) < 2:
        raise ValueError("need at least two candidate N values to compute marginal gain")

    for i in range(1, len(sorted_ns)):
        marginal = worst_p95_by_n[sorted_ns[i - 1]] - worst_p95_by_n[sorted_ns[i]]

        # In theory the worst-case P95 |drift| curve decreases monotonically in N
        # (a larger subset drawn from the full logs is by construction closer to the
        # full set), so ``marginal`` should be non-negative.
        # However, we add an edge case scenario in case the P95 |drift| curve ticks up.
        # MINCE does not select an n* in that case.
        if 0.0 <= marginal < tau:
            return sorted_ns[i]

    raise ValueError(
        f"marginal gain never fell below tau={tau} across candidate Ns {sorted_ns}: "
        f"the sweep has not converged. Plot the P95 |drift| curve to see if it is "
        f"decreasing monotonically. Extend the candidate-N range (or loosen tau) "
        f"before selecting n*."
    )
