#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Monte-Carlo subset sizing: sweep candidate subset sizes -> n* inputs.

For each candidate N, draw B random subsets (without replacement) from the full
benchmark and measure how far each subset's accuracy drifts from the full-benchmark
score. ``run_sizing`` returns a result bundle (worst-case P95 |drift| per N, etc.)
that the n* marginal-gain rule (selection.py) consumes.

Two layers live here:

  random_subset                         the shared random-draw primitive, also used
                                        by the freeze step (subset.py).
  precompute_vectors_* / subsample_drift_*  vectorized (numpy) accuracy/drift, the
                                        performance-critical twins of the plain-Python
                                        aggregations in mince_metrics.py.
  run_subsampling_* / run_model / run_sizing  the sweep itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quark.contrib.mince.config import BenchmarkConfig
from quark.contrib.mince.data_loader import BenchmarkItem, load_benchmark_items
from quark.contrib.mince.selection import compute_worst_p95_by_n

# model -> N -> metric -> stats ("mean_abs"/"std"/"p95_abs"/"max_abs")
ResultsByN = dict[int, dict[str, dict[str, float]]]


def random_subset(
    n_items: int,
    n: int,
    rng: np.random.Generator,
    *,
    sort: bool = False,
) -> np.ndarray | list[int]:
    """Draw ``n`` distinct positions in ``[0, n_items)`` without replacement.

    The single random-draw primitive shared by the Monte-Carlo sizing sweep
    (unsorted hot path) and the freeze step (``subset.py``, sorted artifact). The
    RNG is consumed by one ``rng.choice(n_items, size=n, replace=False)`` call
    regardless of ``sort``, so drift curves stay byte-for-byte identical to the
    original study.

    Args:
        n_items: population size (total number of benchmark items).
        n: subset size to draw.
        rng: numpy Generator, seeded by the caller for reproducibility.
        sort: if True, return a sorted Python list; otherwise the raw ndarray.
    """
    idx = rng.choice(n_items, size=n, replace=False)
    return sorted(idx.tolist()) if sort else idx


# ---------------------------------------------------------------------------
# Vectorized helpers for Monte Carlo (performance-critical)
# ---------------------------------------------------------------------------


def precompute_vectors_ifeval(
    items: list[BenchmarkItem],
    model_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Pre-compute flat numpy arrays for fast IFEVAL subsampling.

    Returns (vectors, full_accs) where vectors contains:
      prompt_strict, prompt_loose (shape N, float),
      inst_strict_pass, inst_loose_pass, inst_count (shape N, float).
    """
    ps = np.array([float(it.model_results[model_name]["prompt_strict"]) for it in items])
    pl = np.array([float(it.model_results[model_name]["prompt_loose"]) for it in items])
    isp = np.array([sum(it.model_results[model_name]["inst_strict"]) for it in items], dtype=float)
    ilp = np.array([sum(it.model_results[model_name]["inst_loose"]) for it in items], dtype=float)
    ic = np.array([len(it.model_results[model_name]["inst_strict"]) for it in items], dtype=float)

    full_accs = {
        "prompt_strict": float(np.mean(ps)),
        "prompt_loose": float(np.mean(pl)),
        "inst_strict": float(np.sum(isp) / np.sum(ic)) if np.sum(ic) else 0.0,
        "inst_loose": float(np.sum(ilp) / np.sum(ic)) if np.sum(ic) else 0.0,
    }

    vectors = {
        "prompt_strict": ps,
        "prompt_loose": pl,
        "inst_strict_pass": isp,
        "inst_loose_pass": ilp,
        "inst_count": ic,
    }
    return vectors, full_accs


def precompute_vectors_simple(
    items: list[BenchmarkItem],
    model_name: str,
    metric_name: str,
) -> tuple[np.ndarray, float]:
    """Pre-compute a flat numpy array for fast simple-metric subsampling.

    Returns (values_array, full_accuracy).
    """
    vals = np.array([float(it.model_results[model_name][metric_name]) for it in items])
    return vals, float(np.mean(vals))


def subsample_drift_ifeval(
    vectors: dict[str, np.ndarray],
    full_accs: dict[str, float],
    indices: np.ndarray,
) -> dict[str, float]:
    """Compute IFEVAL metric drifts for a subset given precomputed vectors."""
    ps_sub = np.mean(vectors["prompt_strict"][indices])
    pl_sub = np.mean(vectors["prompt_loose"][indices])
    ic_sub = np.sum(vectors["inst_count"][indices])
    is_sub = np.sum(vectors["inst_strict_pass"][indices]) / ic_sub if ic_sub else 0.0
    il_sub = np.sum(vectors["inst_loose_pass"][indices]) / ic_sub if ic_sub else 0.0

    return {
        "prompt_strict": float(ps_sub) - full_accs["prompt_strict"],
        "prompt_loose": float(pl_sub) - full_accs["prompt_loose"],
        "inst_strict": float(is_sub) - full_accs["inst_strict"],
        "inst_loose": float(il_sub) - full_accs["inst_loose"],
    }


def subsample_drift_simple(
    values: np.ndarray,
    full_acc: float,
    indices: np.ndarray,
) -> float:
    """Compute drift for a simple metric given precomputed values."""
    return float(np.mean(values[indices])) - full_acc


# ---------------------------------------------------------------------------
# Monte-Carlo sweep
# ---------------------------------------------------------------------------


def run_subsampling_ifeval(
    items: list[BenchmarkItem],
    model_name: str,
    candidate_ns: list[int],
    B: int,
    rng: np.random.Generator,
) -> tuple[ResultsByN, dict[str, float]]:
    """Vectorized MC for IFEVAL's 4-metric system."""
    vectors, full_accs = precompute_vectors_ifeval(items, model_name)
    n_items = len(items)
    metrics = list(full_accs.keys())

    results: ResultsByN = {}
    for n in candidate_ns:
        drifts = {m: np.empty(B) for m in metrics}
        for b in range(B):
            idx = random_subset(n_items, n, rng)
            d = subsample_drift_ifeval(vectors, full_accs, idx)
            for m in metrics:
                drifts[m][b] = d[m]
        results[n] = {m: _summ(drifts[m]) for m in metrics}
    return results, full_accs


def run_subsampling_simple(
    items: list[BenchmarkItem],
    model_name: str,
    metric_name: str,
    candidate_ns: list[int],
    B: int,
    rng: np.random.Generator,
) -> tuple[ResultsByN, dict[str, float]]:
    """Vectorized MC for simple single-metric benchmarks."""
    vals, full_acc = precompute_vectors_simple(items, model_name, metric_name)
    n_items = len(items)

    results: ResultsByN = {}
    for n in candidate_ns:
        d = np.empty(B)
        for b in range(B):
            idx = random_subset(n_items, n, rng)
            d[b] = subsample_drift_simple(vals, full_acc, idx)
        results[n] = {metric_name: _summ(d)}
    return results, {metric_name: full_acc}


def _summ(d: np.ndarray) -> dict[str, float]:
    return {
        "mean_abs": float(np.mean(np.abs(d))),
        "std": float(np.std(d)),
        "p95_abs": float(np.percentile(np.abs(d), 95)),
        "max_abs": float(np.max(np.abs(d))),
    }


def run_model(
    items: list[BenchmarkItem],
    model_name: str,
    benchmark: BenchmarkConfig,
    candidate_ns: list[int],
    B: int,
    rng: np.random.Generator,
) -> tuple[ResultsByN, dict[str, float]]:
    """Dispatch to the right MC runner based on benchmark type."""
    if benchmark.name == "ifeval":
        return run_subsampling_ifeval(items, model_name, candidate_ns, B, rng)

    all_results: ResultsByN = {}
    all_full: dict[str, float] = {}
    for metric_name in benchmark.metric_names:
        results, full_accs = run_subsampling_simple(items, model_name, metric_name, candidate_ns, B, rng)
        for n in results:
            all_results.setdefault(n, {}).update(results[n])
        all_full.update(full_accs)
    return all_results, all_full


def run_sizing(
    benchmark: BenchmarkConfig,
    model_paths: dict[str, str],
    B: int = 10000,
    seed: int = 42,
    candidate_ns: list[int] | None = None,
) -> dict[str, Any]:
    """Run the full MC sweep across all models. Returns a result bundle.

    The RNG is reset to ``seed`` per model
    """
    candidate_ns = candidate_ns or benchmark.candidate_ns
    items = load_benchmark_items(benchmark, model_paths)

    n_items = len(items)
    bad = [n for n in candidate_ns if n > n_items]
    if bad:
        raise ValueError(f"Candidate N values {bad} exceed total items ({n_items})")

    all_model_results: dict[str, ResultsByN] = {}
    full_accs_by_model: dict[str, dict[str, float]] = {}
    for model_name in model_paths:
        rng = np.random.default_rng(seed)
        results, full_accs = run_model(items, model_name, benchmark, candidate_ns, B, rng)
        all_model_results[model_name] = results
        full_accs_by_model[model_name] = full_accs

    worst_p95_by_n = compute_worst_p95_by_n(all_model_results, candidate_ns, benchmark.metric_names)

    return {
        "benchmark": benchmark,
        "models": list(model_paths),
        "candidate_ns": candidate_ns,
        "all_model_results": all_model_results,
        "full_accs_by_model": full_accs_by_model,
        "worst_p95_by_n": worst_p95_by_n,
        "B": B,
        "seed": seed,
    }
