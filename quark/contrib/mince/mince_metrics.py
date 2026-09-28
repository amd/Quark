#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Score a set of items and measure how a subset drifts from the full benchmark.

  compute_accuracy   all metrics for a model over a set of items.
  compute_drift      per-metric absolute drift |subset - full|.
  compute_drift_table freeze-time per-model full-vs-subset accuracy + drift table.

The vectorized twins used by the Monte-Carlo sweep live in montecarlo.py.
"""

from __future__ import annotations

import numpy as np

from quark.contrib.mince.config import BenchmarkConfig
from quark.contrib.mince.data_loader import BenchmarkItem


def compute_accuracy(
    items: list[BenchmarkItem],
    model_name: str,
    benchmark: BenchmarkConfig,
) -> dict[str, float]:
    """Compute all metrics for a model on a set of items.

    Returns a dict mapping metric_name -> value.
    """
    if benchmark.name == "ifeval":
        return _compute_ifeval_metrics(items, model_name)
    return _compute_simple_metrics(items, model_name, benchmark)


def compute_drift(
    full_accs: dict[str, float],
    subset_accs: dict[str, float],
) -> dict[str, float]:
    """Compute per-metric absolute drift: |subset - full|."""
    return {m: abs(subset_accs[m] - full_accs[m]) for m in full_accs}


def compute_drift_table(
    benchmark: BenchmarkConfig,
    items: list[BenchmarkItem],
    indices: list[int],
    model_paths: dict[str, str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-model full/subset accuracy + drift on the frozen subset.

    Returns ``{model: {metric: {"full", "subset", "drift"}}}`` where ``drift`` is
    the absolute deviation ``|subset - full|``. This is the only model-specific
    content of a freeze run — the subset indices themselves depend only on
    (total_items, seed), so they are identical across models.
    """
    sub_items = [items[i] for i in indices]
    table: dict[str, dict[str, dict[str, float]]] = {}
    for model in model_paths:
        full = compute_accuracy(items, model, benchmark)
        sub = compute_accuracy(sub_items, model, benchmark)
        drift = compute_drift(full, sub)
        table[model] = {m: {"full": full[m], "subset": sub[m], "drift": drift[m]} for m in benchmark.metric_names}
    return table


# ---------------------------------------------------------------------------
# IFEVAL: 4-metric system
# ---------------------------------------------------------------------------


def _compute_ifeval_metrics(
    items: list[BenchmarkItem],
    model_name: str,
) -> dict[str, float]:
    n = len(items)
    if n == 0:
        return dict.fromkeys(["prompt_strict", "prompt_loose", "inst_strict", "inst_loose"], 0.0)

    ps = sum(1 for it in items if it.model_results[model_name]["prompt_strict"]) / n
    pl = sum(1 for it in items if it.model_results[model_name]["prompt_loose"]) / n

    inst_s_pass = sum(sum(it.model_results[model_name]["inst_strict"]) for it in items)
    inst_l_pass = sum(sum(it.model_results[model_name]["inst_loose"]) for it in items)
    inst_total = sum(len(it.model_results[model_name]["inst_strict"]) for it in items)

    return {
        "prompt_strict": ps,
        "prompt_loose": pl,
        "inst_strict": inst_s_pass / inst_total if inst_total else 0.0,
        "inst_loose": inst_l_pass / inst_total if inst_total else 0.0,
    }


# ---------------------------------------------------------------------------
# Simple accuracy: MMLU, GSM8K, MMLU-Pro
# ---------------------------------------------------------------------------


def _compute_simple_metrics(
    items: list[BenchmarkItem],
    model_name: str,
    benchmark: BenchmarkConfig,
) -> dict[str, float]:
    if not items:
        return dict.fromkeys(benchmark.metric_names, 0.0)

    result = {}
    for metric in benchmark.metric_names:
        vals = [it.model_results[model_name][metric] for it in items]
        result[metric] = float(np.mean(vals))
    return result
