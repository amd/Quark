#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""MINCE — Monte-Carlo Informed N-sizing for Compact Evaluation.

An importable core for sizing a representative benchmark subset (n*)
from bf16 per-item evaluation logs, freezing it into a reproducible ID-based
artifact, and reusing it to evaluate downstream model variants within a bounded
accuracy drift.

Layout:
  config.py         benchmark registry (BENCHMARKS, BenchmarkConfig)
  data_loader.py    load lm-eval sample logs -> BenchmarkItem list
  mince_metrics.py  accuracy/drift aggregation (+ compute_drift_table)
  montecarlo.py     random-draw primitive + vectorized MC sizing sweep (run_sizing)
  selection.py      marginal-gain n* rule (select_n_star)
  subset.py         frozen-subset artifact + lm-eval --samples map

The runnable CLIs (size.py, freeze.py, validate.py, extract_inputs.py) live under
examples/contrib/mince/ and import this package.
"""

from quark.contrib.mince.config import BENCHMARKS, BenchmarkConfig
from quark.contrib.mince.data_loader import BenchmarkItem, load_benchmark_items
from quark.contrib.mince.mince_metrics import compute_accuracy, compute_drift, compute_drift_table
from quark.contrib.mince.montecarlo import random_subset, run_sizing
from quark.contrib.mince.selection import DEFAULT_TAU, compute_worst_p95_by_n, select_n_star
from quark.contrib.mince.subset import build_frozen_subset, build_samples, build_subset_inputs

__all__ = [
    "BENCHMARKS",
    "BenchmarkConfig",
    "BenchmarkItem",
    "load_benchmark_items",
    "compute_accuracy",
    "compute_drift",
    "compute_drift_table",
    "random_subset",
    "run_sizing",
    "DEFAULT_TAU",
    "compute_worst_p95_by_n",
    "select_n_star",
    "build_frozen_subset",
    "build_samples",
    "build_subset_inputs",
]
