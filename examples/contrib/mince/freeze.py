#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""MINCE freeze CLI: turn n* into a reproducible subset artifact + lm-eval samples map.

Thin wrapper around ``quark.contrib.mince.subset``: draws a random subset of size
n at a fixed seed from the benchmark's canonical item order, then writes two files:

  frozen_subset_seed<seed>.json  the subset record: the positional ``indices``,
                                 per-item content ids (doc_id/key/subject + a
                                 question snippet), the draw metadata (seed, n,
                                 reduction, lm_eval_version, ...) and the
                                 freeze-time full-vs-subset drift table.
  subset_samples.json            the lm-eval ``--samples`` map for this subset.

Prints full-vs-subset accuracy drift as a sanity check.

Usage:
  python freeze.py --benchmark gsm8k --n 400 --model-dirs <path to bf16_model_log_dir>
"""

from __future__ import annotations

import argparse
import json
import os

from quark.common.utils.log import ScreenLogger
from quark.contrib.mince.config import BENCHMARKS
from quark.contrib.mince.subset import build_frozen_subset, build_samples

logger = ScreenLogger(__name__)


def print_drift_table(benchmark, artifact):
    """Print the persisted per-model full-vs-subset drift table (sanity check)."""
    print(f"\n{'=' * 72}")
    print(f"  {benchmark.name.upper()} full vs subset (n={artifact['n']})")
    print(f"{'=' * 72}")
    abs_drifts = []
    for model, metrics in artifact["drift"].items():
        print(f"\n  {model}")
        for m in benchmark.metric_names:
            d = metrics[m]
            abs_drift = abs(d["drift"])
            abs_drifts.append(abs_drift)
            print(
                f"    {m:<16} full={d['full'] * 100:6.2f}  "
                f"subset={d['subset'] * 100:6.2f}  |drift|={abs_drift * 100:5.2f} pp"
            )
    if abs_drifts:
        mean_abs = sum(abs_drifts) / len(abs_drifts)
        print(f"\n  mean |drift| = {mean_abs * 100:.2f} pp   max |drift| = {max(abs_drifts) * 100:.2f} pp")


def _sanitize(name: str) -> str:
    """Make a model name safe to use as a path component."""
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in name)


def models_label(model_paths) -> str:
    """A short filesystem label for the run: the model name, or '<k>models'."""
    names = list(model_paths)
    return _sanitize(names[0]) if len(names) == 1 else f"{len(names)}models"


def parse_model_dirs(args):
    """Resolve the single ``model_name -> logs_dir`` mapping from ``--model-dirs``.

    The value is the bf16 model's logs directory; the label is taken from its
    basename. MINCE currently calibrates on one bf16 model; multi-model support
    will come in a later PR.
    """
    spec = args.model_dirs.strip().rstrip("/")
    return {os.path.basename(spec): spec}


def main():
    parser = argparse.ArgumentParser(description="MINCE freeze subset artifact")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--n", type=int, required=True, help="Subset size (usually n*)")
    parser.add_argument(
        "--model-dirs",
        required=True,
        help="Single bf16 model's logs directory (the label is taken from the directory name)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", default=None, help="Directory for the artifact (default ./mince_frozen/<benchmark>/<model-label>)"
    )
    args = parser.parse_args()

    benchmark = BENCHMARKS[args.benchmark]
    model_paths = parse_model_dirs(args)

    logger.info(f"Loading {benchmark.name} items ({list(model_paths.keys())})...")
    artifact, items = build_frozen_subset(benchmark, model_paths, args.n, args.seed)
    logger.info(
        f"{artifact['total_items']} items -> subset n={artifact['n']} "
        f"({artifact['reduction_pct']}% reduction, seed={artifact['seed']})"
    )

    print_drift_table(benchmark, artifact)

    out_dir = args.out_dir or os.path.join("mince_frozen", benchmark.name, models_label(model_paths))
    os.makedirs(out_dir, exist_ok=True)

    samples = build_samples(artifact)

    path = os.path.join(out_dir, f"frozen_subset_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
    logger.info(f"Saved artifact: {path}")

    samples_path = os.path.join(out_dir, "subset_samples.json")
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2)
    logger.info(f"Saved lm-eval --samples map: {samples_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
