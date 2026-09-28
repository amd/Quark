#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""MINCE validate: drift between a subset run and the full baseline.

A pure diff of two lm-eval ``results.json`` files (no model, no GPU, no
``lm_eval`` import): read ``results[<benchmark>]`` from each and report
per-metric absolute drift = |subset - full|.

Usage:
  python validate.py --benchmark gsm8k \\
    --subset-results eval_test_logs/<model>-GSM8K/*/results_*.json \\
    --baseline-results bf16_test_logs/<bf16_model>-GSM8K/*/results_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from quark.common.utils.log import ScreenLogger
from quark.contrib.mince.mince_metrics import compute_drift

logger = ScreenLogger(__name__)


def load_metrics(results_path: str, benchmark: str) -> dict[str, float]:
    """Return the score metrics for ``benchmark`` from a results.json.

    (``results_path`` may be a glob; the latest match is used.)

    lm-eval names every score ``"<metric>,<filter>"`` (e.g. ``acc,none``,
    ``exact_match,custom-extract``). We keep only those comma-keyed entries,
    dropping their ``_stderr`` twins. Everything else in the task dict is
    bookkeeping, not a score, and has no comma in its key.
    """
    path = (sorted(glob.glob(results_path)) or [results_path])[-1]
    with open(path) as f:
        task = json.load(f)["results"][benchmark]
    return {
        k: float(v)
        for k, v in task.items()
        if "," in k and "_stderr" not in k and isinstance(v, int | float) and not isinstance(v, bool)
    }


def run_validate(benchmark: str, subset_results: str, baseline_results: str):
    """Return ``(full, subset, drift)`` dicts over the metrics shared by both."""
    subset = load_metrics(subset_results, benchmark)
    full = load_metrics(baseline_results, benchmark)
    shared = [m for m in full if m in subset]
    if not shared:
        raise ValueError(f"no shared metrics for {benchmark!r}: full={sorted(full)} subset={sorted(subset)}")
    # Drop metrics that are 0 in *both* runs (e.g. gsm8k strict-match under a chat
    # template): they carry no signal and only confuse the drift table.
    measured = [m for m in shared if not (full[m] == 0.0 and subset[m] == 0.0)]
    if measured:
        shared = measured
    drift = compute_drift({m: full[m] for m in shared}, {m: subset[m] for m in shared})
    return full, subset, drift


def main():
    p = argparse.ArgumentParser(description="MINCE subset-vs-full drift")
    p.add_argument("--benchmark", required=True, help="task name as it appears in results.json (e.g. gsm8k)")
    p.add_argument("--subset-results", required=True, help="frozen-subset run results.json (glob ok)")
    p.add_argument("--baseline-results", required=True, help="full bf16 results.json (glob ok)")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    full, subset, drift = run_validate(args.benchmark, args.subset_results, args.baseline_results)

    print(f"\n  MINCE validate — {args.benchmark}")
    print(f"  {'metric':<32} {'full':>8} {'subset':>8} {'|drift|(pp)':>12}")
    for m, d in drift.items():
        print(f"  {m:<32} {full[m] * 100:>8.2f} {subset[m] * 100:>8.2f} {d * 100:>12.2f}")
    ad = list(drift.values())
    print(f"  mean|drift|={sum(ad) / len(ad) * 100:.2f}pp  max|drift|={max(ad) * 100:.2f}pp")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, f"{args.benchmark}_drift_report.json")
        with open(path, "w") as f:
            json.dump(
                {"benchmark": args.benchmark, "full_metrics": full, "subset_metrics": subset, "drift": drift},
                f,
                indent=2,
            )
        logger.info(f"Saved: {path}")


if __name__ == "__main__":
    main()
