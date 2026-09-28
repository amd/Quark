#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""MINCE sizing CLI: Monte-Carlo sweep over candidate subset sizes -> n*.

Thin wrapper around ``quark.contrib.mince.montecarlo`` / ``selection``:

1. Consumes bf16 per-item evaluation logs
2. For each candidate N, draws B random subsets (without replacement) and measures how
far each subset's accuracy drifts from the full-benchmark score.
3. Reports P95 |drift| + marginal gain per N, then applies the marginal-gain rule to pick n*.

Usage:
  python size.py --benchmark gsm8k --model-dirs <path to bf16_model_log_dir>
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from quark.common.utils.log import ScreenLogger
from quark.contrib.mince.config import BENCHMARKS
from quark.contrib.mince.montecarlo import run_sizing
from quark.contrib.mince.selection import DEFAULT_TAU, select_n_star

logger = ScreenLogger(__name__)


def print_worst_case(bundle):
    benchmark = bundle["benchmark"]
    worst = bundle["worst_p95_by_n"]
    sorted_ns = sorted(worst)
    print(f"\n{'=' * 60}")
    print("WORST-CASE P95 |DRIFT| (across all models and metrics)")
    print(f"{'=' * 60}")
    print(f"  {'N':>6}  {'% of full':>9}  {'worst P95':>10}  {'marginal':>10}")
    prev = None
    for n in sorted_ns:
        pct = n / benchmark.total_items * 100
        marg = "" if prev is None else f"{prev - worst[n]:+.4f}"
        print(f"  {n:>6}  {pct:>8.1f}%  {worst[n]:>10.4f}  {marg:>10}")
        prev = worst[n]


def plot_sizing(bundle, out_path):
    """Save a marginal-gain plot to help *choose* ``tau``.

    Shows the per-step marginal gain (the drop in worst-case P95 |drift| from the
    previous candidate N) as bars vs N. It is intentionally tau-agnostic — no tau
    line, no n* marker — since the whole point is to let you decide tau: pick the
    pp-per-step level below which you consider returns diminishing, and n* is the
    first N whose bar sits below that level.

    matplotlib is imported lazily so it stays an optional dependency; if it is not
    installed the run continues and only the plot is skipped.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot (pip install matplotlib to enable --plot)")
        return None

    benchmark = bundle["benchmark"]
    worst = bundle["worst_p95_by_n"]
    ns = sorted(worst)
    marg_ns = ns[1:]
    marg_pp = [(worst[ns[i - 1]] - worst[ns[i]]) * 100 for i in range(1, len(ns))]
    width = (min((b - a) for a, b in zip(ns, ns[1:], strict=False)) * 0.6) if len(ns) > 1 else 1.0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(marg_ns, marg_pp, width=width, color="#1f77b4")
    ax.set_ylabel("marginal gain in worst-case P95 |drift| (pp per step)")
    ax.set_xlabel("subset size N")
    ax.set_title(f"MINCE sizing — {benchmark.name}  (B={bundle['B']}, seed={bundle['seed']})")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved sizing plot: {out_path}")
    return out_path


def save_report(bundle, n_star, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    benchmark = bundle["benchmark"]
    report = {
        "generated": datetime.now().isoformat(),
        "benchmark": benchmark.name,
        "models": bundle["models"],
        "total_items": benchmark.total_items,
        "B": bundle["B"],
        "seed": bundle["seed"],
        "candidate_ns": sorted(bundle["candidate_ns"]),
        "metrics": benchmark.metric_names,
        "worst_p95_by_n": {str(k): v for k, v in bundle["worst_p95_by_n"].items()},
        "n_star": n_star,
        "full_accs": bundle["full_accs_by_model"],
    }
    path = os.path.join(out_dir, "sizing_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved sizing report: {path}")
    return path


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
    parser = argparse.ArgumentParser(description="MINCE Monte-Carlo sizing")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument(
        "--model-dirs",
        required=True,
        help="Single bf16 model's logs directory (the label is taken from the directory name)",
    )
    parser.add_argument("--B", type=int, default=10000, help="MC draws per N")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--candidate-ns", default=None, help="Override candidate N values (comma-separated)")
    parser.add_argument(
        "--tau", type=float, default=DEFAULT_TAU, help="Marginal-gain threshold for n* (default 0.01 = 1pp)"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for the sizing report (default ./mince_sizing/<benchmark>/<model-label>)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a sizing plot (P95 curve + marginal gain vs tau) to help choose --tau. Requires matplotlib.",
    )
    parser.add_argument(
        "--plot-path", default=None, help="Where to write the plot (default <out-dir>/sizing_plot.png). Implies --plot."
    )
    args = parser.parse_args()

    benchmark = BENCHMARKS[args.benchmark]
    model_paths = parse_model_dirs(args)
    candidate_ns = [int(x) for x in args.candidate_ns.split(",")] if args.candidate_ns else None

    logger.info(f"Benchmark: {benchmark.name} ({benchmark.total_items} items)")
    logger.info(f"Models: {list(model_paths.keys())}  B={args.B}  seed={args.seed}")

    bundle = run_sizing(benchmark, model_paths, B=args.B, seed=args.seed, candidate_ns=candidate_ns)
    print_worst_case(bundle)

    out_dir = args.out_dir or os.path.join("mince_sizing", benchmark.name, models_label(model_paths))

    # Resolve n* but tolerate non-convergence so the diagnostic plot can still
    # render (that is exactly when a user wants to see the curve to pick tau).
    try:
        n_star = select_n_star(bundle["worst_p95_by_n"], tau=args.tau)
        n_star_err = None
    except ValueError as e:
        n_star, n_star_err = None, e

    if args.plot or args.plot_path:
        os.makedirs(out_dir, exist_ok=True)
        plot_path = args.plot_path or os.path.join(out_dir, "sizing_plot.png")
        plot_sizing(bundle, plot_path)

    # Preserve the "MINCE does not select an n* if the sweep hasn't converged"
    # contract — but only after the plot has been written.
    if n_star_err is not None:
        raise n_star_err

    logger.info(f"n* = {n_star}  ({n_star / benchmark.total_items * 100:.1f}% of full, tau={args.tau})")

    save_report(bundle, n_star, out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
