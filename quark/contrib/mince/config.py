#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Benchmark configuration registry for MINCE.

Central lookup (`BENCHMARKS[name]`) describing each benchmark's lm-eval logs.
The rest of the MINCE pipeline reads a BenchmarkConfig rather than hard-coding values:

  data_loader.py    uses `sample_glob` to find the logs, `sample_filter` to pick one
                    row per item, and `total_items` as a completeness guard.
  mince_metrics.py  scores each metric in `metric_names` (IFEval vs simple dispatch).
  montecarlo.py     sweeps `candidate_ns` during Monte-Carlo sizing.
  subset.py         reports subset size relative to `total_items`.

The runnable CLIs (size.py, freeze.py, validate.py, extract_inputs.py) live under
examples/contrib/mince/ and import these package modules.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkConfig:
    """How to locate and score one benchmark's lm-eval sample logs.

    ``name`` is the lm-eval task name and the ``BENCHMARKS`` key. ``metric_names``
    are the keys each loader attaches to ``item.model_results[model]``, and
    ``total_items`` is the full split size, used as a completeness guard at load
    time and as the denominator when reporting subset size.
    """

    name: str
    total_items: int
    metric_names: list[str]
    sample_glob: str
    # Candidate subset sizes swept during Monte-Carlo sizing.
    candidate_ns: list[int] = field(default_factory=list)
    # If set, only load JSONL rows where doc["filter"] matches this value.
    # lm-eval-harness can write multiple filter variants per item (e.g.
    # "strict-match" and "flexible-extract" for GSM8K).
    sample_filter: str = ""


BENCHMARKS: dict[str, BenchmarkConfig] = {
    "ifeval": BenchmarkConfig(
        name="ifeval",
        total_items=541,
        metric_names=["prompt_strict", "prompt_loose", "inst_strict", "inst_loose"],
        sample_glob="samples_ifeval_*.jsonl",
        candidate_ns=[50, 100, 150, 200, 250, 300, 350, 400],
    ),
    "mmlu": BenchmarkConfig(
        name="mmlu",
        total_items=14042,
        metric_names=["acc"],
        sample_glob="samples_mmlu_*_*.jsonl",
        candidate_ns=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000],
    ),
    "gsm8k": BenchmarkConfig(
        name="gsm8k",
        total_items=1319,
        metric_names=["exact_match"],
        sample_glob="samples_gsm8k_*.jsonl",
        candidate_ns=[100, 200, 300, 400, 500, 600, 700, 800],
        sample_filter="flexible-extract",
    ),
    "commonsense_qa": BenchmarkConfig(
        name="commonsense_qa",
        total_items=1221,
        metric_names=["acc"],
        sample_glob="samples_commonsense_qa_*.jsonl",
        candidate_ns=[100, 200, 300, 400, 500, 600, 700, 800],
    ),
    "piqa": BenchmarkConfig(
        name="piqa",
        total_items=1838,
        metric_names=["acc", "acc_norm"],
        sample_glob="samples_piqa_*.jsonl",
        candidate_ns=[200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
    ),
    "mmlu_pro": BenchmarkConfig(
        name="mmlu_pro",
        total_items=12032,
        metric_names=["exact_match"],
        sample_glob="samples_mmlu_pro_*.jsonl",
        candidate_ns=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000],
    ),
}
