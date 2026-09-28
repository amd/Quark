#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""MINCE: render the lm-eval inputs for a frozen subset.

Reads a ``subset_samples.json`` (the ``{task: [indices]}`` map written by
freeze.py) and writes a JSON of the actual lm-eval inputs — the rendered prompt
and gold target — for exactly those frozen items. Useful for inspecting or
shipping the questions the subset consists of.

Usage:
    python extract_inputs.py \
      --samples mince_frozen/mmlu/<model>/subset_samples.json \
      --out subset_inputs.json --num-fewshot 5
"""

from __future__ import annotations

import argparse
import json

from quark.common.utils.log import ScreenLogger
from quark.contrib.mince.subset import build_subset_inputs

logger = ScreenLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Render lm-eval inputs for a MINCE frozen subset")
    p.add_argument("--samples", required=True, help="Path to subset_samples.json (from freeze.py)")
    p.add_argument("--out", required=True, help="Where to write the subset inputs JSON")
    p.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Few-shot count to render — match your bf16 run (e.g. 0 for gsm8k, 5 for mmlu). Default 0.",
    )
    p.add_argument("--fewshot-seed", type=int, default=1234, help="Seed for few-shot sampling (lm-eval default 1234)")
    args = p.parse_args()

    with open(args.samples) as f:
        samples = json.load(f)

    inputs = build_subset_inputs(samples, num_fewshot=args.num_fewshot, fewshot_seed=args.fewshot_seed)

    with open(args.out, "w") as f:
        json.dump(inputs, f, indent=2, default=str)

    total = sum(len(v) for v in inputs.values())
    logger.info(f"Wrote {total} inputs across {len(inputs)} task(s) -> {args.out}")


if __name__ == "__main__":
    main()
