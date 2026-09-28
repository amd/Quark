#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Non-pytest helpers shared by the MINCE test suite.

These are plain functions and type aliases used to build tiny, deterministic
lm-eval-style sample JSONLs. They are kept out of ``conftest.py`` (which is
reserved for pytest fixtures/plugins) and imported where needed.
"""

from __future__ import annotations

import json
import os
from typing import Any

from quark.contrib.mince.config import BenchmarkConfig

# A prepared (config, model_name -> sample-dir) pair ready for load_benchmark_items.
Fixture = tuple[BenchmarkConfig, dict[str, str]]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def ifeval_rows(seed: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(8):
        # Deterministic but varied pass/fail so metrics are non-trivial.
        strict = (i + seed) % 2
        loose = (i + seed + 1) % 2
        rows.append(
            {
                "doc_id": i,
                "doc": {
                    "key": 1000 + i,
                    "prompt": f"prompt {i}",
                    "instruction_id_list": [f"cat_{i % 3}", "extra"],
                },
                "prompt_level_strict_acc": float(strict),
                "prompt_level_loose_acc": float(loose),
                "inst_level_strict_acc": [strict, (i + seed) % 2],
                "inst_level_loose_acc": [loose, (i + seed + 1) % 2],
            }
        )
    return rows
