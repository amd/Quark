#!/usr/bin/env python3
#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Commit newly passed llama.cpp e2e format tests (poll mode)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.run_all_llama_cpp_format_tests import (
    _commit_validated_format,
    _format_passed,
)

RESULTS = Path("/home/l/work/gguf/qwen35-awq/format-test-results.json")
REPO = Path(__file__).resolve().parents[1]
VALIDATED = REPO / "test/test_for_torch/llama_cpp_export_e2e_validated.json"
SKIP_LOAD = ["f16", "bf16", "f32"]


def commit_new() -> int:
    if not RESULTS.exists():
        return 0
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    existing = {}
    if VALIDATED.exists():
        existing = json.loads(VALIDATED.read_text(encoding="utf-8"))

    committed = 0
    for fmt, entry in sorted(results.items()):
        if fmt in existing:
            continue
        if not _format_passed(entry, SKIP_LOAD):
            continue
        _commit_validated_format(fmt, entry, repo_root=REPO)
        committed += 1
    return committed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", type=int, default=0, help="Poll interval seconds")
    args = parser.parse_args()

    if args.watch <= 0:
        n = commit_new()
        print(f"committed {n} new format(s)")
        return

    print(f"watching {RESULTS} every {args.watch}s", flush=True)
    while True:
        try:
            n = commit_new()
            if n:
                print(f"committed {n} new format(s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"poll error: {exc}", flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
