#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""CLI entrypoint for managed workspace maintenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quark.experimental.torch.quant_perf.workspace.gc import gc_workspaces


def run_gc_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="quark-quant-perf gc")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--branches", action="store_true")
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    args = parser.parse_args(argv)
    roots = [Path(root).resolve() for root in args.root]
    if not roots:
        roots = [Path("/tmp/quark_quant_perf_worktrees")]
        roots.extend(path for path in Path("runs").glob("*/workspaces/managed") if path.is_dir())
    result = gc_workspaces(
        roots,
        dry_run=args.dry_run,
        remove_empty_branches=args.branches,
        max_age_hours=args.max_age_hours,
    )
    print(json.dumps(result, indent=2))
    return 1 if result["errors"] else 0
