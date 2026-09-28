#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Small helpers shared by GEAK candidate tracking and patch reporting."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

_MANGLE_SUFFIX_RE = re.compile(r"(_[0-9a-f]{8,}|_\d+d\d+d\d+.*|\[.*\])$")


def normalize_kernel_sig(op_name: str) -> str:
    """Strips shape/hash suffixes, keeping the operator family name -- this
    is the KB lookup key, distinct from kernel_source.py's demangling (which
    targets grep, not cross-run signature matching).

    This is a lossy family key. Use make_kernel_id for candidate identity and
    artifact directories.
    """
    base = op_name.rsplit("::", 1)[-1]
    sig = _MANGLE_SUFFIX_RE.sub("", base).strip("_")
    return sig[:80]


def make_kernel_id(op_name: str) -> str:
    """Identify a complete runtime signature without conflating template variants."""
    stem = op_name.split("<", 1)[0].split("(", 1)[0].rsplit("::", 1)[-1]
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.-")[:63] or "kernel"
    return f"{prefix}-{hashlib.sha256(op_name.encode()).hexdigest()[:16]}"


def kernel_record_id(record: dict[str, Any]) -> str:
    """Read legacy records by their full name, never by a shortened alias."""
    name = record.get("kernel_name") or record.get("name")
    return make_kernel_id(str(name)) if name else str(record.get("kernel_id") or record.get("kernel_sig") or "")


def kernel_candidate_status(report: dict[str, Any]) -> str:
    """Separate explicit rejection from incomplete evidence and E2E candidates."""
    correctness = report.get("round_evaluation", {}).get("correctness", {}).get("success")
    director_status = report.get("director_status")
    if (
        correctness is False
        or director_status == "rejected"
        or report.get("patch_validation", {}).get("status") == "rejected"
        or report.get("micro_speedup_source") == "invalid_workload"
    ):
        return "rejected"
    if director_status is not None and director_status not in {"accepted", "flagged"}:
        return "incomplete"
    speedup = report.get("verified_speedup")
    if speedup is not None and speedup <= 1.0 and (report.get("final") or correctness is True):
        return "rejected"
    if correctness is not True:
        return "incomplete"
    patch = report.get("best_patch", "")
    if not patch or not Path(patch).is_file() or Path(patch).stat().st_size == 0:
        return "incomplete"
    if speedup is None and report.get("micro_speedup_source") != "unmeasured_verified_patch":
        return "incomplete"
    return "candidate"


def is_kernel_candidate(report: dict[str, Any]) -> bool:
    """Return whether a validated patch is eligible for full model testing."""
    return kernel_candidate_status(report) == "candidate"


def summarize_patch(best_patch: str) -> str:
    """Best-effort one-line summary of a patch when GEAK's report doesn't
    already provide approach_summary: the first non-empty diff/commit-message
    line."""
    try:
        with open(best_patch) as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith(("diff ", "index ", "---", "+++", "@@")):
                    return stripped.lstrip("+-# ")
    except OSError:
        pass
    return "no summary available"


def aggregate_gain(results: list[dict[str, Any]]) -> float:
    """Multiple KEPT GEAK patches compose multiplicatively (each applies to
    an independent kernel in the same serving pipeline)."""
    gain = 1.0
    for r in results:
        speedup = r.get("verified_speedup")
        if isinstance(speedup, int | float) and speedup > 0.0:
            gain *= float(speedup)
    return gain
