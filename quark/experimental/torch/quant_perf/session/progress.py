#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""progress.json: a human-facing, poll-anytime status file, separate from
state.json (which is the program's own resume checkpoint -- "facts").
progress.json is "narrative" and its schema can evolve independently.

Design ref: IMPL_SPEC §4.8.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.session.persistence import write_json_atomic

_MAX_WARNINGS = 50


def write_progress(session_dir: Path, warning: str | None = None, **fields: Any) -> None:
    """Update progress through an atomic, last-writer-wins replacement.

    Callers only pass the fields they are updating; everything else from the
    previously observed document is preserved. ``warning``, if given, is
    appended to the rolling warnings list capped at the most recent 50.
    """
    session_dir = Path(session_dir)
    path = session_dir / "progress.json"
    if path.exists():
        data: dict[str, Any] = json.loads(path.read_text())
    else:
        now = datetime.now(UTC).isoformat()
        data = {"warnings": [], "started_at": now, "_started_ts": time.time()}

    warnings = data.get("warnings", [])
    if warning is not None:
        warnings = (warnings + [warning])[-_MAX_WARNINGS:]

    data.update(fields)
    data["warnings"] = warnings
    data["updated_at"] = datetime.now(UTC).isoformat()
    data["elapsed_seconds"] = int(time.time() - data.get("_started_ts", time.time()))

    write_json_atomic(path, data)


def read_progress(session_dir: Path) -> dict[str, Any] | None:
    path = Path(session_dir) / "progress.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    repair = data.get("repair") or {}
    if repair.get("status") == "running":
        # Project live durations at read time; no heartbeat or checkpoint writes.
        now = time.time()
        for start, elapsed in (("started_at", "elapsed_seconds"), ("repair_started_at", "total_elapsed_seconds")):
            if repair.get(start):
                repair[elapsed] = max(0.0, now - datetime.fromisoformat(repair[start]).timestamp())
    return data
