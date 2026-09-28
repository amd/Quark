#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .types import KnowledgeBundle


def append_query_audit(
    session_dir: str | Path,
    *,
    consumer: str,
    context_hash: str,
    bundle: KnowledgeBundle,
    round_id: int | None = None,
    error_signature: str = "",
) -> None:
    path = Path(session_dir) / "knowledge" / "query_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, object] = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "consumer": consumer,
        "context_hash": context_hash,
        "knowledge_ids": list(bundle.source_ids),
    }
    if round_id is not None:
        row.update(round=round_id, error_signature=error_signature)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
