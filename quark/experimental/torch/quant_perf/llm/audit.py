#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def append_llm_call(
    session_dir: str | Path,
    *,
    call_type: str,
    model: str,
    round_id: int,
    prompt: str,
    output: str,
    outcome: str,
    knowledge_ids: list[str] | None = None,
) -> None:
    path = Path(session_dir) / "llm_calls.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "call_type": call_type,
        "model": model,
        "round": int(round_id),
        "prompt_hash": _hash(prompt),
        "output_hash": _hash(output),
        "outcome": outcome,
        "knowledge_ids": list(knowledge_ids or []),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
