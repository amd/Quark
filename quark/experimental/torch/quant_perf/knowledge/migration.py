#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""One-time repair signature migration, optionally using original session evidence."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.repair.evidence import extract_failure_evidence
from quark.experimental.torch.quant_perf.repair.signatures import (
    FAILURE_SIGNATURE_VERSION,
    failure_signature,
    similar_failure_pattern,
)


def _signature_from_record(record: dict[str, Any]) -> str:
    fields = ("exception_type", "exception_message", "root_file", "root_function")
    if not all(isinstance(record.get(field), str) for field in fields):
        return ""
    return failure_signature(*(record[field] for field in fields))


def _unversioned_signature(value: str) -> str:
    # The former traceback format preserved numerical values. Earlier
    # "Error: ... N ..." signatures lost information and cannot be inferred.
    parts = value.split("|", 3)
    if not re.fullmatch(r"[\w.]*(?:Error|Exception|Warning)|UnknownFailure", parts[0]):
        return ""
    if len(parts) == 2:
        return failure_signature(parts[0], parts[1], "", "")
    if len(parts) in {3, 4} and parts[1].endswith((".py", ".cpp", ".cu", ".hip")):
        return failure_signature(parts[0], parts[-1], parts[1], parts[2] if len(parts) == 4 else "")
    return ""


def _session_evidence(sessions: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    records = {}
    for directory in sessions:
        directory = directory.resolve()
        state = json.loads((directory / "state.json").read_text())
        for index, journey in enumerate(state.get("repair_journey", []), 1):
            record = dict(journey.get("failure_evidence") or {})
            if not _signature_from_record(record):
                paths = record.get("evidence_paths") or journey.get("evidence_paths") or []
                diagnostic = str(journey.get("error") or "")
                for name in paths:
                    path = Path(name)
                    path = (path if path.is_absolute() else directory / path).resolve()
                    if path.is_relative_to(directory) and path.is_file():
                        diagnostic = path.read_text()
                        break
                evidence = extract_failure_evidence(diagnostic)
                if not evidence.root_file or not evidence.exception_type:
                    continue
                record = {
                    field: getattr(evidence, field)
                    for field in (
                        "exception_type",
                        "exception_message",
                        "root_file",
                        "root_function",
                        "root_source",
                    )
                }
            records[f"{state['session_id']}:repair:{index}"] = record
    return records


def migrate_repair_signatures(db: sqlite3.Connection, sessions: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Update recoverable records in the caller's transaction; retain unresolved rows."""
    evidence_by_id = _session_evidence(sessions)
    migrated = 0
    pending = []
    for row in db.execute(
        "SELECT * FROM repair_experience WHERE signature_version < ?", (FAILURE_SIGNATURE_VERSION,)
    ).fetchall():
        payload = json.loads(row["payload"])
        record = evidence_by_id.get(row["record_id"]) or payload.get("failure_evidence") or {}
        signature = _signature_from_record(record) or _unversioned_signature(row["error_signature"])
        if not signature:
            pending.append({"record_id": row["record_id"], "reason": "original failure evidence required"})
            continue
        if record:
            payload["failure_evidence"] = record
        db.execute(
            "UPDATE repair_experience SET error_signature=?, error_pattern=?, signature_version=?, payload=? WHERE record_id=?",
            (
                signature,
                similar_failure_pattern(signature),
                FAILURE_SIGNATURE_VERSION,
                json.dumps(payload, sort_keys=True),
                row["record_id"],
            ),
        )
        migrated += 1
    return {"migrated": migrated, "pending": pending}
