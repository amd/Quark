#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""SQLite-backed runtime experience and operational health storage."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.repair.signatures import FAILURE_SIGNATURE_VERSION, similar_failure_pattern

SCHEMA_VERSION = 3
logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quantization_experience (
    record_id TEXT PRIMARY KEY,
    source_session_id TEXT NOT NULL,
    context_fingerprint TEXT NOT NULL,
    model_arch TEXT NOT NULL,
    framework TEXT NOT NULL,
    gpu_type TEXT NOT NULL,
    outcome TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    payload TEXT NOT NULL,
    observation_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS repair_experience (
    record_id TEXT PRIMARY KEY,
    source_session_id TEXT NOT NULL,
    context_fingerprint TEXT NOT NULL,
    framework TEXT NOT NULL,
    framework_version TEXT NOT NULL,
    arch_fingerprint TEXT NOT NULL,
    quant_signature TEXT NOT NULL,
    failure_mode TEXT NOT NULL,
    error_signature TEXT NOT NULL,
    signature_version INTEGER NOT NULL DEFAULT 1,
    error_pattern TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    payload TEXT NOT NULL,
    observation_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS kernel_optimization_experience (
    record_id TEXT PRIMARY KEY,
    source_session_id TEXT NOT NULL,
    context_fingerprint TEXT NOT NULL,
    kernel_signature TEXT NOT NULL,
    bound_type TEXT NOT NULL,
    quant_signature TEXT NOT NULL,
    outcome TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    payload TEXT NOT NULL,
    observation_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge_review (
    experience_key TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS baseline_health_experience (
    fingerprint TEXT NOT NULL,
    framework TEXT NOT NULL,
    framework_commit TEXT NOT NULL,
    outcome TEXT NOT NULL,
    failure_class TEXT NOT NULL,
    cache_policy TEXT NOT NULL,
    diagnosis TEXT NOT NULL,
    recovery TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def _decode_payload(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["payload"] = json.loads(str(result["payload"]) or "{}")
    return result


class ExperienceStore:
    def __init__(self, path: str | Path):
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path))
        self.path = db_path
        self._db.row_factory = sqlite3.Row
        self.migration_summary: dict[str, Any] = {"migrated": 0, "pending": []}
        try:
            existing = self._db.execute("SELECT 1 FROM sqlite_master WHERE name='schema_metadata'").fetchone()
            version_row = (
                self._db.execute("SELECT value FROM schema_metadata WHERE name='schema_version'").fetchone()
                if existing
                else None
            )
            version = int(version_row[0]) if version_row else 0
            if version > SCHEMA_VERSION:
                raise ValueError(f"experience database schema {version} is newer than supported {SCHEMA_VERSION}")
            if version == SCHEMA_VERSION:
                return
            if existing and version < SCHEMA_VERSION:
                self.backup(version)
            with self._db:
                self._db.execute("BEGIN IMMEDIATE")
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        self._db.execute(statement)
                if version < SCHEMA_VERSION:
                    from .migration import migrate_repair_signatures

                    columns = {row[1] for row in self._db.execute("PRAGMA table_info(repair_experience)")}
                    if "signature_version" not in columns:
                        self._db.execute(
                            "ALTER TABLE repair_experience ADD COLUMN signature_version INTEGER NOT NULL DEFAULT 1"
                        )
                        self._db.execute(
                            "ALTER TABLE repair_experience ADD COLUMN error_pattern TEXT NOT NULL DEFAULT ''"
                        )
                    self.migration_summary = migrate_repair_signatures(self._db)
                    self._db.execute(
                        "INSERT OR REPLACE INTO schema_metadata(name, value) VALUES ('schema_version', ?)",
                        (str(SCHEMA_VERSION),),
                    )
            if self.migration_summary["pending"]:
                logger.warning(
                    "%d repair records need original evidence; use knowledge migrate --session DIR",
                    len(self.migration_summary["pending"]),
                )
        except Exception:
            self._db.close()
            raise

    def backup(self, version: int = SCHEMA_VERSION) -> Path:
        path = self.path.with_name(f"{self.path.name}.v{version}.{time.time_ns()}.bak")
        with sqlite3.connect(str(path)) as target:
            self._db.backup(target)
        return path

    def migrate_repair(self, sessions: tuple[Path, ...]) -> dict[str, Any]:
        from .migration import migrate_repair_signatures

        if self._db.execute(
            "SELECT 1 FROM repair_experience WHERE signature_version < ? LIMIT 1", (FAILURE_SIGNATURE_VERSION,)
        ).fetchone():
            self.backup()
        with self._db:
            return migrate_repair_signatures(self._db, sessions)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> ExperienceStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _upsert(
        self,
        *,
        table: str,
        values: dict[str, Any],
    ) -> None:
        columns = list(values)
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(
            f"{column}=excluded.{column}" for column in columns if column not in {"record_id", "observation_count"}
        )
        self._db.execute(
            f"INSERT INTO {table} "
            f"({', '.join(columns)}, observation_count) "
            f"VALUES ({placeholders}, 1) "
            "ON CONFLICT(record_id) DO UPDATE SET "
            f"{updates}, "
            "last_seen_at=CURRENT_TIMESTAMP",
            tuple(values[column] for column in columns),
        )
        self._db.commit()

    def record_quantization_experience(
        self,
        *,
        record_id: str,
        source_session_id: str,
        context_fingerprint: str,
        model_arch: str,
        framework: str,
        gpu_type: str,
        outcome: str,
        verification_status: str,
        payload: dict[str, Any],
    ) -> None:
        self._upsert(
            table="quantization_experience",
            values={
                "record_id": record_id,
                "source_session_id": source_session_id,
                "context_fingerprint": context_fingerprint,
                "model_arch": model_arch,
                "framework": framework,
                "gpu_type": gpu_type,
                "outcome": outcome,
                "verification_status": verification_status,
                "payload": json.dumps(payload, sort_keys=True),
            },
        )

    def find_quantization_experience(
        self,
        *,
        model_arch: str,
        framework: str,
        gpu_type: str,
    ) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM quantization_experience "
            "WHERE model_arch=? AND framework=? AND gpu_type=? "
            "ORDER BY last_seen_at DESC, record_id",
            (model_arch, framework, gpu_type),
        ).fetchall()
        return [_decode_payload(dict(row)) for row in rows]

    def record_repair_experience(
        self,
        *,
        record_id: str,
        source_session_id: str,
        context_fingerprint: str,
        framework: str,
        framework_version: str,
        arch_fingerprint: str,
        quant_signature: str,
        failure_mode: str,
        error_signature: str,
        outcome: str,
        verification_status: str,
        payload: dict[str, Any],
    ) -> None:
        if error_signature and not error_signature.startswith(f"v{FAILURE_SIGNATURE_VERSION}|"):
            raise ValueError("repair experience requires a canonical failure signature")
        self._upsert(
            table="repair_experience",
            values={
                "record_id": record_id,
                "source_session_id": source_session_id,
                "context_fingerprint": context_fingerprint,
                "framework": framework,
                "framework_version": framework_version,
                "arch_fingerprint": arch_fingerprint,
                "quant_signature": quant_signature,
                "failure_mode": failure_mode,
                "error_signature": error_signature,
                "signature_version": FAILURE_SIGNATURE_VERSION,
                "error_pattern": similar_failure_pattern(error_signature),
                "outcome": outcome,
                "verification_status": verification_status,
                "payload": json.dumps(payload, sort_keys=True),
            },
        )

    def find_repair_guidance(
        self,
        *,
        framework: str,
        framework_version: str,
        arch_fingerprint: str,
        quant_signature: str,
        failure_mode: str,
        error_signature: str,
    ) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM repair_experience "
            "WHERE framework=? AND framework_version=? "
            "AND arch_fingerprint=? AND quant_signature=? "
            "AND failure_mode=? "
            "AND (?='' OR (signature_version=? AND error_signature=?)) "
            "ORDER BY (verification_status='verified') DESC, "
            "last_seen_at DESC LIMIT 1",
            (
                framework,
                framework_version,
                arch_fingerprint,
                quant_signature,
                failure_mode,
                error_signature,
                FAILURE_SIGNATURE_VERSION,
                error_signature,
            ),
        ).fetchone()
        if row is not None:
            return {**_decode_payload(dict(row)), "match_type": "exact"}
        pattern = similar_failure_pattern(error_signature)
        if not pattern:
            return None
        row = self._db.execute(
            "SELECT * FROM repair_experience WHERE framework=? AND framework_version=? "
            "AND arch_fingerprint=? AND quant_signature=? AND failure_mode=? "
            "AND signature_version=? AND error_pattern=? "
            "ORDER BY (verification_status='verified') DESC, last_seen_at DESC, record_id LIMIT 1",
            (
                framework,
                framework_version,
                arch_fingerprint,
                quant_signature,
                failure_mode,
                FAILURE_SIGNATURE_VERSION,
                pattern,
            ),
        ).fetchone()
        return {**_decode_payload(dict(row)), "match_type": "related"} if row is not None else None

    def record_kernel_optimization_experience(
        self,
        *,
        record_id: str,
        source_session_id: str,
        context_fingerprint: str,
        kernel_signature: str,
        bound_type: str,
        quant_signature: str,
        outcome: str,
        verification_status: str,
        payload: dict[str, Any],
    ) -> None:
        self._upsert(
            table="kernel_optimization_experience",
            values={
                "record_id": record_id,
                "source_session_id": source_session_id,
                "context_fingerprint": context_fingerprint,
                "kernel_signature": kernel_signature,
                "bound_type": bound_type,
                "quant_signature": quant_signature,
                "outcome": outcome,
                "verification_status": verification_status,
                "payload": json.dumps(payload, sort_keys=True),
            },
        )

    def find_kernel_optimization_experience(
        self,
        *,
        kernel_signature: str,
        bound_type: str,
        quant_signature: str,
    ) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM kernel_optimization_experience "
            "WHERE kernel_signature=? AND bound_type=? "
            "AND quant_signature=? "
            "ORDER BY last_seen_at DESC, record_id",
            (kernel_signature, bound_type, quant_signature),
        ).fetchall()
        return [_decode_payload(dict(row)) for row in rows]

    def record_baseline_health(
        self,
        *,
        fingerprint: str,
        framework: str,
        framework_commit: str,
        outcome: str,
        failure_class: str,
        cache_policy: str,
        diagnosis: str,
        recovery: str = "",
    ) -> None:
        self._db.execute(
            "INSERT INTO baseline_health_experience "
            "(fingerprint, framework, framework_commit, outcome, "
            "failure_class, cache_policy, diagnosis, recovery) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fingerprint,
                framework,
                framework_commit,
                outcome,
                failure_class,
                cache_policy,
                diagnosis[:4000],
                recovery,
            ),
        )
        self._db.commit()

    def baseline_hard_failure(self, fingerprint: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT outcome, failure_class, cache_policy, diagnosis, recovery "
            "FROM baseline_health_experience WHERE fingerprint=? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (fingerprint,),
        ).fetchone()
        if row is None or row["outcome"] != "failed":
            return None
        if row["cache_policy"] != "hard":
            return None
        return dict(row)

    def list_reviewable_experience(
        self,
        *,
        session_id: str = "",
        domain: str = "",
        include_reviewed: bool = False,
    ) -> list[dict[str, Any]]:
        tables = {
            "quantization": "quantization_experience",
            "repair": "repair_experience",
            "kernel_optimization": ("kernel_optimization_experience"),
        }
        selected = {domain: tables[domain]} if domain else tables
        rows: list[dict[str, Any]] = []
        for row_domain, table in selected.items():
            sql = f"SELECT * FROM {table}"
            params: tuple[Any, ...] = ()
            if session_id:
                sql += " WHERE source_session_id=?"
                params = (session_id,)
            sql += " ORDER BY created_at, record_id"
            for row in self._db.execute(sql, params).fetchall():
                item = _decode_payload(dict(row))
                key = f"{row_domain}:{item['record_id']}"
                review = self._db.execute(
                    "SELECT decision, reason FROM knowledge_review WHERE experience_key=?",
                    (key,),
                ).fetchone()
                decision = str(review["decision"]) if review else ""
                if decision and not include_reviewed:
                    continue
                item.update(
                    {
                        "domain": row_domain,
                        "experience_key": key,
                        "review_decision": decision,
                        "review_reason": (str(review["reason"]) if review else ""),
                    }
                )
                rows.append(item)
        return rows

    def get_experience(self, experience_key: str) -> dict[str, Any]:
        domain, separator, record_id = experience_key.partition(":")
        tables = {
            "quantization": "quantization_experience",
            "repair": "repair_experience",
            "kernel_optimization": ("kernel_optimization_experience"),
        }
        if not separator or domain not in tables:
            raise KeyError(experience_key)
        row = self._db.execute(
            f"SELECT * FROM {tables[domain]} WHERE record_id=?",
            (record_id,),
        ).fetchone()
        if row is None:
            raise KeyError(experience_key)
        result = _decode_payload(dict(row))
        result["domain"] = domain
        result["experience_key"] = experience_key
        return result

    def record_review_decision(
        self,
        *,
        experience_key: str,
        domain: str,
        decision: str,
        reason: str = "",
    ) -> None:
        self._db.execute(
            "INSERT INTO knowledge_review "
            "(experience_key, domain, decision, reason) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(experience_key) DO UPDATE SET "
            "domain=excluded.domain, decision=excluded.decision, "
            "reason=excluded.reason, reviewed_at=CURRENT_TIMESTAMP",
            (experience_key, domain, decision, reason),
        )
        self._db.commit()
