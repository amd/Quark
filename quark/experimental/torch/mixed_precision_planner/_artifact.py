#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._serialization import StrictSchema, atomic_write_json, load_json_object, sha256_json
from .errors import ArtifactCompatibilityError, SchemaValidationError


@dataclass(frozen=True, slots=True)
class Artifact(StrictSchema):
    """Versioned JSON envelope shared by mixed-precision stage artifacts."""

    schema_version: int
    artifact_type: str
    artifact_id: str
    created_at: str
    model_fingerprint: str | None
    input_fingerprints: dict[str, str]
    upstream: dict[str, str]
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise SchemaValidationError(f"Unsupported artifact schema version: {self.schema_version}.")
        if not self.artifact_type:
            raise SchemaValidationError("artifact_type must not be empty.")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaValidationError("created_at must be an ISO-8601 timestamp.") from exc
        if self.model_fingerprint == "":
            raise SchemaValidationError("model_fingerprint must be null or non-empty.")
        if any(not key or not value.startswith("sha256:") for key, value in self.input_fingerprints.items()):
            raise SchemaValidationError("Input fingerprints must use non-empty names and SHA-256 values.")
        if any(not key or not value.startswith("sha256:") for key, value in self.upstream.items()):
            raise SchemaValidationError("Upstream artifact ids must use non-empty names and SHA-256 values.")
        self.validate_integrity()

    @staticmethod
    def fingerprint_inputs(inputs: Mapping[str, object]) -> dict[str, str]:
        if any(not isinstance(key, str) or not key for key in inputs):
            raise SchemaValidationError("Artifact input names must be non-empty strings.")
        return {key: sha256_json(value) for key, value in sorted(inputs.items())}

    def semantic_dict(self) -> dict[str, object]:
        value = self.to_dict()
        value.pop("artifact_id")
        value.pop("created_at")
        return value

    def compute_artifact_id(self) -> str:
        return sha256_json(self.semantic_dict())

    def validate_integrity(self) -> None:
        if not self.artifact_id or self.artifact_id != self.compute_artifact_id():
            raise SchemaValidationError(f"Invalid artifact id for {self.artifact_type}.")

    @classmethod
    def create(
        cls,
        artifact_type: str,
        payload: Mapping[str, object],
        *,
        model_fingerprint: str | None = None,
        inputs: Mapping[str, object] | None = None,
        upstream: Mapping[str, str] | None = None,
        created_at: str | None = None,
    ) -> Artifact:
        resolved_created_at = created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
        semantic: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": artifact_type,
            "model_fingerprint": model_fingerprint,
            "input_fingerprints": cls.fingerprint_inputs(inputs or {}),
            "upstream": dict(upstream or {}),
            "payload": dict(payload),
        }
        return cls(
            artifact_id=sha256_json(semantic),
            created_at=resolved_created_at,
            schema_version=1,
            artifact_type=artifact_type,
            model_fingerprint=model_fingerprint,
            input_fingerprints=cls.fingerprint_inputs(inputs or {}),
            upstream=dict(upstream or {}),
            payload=dict(payload),
        )

    @classmethod
    def load(cls, path: str | Path, *, expected_type: str | None = None) -> Artifact:
        artifact = cls.from_dict(load_json_object(path))
        if expected_type is not None and artifact.artifact_type != expected_type:
            raise SchemaValidationError(f"Expected artifact type {expected_type!r}, got {artifact.artifact_type!r}.")
        return artifact

    def save(self, path: str | Path) -> None:
        self.validate_integrity()
        atomic_write_json(path, self.to_dict())

    def validate_compatibility(
        self,
        *,
        model_fingerprint: str | None,
        inputs: Mapping[str, object],
        upstream: Mapping[str, str],
    ) -> None:
        self.validate_integrity()
        mismatches: list[str] = []
        if self.model_fingerprint != model_fingerprint:
            mismatches.append("model")
        if self.input_fingerprints != self.fingerprint_inputs(inputs):
            mismatches.append("inputs")
        if self.upstream != dict(upstream):
            mismatches.append("upstream")
        if mismatches:
            raise ArtifactCompatibilityError(
                f"Incompatible {self.artifact_type} artifact: {', '.join(mismatches)} changed."
            )


__all__ = ["Artifact"]
