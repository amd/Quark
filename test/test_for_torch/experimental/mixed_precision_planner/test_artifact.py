#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from quark.experimental.torch.mixed_precision_planner._artifact import Artifact
from quark.experimental.torch.mixed_precision_planner.errors import (
    ArtifactCompatibilityError,
    SchemaValidationError,
)


def test_artifact_round_trip(tmp_path: Path) -> None:
    upstream = Artifact.create("space", {})
    artifact = Artifact.create(
        "example",
        {"value": 1},
        model_fingerprint="model",
        inputs={"strategy": {"seed": 42}},
        upstream={"space": upstream.artifact_id},
    )
    path = tmp_path / "artifact.json"
    artifact.save(path)
    assert Artifact.load(path, expected_type="example") == artifact


def test_artifact_id_is_stable() -> None:
    first = Artifact.create(
        "example",
        {"a": 1, "b": 2},
        inputs={"x": 1, "y": 2},
        created_at="2026-01-01T00:00:00Z",
    )
    second = Artifact.create(
        "example",
        {"b": 2, "a": 1},
        inputs={"y": 2, "x": 1},
        created_at="2026-02-01T00:00:00Z",
    )
    assert first.artifact_id == second.artifact_id


def test_artifact_rejects_tampering_and_wrong_type(tmp_path: Path) -> None:
    artifact = Artifact.create("example", {"value": 1})
    tampered = artifact.to_dict()
    tampered["payload"] = {"value": 2}
    with pytest.raises(SchemaValidationError):
        Artifact.from_dict(tampered)

    path = tmp_path / "artifact.json"
    artifact.save(path)
    with pytest.raises(SchemaValidationError):
        Artifact.load(path, expected_type="other")


def test_artifact_revalidates_mutable_payload_at_boundaries(tmp_path: Path) -> None:
    artifact = Artifact.create("example", {"value": 1})
    artifact.payload["value"] = 2
    with pytest.raises(SchemaValidationError, match="Invalid artifact id"):
        artifact.save(tmp_path / "artifact.json")
    with pytest.raises(SchemaValidationError, match="Invalid artifact id"):
        artifact.validate_compatibility(model_fingerprint=None, inputs={}, upstream={})


def test_artifact_compatibility() -> None:
    upstream = Artifact.create("space", {})
    artifact = Artifact.create(
        "example",
        {},
        model_fingerprint="model",
        inputs={"seed": 42},
        upstream={"space": upstream.artifact_id},
    )
    artifact.validate_compatibility(
        model_fingerprint="model",
        inputs={"seed": 42},
        upstream={"space": upstream.artifact_id},
    )
    with pytest.raises(ArtifactCompatibilityError, match="model"):
        artifact.validate_compatibility(
            model_fingerprint="other",
            inputs={"seed": 42},
            upstream={"space": upstream.artifact_id},
        )
    with pytest.raises(ArtifactCompatibilityError, match="inputs"):
        artifact.validate_compatibility(
            model_fingerprint="model",
            inputs={"seed": 43},
            upstream={"space": upstream.artifact_id},
        )
    with pytest.raises(ArtifactCompatibilityError, match="upstream"):
        artifact.validate_compatibility(
            model_fingerprint="model",
            inputs={"seed": 42},
            upstream={},
        )


def test_artifact_rejects_invalid_input_names() -> None:
    with pytest.raises(SchemaValidationError, match="input names"):
        Artifact.create("example", {}, inputs={"": 1})


def test_artifact_rejects_invalid_json_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="Duplicate JSON key"):
        Artifact.load(path)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"schema_version": 2}, SchemaValidationError),
        ({"artifact_type": ""}, SchemaValidationError),
        ({"created_at": ""}, SchemaValidationError),
        ({"model_fingerprint": ""}, SchemaValidationError),
        ({"input_fingerprints": {"x": ""}}, SchemaValidationError),
        ({"upstream": {"x": ""}}, SchemaValidationError),
    ],
)
def test_artifact_rejects_invalid_envelope(changes: dict[str, object], error: type[Exception]) -> None:
    artifact = Artifact.create("example", {})
    with pytest.raises(error):
        replace(artifact, artifact_id="", **changes)
