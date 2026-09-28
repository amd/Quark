#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

from quark.experimental.torch.mixed_precision_planner._serialization import (
    StrictSchema,
    atomic_write_json,
    canonical_json_text,
    load_json_object,
)
from quark.experimental.torch.mixed_precision_planner.errors import SchemaValidationError


@dataclass(frozen=True, slots=True)
class Nested(StrictSchema):
    value: int


@dataclass(frozen=True, slots=True)
class Sample(StrictSchema):
    version: Literal[1]
    name: Literal["sample"]
    values: list[int]
    pair: tuple[int, str]
    mapping: dict[str, float]
    optional: str | None
    nested: Nested
    anything: Any
    flag: bool


class NotDataclass(StrictSchema):
    pass


def sample_dict() -> dict[str, object]:
    return {
        "version": 1,
        "name": "sample",
        "values": [1],
        "pair": [2, "x"],
        "mapping": {"score": 1},
        "optional": None,
        "nested": {"value": 3},
        "anything": {"free": True},
        "flag": True,
    }


def test_strict_schema_round_trip() -> None:
    value = Sample.from_dict(sample_dict())
    assert value.mapping["score"] == 1.0
    assert Sample.from_dict(value.to_dict()) == value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("name", "other"),
        ("values", "bad"),
        ("values", ["bad"]),
        ("pair", "bad"),
        ("pair", [1]),
        ("mapping", "bad"),
        ("optional", 1),
        ("nested", "bad"),
        ("flag", 1),
        ("anything", math.inf),
    ],
)
def test_strict_schema_rejects_wrong_types(field: str, value: object) -> None:
    data = sample_dict()
    data[field] = value
    with pytest.raises(SchemaValidationError):
        Sample.from_dict(data)


def test_strict_schema_rejects_unknown_missing_and_non_string_fields() -> None:
    data = sample_dict()
    data["unknown"] = True
    with pytest.raises(SchemaValidationError, match="unknown fields"):
        Sample.from_dict(data)
    with pytest.raises(SchemaValidationError, match="missing required"):
        Sample.from_dict({})
    with pytest.raises(SchemaValidationError, match="keys must be strings"):
        Sample.from_dict({1: "value"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dataclass"):
        NotDataclass().to_dict()


def test_json_validation_and_atomic_write(tmp_path: Path) -> None:
    assert canonical_json_text({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    with pytest.raises(SchemaValidationError, match="finite"):
        canonical_json_text({"value": math.nan})
    with pytest.raises(SchemaValidationError, match="keys must be strings"):
        canonical_json_text({1: "value"})
    with pytest.raises(SchemaValidationError, match="Unsupported JSON value"):
        canonical_json_text(object())

    path = tmp_path / "nested" / "value.json"
    atomic_write_json(path, {"b": 1, "a": 2})
    assert load_json_object(path) == {"a": 2, "b": 1}

    path.write_text("{", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="Invalid JSON"):
        load_json_object(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="Expected a JSON object"):
        load_json_object(path)
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="Invalid JSON constant"):
        load_json_object(path)
    path.write_text('{"value": 1e999}', encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="finite"):
        load_json_object(path)
