#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import types
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Self, Union, get_args, get_origin, get_type_hints

from .errors import SchemaValidationError


def _json_value(value: object) -> object:
    if isinstance(value, StrictSchema):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError("JSON values must be finite.")
        return value
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError("JSON object keys must be strings.")
            result[key] = _json_value(item)
        return result
    raise SchemaValidationError(f"Unsupported JSON value type: {type(value).__name__}.")


def canonical_json_text(value: object) -> str:
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_json(value: object) -> str:
    digest = hashlib.sha256(canonical_json_text(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaValidationError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SchemaValidationError(f"Invalid JSON constant: {value}.")


def load_json_object(path: str | Path) -> dict[str, object]:
    file_path = Path(path)
    try:
        value = json.loads(
            file_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"Invalid JSON in {file_path}: {exc.msg}.") from exc
    if not isinstance(value, dict):
        raise SchemaValidationError(f"Expected a JSON object in {file_path}.")
    normalized = _json_value(value)
    assert isinstance(normalized, dict)
    return normalized


def atomic_write_json(path: str | Path, value: object) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, file_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _decode_union(value: object, annotation: object, path: str) -> object:
    errors: list[str] = []
    for candidate in get_args(annotation):
        try:
            return _decode_value(value, candidate, path)
        except SchemaValidationError as exc:
            errors.append(str(exc))
    raise SchemaValidationError(f"{path} does not match any allowed type: {'; '.join(errors)}")


def _decode_value(value: object, annotation: object, path: str) -> object:
    if annotation is Any:
        return _json_value(value)

    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return _decode_union(value, annotation, path)

    if origin is Literal:
        choices = get_args(annotation)
        if not any(type(value) is type(choice) and value == choice for choice in choices):
            raise SchemaValidationError(f"{path} must be one of {choices}, got {value!r}.")
        return value

    if origin is list:
        if not isinstance(value, list):
            raise SchemaValidationError(f"{path} must be a list.")
        (item_type,) = get_args(annotation)
        return [_decode_value(item, item_type, f"{path}[{index}]") for index, item in enumerate(value)]

    if origin is tuple:
        if not isinstance(value, list):
            raise SchemaValidationError(f"{path} must be a list.")
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode_value(item, args[0], f"{path}[{index}]") for index, item in enumerate(value))
        if len(args) != len(value):
            raise SchemaValidationError(f"{path} must contain {len(args)} items.")
        return tuple(
            _decode_value(item, item_type, f"{path}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, args, strict=True))
        )

    if origin is dict:
        if not isinstance(value, dict):
            raise SchemaValidationError(f"{path} must be an object.")
        key_type, item_type = get_args(annotation)
        if key_type is not str:
            raise SchemaValidationError(f"{path} uses an unsupported key type.")
        if any(not isinstance(key, str) for key in value):
            raise SchemaValidationError(f"{path} keys must be strings.")
        return {key: _decode_value(item, item_type, f"{path}.{key}") for key, item in value.items()}

    if annotation is type(None):
        if value is not None:
            raise SchemaValidationError(f"{path} must be null.")
        return None

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except (TypeError, ValueError) as exc:
            allowed = tuple(item.value for item in annotation)
            raise SchemaValidationError(f"{path} must be one of {allowed}, got {value!r}.") from exc

    if isinstance(annotation, type) and issubclass(annotation, StrictSchema):
        if not isinstance(value, dict):
            raise SchemaValidationError(f"{path} must be an object.")
        return annotation.from_dict(value, path=path)

    if annotation is bool:
        if not isinstance(value, bool):
            raise SchemaValidationError(f"{path} must be a boolean.")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaValidationError(f"{path} must be an integer.")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SchemaValidationError(f"{path} must be a number.")
        decoded = float(value)
        if not math.isfinite(decoded):
            raise SchemaValidationError(f"{path} must be finite.")
        return decoded
    if annotation is str:
        if not isinstance(value, str):
            raise SchemaValidationError(f"{path} must be a string.")
        return value

    raise SchemaValidationError(f"{path} uses unsupported annotation {annotation!r}.")


class StrictSchema:
    def to_dict(self) -> dict[str, object]:
        if not is_dataclass(self):
            raise TypeError(f"{type(self).__name__} must be a dataclass.")
        return {field.name: _json_value(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object], *, path: str | None = None) -> Self:
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass.")
        current_path = path or cls.__name__
        if any(not isinstance(key, str) for key in data):
            raise SchemaValidationError(f"{current_path} keys must be strings.")
        field_map = {field.name: field for field in fields(cls)}
        unknown = sorted(set(data) - set(field_map))
        if unknown:
            raise SchemaValidationError(f"{current_path} contains unknown fields: {unknown}.")

        type_hints = get_type_hints(cls)
        values: dict[str, object] = {}
        missing: list[str] = []
        for name, field in field_map.items():
            if name not in data:
                if field.default is MISSING and field.default_factory is MISSING:
                    missing.append(name)
                continue
            values[name] = _decode_value(data[name], type_hints[name], f"{current_path}.{name}")
        if missing:
            raise SchemaValidationError(f"{current_path} is missing required fields: {sorted(missing)}.")
        try:
            return cls(**values)
        except SchemaValidationError:
            raise
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(f"{current_path} is invalid: {exc}") from exc


__all__ = [
    "StrictSchema",
    "atomic_write_json",
    "canonical_json_text",
    "load_json_object",
    "sha256_json",
]
