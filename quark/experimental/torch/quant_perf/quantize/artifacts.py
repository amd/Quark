#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Shared validation for exported quantized checkpoints."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any


def _valid_safetensors(path: Path) -> bool:
    try:
        size = path.stat().st_size
        if size <= 8:
            return False
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            if header_size <= 0 or header_size > size - 8:
                return False
            header = json.loads(handle.read(header_size))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(header, dict):
        return False
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    if not tensors:
        return False
    payload_size = size - 8 - header_size
    for tensor in tensors:
        if not isinstance(tensor, dict):
            return False
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > payload_size
        ):
            return False
    return True


def _weight_files(root: Path) -> tuple[list[Path], list[str]]:
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return sorted(root.glob("*.safetensors")), []
    try:
        payload: Any = json.loads(index.read_text())
        weight_map = payload.get("weight_map")
    except (OSError, json.JSONDecodeError, AttributeError):
        return [], ["invalid_safetensors_index"]
    if not isinstance(weight_map, dict) or not weight_map:
        return [], ["invalid_safetensors_index"]
    files = []
    for value in sorted(set(weight_map.values())):
        if not isinstance(value, str):
            return [], ["invalid_safetensors_index"]
        candidate = root / value
        try:
            candidate.resolve().relative_to(root.resolve())
        except ValueError:
            return [], ["invalid_safetensors_index"]
        files.append(candidate)
    return files, []


def validate_quant_checkpoint(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir)
    errors: list[str] = []
    config_path = root / "config.json"
    config_valid = False
    with contextlib.suppress(OSError, json.JSONDecodeError):
        config_valid = isinstance(json.loads(config_path.read_text()), dict)
    if not config_valid:
        errors.append("invalid_config")

    weight_files, index_errors = _weight_files(root)
    errors.extend(index_errors)
    has_safetensors = bool(weight_files)
    if not has_safetensors:
        errors.append("missing_safetensors")
    elif any(not path.is_file() or not _valid_safetensors(path) for path in weight_files):
        errors.append("invalid_safetensors")

    return {
        "status": "success" if not errors else "failed",
        "quantized_model_dir": str(root),
        "has_safetensors": has_safetensors,
        "has_config": config_valid,
        "errors": list(dict.fromkeys(errors)),
    }
