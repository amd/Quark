#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Perform a dependency-free structural inspection of a Quark Torch export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, f"missing {path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"cannot read {path.name}: {error}"
    return value if isinstance(value, dict) else None, None if isinstance(
        value, dict
    ) else f"{path.name} is not an object"


def quantization_keys(value: object, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "quant" in str(key).lower() or str(key).lower() in {"bits", "dtype", "scheme"}:
                found.append(path)
            found.extend(quantization_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value[:100]):
            found.extend(quantization_keys(child, f"{prefix}[{index}]"))
    return found


def inspect(source: Path | None, output: Path) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    if not output.is_dir():
        return {"status": "failed", "errors": [f"not a directory: {output}"]}

    config, config_error = load_json(output / "config.json")
    if config_error:
        errors.append(config_error)
    weight_files = sorted(path.name for path in output.glob("*.safetensors"))
    weight_files += sorted(path.name for path in output.glob("*.bin"))
    indexes = sorted(path.name for path in output.glob("*.index.json"))
    if not weight_files:
        errors.append("no .safetensors or .bin weight files found")
    markers = sorted(set(quantization_keys(config or {})))
    if not markers:
        warnings.append("no quantization-related key found in config.json")

    missing_index_shards: list[str] = []
    for index_name in indexes:
        index, error = load_json(output / index_name)
        if error or not index:
            warnings.append(error or f"empty {index_name}")
            continue
        for shard in set((index.get("weight_map") or {}).values()):
            if isinstance(shard, str) and not (output / shard).is_file():
                missing_index_shards.append(shard)
    if missing_index_shards:
        errors.append("index references missing shards")

    source_aux_missing: list[str] = []
    if source and source.is_dir():
        for candidate in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            if (source / candidate).is_file() and not (output / candidate).is_file():
                source_aux_missing.append(candidate)
        if source_aux_missing:
            warnings.append("some source auxiliary files are absent from output")

    return {
        "status": "failed" if errors else "ok",
        "quantized_model_dir": str(output.resolve()),
        "config_present": config is not None,
        "quantization_markers": markers,
        "weight_files": weight_files,
        "weight_bytes": sum((output / name).stat().st_size for name in weight_files),
        "index_files": indexes,
        "missing_index_shards": sorted(set(missing_index_shards)),
        "source_auxiliary_missing": source_aux_missing,
        "warnings": warnings,
        "errors": errors,
        "scope": "structural inspection only; consumer load, inference, and accuracy remain required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", type=Path)
    parser.add_argument("--quantized-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.source_model_dir, args.quantized_model_dir)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
