#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Inspect an ONNX artifact and compare its public I/O with a float source model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def value_info_signature(items) -> list[dict[str, object]]:
    result = []
    for item in items:
        tensor = item.type.tensor_type
        dims: list[int | str | None] = []
        for dim in tensor.shape.dim:
            dims.append(dim.dim_value or dim.dim_param or None)
        result.append({"name": item.name, "elem_type": tensor.elem_type, "shape": dims})
    return result


def model_summary(path: Path) -> tuple[dict[str, object] | None, list[str]]:
    errors: list[str] = []
    try:
        import onnx
    except ImportError as error:
        return None, [f"onnx is not installed: {error}"]
    try:
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model, full_check=False)
    except Exception as error:  # ONNX exposes several checker/load exception types.
        return None, [f"cannot load/check {path}: {error}"]

    op_counts = Counter(node.op_type for node in model.graph.node)
    domains = Counter((node.domain or "ai.onnx") for node in model.graph.node)
    quant_ops = {
        name: count
        for name, count in op_counts.items()
        if name in {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "MatMulNBits"} or "Quant" in name
    }
    external_files = sorted(
        {
            entry.value
            for initializer in model.graph.initializer
            for entry in initializer.external_data
            if entry.key == "location"
        }
    )
    missing_external = [name for name in external_files if not (path.parent / name).is_file()]
    if missing_external:
        errors.append("external data files are missing")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "ir_version": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "inputs": value_info_signature(model.graph.input),
        "outputs": value_info_signature(model.graph.output),
        "operator_counts": dict(sorted(op_counts.items())),
        "domain_counts": dict(sorted(domains.items())),
        "quantization_operator_counts": dict(sorted(quant_ops.items())),
        "quark_domain_nodes": sum(count for domain, count in domains.items() if domain.startswith("com.amd.quark")),
        "external_data_files": external_files,
        "missing_external_data_files": missing_external,
    }, errors


def inspect(source: Path | None, quantized: Path) -> dict[str, object]:
    quant_summary, errors = model_summary(quantized)
    if quant_summary is None:
        return {"status": "failed", "errors": errors}
    warnings: list[str] = []
    marker_count = sum(quant_summary["quantization_operator_counts"].values()) + quant_summary["quark_domain_nodes"]
    if marker_count == 0:
        warnings.append("no quantization operator or com.amd.quark node was detected")

    comparison = None
    if source:
        source_summary, source_errors = model_summary(source)
        errors.extend(source_errors)
        if source_summary:
            comparison = {
                "inputs_equal": source_summary["inputs"] == quant_summary["inputs"],
                "outputs_equal": source_summary["outputs"] == quant_summary["outputs"],
                "source_size_bytes": source_summary["size_bytes"],
                "quantized_size_bytes": quant_summary["size_bytes"],
            }
            if not comparison["inputs_equal"] or not comparison["outputs_equal"]:
                warnings.append("public input/output signatures differ from the source model")

    return {
        "status": "failed" if errors else "ok",
        "quantized": quant_summary,
        "comparison": comparison,
        "warnings": warnings,
        "errors": errors,
        "scope": "structure and ONNX schema only; provider load, inference, accuracy, and performance remain required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path)
    parser.add_argument("--quantized-model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.source_model, args.quantized_model)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
