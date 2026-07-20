#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Map Quark weight-only schemes to llama.cpp GGUF quant types (layout passthrough)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuarkNativeScheme:
    dtype: str
    symmetric: bool
    group_size: int
    pack_method: str

    @property
    def gguf_format(self) -> str:
        if self.dtype == "uint4" and not self.symmetric and self.group_size == 32:
            return "q4_1"
        if self.dtype == "int4" and self.symmetric and self.group_size == 32:
            return "q4_0"
        raise ValueError(
            f"No native GGUF passthrough for Quark scheme "
            f"dtype={self.dtype!r} symmetric={self.symmetric} group_size={self.group_size}. "
            "Supported: uint4_wo_32 -> q4_1, int4_wo_32 -> q4_0."
        )


def read_quark_export_config(quark_dir: Path) -> tuple[dict, dict]:
    config_path = quark_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {quark_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    qcfg = config.get("quantization_config") or {}
    export_cfg = qcfg.get("export") or {}
    weight = (qcfg.get("global_quant_config") or {}).get("weight") or {}
    return weight, export_cfg


def detect_native_scheme(quark_dir: Path) -> QuarkNativeScheme:
    weight, export_cfg = read_quark_export_config(quark_dir)
    dtype = weight.get("dtype")
    group_size = int(weight.get("group_size", 0))
    symmetric = bool(weight.get("symmetric", False))
    pack_method = export_cfg.get("pack_method", "reorder")
    if dtype not in {"uint4", "int4"} or group_size <= 0:
        raise ValueError(
            f"Cannot detect native GGUF scheme from {quark_dir / 'config.json'}. "
            "Expected global_quant_config.weight with dtype uint4/int4 and group_size."
        )
    return QuarkNativeScheme(
        dtype=dtype,
        symmetric=symmetric,
        group_size=group_size,
        pack_method=pack_method,
    )


def validate_native_passthrough(quark_dir: Path, export_format: str) -> QuarkNativeScheme:
    scheme = detect_native_scheme(quark_dir)
    expected = scheme.gguf_format
    if export_format.lower() != expected:
        raise ValueError(
            f"Quark checkpoint in {quark_dir} maps to GGUF format {expected!r} "
            f"(dtype={scheme.dtype}, symmetric={scheme.symmetric}, group_size={scheme.group_size}), "
            f"but export_format={export_format!r} was requested. "
            "Native passthrough avoids re-quantization only when formats match."
        )
    return scheme
