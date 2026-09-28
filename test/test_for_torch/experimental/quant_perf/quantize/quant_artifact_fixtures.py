#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from pathlib import Path


def write_minimal_safetensors(path: Path) -> None:
    header = b'{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    padding = b" " * ((8 - len(header) % 8) % 8)
    encoded = header + padding
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"\0\0\0\0")


def write_valid_quant_checkpoint(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}")
    write_minimal_safetensors(root / "model.safetensors")
