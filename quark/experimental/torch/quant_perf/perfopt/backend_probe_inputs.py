#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Model-specific checkpoint inputs for the generic backend probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class BackendProbeInputs:
    projections: dict[str, dict[str, torch.Tensor]]


def backend_probe_input_support_reason(model_arch: str) -> str:
    if model_arch == "gemma4":
        return ""
    return f"unsupported_model_arch:{model_arch or 'unknown'}"


def load_backend_probe_inputs(
    model_arch: str,
    model_dir: str | Path,
) -> BackendProbeInputs:
    unsupported_reason = backend_probe_input_support_reason(model_arch)
    if unsupported_reason:
        raise RuntimeError(unsupported_reason)
    return load_gemma4_probe_inputs(model_dir)


def load_gemma4_probe_inputs(
    model_dir: str | Path,
) -> BackendProbeInputs:
    root = Path(model_dir)
    prefix = "model.language_model.layers.0.mlp"
    names = {
        "gate_weight": f"{prefix}.gate_proj.weight",
        "gate_scale": f"{prefix}.gate_proj.weight_scale",
        "up_weight": f"{prefix}.up_proj.weight",
        "up_scale": f"{prefix}.up_proj.weight_scale",
        "down_weight": f"{prefix}.down_proj.weight",
        "down_scale": f"{prefix}.down_proj.weight_scale",
    }
    loaded: dict[str, torch.Tensor] = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:  # type: ignore[no-untyped-call]
            keys = set(handle.keys())
            for label, name in names.items():
                if label not in loaded and name in keys:
                    loaded[label] = handle.get_tensor(name)
        if len(loaded) == len(names):
            break
    missing = sorted(set(names) - set(loaded))
    if missing:
        raise RuntimeError("Gemma4 backend probe checkpoint is missing tensors: " + ", ".join(missing))
    return BackendProbeInputs(
        projections={
            "gate_up": {
                "weight": torch.cat(
                    [loaded["gate_weight"], loaded["up_weight"]],
                    dim=0,
                ),
                "scale": torch.cat(
                    [loaded["gate_scale"], loaded["up_scale"]],
                    dim=0,
                ),
            },
            "down": {
                "weight": loaded["down_weight"],
                "scale": loaded["down_scale"],
            },
        }
    )
