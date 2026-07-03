#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Collect calibrated NVFP4 input min/max/scale from a Quark-prepared model.

The NVFP4 input quantizer has two stages:

* stage 0: FP4 per-group (group_size=16, DYNAMIC) — per-group micro scale,
  recomputed every forward, so not a fixed calibration value.
* stage 1: FP8E4M3 per-tensor (STATIC, min_max) — the calibrated per-tensor
  global scale. This is what we export: a scalar min, max, scale.

:func:`collect_input_minmax` reads the per-tensor STATIC stage's observer
(min_val / max_val) and its computed ``scale`` buffer for every input quantizer.
RAM-resident blocks are read from live modules; disk-offloaded blocks are
reconstructed from the saved state written by ``dsv4_offload``.
"""

from __future__ import annotations

import math
import re as _re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from quark.torch.quantization.nn.modules.quantize_linear import QuantMixin
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

_ROUTED_PROXY_RE = _re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.proxy$")
_SHARED_PROXY_RE = _re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.(w[123])\.proxy$")


def _is_static_pertensor_stage(stage) -> bool:
    """True if this fake-quant stage holds a scalar per-tensor min/max observer."""
    obs = getattr(stage, "observer", None)
    if obs is None:
        return False
    mn = getattr(obs, "min_val", None)
    # Per-tensor observer stores a 0-dim (scalar) min_val; per-group is an array.
    return isinstance(mn, torch.Tensor) and mn.dim() == 0


def _stage_minmax_scale(stage) -> dict | None:
    """Read {min, max, amax, scale} from a static per-tensor stage, or None.

    Returns None for never-routed observers (min=+inf, max=-inf).
    """
    obs = getattr(stage, "observer", None)
    if obs is None:
        return None
    mn = getattr(obs, "min_val", None)
    mx = getattr(obs, "max_val", None)
    if not (isinstance(mn, torch.Tensor) and isinstance(mx, torch.Tensor)):
        return None
    mnf, mxf = float(mn.float().item()), float(mx.float().item())
    if not (math.isfinite(mnf) and math.isfinite(mxf)):
        return None
    rec = {"min": mnf, "max": mxf, "amax": max(abs(mnf), abs(mxf))}
    sc = stage._buffers.get("scale")
    if isinstance(sc, torch.Tensor) and sc.numel() == 1:
        rec["scale"] = float(sc.float().item())
    return rec


def collect_input_minmax(
    model: nn.Module,
    offload_dir: Path | None,
    n_ram_blocks: int,
) -> dict[str, dict[str, float]]:
    """Return {layer_name: {min, max, amax, scale}} for every input quantizer.

    Values come from the NVFp4 input quantizer's per-tensor STATIC stage:
      - min / max : observer.min_val / max_val (scalars)
      - amax      : max(|min|, |max|)
      - scale     : the static stage's computed per-tensor scale buffer

    RAM-resident blocks are read from live modules.  Disk-offloaded blocks have
    their state in ``offload_dir/block_<idx>.pt`` with keys
    ``<rel_name>@@obs.{min,max}_val`` and ``<rel_name>@@buf.scale``.
    """
    result: dict[str, dict[str, float]] = {}

    layers = model.layers
    n_blocks = len(layers)
    disk_block_ids = set(range(n_ram_blocks, n_blocks)) if offload_dir is not None else set()

    # [1] Live (RAM-resident) observers — use the static per-tensor stage.
    for name, mod in model.named_modules():
        if not isinstance(mod, QuantMixin):
            continue
        iq = getattr(mod, "_input_quantizer", None)
        if iq is None:
            continue
        stages = [iq] if isinstance(iq, ScaledFakeQuantize) else list(iq)
        for stage in stages:
            if not _is_static_pertensor_stage(stage):
                continue
            rec = _stage_minmax_scale(stage)
            if rec is not None:
                result[name] = rec
            break

    # [2] Disk-offloaded blocks: reconstruct from saved state.
    for idx in disk_block_ids:
        path = offload_dir / f"block_{idx}.pt"
        if not path.exists():
            continue
        st = torch.load(path, map_location="cpu", weights_only=True)
        block_prefix = f"layers.{idx}."
        # Group saved keys by relative module name (per fake-quant stage).
        rel: dict[str, dict[str, float]] = {}
        for key, tensor in st.items():
            rel_name, attr_key = key.split("@@", 1)
            if attr_key == "obs.min_val" and tensor.dim() == 0:
                rel.setdefault(rel_name, {})["min"] = float(tensor.float().item())
            elif attr_key == "obs.max_val" and tensor.dim() == 0:
                rel.setdefault(rel_name, {})["max"] = float(tensor.float().item())
            elif attr_key == "buf.scale" and tensor.numel() == 1:
                rel.setdefault(rel_name, {})["scale"] = float(tensor.float().item())
        for rel_name, rec in rel.items():
            # Only the static per-tensor stage has scalar min AND max.
            if "min" not in rec or "max" not in rec:
                continue
            if not (math.isfinite(rec["min"]) and math.isfinite(rec["max"])):
                continue
            rec["amax"] = max(abs(rec["min"]), abs(rec["max"]))
            layer_rel = rel_name.split("._input_quantizer")[0]
            result[block_prefix + layer_rel] = rec
        del st

    return result


# ---------------------------------------------------------------------------
# Convert collected input_scale -> NVFP4 safetensors layout
# ---------------------------------------------------------------------------
#
# The collected `scale_map` keys look like the wrapped proxy module names:
#     layers.<L>.ffn.experts.<E>.<wK>.proxy            (routed)
#     layers.<L>.ffn.shared_experts.<wK>.proxy         (shared)
# The HF checkpoint stores each as its own F32 scalar tensor named:
#     layers.<L>.ffn.experts.<E>.<wK>.input_scale
#     layers.<L>.ffn.shared_experts.<wK>.input_scale
# So conversion is rename + reshape + (small) fill of never-routed experts.


def build_input_scale_tensors(scale_map, n_experts_per_layer):
    """Convert {proxy_name: scale} -> {hf_key: F32 scalar tensor} + report.

    - routed experts: emit all (layer, expert, proj); fill missing ones with
      the max over calibrated experts in that (layer, proj).
    - shared experts: emit one per (layer, proj) when present (always active).
    """
    by_group = defaultdict(dict)  # (L, proj) -> {expert: scale}
    shared = {}  # (L, proj) -> scale
    layers_seen = set()
    for k, v in scale_map.items():
        m = _ROUTED_PROXY_RE.match(k)
        if m:
            layer, expert, proj = int(m.group(1)), int(m.group(2)), m.group(3)
            by_group[(layer, proj)][expert] = float(v)
            layers_seen.add(layer)
            continue
        ms = _SHARED_PROXY_RE.match(k)
        if ms:
            layer, proj = int(ms.group(1)), ms.group(2)
            shared[(layer, proj)] = float(v)
            layers_seen.add(layer)

    if not by_group:
        raise ValueError("no routed-expert scales collected; nothing to convert")

    n_layers = max(layers_seen) + 1
    projections = ("w1", "w2", "w3")
    fill_value = {g: max(e.values()) for g, e in by_group.items()}

    tensors = {}
    n_calibrated = n_filled = n_shared = 0
    filled_groups = defaultdict(int)
    for layer in range(n_layers):
        for expert in range(n_experts_per_layer):
            for proj in projections:
                group = (layer, proj)
                if group not in fill_value:
                    raise ValueError(f"layer {layer} proj {proj}: no calibrated expert; cannot derive a fill value")
                experts = by_group[group]
                if expert in experts:
                    val = experts[expert]
                    n_calibrated += 1
                else:
                    val = fill_value[group]
                    n_filled += 1
                    filled_groups[group] += 1
                key = f"layers.{layer}.ffn.experts.{expert}.{proj}.input_scale"
                tensors[key] = torch.tensor(val, dtype=torch.float32)

    for layer in range(n_layers):
        for proj in projections:
            if (layer, proj) not in shared:
                continue
            key = f"layers.{layer}.ffn.shared_experts.{proj}.input_scale"
            tensors[key] = torch.tensor(shared[(layer, proj)], dtype=torch.float32)
            n_shared += 1

    report = {
        "n_layers": n_layers,
        "n_experts_per_layer": n_experts_per_layer,
        "n_total": len(tensors),
        "n_calibrated": n_calibrated,
        "n_filled": n_filled,
        "n_shared": n_shared,
        "filled_groups": dict(filled_groups),
        "fill_value": fill_value,
    }
    return tensors, report
