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
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from quark.torch.quantization.nn.modules.quantize_linear import QuantMixin
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

# Every collected input-quantizer name ends in ``.proxy`` (the NativeLinear
# submodule Quark quantizes); the checkpoint key swaps ``.proxy`` -> ``.input_scale``.
_PROXY_SUFFIX = ".proxy"


def _template(name: str) -> str:
    """Group key for the never-fired fill: mask whole-number path segments so
    siblings differing only by a layer/expert index collapse together, e.g.
    ``layers.3.ffn.experts.17.w1`` -> ``layers.*.ffn.experts.*.w1`` (the ``w1``
    suffix is kept, so the fill stays projection-specific).
    """
    return ".".join("*" if seg.isdigit() else seg for seg in name.split("."))


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
# The set of keys is driven entirely by which modules were wrapped (i.e. by the
# ``--exclude_layers`` passed to Stage 2), not by any hardcoded layer/expert layout.


def build_input_scale_tensors(scale_map, wrapped_names, n_experts_per_layer=None):
    """Convert {proxy_name: scale} -> {hf_key: F32 scalar tensor} + report.

    ``wrapped_names`` is the full set of wrapped proxy module names (everything
    Quark calibrated, including sparse experts that never fired during calibration
    and so are absent from ``scale_map``). Each name maps 1:1 to an HF key by
    swapping the trailing ``.proxy`` for ``.input_scale``: names in ``scale_map``
    take their calibrated value; names missing from it (never fired) are filled with
    the max calibrated scale among names sharing their template (see
    :func:`_template`), which never clips.

    ``n_experts_per_layer`` (optional) is only a sanity cross-check: it does not
    affect the output. When given, per-expert templates (those varying over >=2
    numeric path segments) are expected to emit ``n_layers * n_experts_per_layer``
    keys; any shortfall is reported in ``report['expert_count_mismatches']``.
    """

    # Fill key: (layer_index, per-layer template) so never-fired experts are filled
    # with the max scale from calibrated siblings in the SAME layer only.
    # Example: "layers.3.ffn.experts.17.w1.proxy" -> (3, "layers.*.ffn.experts.*.w1.proxy")
    def _fill_key(name: str) -> tuple:
        segs = name.split(".")
        layer = next((int(s) for s in segs if s.isdigit()), None)
        return (layer, _template(name))

    by_fill_key = defaultdict(list)
    for name, sc in scale_map.items():
        by_fill_key[_fill_key(name)].append(float(sc))
    fill_value = {k: max(v) for k, v in by_fill_key.items()}

    tensors = {}
    n_calibrated = n_filled = 0
    filled_templates = defaultdict(int)
    for name in wrapped_names:
        if not name.endswith(_PROXY_SUFFIX):
            continue
        key = name[: -len(_PROXY_SUFFIX)] + ".input_scale"
        if name in scale_map:
            val = float(scale_map[name])
            n_calibrated += 1
        else:
            fk = _fill_key(name)
            if fk not in fill_value:
                raise ValueError(
                    f"{name}: never fired and no calibrated sibling in layer {fk[0]} "
                    f"template '{fk[1]}'; cannot derive a fill value"
                )
            val = fill_value[fk]
            n_filled += 1
            filled_templates[fk[1]] += 1
        tensors[key] = torch.tensor(val, dtype=torch.float32)

    report = {
        "n_total": len(tensors),
        "n_calibrated": n_calibrated,
        "n_filled": n_filled,
        "filled_templates": dict(filled_templates),
    }
    if n_experts_per_layer:
        report["expert_count_mismatches"] = _check_expert_count(tensors, n_experts_per_layer)
    return tensors, report


def _check_expert_count(tensors, n_experts_per_layer):
    """Cross-check emitted per-expert keys against ``n_experts_per_layer``.

    Per-expert templates vary over >=2 numeric path segments; the leading numeric
    segment is the layer index. Each such template should emit
    ``n_layers * n_experts_per_layer`` keys. Returns
    [(template, expected, actual), ...] for any that don't.
    """
    by_template = defaultdict(list)
    for key in tensors:
        by_template[_template(key)].append(key)
    mismatches = []
    for tmpl, keys in by_template.items():
        if tmpl.count("*") < 2:
            continue
        layers = {next(s for s in k.split(".") if s.isdigit()) for k in keys}
        expected = len(layers) * n_experts_per_layer
        if len(keys) != expected:
            mismatches.append((tmpl, expected, len(keys)))
    return mismatches
