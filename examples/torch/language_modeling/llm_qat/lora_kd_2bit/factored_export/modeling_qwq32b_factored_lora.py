# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Self-loading QwQ-32B (Qwen2 architecture) variant with FACTORED rotation.

Mirror of `modeling_phi4_factored_lora.py` for QwQ-32B:
  - One dense rotation matrix R per unique `in_features` (only 2 for QwQ-32B:
    in=5120 and in=27648), stored at the TOP-LEVEL model.
  - Per quantized projection: only `awq_s_vec` (fp16 [in]) — no perm, no sign,
    no per-projection rotation matrix.
  - Forward per projection:  (x / awq_s_vec) @ R_for_in_features

Mathematically identical to the dense export `qwq-32b-2bit-lora-kd-linear-v21`
(the SRHT params are shared per `in_features` in v21 PTQ, so we factor them
out). q/k/v projections carry a bias (`bias_param`), preserved here.

Per-projection storage (×448 = 64 layers × 7 proj types):
  awq_s_vec            fp16 [in]            (per projection)
  linear.packed_levels uint8 [out, in/4]    (unchanged)
  linear.group_scale   fp16  [out, in/64]   (unchanged)
  lora_A               fp16 [r=64, in]      (unchanged)
  lora_B               fp16 [out, r=64]     (unchanged)
  bias_param           fp16 [out]           (q/k/v only)

Top-level (model-wide):
  shared_R_5120        fp16 [5120, 5120]      52 MB    (q/k/v/o/gate/up, in=5120)
  shared_R_27648       fp16 [27648, 27648]   1.53 GB   (down_proj, in=27648)

Self-loads via:
    AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)

config.json must contain:
    "architectures": ["Qwen2FactoredLoraForCausalLM"]
    "auto_map": {"AutoModelForCausalLM": "modeling_qwq32b_factored_lora.Qwen2FactoredLoraForCausalLM"}
    "srht_awq_lora": {"rank": 64, "alpha": 128, "group_size": 64,
                      "target_suffixes": ["q_proj","k_proj","v_proj","o_proj",
                                          "gate_proj","up_proj","down_proj"]}
    "factored_rotation_dims": [5120, 27648]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

DEFAULT_LINEAR_LEVELS: list[float] = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]
DEFAULT_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════


def _is_quantized_linear_name(name, suffixes):
    return any(name.endswith(s) for s in suffixes) and "embed" not in name and "lm_head" not in name


def _get_parent_and_attr(model, qualname):
    parts = qualname.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def pack_int2(idx):
    assert idx.shape[-1] % 4 == 0
    idx = idx.to(torch.uint8) & 0x3
    a = idx[..., 0::4]
    b = idx[..., 1::4] << 2
    c = idx[..., 2::4] << 4
    d = idx[..., 3::4] << 6
    return (a | b | c | d).contiguous()


def unpack_int2(packed, in_dim):
    p = packed.to(torch.uint8)
    leading = p.shape[:-1]
    out = torch.empty(*leading, in_dim, dtype=torch.long, device=p.device)
    out[..., 0::4] = (p & 0x3).long()
    out[..., 1::4] = ((p >> 2) & 0x3).long()
    out[..., 2::4] = ((p >> 4) & 0x3).long()
    out[..., 3::4] = ((p >> 6) & 0x3).long()
    return out


# ════════════════════════════════════════════════════════════════════
# 2-bit quantized linear
# ════════════════════════════════════════════════════════════════════


class Quantized2BitLinear(nn.Module):
    def __init__(self, in_features, out_features, group_size=64, levels=None, device=None, dtype=None):
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError(f"in % group_size: {in_features} % {group_size}")
        if in_features % 4 != 0:
            raise ValueError(f"in must be divisible by 4: {in_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        n_groups = in_features // group_size
        if levels is None:
            levels = DEFAULT_LINEAR_LEVELS
        self._levels = [float(v) for v in levels]
        self.register_buffer(
            "packed_levels",
            torch.zeros(out_features, in_features // 4, dtype=torch.uint8, device=device),
            persistent=True,
        )
        self.register_buffer(
            "group_scale",
            torch.zeros(out_features, n_groups, dtype=(dtype or torch.float16), device=device),
            persistent=True,
        )

    def _materialize(self, dtype):
        idx = unpack_int2(self.packed_levels, self.in_features)
        levels = torch.tensor(self._levels, device=idx.device, dtype=dtype)
        w_q = levels[idx]
        g = self.group_scale.to(dtype).repeat_interleave(self.group_size, dim=1)
        return (w_q * g).contiguous()

    def forward(self, x):
        return F.linear(x, self._materialize(dtype=x.dtype))

    @property
    def weight(self):
        return self._materialize(dtype=torch.float32)


# ════════════════════════════════════════════════════════════════════
# Factored AWQ-undo + shared SRHT (replaces dense pre_linear)
# ════════════════════════════════════════════════════════════════════


class FactoredAWQRotateLinearWithLoRA(nn.Module):
    """x_rot = (x / awq_s_vec) @ shared_R_<in>; y = linear(x_rot) + lora + bias."""

    def __init__(
        self,
        in_features,
        out_features,
        lora_rank,
        lora_scaling,
        group_size=64,
        levels=None,
        bias=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_scaling = float(lora_scaling)

        self.register_buffer(
            "awq_s_vec", torch.ones(in_features, dtype=(dtype or torch.float16), device=device), persistent=True
        )

        self._R_lookup = None  # set post-init; called fresh each forward

        self.linear = Quantized2BitLinear(
            in_features, out_features, group_size=group_size, levels=levels, device=device, dtype=dtype
        )
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.bias_param = None

        self.lora_A = nn.Linear(in_features, lora_rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(lora_rank, out_features, bias=False, device=device, dtype=dtype)

    def _attach_lookup(self, fn):
        self._R_lookup = fn

    def forward(self, x):
        if self._R_lookup is None:
            raise RuntimeError(f"R lookup not attached for in_features={self.in_features}.")
        R = self._R_lookup()
        x_rot = x / self.awq_s_vec.to(x.dtype)
        x_rot = x_rot @ R.to(x.dtype)
        y_main = self.linear(x_rot)
        y_delta = self.lora_B(self.lora_A(x)) * self.lora_scaling
        out = y_main + y_delta
        if self.bias_param is not None:
            out = out + self.bias_param
        return out

    @property
    def weight(self):
        return self.linear._materialize(dtype=torch.float32)

    @property
    def bias(self):
        return self.bias_param


# ════════════════════════════════════════════════════════════════════
# Top-level: Qwen2 + the swap + shared rotations
# ════════════════════════════════════════════════════════════════════


def _swap_quantized_linears_with_lora(model, lora_rank, lora_scaling, group_size, target_suffixes, levels):
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and _is_quantized_linear_name(name, target_suffixes):
            targets.append((name, mod))
    n = 0
    for name, mod in targets:
        wrapper = FactoredAWQRotateLinearWithLoRA(
            in_features=mod.in_features,
            out_features=mod.out_features,
            lora_rank=lora_rank,
            lora_scaling=lora_scaling,
            group_size=group_size,
            levels=levels,
            bias=mod.bias is not None,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, wrapper)
        n += 1
    return n


class Qwen2FactoredLoraForCausalLM(Qwen2ForCausalLM):
    """QwQ-32B (Qwen2) with quantized projections wrapped in
    FactoredAWQRotateLinearWithLoRA. Shared rotations live on this top-level
    object; each wrapper holds a callable that fetches the live buffer.

    Config additions:
        srht_awq_lora          : { rank, alpha, group_size, target_suffixes, levels }
        factored_rotation_dims : list of unique in_features (e.g. [5120, 27648])
    """

    def __init__(self, config):
        super().__init__(config)

        lora_cfg = getattr(config, "srht_awq_lora", None) or {}
        if isinstance(lora_cfg, dict):
            rank = int(lora_cfg.get("rank", 64))
            alpha = int(lora_cfg.get("alpha", 128))
            group_size = int(lora_cfg.get("group_size", 64))
            target_suffixes = tuple(lora_cfg.get("target_suffixes", DEFAULT_TARGET_SUFFIXES))
            levels = lora_cfg.get("levels", DEFAULT_LINEAR_LEVELS)
        else:
            rank, alpha, group_size = 64, 128, 64
            target_suffixes = DEFAULT_TARGET_SUFFIXES
            levels = DEFAULT_LINEAR_LEVELS
        scaling = (alpha / rank) if rank > 0 else 0.0

        rotation_dims = list(getattr(config, "factored_rotation_dims", []) or [5120, 27648])
        self._rotation_dims = rotation_dims
        for d in rotation_dims:
            self.register_buffer(f"shared_R_{d}", torch.zeros(d, d, dtype=torch.float16), persistent=True)

        n = _swap_quantized_linears_with_lora(
            self,
            lora_rank=rank,
            lora_scaling=scaling,
            group_size=group_size,
            target_suffixes=target_suffixes,
            levels=levels,
        )

        self._attach_shared_rotations()
        if n == 0:
            import logging

            logging.getLogger(__name__).warning("[Qwen2FactoredLoraForCausalLM] no quantized linears swapped.")

    def _attach_shared_rotations(self):
        parent = self
        for _, mod in self.named_modules():
            if isinstance(mod, FactoredAWQRotateLinearWithLoRA):
                d = mod.in_features
                attr = f"shared_R_{d}"
                if not hasattr(parent, attr):
                    raise RuntimeError(f"No {attr} buffer; available dims: {self._rotation_dims}")
                mod._attach_lookup(lambda _attr=attr: getattr(parent, _attr))
