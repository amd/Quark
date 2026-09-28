# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Self-loading Phi-4 (phi3) STRUCTURED SRHT+AWQ+LoRA variant whose 2-bit weight
half is dequantized through **Quark's own** packing + dequant kernels instead of
hand-rolled helpers:

  - `linear.packed_levels`      uint8 [out, in/4]   packed by `quark...Pack_uint2`
  - `linear.weight_scale`       fp16  [out, in/64]   = (2/3) * per-group scale
  - `linear.weight_zero_point`  fp16  [out, in/64]   = 1.5   (float zero-point)

The Lloyd-Max levels {-1,-1/3,1/3,1} are *uniform* (step 2/3), i.e. an affine
quant with zero_point=1.5 and scale=(2/3)*group_scale, so the exact 4-level
codebook is reproduced by Quark's affine dequant:
    W = (uint2_idx - 1.5) * (2/3 * group_scale)   == levels[idx] * group_scale

The SRHT/AWQ `pre_linear`, the shared/structured rotation, and the LoRA branch
remain custom (Quark's native format cannot express those — see PR notes).

Self-loads via: AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

from quark.torch.kernel import dequantize
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.utils.pack import Pack_uint2

DEFAULT_TARGET_SUFFIXES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
# Affine encoding of the Lloyd-Max 4-level codebook {-1,-1/3,1/3,1}.
LEVEL_ZERO_POINT = 1.5
LEVEL_SCALE = 2.0 / 3.0


def _is_quantized_linear_name(name: str, suffixes: tuple[str, ...]) -> bool:
    return any(name.endswith(s) for s in suffixes) and "embed" not in name and "lm_head" not in name


def _get_parent_and_attr(model: nn.Module, qualname: str) -> tuple[nn.Module, str]:
    parts = qualname.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _largest_pow2_factor(n: int) -> int:
    return n & (-n) if n else 0


def srht_block(in_dim: int) -> int:
    return min(_largest_pow2_factor(in_dim), 1024)


def _block_fwht(x: torch.Tensor, blk: int) -> torch.Tensor:
    *lead, n = x.shape
    nblk = n // blk
    x = x.reshape(*lead, nblk, blk)
    h = 1
    while h < blk:
        x = x.reshape(*lead, nblk, blk // (2 * h), 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(*lead, nblk, blk)
        h *= 2
    x = x.reshape(*lead, n)
    return x * (1.0 / math.sqrt(blk))


# ════════════════════════════════════════════════════════════════════
# 2-bit quantized linear — dequant via Quark's Pack_uint2 + dequantize
# ════════════════════════════════════════════════════════════════════


class Quantized2BitLinearQuark(nn.Module):
    def __init__(self, in_features: int, out_features: int, group_size: int = 64, device=None, dtype=None):
        super().__init__()
        if in_features % group_size != 0 or in_features % 4 != 0:
            raise ValueError(f"bad in_features {in_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        n_groups = in_features // group_size
        # Quark's uint2 packer (no per-group transpose → packs the `in` axis).
        self._packer = Pack_uint2(qscheme=None, dtype="uint2")
        self.register_buffer(
            "packed_levels",
            torch.zeros(out_features, in_features // 4, dtype=torch.uint8, device=device),
            persistent=True,
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros(out_features, n_groups, dtype=(dtype or torch.float16), device=device),
            persistent=True,
        )
        self.register_buffer(
            "weight_zero_point",
            torch.full((out_features, n_groups), LEVEL_ZERO_POINT, dtype=(dtype or torch.float16), device=device),
            persistent=True,
        )

    def _materialize(self, dtype: torch.dtype) -> torch.Tensor:
        # qscheme=None → no transpose; in_features % 4 == 0 so unpack yields [out, in] exactly.
        idx = self._packer.unpack(self.packed_levels)
        W = dequantize(
            Dtype.uint2.value,
            idx.to(torch.float32),
            self.weight_scale.to(torch.float32),
            self.weight_zero_point.to(torch.float32),
            1,
            self.group_size,
            QSchemeType.per_group.value,
        )
        return W.to(dtype).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self._materialize(dtype=x.dtype))

    @property
    def weight(self):
        return self._materialize(dtype=torch.float32)


class StructuredSRHTAWQ(nn.Module):
    def __init__(self, in_features: int, dtype=torch.float16, device=None):
        super().__init__()
        self.in_features = in_features
        self.blk = srht_block(in_features)
        self.register_buffer("awq_s_vec", torch.ones(in_features, dtype=dtype, device=device), persistent=True)
        self.register_buffer("srht_perm", torch.arange(in_features, dtype=torch.int16, device=device), persistent=True)
        self.register_buffer("srht_sign", torch.ones(in_features, dtype=torch.int8, device=device), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.awq_s_vec.to(x.dtype)
        x = torch.index_select(x, -1, self.srht_perm.long())
        x = x * self.srht_sign.to(x.dtype)
        return _block_fwht(x, self.blk)


class SRHTAWQStructuredLinearWithLoRA(nn.Module):
    def __init__(
        self, in_features, out_features, lora_rank, lora_scaling, group_size=64, bias=False, device=None, dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_scaling = float(lora_scaling)
        self.pre_linear = StructuredSRHTAWQ(in_features, dtype=dtype or torch.float16, device=device)
        self.linear = Quantized2BitLinearQuark(
            in_features, out_features, group_size=group_size, device=device, dtype=dtype
        )
        self.bias_param = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) if bias else None
        self.lora_A = nn.Linear(in_features, lora_rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(lora_rank, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(self.pre_linear(x))
        y = y + self.lora_B(self.lora_A(x)) * self.lora_scaling
        if self.bias_param is not None:
            y = y + self.bias_param
        return y

    @property
    def weight(self):
        return self.linear._materialize(dtype=torch.float32)

    @property
    def bias(self):
        return self.bias_param


def _swap(model, rank, scaling, group_size, suffixes):
    targets = [
        (n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and _is_quantized_linear_name(n, suffixes)
    ]
    for name, mod in targets:
        w = SRHTAWQStructuredLinearWithLoRA(
            mod.in_features,
            mod.out_features,
            rank,
            scaling,
            group_size,
            bias=mod.bias is not None,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, w)
    return len(targets)


class Phi3StructuredQuarkForCausalLM(Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        lc = getattr(config, "srht_awq_lora", None) or {}
        rank = int(lc.get("rank", 64))
        alpha = int(lc.get("alpha", 128))
        group_size = int(lc.get("group_size", 64))
        suffixes = tuple(lc.get("target_suffixes", DEFAULT_TARGET_SUFFIXES))
        scaling = (alpha / rank) if rank > 0 else 0.0
        n = _swap(self, rank, scaling, group_size, suffixes)
        if n == 0:
            import logging

            logging.getLogger(__name__).warning("[Phi3StructuredQuarkForCausalLM] no linears swapped.")
