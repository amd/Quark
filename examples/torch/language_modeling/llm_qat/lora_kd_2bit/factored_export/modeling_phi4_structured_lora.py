# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Self-loading Phi-4 (phi3 architecture) variant with STRUCTURED SRHT + AWQ-undo
(no dense `pre_linear` matrix).

This is a drop-in replacement for `modeling_phi4_dense_lora.py` that produces
identical math but a ~6x smaller checkpoint / ONNX file. For each of the 160
quantized projections, the dense `nn.Linear[in, in]` `pre_linear` is replaced
with three small structured buffers and a block-Hadamard transform that lowers
to stock ONNX ops only (Gather + Mul + Reshape + Add + Sub).

Per quantized projection:
  pre_linear.awq_s_vec   fp16  [in]            AWQ-undo scaling
  pre_linear.srht_perm   int16 [in]            SRHT permutation indices (max in=17920 < 32768)
  pre_linear.srht_sign   int8  [in]            SRHT ±1 sign flips
  linear.packed_levels   uint8 [out, in/4]     2-bit packed (unchanged)
  linear.group_scale     fp16  [out, in/64]    per-row × per-group scale (unchanged)
  lora_A                 fp16 [r=64, in]       (unchanged)
  lora_B                 fp16 [out, r=64]      (unchanged)

The block size for each projection matches TwoBitScalar's `_global_rotate`:
    blk = min(largest_pow2_factor(in), 1024)
e.g. in=5120  -> blk=1024 (5 blocks of size 1024, 10 butterfly stages)
     in=17920 -> blk=512  (35 blocks of size 512,  9 butterfly stages)

Forward (per projection):
    x_rot   = (x / awq_s_vec)[..., srht_perm] * srht_sign
    x_rot   = block_fwht(x_rot, blk) / sqrt(blk)
    y_main  = linear(x_rot)                               # 2-bit matmul
    y_delta = lora_B(lora_A(x)) * lora_scaling            # LoRA on raw x
    out     = y_main + y_delta

Self-loads via:
    AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)

config.json must contain:
    "architectures": ["Phi3StructuredLoraForCausalLM"]
    "auto_map": {"AutoModelForCausalLM":
                 "modeling_phi4_structured_lora.Phi3StructuredLoraForCausalLM"}
    "srht_awq_lora": {"rank": 64, "alpha": 128, "group_size": 64,
                      "target_suffixes": ["qkv_proj","o_proj","gate_up_proj","down_proj"]}
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

# ════════════════════════════════════════════════════════════════════
# Constants and small helpers
# ════════════════════════════════════════════════════════════════════

DEFAULT_LINEAR_LEVELS: list[float] = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]
DEFAULT_TARGET_SUFFIXES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


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
    """Block size for the block-Hadamard transform: matches TwoBitScalar.
    blk = min(largest_pow2_factor(in_dim), 1024).
    """
    return min(_largest_pow2_factor(in_dim), 1024)


# ════════════════════════════════════════════════════════════════════
# Block Hadamard (butterfly), stock ONNX ops
# ════════════════════════════════════════════════════════════════════


def _block_fwht(x: torch.Tensor, blk: int) -> torch.Tensor:
    """In-place-style block Hadamard along the last dim, in blocks of size `blk`.

    `blk` must be a power of two and divide x.shape[-1]. Output is normalized
    by 1/sqrt(blk) to match TwoBitScalar's `_global_rotate` convention.

    Lowers cleanly to ONNX: Reshape + (Add, Sub, Concat) per stage. log2(blk)
    stages total. No initializer is produced for the Hadamard matrix.
    """
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
# 2-bit packed Linear (4 levels per group, packed int2) -- unchanged
# ════════════════════════════════════════════════════════════════════


def pack_int2(idx: torch.Tensor) -> torch.Tensor:
    assert idx.shape[-1] % 4 == 0, f"in_dim {idx.shape[-1]} must be divisible by 4"
    idx = idx.to(torch.uint8) & 0x3
    a = idx[..., 0::4]
    b = idx[..., 1::4] << 2
    c = idx[..., 2::4] << 4
    d = idx[..., 3::4] << 6
    return (a | b | c | d).contiguous()


def unpack_int2(packed: torch.Tensor, in_dim: int) -> torch.Tensor:
    p = packed.to(torch.uint8)
    leading = p.shape[:-1]
    out = torch.empty(*leading, in_dim, dtype=torch.long, device=p.device)
    out[..., 0::4] = (p & 0x3).long()
    out[..., 1::4] = ((p >> 2) & 0x3).long()
    out[..., 2::4] = ((p >> 4) & 0x3).long()
    out[..., 3::4] = ((p >> 6) & 0x3).long()
    return out


class Quantized2BitLinear(nn.Module):
    """2-bit weight Linear (4 levels per group). Identical math/state_dict keys
    to the existing dense-export modeling file."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 64,
        levels: list[float] | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError(f"in_features {in_features} % group_size {group_size} != 0")
        if in_features % 4 != 0:
            raise ValueError(f"in_features {in_features} must be divisible by 4")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        n_groups = in_features // group_size

        if levels is None:
            levels = DEFAULT_LINEAR_LEVELS
        if len(levels) != 4:
            raise ValueError(f"`levels` must have 4 entries, got {len(levels)}")
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

    def _materialize(self, dtype: torch.dtype) -> torch.Tensor:
        idx = unpack_int2(self.packed_levels, self.in_features)
        levels = torch.tensor(self._levels, device=idx.device, dtype=dtype)
        w_q = levels[idx]
        g = self.group_scale.to(dtype).repeat_interleave(self.group_size, dim=1)
        return (w_q * g).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._materialize(dtype=x.dtype)
        return F.linear(x, W)

    @property
    def weight(self):
        return self._materialize(dtype=torch.float32)


# ════════════════════════════════════════════════════════════════════
# Structured AWQ-undo + SRHT (replaces dense pre_linear)
# ════════════════════════════════════════════════════════════════════


class StructuredSRHTAWQ(nn.Module):
    """Computes  x_rot = block_fwht((x / awq_s_vec)[..., perm] * sign, blk) / sqrt(blk).

    Drop-in replacement for `nn.Linear(in, in)` whose .weight is
    `(rotate_last_dim(diag(1/awq_s), perm, signs, blk)).T`. Stores three small
    vectors instead of an [in,in] matrix. ~6000x smaller for in=17920.
    """

    def __init__(self, in_features: int, dtype=torch.float16, device=None):
        super().__init__()
        self.in_features = in_features
        self.blk = srht_block(in_features)
        if in_features % self.blk != 0:
            # Sanity: blk = largest_pow2_factor(in) divides in by construction.
            raise ValueError(f"in_features {in_features} not divisible by blk {self.blk}")
        # srht_perm uses int16 -- valid as long as in_features <= 32767 (max signed int16).
        # Phi-4 max in=17920; QwQ-32B max in=27648; both fit. Add the guard for safety.
        if in_features > 32767:
            raise ValueError(
                f"in_features {in_features} > 32767; srht_perm int16 would overflow. "
                f"Use a wider dtype if this is needed."
            )

        self.register_buffer(
            "awq_s_vec",
            torch.ones(in_features, dtype=dtype, device=device),
            persistent=True,
        )
        self.register_buffer(
            "srht_perm",
            torch.arange(in_features, dtype=torch.int16, device=device),
            persistent=True,
        )
        self.register_buffer(
            "srht_sign",
            torch.ones(in_features, dtype=torch.int8, device=device),
            persistent=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in]. fp16 in deployment.
        x = x / self.awq_s_vec.to(x.dtype)  # Div
        x = torch.index_select(x, -1, self.srht_perm.long())  # Gather along last dim
        x = x * self.srht_sign.to(x.dtype)  # Mul
        x = _block_fwht(x, self.blk)  # log2(blk) butterfly + /sqrt(blk)
        return x

    @property
    def weight(self):
        """Materialise the equivalent dense `[in, in]` weight (for inspection /
        parity verification only — not used in forward)."""
        in_dim = self.in_features
        dev = self.awq_s_vec.device
        eye = torch.eye(in_dim, device=dev, dtype=torch.float32)
        eye = eye / self.awq_s_vec.to(torch.float32)  # column-scale by 1/awq_s
        eye = eye[:, self.srht_perm.long()]  # permute columns
        eye = eye * self.srht_sign.to(torch.float32)  # column sign-flip
        eye = _block_fwht(eye.contiguous(), self.blk)  # block Hadamard
        return eye.t().contiguous()  # nn.Linear convention: forward(x) = x @ weight.T


# ════════════════════════════════════════════════════════════════════
# Wrapper: structured pre_linear + Quantized2BitLinear + LoRA
# ════════════════════════════════════════════════════════════════════


class SRHTAWQStructuredLinearWithLoRA(nn.Module):
    """Same top-level wiring as `SRHTAWQDenseLinearWithLoRA` in the dense file,
    but with `pre_linear` being a `StructuredSRHTAWQ` instead of `nn.Linear`.
    LoRA branch and the 2-bit `linear` are byte-identical."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int,
        lora_scaling: float,
        group_size: int = 64,
        levels: list[float] | None = None,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_scaling = float(lora_scaling)

        self.pre_linear = StructuredSRHTAWQ(in_features, dtype=dtype or torch.float16, device=device)
        self.linear = Quantized2BitLinear(
            in_features,
            out_features,
            group_size=group_size,
            levels=levels,
            device=device,
            dtype=dtype,
        )
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.bias_param = None

        self.lora_A = nn.Linear(in_features, lora_rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(lora_rank, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_main = self.linear(self.pre_linear(x))
        y_delta = self.lora_B(self.lora_A(x)) * self.lora_scaling
        out = y_main + y_delta
        if self.bias_param is not None:
            out = out + self.bias_param
        return out

    @property
    def weight(self) -> torch.Tensor:
        return self.linear._materialize(dtype=torch.float32)

    @property
    def bias(self):
        return self.bias_param


# ════════════════════════════════════════════════════════════════════
# Top-level model: Phi3 + the swap
# ════════════════════════════════════════════════════════════════════


def _swap_quantized_linears_with_lora(
    model: nn.Module,
    lora_rank: int,
    lora_scaling: float,
    group_size: int,
    target_suffixes: tuple[str, ...],
    levels: list[float] | None = None,
) -> int:
    targets: list[tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and _is_quantized_linear_name(name, target_suffixes):
            targets.append((name, mod))
    n = 0
    for name, mod in targets:
        wrapper = SRHTAWQStructuredLinearWithLoRA(
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


class Phi3StructuredLoraForCausalLM(Phi3ForCausalLM):
    """Phi-4/Phi-3 with quantized projections replaced by
    `SRHTAWQStructuredLinearWithLoRA`. Reads LoRA rank/alpha and group_size
    from `config.srht_awq_lora`."""

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
        n = _swap_quantized_linears_with_lora(
            self,
            lora_rank=rank,
            lora_scaling=scaling,
            group_size=group_size,
            target_suffixes=target_suffixes,
            levels=levels,
        )
        if n == 0:
            import logging

            logging.getLogger(__name__).warning(
                "[Phi3StructuredLoraForCausalLM] no quantized linears swapped — check architecture."
            )
