# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Self-loading Phi-4 (phi3) FACTORED (shared-rotation) + LoRA variant whose 2-bit
weight half dequantizes through **Quark's own** Pack_uint2 + dequantize kernels.

Same as `modeling_phi4_factored_lora.Phi3FactoredLoraForCausalLM` (one shared
rotation `shared_R_<in>` per unique in_features, per-projection `awq_s_vec`,
separate LoRA), except the quantized `linear` stores:

  linear.packed_levels      uint8 [out, in/4]   packed by quark...Pack_uint2
  linear.weight_scale       fp16  [out, in/64]   = (2/3) * per-group scale
  linear.weight_zero_point  fp16  [out, in/64]   = 1.5   (float zero-point)

so W = (uint2_idx - 1.5) * (2/3 * group_scale) == levels[idx] * group_scale.

Self-loads via: AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

from quark.torch.kernel import dequantize
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.utils.pack import Pack_uint2

DEFAULT_TARGET_SUFFIXES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
LEVEL_ZERO_POINT = 1.5
LEVEL_SCALE = 2.0 / 3.0


def _is_quantized_linear_name(name, suffixes):
    return any(name.endswith(s) for s in suffixes) and "embed" not in name and "lm_head" not in name


def _get_parent_and_attr(model, qualname):
    parts = qualname.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


class Quantized2BitLinearQuark(nn.Module):
    def __init__(self, in_features, out_features, group_size=64, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        n_groups = in_features // group_size
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

    def _materialize(self, dtype):
        idx = self._packer.unpack(self.packed_levels)  # [out, in]
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

    def forward(self, x):
        return torch.nn.functional.linear(x, self._materialize(dtype=x.dtype))

    @property
    def weight(self):
        return self._materialize(dtype=torch.float32)


class FactoredAWQRotateLinearWithLoRA(nn.Module):
    """Shared-rotation wrapper: x_rot = (x / awq_s_vec) @ R; y = linear(x_rot) + LoRA."""

    def __init__(
        self, in_features, out_features, lora_rank, lora_scaling, group_size=64, bias=False, device=None, dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_scaling = float(lora_scaling)
        self.register_buffer(
            "awq_s_vec", torch.ones(in_features, dtype=(dtype or torch.float16), device=device), persistent=True
        )
        self._R_lookup = None
        self.linear = Quantized2BitLinearQuark(
            in_features, out_features, group_size=group_size, device=device, dtype=dtype
        )
        self.bias_param = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) if bias else None
        self.lora_A = nn.Linear(in_features, lora_rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(lora_rank, out_features, bias=False, device=device, dtype=dtype)

    def _attach_lookup(self, fn):
        self._R_lookup = fn

    def forward(self, x):
        if self._R_lookup is None:
            raise RuntimeError(f"R lookup not attached for in_features={self.in_features}.")
        R = self._R_lookup()
        x_rot = (x / self.awq_s_vec.to(x.dtype)) @ R.to(x.dtype)
        y = self.linear(x_rot) + self.lora_B(self.lora_A(x)) * self.lora_scaling
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
        w = FactoredAWQRotateLinearWithLoRA(
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


class Phi3FactoredQuarkForCausalLM(Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        lc = getattr(config, "srht_awq_lora", None) or {}
        rank = int(lc.get("rank", 64))
        alpha = int(lc.get("alpha", 128))
        group_size = int(lc.get("group_size", 64))
        suffixes = tuple(lc.get("target_suffixes", DEFAULT_TARGET_SUFFIXES))
        scaling = (alpha / rank) if rank > 0 else 0.0
        rotation_dims = list(getattr(config, "factored_rotation_dims", []) or [5120, 17920])
        self._rotation_dims = rotation_dims
        for d in rotation_dims:
            self.register_buffer(f"shared_R_{d}", torch.zeros(d, d, dtype=torch.float16), persistent=True)
        n = _swap(self, rank, scaling, group_size, suffixes)
        self._attach_shared_rotations()
        if n == 0:
            import logging

            logging.getLogger(__name__).warning("[Phi3FactoredQuarkForCausalLM] no linears swapped.")

    def _attach_shared_rotations(self):
        parent = self
        for _, mod in self.named_modules():
            if isinstance(mod, FactoredAWQRotateLinearWithLoRA):
                attr = f"shared_R_{mod.in_features}"
                if not hasattr(parent, attr):
                    raise RuntimeError(f"No {attr} buffer; dims: {self._rotation_dims}")
                mod._attach_lookup(lambda _attr=attr: getattr(parent, _attr))
