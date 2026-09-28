# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Weight fake-quant for full-weight QAD: GPTQ-style groups along in_features.

Not full GPTQ (no Hessian); for each group of ``group_size`` consecutive input
columns we use max-abs scale and the same 4 symmetric levels as TwoBitScalar
({-1,-1/3,1/3,1} in normalized space). Straight-through estimator on gradients.

When ``learned_scales=True``, per-group scales become trainable parameters
(LSQ-style) instead of being derived from max(|w|) each forward pass. This
gives the optimizer a direct continuous lever on quantization quality.

When ``learned_rounding=True``, each weight element gets a learnable rounding
logit (FlexRound / TesseraQ-style). Instead of always snapping to the nearest
of the 4 levels, the rounding direction (up vs down) is learned via a sigmoid
parameterisation.  A temperature parameter controls soft-to-hard annealing.
At save time, rounding decisions are hardened and baked into the weights,
producing standard {-1,-1/3,1/3,1} * scale values for NPU deployment.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ---------------------------------------------------------------------------
# RTN (round-to-nearest) path — used by v1 / v2 and as default
# ---------------------------------------------------------------------------


class _STEWeightGroup2Bit(Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor, group_size: int) -> torch.Tensor:
        ctx.group_size = int(group_size)
        o, i = w.shape
        g = ctx.group_size
        pad = (g - (i % g)) % g
        if pad:
            w_pad = F.pad(w, (0, pad))
        else:
            w_pad = w
        i2 = w_pad.shape[1]
        x = w_pad.view(o, i2 // g, g)
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        u = (x / scale).clamp(-1.5, 1.5)
        u_abs = u.abs()
        u_q_abs = torch.where(
            u_abs < 2.0 / 3.0,
            torch.full_like(u_abs, 1.0 / 3.0),
            torch.full_like(u_abs, 1.0),
        )
        u_q = torch.sign(u) * u_q_abs
        dq = (u_q * scale).view(o, i2)
        if pad:
            dq = dq[:, :i]
        return dq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def fake_quant_weight_group_gptq_style(w: torch.Tensor, group_size: int) -> torch.Tensor:
    return _STEWeightGroup2Bit.apply(w, group_size)


# ---------------------------------------------------------------------------
# Learned-scale path — LSQ-style: scale is a trainable parameter
# ---------------------------------------------------------------------------


def _snap_to_levels(u: torch.Tensor) -> torch.Tensor:
    """Snap normalized values to {-1, -1/3, 1/3, 1}."""
    u_abs = u.abs()
    u_q_abs = torch.where(
        u_abs < 2.0 / 3.0,
        torch.full_like(u_abs, 1.0 / 3.0),
        torch.ones_like(u_abs),
    )
    return u.sign() * u_q_abs


class _STELearnedScale2Bit(Function):
    """Custom autograd: STE for weight, LSQ gradient for log_scale."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, log_scale: torch.Tensor, group_size: int) -> torch.Tensor:
        o, i = w.shape
        g = group_size
        n_groups = i // g
        scale = log_scale.exp().view(o, n_groups, 1)
        x = w.view(o, n_groups, g)
        u = (x / scale).clamp(-1.5, 1.5)
        u_q = _snap_to_levels(u)
        w_q = (u_q * scale).view(o, i)
        ctx.save_for_backward(u_q, scale)
        ctx.group_size = g
        return w_q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        u_q, scale = ctx.saved_tensors
        g = ctx.group_size
        o, i = grad_output.shape
        n_groups = i // g
        # STE for weight
        grad_w = grad_output
        # LSQ gradient for log_scale:
        # dL/d(log_scale) = dL/dw_q * dw_q/d(scale) * scale   [chain rule for exp]
        # dw_q/d(scale) = u_q  (level assignment treated as fixed)
        grad_grouped = grad_output.view(o, n_groups, g)
        grad_log_scale = (grad_grouped * u_q).sum(dim=-1) * scale.squeeze(-1)
        return grad_w, grad_log_scale.view(-1), None


def fake_quant_learned_scale(w: torch.Tensor, log_scale: torch.Tensor, group_size: int) -> torch.Tensor:
    return _STELearnedScale2Bit.apply(w, log_scale, group_size)


def bake_with_learned_scale(w: torch.Tensor, log_scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Apply fake-quant with learned scale (no autograd). Used at save time."""
    o, i = w.shape
    g = group_size
    n_groups = i // g
    scale = log_scale.exp().view(o, n_groups, 1)
    x = w.view(o, n_groups, g)
    u = (x / scale).clamp(-1.5, 1.5)
    u_q = _snap_to_levels(u)
    return (u_q * scale).view(o, i)


# ---------------------------------------------------------------------------
# Learned-rounding path — FlexRound / TesseraQ-style adaptive rounding
# ---------------------------------------------------------------------------
# The 4 levels are {-1, -1/3, 1/3, 1}.  For a given normalised value u,
# the two candidate levels (floor and ceil) are determined by which interval
# u falls into.  Instead of always snapping to the nearest, we learn a
# per-element sigmoid(logit/temp) that interpolates between floor and ceil.
#
# Intervals and their (floor, ceil) pairs:
#   u < -2/3   → (-1,   -1/3)
#   -2/3 ≤ u < 0  → (-1/3,  1/3)    [straddles zero]
#   0 ≤ u < 2/3   → (-1/3,  1/3)    [straddles zero — mapped via sign]
#   u ≥ 2/3   → (1/3,   1)
#
# Actually for symmetry we work with |u| and apply sign at the end:
#   |u| < 2/3  → floor_abs=1/3, ceil_abs=1/3   (same level — no rounding needed)
#
# Wait — with only 2 positive levels {1/3, 1}, a positive u can only snap to
# one of those.  The rounding decision is: snap to 1/3 or snap to 1?
# The boundary is at 2/3.  Learned rounding shifts this boundary per-element.

_L0, _L1, _L2, _L3 = -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0
_B1, _B2, _B3 = -2.0 / 3.0, 0.0, 2.0 / 3.0


def _floor_ceil_levels(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each element in u, find the (floor, ceil) level pair from {-1,-1/3,1/3,1}.

    Uses scalar-compare torch.where chains instead of bucketize+indexing to
    avoid materialising large int64 index tensors (saves ~3x element memory).
    """
    fl = torch.where(u < _B1, _L0, torch.where(u < _B2, _L1, torch.where(u < _B3, _L2, _L3)))
    cl = torch.where(u < _B1, _L1, torch.where(u < _B2, _L2, torch.where(u < _B3, _L3, _L3)))
    return fl, cl


def _init_rounding_logits(w: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Initialize rounding logits so sigmoid(logit) matches RTN (nearest level).

    For each weight element, compute the fractional position between floor and
    ceil levels, then invert through sigmoid: logit = log(frac / (1 - frac)).
    This means the initial rounding exactly reproduces round-to-nearest.
    """
    o, i = w.shape
    g = group_size
    n_groups = i // g
    s = scale.view(o, n_groups, 1)
    x = w.view(o, n_groups, g)
    u = (x / s).clamp(-1.5, 1.5)

    fl, cl = _floor_ceil_levels(u)
    span = (cl - fl).clamp(min=1e-8)
    frac = ((u - fl) / span).clamp(1e-4, 1.0 - 1e-4)
    logit = torch.log(frac / (1.0 - frac))
    return logit.view(o, i)


class _STELearnedRounding2Bit(Function):
    """Autograd: STE for weight, differentiable gradient for rounding logits and log_scale.

    Memory-optimised: computes u_q directly via fused torch.where without
    materialising separate fl/cl/span tensors.  Saves u (to recompute span
    in backward) instead of a full-size span tensor — same tensor count but
    avoids 2 extra peak intermediates during the forward pass.
    """

    @staticmethod
    def forward(
        ctx, w: torch.Tensor, log_scale: torch.Tensor, rounding_logit: torch.Tensor, group_size: int, temperature: float
    ) -> torch.Tensor:
        o, i = w.shape
        g = group_size
        n_groups = i // g
        scale = log_scale.exp().view(o, n_groups, 1)

        x = w.view(o, n_groups, g)
        u = (x / scale).clamp(-1.5, 1.5)

        v = torch.sigmoid(rounding_logit.view(o, n_groups, g) / max(temperature, 1e-6))

        # Fused u_q = fl + v * span without materialising fl, cl, span.
        # All inter-level spans are 2/3; the top-saturated zone has span 0.
        _s = 2.0 / 3.0
        u_q = torch.where(
            u < _B1,
            -1.0 + v * _s,
            torch.where(u < _B2, _L1 + v * _s, torch.where(u < _B3, _L2 + v * _s, torch.ones_like(u))),
        )

        w_q = (u_q * scale).view(o, i)

        ctx.save_for_backward(u_q, scale, v, u)
        ctx.group_size = g
        ctx.temperature = temperature
        return w_q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        u_q, scale, v, u = ctx.saved_tensors
        g = ctx.group_size
        temp = ctx.temperature
        o, i = grad_output.shape
        n_groups = i // g

        grad_w = grad_output
        grad_grouped = grad_output.view(o, n_groups, g)

        # Recompute span from u: 2/3 everywhere except top zone (u >= 2/3)
        span = torch.where(u < _B3, 2.0 / 3.0, 0.0)

        # dL/d(logit) = dL/dw_q * span * scale * v*(1-v) / temp
        dsig = v * (1.0 - v) / max(temp, 1e-6)
        grad_rounding = (grad_grouped * span * scale * dsig).view(o, i)
        del span, dsig

        # dL/d(log_scale) = sum_g(dL/dw_q * u_q) * scale
        grad_log_scale = (grad_grouped * u_q).sum(dim=-1) * scale.squeeze(-1)

        return grad_w, grad_log_scale.view(-1), grad_rounding, None, None


def fake_quant_learned_rounding(
    w: torch.Tensor, log_scale: torch.Tensor, rounding_logit: torch.Tensor, group_size: int, temperature: float = 1.0
) -> torch.Tensor:
    """Differentiable fake-quant with learned rounding and learned scale."""
    return _STELearnedRounding2Bit.apply(w, log_scale, rounding_logit, group_size, temperature)


def bake_with_learned_rounding(
    w: torch.Tensor, log_scale: torch.Tensor, rounding_logit: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Harden rounding decisions and bake into weights. Used at save time.

    Produces exact {-1, -1/3, 1/3, 1} * scale values for NPU deployment.
    """
    o, i = w.shape
    g = group_size
    n_groups = i // g
    scale = log_scale.exp().view(o, n_groups, 1)
    x = w.view(o, n_groups, g)
    u = (x / scale).clamp(-1.5, 1.5)

    fl, cl = _floor_ceil_levels(u)
    rl = rounding_logit.view(o, n_groups, g)
    v_hard = (rl > 0).float()  # hard binary decision
    u_q = fl + v_hard * (cl - fl)
    return (u_q * scale).view(o, i)


# ---------------------------------------------------------------------------
# Module wrapper
# ---------------------------------------------------------------------------


class Linear2BitGroupSTE(nn.Module):
    """nn.Linear with grouped 2-bit weight fake-quant (forward only).

    Supports three modes (all backward-compatible; default is RTN):

    1. **RTN** (default): max-abs scale, round-to-nearest.  Same as v1/v2.
    2. **Learned scales only** (``learned_scales=True``): per-group LSQ-style
       trainable scales, still round-to-nearest.
    3. **Learned rounding** (``learned_rounding=True``): per-element rounding
       logits (FlexRound-style) with temperature annealing, combined with
       learned scales.  Highest quality; produces exact {-1,-1/3,1/3,1}*scale
       weights when baked at save time.
    """

    def __init__(
        self, linear: nn.Linear, group_size: int, learned_scales: bool = False, learned_rounding: bool = False
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.learned_scales = learned_scales or learned_rounding
        self.learned_rounding = learned_rounding
        self._rounding_temperature = 1.0
        self.weight = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        if self.learned_scales:
            assert linear.in_features % group_size == 0, (
                f"learned_scales requires in_features ({linear.in_features}) divisible by group_size ({group_size})"
            )
            w = linear.weight.data.float()
            o, i = w.shape
            n_groups = i // group_size
            x = w.view(o, n_groups, group_size)
            init_scale = x.abs().amax(dim=-1).clamp(min=1e-8)  # [o, n_groups]
            self.log_scale = nn.Parameter(init_scale.log().to(linear.weight.dtype).reshape(-1))

        if self.learned_rounding:
            w = linear.weight.data.float()
            o, i = w.shape
            n_groups = i // group_size
            s = w.view(o, n_groups, group_size).abs().amax(dim=-1).clamp(min=1e-8)
            init_logit = _init_rounding_logits(w, s, group_size)
            self.rounding_logit = nn.Parameter(init_logit.to(linear.weight.dtype))

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, group_size: int, learned_scales: bool = False, learned_rounding: bool = False
    ) -> Linear2BitGroupSTE:
        return cls(linear, group_size, learned_scales=learned_scales, learned_rounding=learned_rounding)

    def set_rounding_temperature(self, temp: float) -> None:
        """Set the rounding temperature for soft-to-hard annealing."""
        self._rounding_temperature = temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learned_rounding:
            wq = fake_quant_learned_rounding(
                self.weight,
                self.log_scale,
                self.rounding_logit,
                self.group_size,
                self._rounding_temperature,
            )
        elif self.learned_scales:
            wq = fake_quant_learned_scale(self.weight, self.log_scale, self.group_size)
        else:
            wq = fake_quant_weight_group_gptq_style(self.weight, self.group_size)
        return F.linear(x, wq, self.bias)


def replace_linears_with_group_2bit_ste(
    root: nn.Module,
    group_size: int,
    exclude_globs: Iterable[str] = ("*embed_tokens*", "*lm_head*"),
    include_substrings: tuple[str, ...] | None = ("proj",),
    learned_scales: bool = False,
    learned_rounding: bool = False,
) -> list[str]:
    """Replace nn.Linear modules in-place. Returns list of replaced module names."""

    def _excluded(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in exclude_globs)

    def _included(name: str) -> bool:
        if include_substrings is None:
            return True
        return any(s in name for s in include_substrings)

    replaced: list[str] = []

    def _recurse(module: nn.Module, prefix: str) -> None:
        for child_name, child in list(module.named_children()):
            fq = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and not _excluded(fq) and _included(fq):
                setattr(
                    module,
                    child_name,
                    Linear2BitGroupSTE.from_linear(
                        child,
                        group_size,
                        learned_scales=learned_scales,
                        learned_rounding=learned_rounding,
                    ),
                )
                replaced.append(fq)
            else:
                _recurse(child, fq)

    _recurse(root, "")
    return replaced


# ---------------------------------------------------------------------------
# MoE expert fake-quant (gpt-oss) — packed 3D nn.Parameter experts
# ---------------------------------------------------------------------------
# gpt-oss stores its experts as 3D parameters inside ``GptOssExperts``:
#   gate_up_proj  [num_experts, hidden,       2*intermediate]   (applied as x @ W)
#   down_proj     [num_experts, intermediate, hidden]           (applied as x @ W)
# These are NOT nn.Linear, so the linear replacement above skips them. Since the
# matmul is ``x @ W`` (not ``x @ W.T``), the contraction / in_features axis is
# dim 0 of each per-expert 2D slice, so groups run along dim 0. Everything here
# is only used when the gpt-oss expert flag is enabled; the dense Linear path is
# completely untouched.


class _STEExpertWeight2Bit(Function):
    """RTN grouped 2-bit fake-quant for a 2D expert weight [in, out], grouped along ``in`` (dim 0)."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, group_size: int) -> torch.Tensor:
        g = int(group_size)
        i, o = w.shape
        pad = (g - (i % g)) % g
        w_pad = F.pad(w, (0, 0, 0, pad)) if pad else w
        i2 = w_pad.shape[0]
        x = w_pad.view(i2 // g, g, o)
        scale = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        u = (x / scale).clamp(-1.5, 1.5)
        u_q = _snap_to_levels(u)
        dq = (u_q * scale).view(i2, o)
        if pad:
            dq = dq[:i]
        return dq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _STEExpertLearnedScale2Bit(Function):
    """LSQ-style learned-scale 2-bit fake-quant for a 2D expert weight [in, out], grouped along dim 0.

    ``log_scale`` has shape [in // group_size, out]. Requires ``in`` divisible by group_size.
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor, log_scale: torch.Tensor, group_size: int) -> torch.Tensor:
        g = group_size
        i, o = w.shape
        n = i // g
        scale = log_scale.exp().view(n, 1, o)
        x = w.view(n, g, o)
        u = (x / scale).clamp(-1.5, 1.5)
        u_q = _snap_to_levels(u)
        w_q = (u_q * scale).view(i, o)
        ctx.save_for_backward(u_q, scale)
        ctx.group_size = g
        return w_q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        u_q, scale = ctx.saved_tensors
        g = ctx.group_size
        i, o = grad_output.shape
        n = i // g
        grad_w = grad_output
        grad_grouped = grad_output.view(n, g, o)
        grad_log_scale = (grad_grouped * u_q).sum(dim=1) * scale.squeeze(1)
        return grad_w, grad_log_scale.view(n, o), None


def _init_expert_log_scale(w: torch.Tensor, group_size: int) -> torch.Tensor:
    """Init per-(group, out) log-scales for a 3D expert tensor [E, in, out] (group along ``in``)."""
    e, i, o = w.shape
    assert i % group_size == 0, (
        f"learned_scales requires expert in_features ({i}) divisible by group_size ({group_size})"
    )
    n = i // group_size
    x = w.float().view(e, n, group_size, o)
    s = x.abs().amax(dim=2).clamp(min=1e-8)  # [E, n, out]
    return s.log().to(w.dtype)


def _bake_expert_3d(w: torch.Tensor, log_scale: torch.Tensor | None, group_size: int) -> torch.Tensor:
    """Bake fake-quant into a 3D expert tensor [E, in, out] (no autograd). Used at save time."""
    e, i, o = w.shape
    g = group_size
    pad = (g - (i % g)) % g
    w_pad = F.pad(w, (0, 0, 0, pad)) if pad else w
    i2 = w_pad.shape[1]
    n = i2 // g
    x = w_pad.view(e, n, g, o)
    if log_scale is not None:
        scale = log_scale.exp().view(e, n, 1, o)
    else:
        scale = x.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    u = (x / scale).clamp(-1.5, 1.5)
    u_q = _snap_to_levels(u)
    dq = (u_q * scale).reshape(e, i2, o)
    if pad:
        dq = dq[:, :i, :]
    return dq


class Experts2BitGroupSTE(nn.Module):
    """gpt-oss ``GptOssExperts`` with grouped 2-bit weight fake-quant on the 3D expert tensors.

    Reproduces the upstream expert forward exactly, applying STE fake-quant to
    ``gate_up_proj`` and ``down_proj`` per hit expert. Biases, router and gating
    are untouched. Supports RTN and learned per-group scales (no learned rounding
    for experts — ``learned_rounding`` is treated as learned scales here).
    """

    def __init__(self, experts: nn.Module, group_size: int, learned_scales: bool = False) -> None:
        super().__init__()
        self.group_size = group_size
        self.learned_scales = learned_scales
        self.num_experts = int(experts.num_experts)
        self.intermediate_size = int(experts.intermediate_size)
        self.hidden_size = int(experts.hidden_size)
        self.alpha = float(getattr(experts, "alpha", 1.702))
        self.limit = float(getattr(experts, "limit", 7.0))

        self.gate_up_proj = nn.Parameter(experts.gate_up_proj.data.clone())
        self.gate_up_proj_bias = nn.Parameter(experts.gate_up_proj_bias.data.clone())
        self.down_proj = nn.Parameter(experts.down_proj.data.clone())
        self.down_proj_bias = nn.Parameter(experts.down_proj_bias.data.clone())

        if learned_scales:
            self.gate_up_log_scale = nn.Parameter(_init_expert_log_scale(self.gate_up_proj.data, group_size))
            self.down_log_scale = nn.Parameter(_init_expert_log_scale(self.down_proj.data, group_size))

    @classmethod
    def from_experts(cls, experts: nn.Module, group_size: int, learned_scales: bool = False) -> Experts2BitGroupSTE:
        return cls(experts, group_size, learned_scales=learned_scales)

    def _q_gate_up(self, e: torch.Tensor) -> torch.Tensor:
        if self.learned_scales:
            return _STEExpertLearnedScale2Bit.apply(self.gate_up_proj[e], self.gate_up_log_scale[e], self.group_size)
        return _STEExpertWeight2Bit.apply(self.gate_up_proj[e], self.group_size)

    def _q_down(self, e: torch.Tensor) -> torch.Tensor:
        if self.learned_scales:
            return _STEExpertLearnedScale2Bit.apply(self.down_proj[e], self.down_log_scale[e], self.group_size)
        return _STEExpertWeight2Bit.apply(self.down_proj[e], self.group_size)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        return (up + 1) * glu

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        with torch.no_grad():
            expert_mask = F.one_hot(router_indices, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = current_state @ self._q_gate_up(expert_idx) + self.gate_up_proj_bias[expert_idx]
            gated_output = self._apply_gate(gate_up)
            out = gated_output @ self._q_down(expert_idx) + self.down_proj_bias[expert_idx]
            weighted_output = out * routing_weights[token_idx, top_k_pos, None]
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
        return next_states


def _looks_like_gptoss_experts(m: nn.Module) -> bool:
    return (
        m.__class__.__name__ == "GptOssExperts"
        and isinstance(getattr(m, "gate_up_proj", None), nn.Parameter)
        and isinstance(getattr(m, "down_proj", None), nn.Parameter)
        and m.gate_up_proj.dim() == 3
    )


def replace_gptoss_experts_with_2bit_ste(
    root: nn.Module,
    group_size: int,
    learned_scales: bool = False,
) -> list[str]:
    """Replace gpt-oss ``GptOssExperts`` modules in-place with 2-bit STE versions.

    Returns the list of replaced module names. No-op on models without gpt-oss
    experts (so it is safe to call unconditionally behind the flag).
    """
    replaced: list[str] = []

    def _recurse(module: nn.Module, prefix: str) -> None:
        for child_name, child in list(module.named_children()):
            fq = f"{prefix}.{child_name}" if prefix else child_name
            if _looks_like_gptoss_experts(child):
                setattr(
                    module,
                    child_name,
                    Experts2BitGroupSTE.from_experts(child, group_size, learned_scales=learned_scales),
                )
                replaced.append(fq)
            else:
                _recurse(child, fq)

    _recurse(root, "")
    return replaced


def bake_experts_2bit_ste(root: nn.Module) -> None:
    """Bake fake-quant into the 3D expert weights of every ``Experts2BitGroupSTE``.

    Hardens gate_up_proj/down_proj to their fake-quantized values and removes the
    learned-scale params so the saved state_dict matches a standard gpt-oss model.
    """
    for m in root.modules():
        if isinstance(m, Experts2BitGroupSTE):
            with torch.no_grad():
                gate_up_ls = getattr(m, "gate_up_log_scale", None) if m.learned_scales else None
                down_ls = getattr(m, "down_log_scale", None) if m.learned_scales else None
                m.gate_up_proj.data.copy_(
                    _bake_expert_3d(
                        m.gate_up_proj.data, gate_up_ls.data if gate_up_ls is not None else None, m.group_size
                    )
                )
                m.down_proj.data.copy_(
                    _bake_expert_3d(m.down_proj.data, down_ls.data if down_ls is not None else None, m.group_size)
                )
            if m.learned_scales:
                if hasattr(m, "gate_up_log_scale"):
                    del m.gate_up_log_scale
                if hasattr(m, "down_log_scale"):
                    del m.down_log_scale
                m.learned_scales = False


def strip_group_2bit_ste_to_linear(root: nn.Module) -> None:
    """Replace Linear2BitGroupSTE with plain nn.Linear, baking the 2-bit fake-quantized weights.

    For learned_rounding modules, rounding decisions are hardened (sigmoid > 0.5
    → round up, else round down) so the saved weights are exact
    {-1, -1/3, 1/3, 1} * scale values suitable for NPU deployment.
    """

    def _recurse(module: nn.Module, prefix: str) -> None:
        for child_name, child in list(module.named_children()):
            if isinstance(child, Linear2BitGroupSTE):
                lin = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
                lin = lin.to(device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    if child.learned_rounding:
                        lin.weight.copy_(
                            bake_with_learned_rounding(
                                child.weight,
                                child.log_scale,
                                child.rounding_logit,
                                child.group_size,
                            )
                        )
                    elif child.learned_scales:
                        lin.weight.copy_(bake_with_learned_scale(child.weight, child.log_scale, child.group_size))
                    else:
                        lin.weight.copy_(fake_quant_weight_group_gptq_style(child.weight, child.group_size))
                    if child.bias is not None and lin.bias is not None:
                        lin.bias.copy_(child.bias)
                setattr(module, child_name, lin)
            else:
                _recurse(child, f"{prefix}.{child_name}" if prefix else child_name)

    _recurse(root, "")
