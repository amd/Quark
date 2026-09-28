#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""AutoRound wrapper classes: learnable per-weight rounding offset around a quantized linear.

Reads scale/zero_point/quant_min/quant_max/group_size from the layer's Quark ``weight_quantizer``,
same as GPTQ.

``WrapperLinearInt`` (INT4 asymmetric) and ``WrapperLinearMXFP4`` (OCP microscaling FP4) share a
common lifecycle (:class:`_WrapperLinearBase`) but need materially different quantize expressions,
so each owns its own ``_quantize_weight``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from quark.experimental.torch.autoround.mxfp_quant import mxfp4_fake_quantize, mxfp4_scale, round_ste
from quark.torch.kernel.hw_emulation.hw_emulation_interface import fake_quantize_per_channel_affine
from quark.torch.quantization.config.type import Dtype


def attach_int4_fakequant(linear: nn.Module, group_size: int, bits: int = 4) -> nn.Module:
    """TEST-ONLY helper: attach a minimal per-group asymmetric INT fake-quant to `linear`.

    Computes scale/zero_point per output row per group from the linear's current weight,
    stores them as buffers, and replaces `linear.forward` so that it fake-quantizes the
    weight with the SAME math WrapperLinearInt uses at V=0 (round/clamp/dequant), so the two
    are numerically consistent in tests. This is not used in the real processor path,
    which instead relies on `linear.weight_quantizer` (a Quark `ScaledFakeQuantize`).
    """
    weight = linear.weight.data
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, "in_features must be divisible by group_size"
    n_groups = in_features // group_size

    qmax = (2**bits) - 1
    qmin = 0

    w_grouped = weight.reshape(out_features, n_groups, group_size)
    w_min = w_grouped.min(dim=-1, keepdim=True).values
    w_max = w_grouped.max(dim=-1, keepdim=True).values

    scale = (w_max - w_min).clamp(min=1e-8) / qmax
    zero_point = torch.round(-w_min / scale)
    zero_point = zero_point.clamp(qmin, qmax)

    linear.register_buffer("ar_scale", scale)
    linear.register_buffer("ar_zp", zero_point)
    linear.ar_group_size = group_size
    linear.ar_qmax = qmax

    def _forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        out_f, in_f = w.shape
        ng = in_f // self.ar_group_size
        wg = w.reshape(out_f, ng, self.ar_group_size)
        s = self.ar_scale
        zp = self.ar_zp
        w_int = torch.clamp(round_ste(wg / s + zp), 0, self.ar_qmax)
        w_deq = (w_int - zp) * s
        w_deq = w_deq.reshape(out_f, in_f)
        return F.linear(x, w_deq, self.bias)

    linear.forward = _forward.__get__(linear, type(linear))
    return linear


class _WrapperLinearBase(nn.Module):
    """Format-independent lifecycle shared by every AutoRound wrapper: owns the learnable
    rounding offset ``V`` and implements the wrap/freeze contract
    (:class:`AutoRoundProcessor` calls ``_quantize_weight()`` every forward and
    ``get_scale_zero_point()`` once at freeze). Subclasses provide the format-specific quantize
    expression and clip-tuning parameter(s).
    """

    # min_scale/max_scale may only shrink the clip range, never expand/invert it (matches the
    # official repo's WrapperLinear.minmax_scale_bound) -- unconstrained, sign-SGD can drift them
    # to extreme values that overfit the calibration loss at the cost of downstream accuracy.
    MINMAX_SCALE_BOUND = (0.0, 1.0)

    def __init__(self, linear: nn.Module, enable_minmax_tuning: bool = False):
        super().__init__()
        self.linear = linear
        self.value = nn.Parameter(torch.zeros_like(linear.weight))
        self.enable_minmax_tuning = enable_minmax_tuning
        self._setup_activation_quant()
        if enable_minmax_tuning:
            self._setup_minmax_tuning()

    def _setup_minmax_tuning(self) -> None:
        """Create whichever clip-tuning parameter(s) this format needs. Only called when
        ``enable_minmax_tuning`` is True."""
        raise NotImplementedError

    def _setup_activation_quant(self) -> None:
        """Hook for formats that support weight+activation quantization. No-op by default
        (weight-only) — override alongside :meth:`_quantize_activation`."""
        return

    def value_clamped(self) -> torch.Tensor:
        return torch.clamp(self.value, -0.5, 0.5)

    def clip_params(self) -> list[torch.Tensor]:
        """Learnable clip-tuning parameters for the block optimizer's second param group
        (``autoround.py::optimize_wrappers_signed_sgd``). Empty unless minmax tuning is on."""
        return []

    def _quantize_weight(self) -> torch.Tensor:
        raise NotImplementedError

    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Identity by default (weight-only quantization). Formats that also support
        weight+activation quantization override this (no learnable parameters belong here —
        AutoRound only ever tunes weight-side ``V``/clip; activation quantization, when present,
        is a fixed, non-trainable quantize function applied fresh every forward)."""
        return x

    def get_scale_zero_point(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current (scale, zero_point) as 2D [out_features, n_groups] tensors, ready
        to write back into the real ``weight_quantizer`` at freeze. Only valid when minmax on."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self._quantize_activation(x), self._quantize_weight(), self.linear.bias)


class WrapperLinearInt(_WrapperLinearBase):
    """Learnable per-weight rounding offset V, plus optional learnable per-group clip scale(s)
    (AutoRound minmax tuning), for INT weight-only quantization -- symmetric or asymmetric.

    w_int = clamp(round_ste(w / scale + clamp(V, -0.5, 0.5)) + zero_point, quant_min, quant_max)
    w_deq = (w_int - zero_point) * scale

    Forward calls Quark's real kernel directly (``hw_emulation_interface.py::
    fake_quantize_per_channel_affine``, the same one ``weight_quantizer`` dispatches to), so it
    can't drift from what's actually deployed; ``V`` is injected by shifting the kernel's input
    by ``V * scale`` (see ``_quantize_weight``). Backward uses the formula above as an exact STE
    gradient -- INT4 rounding is a single ``round()``, unlike MXFP4's multi-step grid snap in
    ``mxfp_quant.py``, which needs a derived surrogate.

    With ``enable_minmax_tuning``, ``scale``/``zero_point`` are recomputed each forward from the
    group's weight min/max scaled by learnable clip param(s), matching Quark's per-group int
    observer (``observer.py::calculate_int_quant_params``) so tuned qparams can be written back
    at freeze. Symmetric schemes (``weight_quantizer.symmetric``) get a single ``max_scale`` clip
    param and a zero_point fixed at 0 (observer.py:277-280); asymmetric schemes get separate
    ``min_scale``/``max_scale`` and a tuned zero_point (observer.py:268-269, 288-292). Without
    minmax tuning, the layer's existing ``weight_quantizer`` values are used as-is (only V learned).
    """

    _EPS = torch.finfo(torch.float32).eps

    def _read_quant_meta(self) -> tuple[int, int, int, bool]:
        """Read (quant_min, quant_max, group_size, symmetric) from the real weight_quantizer, or
        the test-only ar_* fallback (always asymmetric -- ``attach_int4_fakequant`` only builds an
        asymmetric fake-quant)."""
        weight_quantizer = getattr(self.linear, "weight_quantizer", None)
        if weight_quantizer is not None:
            return (
                int(weight_quantizer.quant_min),
                int(weight_quantizer.quant_max),
                int(weight_quantizer.group_size),
                bool(weight_quantizer.symmetric),
            )
        return 0, int(self.linear.ar_qmax), int(self.linear.ar_group_size), False

    def _setup_minmax_tuning(self) -> None:
        quant_min, quant_max, group_size, symmetric = self._read_quant_meta()
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.group_size = group_size
        self.symmetric = symmetric

        weight = self.linear.weight.detach()
        out_features, in_features = weight.shape
        n_groups = in_features // group_size
        wg = weight.reshape(out_features, n_groups, group_size)

        if symmetric:
            # Matches Quark's per-group symmetric-int observer basis exactly: amax = max(-min,
            # max) per group (observer.py:277-280) -- a single scale, zero_point fixed at 0. Only
            # one clip param (max_scale), like WrapperLinearMXFP4 -- there's no min side to tune.
            wmax = wg.abs().max(dim=-1, keepdim=True).values
            self.register_buffer("wmax", wmax)
            self.max_scale = nn.Parameter(
                torch.ones(out_features, n_groups, 1, device=weight.device, dtype=weight.dtype)
            )
        else:
            # Mirror the observer basis exactly: min_val_neg = clamp(min, max=0),
            # max_val_pos = clamp(max, min=0) (observer.py:268-269). Store the CLAMPED
            # min/max so that at min_scale=max_scale=1.0 the recomputed scale/zp equal
            # the observer's.
            wmin = wg.min(dim=-1, keepdim=True).values.clamp(max=0.0)
            wmax = wg.max(dim=-1, keepdim=True).values.clamp(min=0.0)
            self.register_buffer("wmin", wmin)
            self.register_buffer("wmax", wmax)
            self.min_scale = nn.Parameter(
                torch.ones(out_features, n_groups, 1, device=weight.device, dtype=weight.dtype)
            )
            self.max_scale = nn.Parameter(
                torch.ones(out_features, n_groups, 1, device=weight.device, dtype=weight.dtype)
            )

    def clip_params(self) -> list[torch.Tensor]:
        if not self.enable_minmax_tuning:
            return []
        if self.symmetric:
            return [self.max_scale]
        return [self.min_scale, self.max_scale]

    def _recompute_scale_zp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute per-group scale/zero_point ([out_features, n_groups, 1]) from the clipped
        group min/max, matching the official repo's ``quant_tensor_sym``/``quant_tensor_asym``.

        Symmetric: a single amax-scaled ``max_scale``, zero_point fixed at 0 -- there's nothing to
        clamp on zero_point since it's a constant, not a computed value.

        Asymmetric: zero_point uses ``round_ste`` (not a plain round, so gradient reaches
        min_scale/max_scale through it too) and is left unclamped here -- only the final combined
        index (``round_ste(w/scale+V) + zero_point``, in ``_quantize_weight``) is clamped, matching
        ``clamp(int_w + zp, 0, maxq)``.
        """
        # Project the clip param(s) back into MINMAX_SCALE_BOUND in place, every call — mirrors
        # the official repo's per-forward-pass clamp (auto_round/wrapper.py _qdq_weight).
        self.max_scale.data.clamp_(*self.MINMAX_SCALE_BOUND)
        if self.symmetric:
            wmax_s = self.wmax * self.max_scale
            scale = wmax_s / (float(self.quant_max - self.quant_min) / 2.0)
            scale = torch.clamp(scale, min=self._EPS)
            zero_point = torch.zeros_like(scale)
            return scale, zero_point

        self.min_scale.data.clamp_(*self.MINMAX_SCALE_BOUND)
        wmin_s = self.wmin * self.min_scale
        wmax_s = self.wmax * self.max_scale
        scale = (wmax_s - wmin_s) / float(self.quant_max - self.quant_min)
        scale = torch.clamp(scale, min=self._EPS)
        zero_point = self.quant_min - round_ste(wmin_s / scale)
        return scale, zero_point

    def get_scale_zero_point(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Current (scale, zero_point) as 2D [out_features, n_groups] tensors, for write-back
        into ``weight_quantizer`` at freeze. Only valid when minmax tuning is on. Unlike
        ``_recompute_scale_zp``, zero_point IS clamped to [quant_min, quant_max] here, since it's
        written into a possibly-unsigned-integer buffer that has no tolerance for out-of-range
        values.
        """
        scale, zero_point = self._recompute_scale_zp()
        zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max)
        return scale.squeeze(-1), zero_point.squeeze(-1)

    def _quantize_weight(self) -> torch.Tensor:
        weight = self.linear.weight
        out_features, in_features = weight.shape

        if self.enable_minmax_tuning:
            group_size = self.group_size
            quant_min, quant_max = self.quant_min, self.quant_max
            scale, zero_point = self._recompute_scale_zp()
        else:
            # Production processor path: prefer the layer's own Quark weight quantizer
            # (a ScaledFakeQuantize, populated by calibration) over the test-only ar_* buffers.
            # See NOTES.md Q2 — GPTQ reads scale/zero_point the same way
            # (quark/torch/algorithm/gptq/gptq.py:374-375).
            weight_quantizer = getattr(self.linear, "weight_quantizer", None)
            if weight_quantizer is not None:
                scale = weight_quantizer.scale
                zero_point = weight_quantizer.zero_point
                quant_min = weight_quantizer.quant_min
                quant_max = weight_quantizer.quant_max
                group_size = weight_quantizer.group_size
            else:
                scale = self.linear.ar_scale
                zero_point = self.linear.ar_zp
                quant_min = 0
                quant_max = self.linear.ar_qmax
                group_size = self.linear.ar_group_size

            # Quark's per-group weight scale/zero_point come back as [out_features, n_groups]
            # (see observer.py `_scale.reshape(-1, group_count)`); add the trailing group axis
            # so they broadcast over group_size. The test-only ar_* buffers are already 3D.
            if hasattr(scale, "dim") and scale.dim() == 2:
                scale = scale.unsqueeze(-1)
            if hasattr(zero_point, "dim") and zero_point.dim() == 2:
                zero_point = zero_point.unsqueeze(-1)

        n_groups = in_features // group_size
        wg = weight.reshape(out_features * n_groups, group_size)
        vg = self.value_clamped().reshape(out_features * n_groups, group_size)
        scale_flat = scale.reshape(out_features * n_groups)
        zero_point_flat = zero_point.reshape(out_features * n_groups)

        # Forward: Quark's real kernel directly, so it can't drift from deployment. V is
        # injected by shifting the input by V*scale: round((w + V*scale)*inv_scale) ==
        # round(w*inv_scale + V), a no-op at V=0.
        with torch.no_grad():
            shifted = (wg + vg.detach() * scale_flat.unsqueeze(-1)).detach().contiguous()
            w_deq_exact = fake_quantize_per_channel_affine(
                shifted, scale_flat, zero_point_flat, axis=0, quant_min=quant_min, quant_max=quant_max
            )

        # Backward: round_ste's gradient is already exact here (a single round()), unlike
        # MXFP4's multi-step grid snap, which needs a derived surrogate (mxfp_quant._mxfp4_grid_smooth).
        w_int_smooth = round_ste(wg * (1.0 / scale_flat.unsqueeze(-1)) + vg)
        w_int_smooth = torch.clamp(w_int_smooth + zero_point_flat.unsqueeze(-1), quant_min, quant_max)
        w_deq_smooth = (w_int_smooth - zero_point_flat.unsqueeze(-1)) * scale_flat.unsqueeze(-1)

        w_deq = (w_deq_exact - w_deq_smooth).detach() + w_deq_smooth
        return w_deq.reshape(out_features, in_features)


class WrapperLinearMXFP4(_WrapperLinearBase):
    """Learnable per-weight rounding offset V, plus an optional learnable per-group clip scale,
    for OCP microscaling FP4 (E2M1) weight quantization.

    Quantize math is ``mxfp_quant.mxfp4_fake_quantize``, ported from Quark's own kernel (not the
    official repo's ``data_type/mxfp.py``, which disagrees on both the shared-exponent formula
    and FP4-grid tie-breaking). AutoRound-tuned weights are served by Quark's real kernel at
    freeze/inference, so tuning-time math must match it exactly.

    Unlike ``WrapperLinearInt``, there is only one clip parameter (``max_scale``) and no
    zero-point -- MXFP4 is symmetric, with a single per-group shared-exponent scale.

    If ``linear.input_quantizer`` is a real dynamic MXFP4 quantizer (weight+activation scheme),
    activations are also MXFP4-quantized every forward via that quantizer directly -- no
    learnable parameters or calibration state involved, so Quark's own
    ``DynamicScaledFakeQuantize`` is used as-is rather than reimplemented. Otherwise (default,
    weight-only) activations pass through unchanged.
    """

    def _setup_activation_quant(self) -> None:
        input_quantizer = getattr(self.linear, "input_quantizer", None)
        self.quantize_activation = input_quantizer is not None and getattr(input_quantizer, "dtype", None) == Dtype.fp4

    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize_activation:
            return x
        return self.linear.input_quantizer(x)

    def _read_group_size(self) -> int:
        weight_quantizer = getattr(self.linear, "weight_quantizer", None)
        if weight_quantizer is not None:
            return int(weight_quantizer.group_size)
        return int(self.linear.ar_group_size)

    def _read_scale_calculation_mode(self) -> str:
        """``"even"`` (default, matches Quark's HIP kernel) or ``"floor"`` (matches the official
        repo's formula, see ``mxfp_quant.mxfp4_scale_floor``).

        Read from ``weight_quantizer.observer.scale_calculation_mode``, not the quantizer
        itself: post-calibration, ``weight_quantizer`` is a ``StaticScaledFakeQuantize`` with no
        such attribute of its own -- only the observer keeps it around, for recalibration.
        """
        weight_quantizer = getattr(self.linear, "weight_quantizer", None)
        if weight_quantizer is not None:
            observer = getattr(weight_quantizer, "observer", None)
            mode = getattr(observer, "scale_calculation_mode", None)
            if mode is not None:
                return mode
        return "even"

    def _setup_minmax_tuning(self) -> None:
        group_size = self._read_group_size()
        self.group_size = group_size
        self.scale_calculation_mode = self._read_scale_calculation_mode()

        weight = self.linear.weight.detach()
        out_features, in_features = weight.shape
        n_groups = in_features // group_size
        wg = weight.reshape(out_features, n_groups, group_size)
        amax = wg.abs().max(dim=-1, keepdim=True).values
        self.register_buffer("amax", amax)
        self.max_scale = nn.Parameter(torch.ones(out_features, n_groups, 1, device=weight.device, dtype=weight.dtype))

    def clip_params(self) -> list[torch.Tensor]:
        if not self.enable_minmax_tuning:
            return []
        return [self.max_scale]

    def _recompute_scale(self) -> torch.Tensor:
        """Clamp ``max_scale`` into ``MINMAX_SCALE_BOUND`` every call (mirrors the official
        repo's per-forward clamp and ``WrapperLinearInt._recompute_scale_zp``) -- unconstrained
        sign-SGD can otherwise drift it outside [0, 1], even negative."""
        self.max_scale.data.clamp_(*self.MINMAX_SCALE_BOUND)
        return mxfp4_scale(self.amax * self.max_scale, self.scale_calculation_mode)

    def get_scale_zero_point(self) -> tuple[torch.Tensor, torch.Tensor]:
        """MXFP4 is symmetric (no real zero-point); return zeros, matching
        ``PerBlockMXObserver``'s own placeholder and keeping the write-back shape contract in
        ``AutoRoundProcessor._freeze_block`` the same across both wrapper types."""
        scale = self._recompute_scale()
        scale_2d = scale.squeeze(-1)
        return scale_2d, torch.zeros_like(scale_2d)

    def _quantize_weight(self) -> torch.Tensor:
        weight = self.linear.weight
        out_features, in_features = weight.shape

        if self.enable_minmax_tuning:
            group_size = self.group_size
            scale = self._recompute_scale()
        else:
            group_size = self._read_group_size()
            weight_quantizer = self.linear.weight_quantizer
            scale = weight_quantizer.scale
            if scale.dim() == 2:
                scale = scale.unsqueeze(-1)

        n_groups = in_features // group_size
        wg = weight.reshape(out_features, n_groups, group_size)
        vg = self.value_clamped().reshape(out_features, n_groups, group_size)

        w_deq = mxfp4_fake_quantize(wg, scale, v=vg)
        return w_deq.reshape(out_features, in_features)


# Backward-compatible alias: existing call sites/tests refer to the INT4 wrapper as
# `WrapperLinear` (the name predates MXFP4 support).
WrapperLinear = WrapperLinearInt


def make_wrapper(linear: nn.Module, enable_minmax_tuning: bool = False) -> _WrapperLinearBase:
    """Dispatch to the wrapper class matching ``linear``'s quantization format."""
    weight_quantizer = getattr(linear, "weight_quantizer", None)
    dtype = getattr(weight_quantizer, "dtype", None)
    if dtype == Dtype.fp4:
        return WrapperLinearMXFP4(linear, enable_minmax_tuning)
    return WrapperLinearInt(linear, enable_minmax_tuning)
