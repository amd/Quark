#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""MXFP4 (OCP microscaling FP4, E2M1) quantize/dequantize math for AutoRound tuning.

The differentiable backward surrogates in this module (:func:`mxfp4_scale_floor`,
:func:`_mxfp4_grid_smooth`) are independent re-implementations verified to numerically match the
gradient shape of the official ``auto-round`` repo (https://github.com/intel/auto-round,
Apache License 2.0) -- credit to that project for the ``quant_mx``/``quant_element`` formulas
this code mirrors.

The shared-exponent scale and FP4-grid rounding both call Quark's own production code directly
(rather than re-implementing the bit-level math here), so this module tracks Quark's actual
kernel behavior automatically:

- Scale (``even`` mode): ``quark.torch.quantization.utils.even_round`` -- the same function
  ``PerBlockMXObserver.get_scale`` uses during calibration.
- FP4-grid rounding: ``torch.ops.quark.fake_quantize_to_low_precision_fp`` -- the same compiled
  op the real static/packed quantization dispatch calls, with the same round-half-to-even
  tie-break the official ``auto-round`` repo's ``mantissa_rounding="even"`` uses.

Only ``floor`` scale-calculation mode (a Quark-only variant for apples-to-apples comparison
against the official repo's scale formula; not something the real kernel supports) has no
existing Quark function to call, and is implemented directly here.

Both steps are differentiable via straight-through estimators (STE), so gradient reaches the
AutoRound-learnable ``max_scale`` (through the scale) and ``V`` (through the grid snap).
"""

import torch

from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.utils import calculate_qmin_qmax, even_round, get_dtype_params

# Quark's own source of truth for E2M1's bit layout and representable range, rather than
# hardcoding them here.
FP4_EBITS, FP4_MBITS, FP4_EMAX = get_dtype_params(Dtype.fp4)
_, FP4_MAX_NORM = calculate_qmin_qmax(Dtype.fp4)

# round_mode passed to torch.ops.quark.fake_quantize_to_low_precision_fp at every real call
# site in hw_emulation_interface.py (fake- and real-quantize alike); RoundMode only defines
# ROUND_HALF_TO_EVEN=8, but 0 is what production code actually passes, so match that exactly.
_FP4_ROUND_MODE = 0


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for round(): forward rounds, backward is identity."""
    return (torch.round(x) - x).detach() + x


def floor_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for floor(): forward floors, backward is identity."""
    return (torch.floor(x) - x).detach() + x


def mxfp4_scale_even(max_abs: torch.Tensor) -> torch.Tensor:
    """OCP E8M0 shared-exponent scale using Quark's own ``even_round`` (``PerBlockMXObserver``'s
    default mode) -- not the official repo's plain ``2 ** (floor(log2(amax)) - emax)`` (that's
    Quark's separate ``"floor"`` mode, :func:`mxfp4_scale_floor`).

    Forward calls ``even_round`` directly (its int32 bit manipulation blocks gradient); backward
    is an STE treating the computation as the smooth function it approximates, via ``floor_ste``
    on the log2/exp2 chain, so gradient reaches the tunable ``max_scale`` clip parameter.
    """
    with torch.no_grad():
        scale_exact = even_round(max_abs.detach().to(torch.float32).contiguous(), Dtype.fp4)

    smooth_log2 = torch.log2(torch.clamp(max_abs, min=torch.finfo(torch.float32).tiny))
    smooth = torch.exp2(floor_ste(smooth_log2) - FP4_EMAX)
    return (scale_exact - smooth).detach() + smooth


def mxfp4_scale_floor(max_abs: torch.Tensor) -> torch.Tensor:
    """Quark's ``"floor"`` scale-calculation mode (``PerBlockMXObserver.get_scale``):
    ``2 ** (floor(log2(amax)) - emax)``, no mantissa pre-rounding -- matches the official
    repo's ``quant_mx`` formula exactly, unlike the default ``"even"`` mode
    (:func:`mxfp4_scale_even`). No real kernel dispatches to this, so implemented directly here;
    differentiable via ``floor_ste``.
    """
    eps = torch.finfo(torch.float32).eps
    log2_amax = torch.log2(torch.clamp(max_abs, min=torch.finfo(torch.float32).tiny))
    scale = torch.exp2(floor_ste(log2_amax) - FP4_EMAX)
    return torch.where(max_abs == 0, torch.full_like(scale, eps), scale)


def mxfp4_scale(max_abs: torch.Tensor, scale_calculation_mode: str = "even") -> torch.Tensor:
    """Dispatch to :func:`mxfp4_scale_even` (default, matches Quark's ``PerBlockMXObserver``) or
    :func:`mxfp4_scale_floor` (matches the official auto-round repo's formula exactly) by name,
    mirroring the ``scale_calculation_mode`` field on Quark's ``OCP_MXFP4Spec``/weight_quantizer."""
    if scale_calculation_mode == "floor":
        return mxfp4_scale_floor(max_abs)
    if scale_calculation_mode == "even":
        return mxfp4_scale_even(max_abs)
    raise ValueError(f"Unsupported scale_calculation_mode: {scale_calculation_mode!r}")


def _mxfp4_grid_smooth(x_clamped: torch.Tensor) -> torch.Tensor:
    """Differentiable surrogate for the FP4-grid snap, used ONLY for the STE backward pass (the
    forward value always comes from the real op, :func:`mxfp4_quantize_dequantize`). Mirrors the
    official repo's ``quant_element``: extract each element's own exponent (``floor_ste``),
    rescale into the mantissa's integer domain, round via a differentiable round-half-to-even,
    rescale back. Verified to match the official repo's actual gradient values exactly across
    subnormal/normal/near-saturation test points -- unlike a flat STE (gradient always 1), this
    gives ``V``/``max_scale`` the real *shape* of gradient the official repo's tuning sees.

    Uses the official repo's own bit-counting convention locally (``ebits=2``, ``mbits=3`` --
    mantissa bits including the implicit leading one) rather than ``FP4_EBITS``/``FP4_MBITS``
    above (Quark's convention, excluding it) -- same E2M1 format, different labeling.
    """
    ebits, mbits = 2, 3
    private_exp = floor_ste(torch.log2(torch.abs(x_clamped) + (x_clamped == 0).to(x_clamped.dtype)))
    min_exp = -(2.0 ** float(ebits - 1)) + 2
    private_exp = torch.clamp(private_exp, min=min_exp)

    scaled = x_clamped / torch.exp2(private_exp) * (2.0 ** float(mbits - 2))

    abs_scaled = torch.abs(scaled)
    mask = ((abs_scaled - 0.5) % 2 == 0).to(scaled.dtype)
    rounded = torch.sign(scaled) * (floor_ste(abs_scaled + 0.5) - mask)

    return rounded / (2.0 ** float(mbits - 2)) * torch.exp2(private_exp)


def mxfp4_quantize_dequantize(x_scaled: torch.Tensor) -> torch.Tensor:
    """Quantize an already-scaled tensor (weight/scale, optionally + the AutoRound rounding
    offset V) to the OCP E2M1 grid and immediately dequantize it, i.e. FP4 fake-quantization.

    Forward calls ``torch.ops.quark.fake_quantize_to_low_precision_fp`` directly -- the same
    compiled op Quark's real static/packed quantization dispatch calls, so AutoRound-tuned
    weights are quantized with the exact tie-break rule they'll actually be served with.

    Applies a real (non-STE) ``clamp`` to ``FP4_MAX_NORM`` before both the real op and the smooth
    surrogate, matching the official repo's ``quant_mx`` (real clamp gradient: 0 outside the
    range, before its STE grid-rounding step) -- doesn't change forward numerics (Quark's op
    already saturates internally to the same bound), but fixes the backward: without it, a
    saturated element would still get gradient 1 pushing ``V``/``max_scale`` further, instead of
    the correct 0.

    Backward uses :func:`_mxfp4_grid_smooth` (not a flat identity surrogate) so gradient reaches
    ``V`` and, through the scale, ``max_scale`` with the official repo's actual gradient shape.
    """
    x_clamped = torch.clamp(x_scaled, min=-FP4_MAX_NORM, max=FP4_MAX_NORM)

    with torch.no_grad():
        x32 = x_clamped.detach().to(torch.float32).contiguous()
        dequantized = torch.ops.quark.fake_quantize_to_low_precision_fp(
            x32, FP4_EBITS, FP4_MBITS, FP4_MAX_NORM, _FP4_ROUND_MODE
        ).to(x_scaled.dtype)

    smooth = _mxfp4_grid_smooth(x_clamped)
    return (dequantized - smooth).detach() + smooth


def mxfp4_fake_quantize(x: torch.Tensor, scale: torch.Tensor, v: torch.Tensor | float = 0.0) -> torch.Tensor:
    """Full MXFP4 fake-quantize: dequantized value = ``mxfp4_quantize_dequantize(x/scale + v) *
    scale``. ``scale`` is expected to already be a per-group MXFP4 shared-exponent scale (e.g.
    from :func:`mxfp4_scale`); ``v`` is the AutoRound learnable rounding offset, added before the
    FP4-grid snap (same role as INT4's ``V`` in ``wrapper.py``)."""
    return mxfp4_quantize_dequantize(x / scale + v) * scale
