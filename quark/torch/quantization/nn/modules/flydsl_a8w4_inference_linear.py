#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""FlyDSL-backed native inference linear for A8W4 on gfx950.

This backend consumes MXFP4-packed weights and dynamically quantizes bf16/fp16
activations to per-1x32 FP8 before launching FlyDSL's gfx950 A8W4 GEMM.
"""

from __future__ import annotations

from functools import cache
from typing import Any, ClassVar

import torch
from torch import Tensor

from quark.common.utils.log import ScreenLogger
from quark.torch.kernel.flydsl import (
    get_mxfp8_quant as _get_flydsl_mxfp8_quant,
)
from quark.torch.kernel.flydsl import (
    is_flydsl_quant_available,
)
from quark.torch.quantization.nn.modules.aiter_fp4_inference_linear import (
    _get_mxfp4_float_weight,
    _pack_weight_asm,
)
from quark.torch.quantization.nn.modules.native_inference_linear_common import (
    NativeInferenceLinear,
    NativeInferenceMode,
    _KernelState,
    _require_aiter,
    register_native_backend,
)

logger = ScreenLogger(__name__)


def _e8m0_from_fp32(x: Tensor) -> Tensor:
    """E8M0 exponent of ``x``, rounding the fraction **UP** (ceil).

    ``x`` is ``amax / dtype_max``, so the scale must never be smaller than that ratio.
    With round-to-nearest a mantissa < 1.5 keeps the lower binade, ``amax/scale`` lands in
    (448, 672), and the following ``clamp(+-448)`` truncates the top of every affected
    32-block. That error is *biased toward zero* rather than random, so it appears as a
    coherent output-magnitude loss -- measured gain 0.963 end-to-end, and exactly 7/8 for an
    all-ones block -- which cosine similarity cannot see, letting ``cos > 0.98`` assertions
    pass while ~3.7% of the output magnitude is missing.

    Rounding up is the safe direction: it costs at most one binade of resolution on a
    block's *small* elements, which e4m3's 3 mantissa bits absorb, and never saturates
    against the clamp. Keep in sync with the FlyDSL kernel mirror
    (``quark/torch/kernel/flydsl/mxfp8_quant.py``); ``test/test_for_torch/test_e8m0_ceil.py``
    asserts both against an independently computed ceil rather than against each other.
    """
    u32 = x.view(torch.int32)
    exponent = ((u32 >> 23) & 0xFF).view(torch.uint32).to(torch.uint8)
    nan_case = exponent == 0xFF
    # Any nonzero mantissa bumps the exponent -> ceil(log2 x), not round-to-nearest.
    round_case = ((u32 & 0x7FFFFF) > 0) & (exponent < 0xFF)
    exponent[round_case] += 1
    exponent[nan_case] = 0xFF
    return exponent


def _e8m0_to_fp32(scale_e8m0: Tensor) -> Tensor:
    scale_e8m0 = scale_e8m0.view(torch.uint8)
    zero_case = scale_e8m0 == 0
    nan_case = scale_e8m0 == 0xFF
    scale_f32 = scale_e8m0.to(torch.int32) << 23
    scale_f32[zero_case] = 0x00400000
    scale_f32[nan_case] = 0x7F800001
    return scale_f32.view(torch.float32)


def _shuffle_e8m0_scale(scale: Tensor) -> Tensor:
    m, n = scale.shape
    scale_padded = torch.empty(
        (m + 255) // 256 * 256,
        (n + 7) // 8 * 8,
        dtype=scale.dtype,
        device=scale.device,
    )
    scale_padded[:m, :n] = scale
    scale = scale_padded
    sm, sn = scale.shape
    scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
    scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
    return scale.view(sm, sn)


def _quantize_mxfp8_1x32_eager(x: Tensor) -> tuple[Tensor, Tensor]:
    block = 32
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = float(torch.finfo(fp8_dtype).max)
    shape = x.shape
    x_flat = x.contiguous().view(-1, block).float()
    amax = torch.amax(torch.abs(x_flat), dim=-1).clamp_min(1e-30)
    scale_e8m0 = _e8m0_from_fp32(amax / fp8_max)
    scale_f32 = _e8m0_to_fp32(scale_e8m0).clamp_min(1e-30)
    x_q = (x_flat / scale_f32.view(-1, 1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    x_q = x_q.view(shape).view(torch.uint8).contiguous()
    scale = scale_e8m0.view(shape[0], shape[-1] // block).contiguous()
    return x_q, scale


# Optional native FlyDSL MXFP8 quant kernel (packaged as
# quark.torch.kernel.flydsl.mxfp8_quant, resolved via quark.torch.kernel.flydsl).
# It writes the e8m0 scale directly in the tiled layout (shuffle=True). gfx950 only;
# get_mxfp8_quant() is itself cached and only performs an import, so the callable is
# resolved on demand instead of being memoized a second time here. Importing the
# accessor always succeeds -- it defers `import flydsl` to call time -- so availability
# has to be probed with is_flydsl_quant_available() rather than by testing the accessor
# against None.
@cache
def _flydsl_quant_available() -> bool:
    """Memoized is_flydsl_quant_available(); functools.cache does not cache the raise."""
    return is_flydsl_quant_available()


def _quantize_mxfp8_1x32_flydsl_shuffled(x: Tensor) -> tuple[Tensor, Tensor]:
    """Native FlyDSL per-1x32 MXFP8 quant + tiled e8m0 scale write in one kernel.
    Returns (x_q uint8, x_scale uint8) already shuffled (no _shuffle_e8m0_scale).
    Bit-exact to the eager reference. Raises ImportError, with the install hint, when
    the flydsl wheel is missing."""
    x_q, x_scale = _get_flydsl_mxfp8_quant()(x, shuffle=True)
    return x_q.view(torch.uint8), x_scale.view(torch.uint8).contiguous()


def _validate_a8w4_inputs(
    x: Tensor, weight: Tensor, weight_scale: Tensor, out_dtype: torch.dtype
) -> tuple[int, int, int]:
    if not x.is_cuda or not weight.is_cuda or not weight_scale.is_cuda:
        raise ValueError("FlyDSL A8W4 requires CUDA/ROCm tensors for input, weight, and scale.")
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"FlyDSL A8W4 supports bf16/fp16 outputs, got {out_dtype}.")

    m = x.shape[0]
    n = weight.shape[0]
    k = x.shape[1]
    if k < 256 or k % 256 != 0:
        raise ValueError(f"FlyDSL A8W4 requires K >= 256 and divisible by 256, got K={k}.")
    # An N that is not a multiple of 128 routes to a tile_n < 128, which the MX-scale
    # MFMA path cannot issue (it packs two 16-wide N blocks per instruction). The kernel
    # rejects it, but catch it here so the error names the shape rather than the tile.
    if n < 128 or n % 128 != 0:
        raise ValueError(f"FlyDSL A8W4 requires N >= 128 and divisible by 128, got N={n}.")
    if weight.shape[1] * 2 != k:
        raise ValueError(
            f"FlyDSL A8W4 expects MXFP4-packed weight shape [N, K/2], got weight={tuple(weight.shape)} and K={k}."
        )
    return m, n, k


# --------------------------------------------------------------------------- #
# Kernel config selection
# --------------------------------------------------------------------------- #
# The preshuffle A8W4 GEMM is parameterized by the tile shape plus a handful of
# scheduling knobs. The old fixed 32/128/128 tile only suits very small M (decode);
# for prefill-sized M (e.g. Wan diffusion, M~4.7k tokens) it leaves ~1.5x on the
# table. On gfx950 the consistent winner across the Wan linear shapes is
# 64/256/128 with async global->LDS copies.
#
# Heuristic default for large M (prefill). tile_n=256 needs N % 256 == 0; the
# caller falls back to _safe_base_cfg() when that does not hold.
_PREFILL_CFG: dict[str, Any] = dict(tile_m=64, tile_n=256, tile_k=128, use_async_copy=True)

# Cache the chosen config per (m_bucket, n, k, out_s) and the compiled launch per
# (m_pad, n, k, out_s, cfg). M is a compile-time constant for the FlyDSL kernel, but
# in a given run M is fixed (resolution/frames), so the launch cache stays warm.
_CONFIG_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_LAUNCH_CACHE: dict[tuple[Any, ...], Any] = {}
# Compiled GEMM fast-path callables (flyc.compile) keyed by the same tuple as
# _LAUNCH_CACHE. flyc.compile reuses the pre-built CallState and skips flyc.jit's
# per-call signature bind / globals-drift / cache-key rebuild (~54 us/call wall
# -> ~5 us). The GEMM runs ~550x/forward on the shared w4a8 path, so this speeds
# up both the fused and the eager quant paths. None => not yet compiled.
_GEMM_COMPILED_CACHE: dict[tuple[Any, ...], Any] = {}


def _m_bucket(m: int) -> int:
    """Coarse M bucket so config selection is stable across nearby token counts."""
    if m <= 32:
        return 32
    if m < 512:
        return 256
    return 4096  # prefill-sized


def _cfg_valid_for_shape(cfg: dict[str, Any], n: int, k: int) -> bool:
    """The preshuffle kernel does not mask N/K tiles that overrun the matrix, so a
    tile_n/tile_k that does not divide N/K launches with an out-of-range grid
    (hipErrorInvalidValue). Only accept tiles that tile N and K exactly. (M is
    handled separately by zero-padding to tile_m.)"""
    return (n % cfg["tile_n"] == 0) and (k % cfg["tile_k"] == 0)


def _pad_m(m: int, tile_m: int) -> int:
    return (m + tile_m - 1) // tile_m * tile_m


def _compile_cfg(
    m_pad: int,
    n: int,
    k: int,
    out_s: str,
    cfg: dict[str, Any],
    epilogue: str = "none",
    rank: int = 0,
) -> Any:
    key = (m_pad, n, k, out_s, tuple(sorted(cfg.items())), epilogue, rank)
    launch = _LAUNCH_CACHE.get(key)
    if launch is None:
        from quark.torch.kernel.flydsl import get_a8w4_compile

        compile_fn = get_a8w4_compile()
        kwargs: dict[str, Any] = dict(M=m_pad, N=n, K=k, out_dtype=out_s, epilogue=epilogue, **cfg)
        if epilogue in ("svd", "svd_bias"):
            kwargs["rank"] = rank
        launch = compile_fn(**kwargs)
        _LAUNCH_CACHE[key] = launch
    return launch


def _safe_base_cfg(k: int) -> dict[str, Any]:
    """A guaranteed-valid config for this shape: tile_n/tile_k must divide N/K, else the
    preshuffle kernel overruns the matrix. N is validated to be a multiple of 128, so the
    128-wide tile always divides it, and tile_n never drops below 128 because the MX-scale
    MFMA path cannot issue a narrower N tile."""
    tk = 128 if k % 128 == 0 else (64 if k % 64 == 0 else 32)
    return dict(tile_m=32, tile_n=128, tile_k=tk, use_async_copy=False)


def _select_config(m: int, n: int, k: int, out_s: str) -> dict[str, Any]:
    key = (_m_bucket(m), n, k, out_s)
    cfg = _CONFIG_CACHE.get(key)
    if cfg is not None:
        return cfg
    # Heuristic: prefill-sized M uses the tuned tile, but only when it tiles N and K
    # exactly; otherwise use a shape-safe baseline. tile_n=256 needs N % 256 == 0, which
    # the validator does not guarantee (it only requires a multiple of 128), and
    # tile_k=128 still has to divide K.
    cfg = _PREFILL_CFG if m >= 64 and _cfg_valid_for_shape(_PREFILL_CFG, n, k) else _safe_base_cfg(k)
    _CONFIG_CACHE[key] = cfg
    return cfg


def _quantize_activation_mxfp8(x: Tensor) -> tuple[Tensor, Tensor]:
    """Dynamically quantize x to per-1x32 MXFP8 with the tiled (shuffled) e8m0 scale.

    Uses the fused FlyDSL quant+shuffle kernel. Only the FlyDSL A8W4/SVDQuant GEMMs
    call this and those already require the flydsl wheel, so the kernel is the normal
    path here and the fastest measured (1.5-2.2x over the old Triton default at Wan
    shapes). The pure-torch reference runs instead when the kernel is unavailable or
    the input is not the 2D CUDA K%32==0 shape the kernel accepts; a kernel that is
    present but fails raises rather than silently degrading the numerics.

    Returns (x_q uint8, x_scale uint8) with the scale in the kernel's tiled layout.
    """
    if x.is_cuda and x.dim() == 2 and x.shape[-1] % 32 == 0 and _flydsl_quant_available():
        return _quantize_mxfp8_1x32_flydsl_shuffled(x)  # scale already shuffled
    x_q, x_scale = _quantize_mxfp8_1x32_eager(x)
    x_scale = _shuffle_e8m0_scale(x_scale.view(torch.uint8)).view(torch.uint8).contiguous()
    return x_q, x_scale


def _run_gemm(gemm_key: tuple[Any, ...], launch: Any, gemm_args: tuple[Any, ...]) -> None:
    """Launch the GEMM through flydsl's compiled entry point, memoized per key.

    ``flydsl.compiler.compile`` lowers the traced launcher once; calling ``launch``
    directly re-enters the tracer on every forward.
    """
    compiled = _GEMM_COMPILED_CACHE.get(gemm_key)
    if compiled is None:
        import flydsl.compiler as _flyc

        compiled = _flyc.compile(launch, *gemm_args)
        _GEMM_COMPILED_CACHE[gemm_key] = compiled
    compiled(*gemm_args)


def _gemm_flydsl_a8w4(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    out_dtype: torch.dtype,
    bias: Tensor | None = None,
    epilogue: str = "none",
) -> Tensor:
    from quark.torch.utils.device import get_gpu_arch, is_gfx950

    if not is_gfx950():
        raise RuntimeError(f"FlyDSL A8W4 native linear requires gfx950, got {get_gpu_arch()}.")

    m, n, k = _validate_a8w4_inputs(x, weight, weight_scale, out_dtype)
    out_s = "bf16" if out_dtype == torch.bfloat16 else "fp16"

    cfg = _select_config(m, n, k, out_s)
    m_pad = _pad_m(m, cfg["tile_m"])
    launch = _compile_cfg(m_pad, n, k, out_s, cfg, epilogue)

    if m_pad != m:
        x = torch.nn.functional.pad(x, (0, 0, 0, m_pad - m))

    x_q, x_scale = _quantize_activation_mxfp8(x)
    y = torch.empty(m_pad, n, device=x.device, dtype=out_dtype)
    if bias is not None and epilogue != "none":
        bias_arg = bias.to(out_dtype).contiguous().view(-1)
    else:
        bias_arg = torch.empty(0, device=x.device, dtype=out_dtype)
    # arg_d / arg_l2 are the SVDQuant low-rank up-proj operands, unused here (the
    # plain A8W4 path has no fused SVD epilogue); pass empty tensors so the shared
    # kernel signature is satisfied.
    _svd_empty = torch.empty(0, device=x.device, dtype=out_dtype)
    gemm_args = (
        y.contiguous().view(-1),
        x_q.contiguous().view(-1),
        weight.contiguous().view(-1),
        x_scale.view(-1),
        weight_scale.contiguous().view(torch.uint8).view(-1),
        bias_arg,
        _svd_empty,
        _svd_empty,
        m_pad,
        n,
        torch.cuda.current_stream(),
    )
    # Reuse a flyc.compile fast-path callable per shape/cfg (skips flyc.jit's
    # ~54 us/call signature-bind + globals-drift + cache-key rebuild). Keyed the
    # same way as _LAUNCH_CACHE. First call builds it; later calls dispatch direct.
    gemm_key = (m_pad, n, k, out_s, tuple(sorted(cfg.items())), epilogue)
    _run_gemm(gemm_key, launch, gemm_args)
    return y[:m] if m_pad != m else y


def _gemm_flydsl_svdquant(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    d: Tensor,
    l2: Tensor,
    out_dtype: torch.dtype,
    bias: Tensor | None = None,
) -> Tensor:
    """A8W4 residual GEMM with the SVDQuant low-rank up-proj fused into the epilogue.

    Computes ``y = dequant(quant(x)) @ weight^T + d @ l2^T`` in a single kernel,
    where ``d = x @ L1^T`` (M x rank, pre-computed by the caller) and ``l2`` is the
    SVD factor L2 (N x rank). ``weight``/``weight_scale`` are the MXFP4-packed
    residual R (same layout as _gemm_flydsl_a8w4). Bias, when given, is fused too
    (epilogue="svd_bias"). Falls back to a separate GEMM + torch add on any error.
    """
    from quark.torch.utils.device import get_gpu_arch, is_gfx950

    if not is_gfx950():
        raise RuntimeError(f"FlyDSL SVDQuant native linear requires gfx950, got {get_gpu_arch()}.")

    m, n, k = _validate_a8w4_inputs(x, weight, weight_scale, out_dtype)
    rank = int(l2.shape[-1])
    out_s = "bf16" if out_dtype == torch.bfloat16 else "fp16"
    epilogue = "svd_bias" if (bias is not None) else "svd"

    cfg = _select_config(m, n, k, out_s)
    m_pad = _pad_m(m, cfg["tile_m"])
    launch = _compile_cfg(m_pad, n, k, out_s, cfg, epilogue, rank)

    if m_pad != m:
        x = torch.nn.functional.pad(x, (0, 0, 0, m_pad - m))
        d = torch.nn.functional.pad(d, (0, 0, 0, m_pad - m))

    x_q, x_scale = _quantize_activation_mxfp8(x)
    y = torch.empty(m_pad, n, device=x.device, dtype=out_dtype)
    if bias is not None:
        bias_arg = bias.to(out_dtype).contiguous().view(-1)
    else:
        bias_arg = torch.empty(0, device=x.device, dtype=out_dtype)
    d_arg = d.to(out_dtype).contiguous().view(-1)
    l2_arg = l2.to(out_dtype).contiguous().view(-1)
    gemm_args = (
        y.contiguous().view(-1),
        x_q.contiguous().view(-1),
        weight.contiguous().view(-1),
        x_scale.view(-1),
        weight_scale.contiguous().view(torch.uint8).view(-1),
        bias_arg,
        d_arg,
        l2_arg,
        m_pad,
        n,
        torch.cuda.current_stream(),
    )
    gemm_key = (m_pad, n, k, out_s, tuple(sorted(cfg.items())), epilogue, rank)
    _run_gemm(gemm_key, launch, gemm_args)
    return y[:m] if m_pad != m else y


@register_native_backend(NativeInferenceMode.FLYDSL_A8W4)
class FlyDSLA8W4NativeInferenceLinear(NativeInferenceLinear):
    """FlyDSL gfx950 A8W4 native inference linear.

    Construction mirrors the MXFP4 backend's weight recovery and ASM packing so
    the B operand has the layout expected by FlyDSL's preshuffle A8W4 kernel.
    Forward dynamically quantizes activations to MXFP8 and launches FlyDSL
    directly; unsupported shapes or arches raise instead of falling back.
    """

    expected_mode: ClassVar[NativeInferenceMode] = NativeInferenceMode.FLYDSL_A8W4

    def reset_parameters(self) -> None:
        pass

    def _apply_kernel_state(self, state: _KernelState, *, use_preshuffle: bool = False) -> None:
        del use_preshuffle

        # Same domain the GEMM validates, checked at conversion time so an ineligible
        # layer is left on the eager path (enable_native_inference skips ValueError)
        # rather than raising on every forward.
        if (
            self.in_features < 256
            or self.in_features % 256 != 0
            or self.out_features < 128
            or self.out_features % 128 != 0
        ):
            raise ValueError(
                f"FlyDSL A8W4 kernel requires in_features >=256 and a multiple of 256, and "
                f"out_features >=128 and a multiple of 128; got in={self.in_features} "
                f"out={self.out_features}."
            )

        _require_aiter()
        super()._apply_kernel_state(state)

        weight_float = _get_mxfp4_float_weight(self)
        kernel_weight, kernel_scale = _pack_weight_asm(weight_float)
        del weight_float

        self.register_buffer("_kernel_weight", kernel_weight, persistent=False)
        self._buffers["_kernel_scale"] = kernel_scale

        logger.debug(
            "FlyDSL A8W4 native linear: in=%d out=%d weight=%s scale=%s",
            self.in_features,
            self.out_features,
            tuple(kernel_weight.shape),
            tuple(kernel_scale.shape),
        )

    def _get_kernel_weight(self) -> Tensor:
        return self._kernel_weight

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        del kwargs
        x = args[0]
        original_shape = x.shape
        out_dtype = x.dtype

        # The FlyDSL A8W4 kernel only emits bf16/fp16. The frozen module's
        # _output_dtype can be fp32 (the QDQ weight is stored fp32) or the
        # activation may arrive as fp32 (e.g. Wan norm outputs). Compute the
        # GEMM in bf16 and cast the result back to the caller's dtype.
        kernel_dtype = self._output_dtype if self._output_dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        x_2d = x.to(kernel_dtype).view(-1, self.in_features)

        # Fuse the bias add into the GEMM epilogue when a bias exists (saves a
        # separate elementwise kernel per layer). The kernel adds bias in its
        # bf16/fp16 output space; identical to a post-add for bf16/fp16 out_dtype.
        if self.bias is not None:
            out = _gemm_flydsl_a8w4(
                x_2d,
                self._kernel_weight.view(torch.uint8),
                self._kernel_scale.view(torch.uint8),
                kernel_dtype,
                bias=self.bias,
                epilogue="bias",
            )
            return out.view(*original_shape[:-1], self.out_features).to(out_dtype)

        out = _gemm_flydsl_a8w4(
            x_2d,
            self._kernel_weight.view(torch.uint8),
            self._kernel_scale.view(torch.uint8),
            kernel_dtype,
        )
        return out.view(*original_shape[:-1], self.out_features).to(out_dtype)


__all__ = [
    "FlyDSLA8W4NativeInferenceLinear",
    # Imported by flydsl_svdquant_inference_linear as the residual GEMM.
    "_gemm_flydsl_a8w4",
    "_gemm_flydsl_svdquant",
]
