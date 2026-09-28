# SPDX-License-Identifier: MIT
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Native FlyDSL per-1x32 MXFP8 activation-quant kernel (gfx950).

Standalone counterpart to the fused Triton kernel in
``quark/torch/quantization/nn/modules/mxfp8_triton_quant.py``. Quantizes a
bf16 (M, K) activation tensor to per-1x32 MXFP8 (fp8_e4m3fn elements + E8M0
exponent scale). Extracted from FlyDSL's ``kernels/silu_and_mul_fq.py`` quant
path (drops SiLU + MoE-sorted addressing) and made **bit-exact** to the existing
Triton/eager path:

  amax  = max(|x|) over each 32-block, clamped >= 1e-30
  scale = e8m0(amax / 448)     # exponent of (amax/448), round-up on fraction
  x_q   = clamp(x / 2^(exp-127), +-448) as fp8_e4m3

Design: grid = one workgroup per row (M programs), BLOCK_THREADS threads; each
thread owns ``NBc`` whole consecutive 32-element blocks of its row, so the
per-block amax reduction is entirely in-register (no cross-lane shuffle). Handles
any K % 32 == 0 (Wan uses 3072 / 14336 / 4096 / 256).

``shuffle=True`` (default) scatters each block's E8M0 byte directly into the
FlyDSL A8W4 tiled scale layout (drop-in for ``_shuffle_e8m0_scale``):

    off = (r//32)*(sn*32) + (r//16)%2 + (r%16)*4
        + (c//8)*256      + (c//4)%2*2 + (c%4)*64

with sn = round_up(K//32, 8), buffer sm*sn, sm = round_up(M, 256).
``shuffle=False`` writes the flat (M, K//32) E8M0 scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.numeric import BFloat16, Float32
from flydsl.expr.numeric import Int32 as NInt32
from flydsl.expr.typing import Int32, T
from flydsl.expr.vector import ReductionOp, Vector, full
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

if TYPE_CHECKING:
    from torch import Tensor

BLOCK_THREADS = 256
FP8_MAX = 448.0  # torch.finfo(float8_e4m3fn).max


def build_mxfp8_1x32_quant_module(K: int, shuffle: bool = True) -> Any:
    """Return a JIT launcher for per-1x32 MXFP8 quant of an (M, K) bf16 tensor.

    launch_fn(x, q_out, s_out, M, sn, stream):
      x     : (M, K) bf16 buffer
      q_out : (M, K) fp8_e4m3fn bytes
      s_out : E8M0 scale bytes -- tiled (sm*sn) if shuffle else flat (M, K//32)
      M, sn : runtime i32 (sn = round_up(K//32, 8); unused when not shuffle)
    """
    assert K % 32 == 0, f"K={K} must be divisible by 32"
    NB = K // 32  # blocks per row
    NBc = (NB + BLOCK_THREADS - 1) // BLOCK_THREADS  # blocks per thread (constexpr)

    @flyc.kernel
    def quant_kernel(
        x: fx.Tensor,
        q_out: fx.Tensor,
        s_out: fx.Tensor,
        M: fx.Int32,
        sn: fx.Int32,
    ) -> None:
        bid = fx.block_idx.x  # row index
        tid = fx.thread_idx.x

        x_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
        q_rsrc = buffer_ops.create_buffer_resource(q_out, max_size=True)
        s_rsrc = buffer_ops.create_buffer_resource(s_out, max_size=True)

        row = ArithValue(bid)
        row_base_elem = row * fx.Int32(K)  # element offset of row in x/q

        abs_mask8 = full(8, Int32(0x7FFFFFFF), Int32)
        c_1e30 = arith.constant(1e-30, type=T.f32)
        c0_i32 = arith.constant(0, type=T.i32)
        c1_f32 = arith.constant(1.0, type=T.f32)
        fp8_max_f = arith.constant(FP8_MAX, type=T.f32)
        neg_fp8_max_f = arith.constant(-FP8_MAX, type=T.f32)
        inv_fp8_max = arith.constant(1.0 / FP8_MAX, type=T.f32)

        # Row-block term of the tiled scale offset (constant across this thread's
        # blocks): (row//32)*(sn*32) + (row//16)%2 + (row%16)*4
        if const_expr(shuffle):
            sn_v = ArithValue(sn)
            r_off = (
                (row // fx.Int32(32)) * (sn_v * fx.Int32(32))
                + (row // fx.Int32(16)) % fx.Int32(2)
                + (row % fx.Int32(16)) * fx.Int32(4)
            )

        c_0xFF = arith.constant(0xFF, type=T.i32)
        for j in range_constexpr(NBc):
            blk = ArithValue(tid) + fx.Int32(j * BLOCK_THREADS)  # block idx in row
            # Out-of-range threads (NB not a multiple of BLOCK_THREADS) must not
            # store. Guard with a runtime branch (FlyDSL lowers this to scf.if).
            # NBc==1 and NB%BLOCK_THREADS==0 -> no guard needed (constexpr skip).
            needs_guard = const_expr(not (NBc == 1 and NB % BLOCK_THREADS == 0))
            do_block = arith.cmpi(arith.CmpIPredicate.ult, blk, fx.Int32(NB)) if needs_guard else True
            if do_block:
                elem0 = row_base_elem + blk * fx.Int32(32)  # first elem of block

                # ---- load 32 bf16 as 4 x (8-wide bf16 = 128b) -> f32 ----
                sub = []  # 4 Vectors of 8 f32
                for t in range_constexpr(4):
                    raw = buffer_ops.buffer_load(x_rsrc, elem0 + fx.Int32(t * 8), vec_width=8, dtype=T.bf16)
                    sub.append(Vector(raw, 8, BFloat16).to(Float32))

                # ---- per-block amax (bit-exact abs via mask) ----
                amax = c_1e30
                for t in range_constexpr(4):
                    vabs = (sub[t].bitcast(NInt32) & abs_mask8).bitcast(Float32)
                    cmax = ArithValue(vabs.reduce(ReductionOp.MAX).ir_value())
                    amax = amax.maximumf(cmax)

                # ---- E8M0 scale, bit-exact to _e8m0_from_fp32(amax/FP8_MAX) ----
                ratio = amax * ArithValue(inv_fp8_max)
                u = ratio.bitcast(T.i32)
                exp = (u >> fx.Int32(23)) & fx.Int32(0xFF)
                # ceil, not round-to-nearest: any nonzero mantissa bumps the exponent.
                # Rounding down makes the scale too small, so amax/scale exceeds 448 and
                # the clamp truncates -- a toward-zero bias cosine cannot detect.
                rc = ((u & fx.Int32(0x7FFFFF)) > ArithValue(c0_i32)) & (exp < ArithValue(c_0xFF))
                exp = ArithValue(arith.select(rc, exp + fx.Int32(1), exp))
                exp = ArithValue(arith.minui(exp, c_0xFF))

                # scale = 2^(exp-127) as bit pattern (exp<<23), clamp_min 1e-30
                scale_f = ArithValue((exp << fx.Int32(23)).bitcast(T.f32)).maximumf(c_1e30)
                inv_scale = ArithValue(c1_f32) / scale_f

                # ---- quantize: each 8-wide sub -> 2 dwords (8 fp8 bytes) ----
                for t in range_constexpr(4):
                    vt = sub[t]
                    for w in range_constexpr(2):  # 2 dwords per 8 elems
                        packed = arith.constant(0, type=T.i32)
                        for p in range_constexpr(2):
                            ea = ArithValue(
                                vector.extract(vt.ir_value(), static_position=[w * 4 + p * 2], dynamic_position=[])
                            )
                            eb = ArithValue(
                                vector.extract(vt.ir_value(), static_position=[w * 4 + p * 2 + 1], dynamic_position=[])
                            )
                            sa = ea * inv_scale
                            sb = eb * inv_scale
                            sa = arith.maxnumf(arith.minnumf(sa, fp8_max_f), neg_fp8_max_f)
                            sb = arith.maxnumf(arith.minnumf(sb, fp8_max_f), neg_fp8_max_f)
                            packed = rocdl.cvt_pk_fp8_f32(T.i32, sa, sb, packed, p)
                        # byte offset (fp8=1B): elem0 + t*8 + w*4
                        store_off = elem0 + fx.Int32(t * 8 + w * 4)
                        buffer_ops.buffer_store(
                            packed,
                            q_rsrc,
                            store_off,
                            offset_is_bytes=True,
                        )

                # ---- store E8M0 scale byte ----
                e8 = arith.trunci(T.i8, exp)
                if const_expr(shuffle):
                    c = blk
                    c_off = (
                        (c // fx.Int32(8)) * fx.Int32(256)
                        + (c // fx.Int32(4)) % fx.Int32(2) * fx.Int32(2)
                        + (c % fx.Int32(4)) * fx.Int32(64)
                    )
                    s_off = r_off + c_off
                else:
                    s_off = row * fx.Int32(NB) + blk
                buffer_ops.buffer_store(
                    e8,
                    s_rsrc,
                    s_off,
                    offset_is_bytes=True,
                )

    @flyc.jit
    def launch_quant(
        x: fx.Tensor,
        q_out: fx.Tensor,
        s_out: fx.Tensor,
        M: fx.Int32,
        sn: fx.Int32,
        # FlyDSL's @flyc.jit launcher convention; the default is part of the traced
        # signature, so it cannot be hoisted to a module-level singleton.
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ) -> None:
        idx_rows = ArithValue(M).index_cast(T.index)
        quant_kernel(x, q_out, s_out, M, sn).launch(
            grid=(idx_rows, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_quant


_LAUNCH_CACHE: dict[tuple[int, bool], Any] = {}
# Compiled fast-path callables (flyc.compile) keyed by (K, shuffle). flyc.compile
# returns a CompiledFunction that reuses the pre-built CallState and skips the
# per-call signature bind / globals-drift / cache-key work in flyc.jit.__call__
# (~150 us/call -> ~5 us/call). K is a compile-time constant of the kernel; only
# the runtime args (pointers, M, sn, stream) change between calls.
_COMPILED_CACHE: dict[tuple[int, bool], Any] = {}


def quantize_mxfp8_1x32_flydsl(x: Tensor, shuffle: bool = True) -> tuple[Tensor, Tensor]:
    """x: (M,K) bf16 CUDA tensor, K%32==0. Returns (q uint8 (M,K), s uint8).

    s is the tiled (sm,sn) scale when shuffle=True (drop-in for the FlyDSL A8W4
    kernel; sm=round_up(M,256), sn=round_up(K//32,8)), else flat (M, K//32).
    """
    import torch

    assert x.is_cuda and x.dim() == 2 and x.shape[-1] % 32 == 0
    if str(get_hip_arch()) != "gfx950":
        raise RuntimeError(f"FlyDSL MXFP8 quant requires gfx950, got {get_hip_arch()}")
    x = x.contiguous()
    M, K = x.shape
    NB = K // 32
    q = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    if shuffle:
        sm = (M + 255) // 256 * 256
        sn = (NB + 7) // 8 * 8
        # The kernel writes all M rows; the row-padding tail [M, sm) is never read
        # by the A8W4 GEMM (which pre-pads its activation to m_pad <= sm before
        # quant), so torch.empty is correct and avoids a full-buffer zero-fill.
        # NB%8==0 for Wan shapes -> no column padding to worry about.
        s = torch.empty(sm * sn, device=x.device, dtype=torch.uint8)
    else:
        sm, sn = M, NB
        s = torch.empty(M, NB, device=x.device, dtype=torch.uint8)

    q_u8 = q.view(torch.uint8)
    s_flat = s.view(-1)
    stream = torch.cuda.current_stream()

    compiled = _COMPILED_CACHE.get((K, shuffle))
    if compiled is not None:
        compiled(x, q_u8, s_flat, int(M), int(sn), stream)
        return q_u8, (s.view(sm, sn) if shuffle else s)

    launch = _LAUNCH_CACHE.get((K, shuffle))
    if launch is None:
        launch = build_mxfp8_1x32_quant_module(K, shuffle=shuffle)
        _LAUNCH_CACHE[(K, shuffle)] = launch

    # First call for this (K, shuffle): build the flyc.compile fast-path callable.
    # flyc.compile runs the jit once to compile, then returns a CompiledFunction
    # reusing the CallState. Fall back to the plain launcher if compile is absent.
    try:
        import flydsl.compiler as _flyc

        compiled = _flyc.compile(launch, x, q_u8, s_flat, int(M), int(sn), stream)
        _COMPILED_CACHE[(K, shuffle)] = compiled
    except Exception:  # noqa: BLE001 - fall back to plain jit launcher
        launch(x, q_u8, s_flat, int(M), int(sn), stream)

    return q_u8, (s.view(sm, sn) if shuffle else s)
