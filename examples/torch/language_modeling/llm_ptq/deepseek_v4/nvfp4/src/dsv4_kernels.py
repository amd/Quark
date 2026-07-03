#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Portions of this file are translated from the DeepSeek-V4-Pro checkpoint's
# tilelang kernels (inference/kernel.py).
# Copyright (c) 2023 DeepSeek. Licensed under the MIT License.
#

"""
Triton / PyTorch kernels for DeepSeek-V4-Pro NVFP4 input-scale calibration.

These kernels are translated directly from the DeepSeek-V4-Pro checkpoint's
tilelang kernels (``inference/kernel.py``, Copyright (c) 2023 DeepSeek, MIT) into
triton / PyTorch, so the calibration script does not depend on tilelang. The
control flow and numerics mirror the tilelang originals:

* ``fp8_gemm`` / ``sparse_attn``: triton (FP8 tensor-core), translated directly
  from the tilelang kernels and aligned to within 1 bf16 ULP on real checkpoint
  shapes (median relative error 0).
* ``act_quant`` / ``fp4_act_quant`` / ``hc_split_sinkhorn``: pure-PyTorch,
  bitwise-aligned to tilelang.
* ``fp4_gemm``: pure-PyTorch fallback (not exercised during calibration -- routed
  experts go through the NativeLinear bf16 path).

Importing this module registers a synthetic ``kernel`` module and a
``fast_hadamard_transform`` stub into ``sys.modules`` so the checkpoint's
``model.py`` picks up these implementations instead of the tilelang ones.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
import types as _types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fp8_gemm_triton_kernel(
    A,
    B,
    C,
    Sa,
    Sb,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_sam,
    stride_sak,
    stride_sbn,
    stride_sbk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[M,N] = A_fp8[M,K] @ B_fp8[N,K]^T, per-128 block scale on both, fp32 accum.

    Mirrors the checkpoint's tilelang ``fp8_gemm_kernel``: per K-block FP8 dot with
    fp32 accumulation, the (scale_a * scale_b) applied per 128-K-block, bf16 output.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            A + offs_m[:, None] * stride_am + (kb * BLOCK_K + offs_k)[None, :] * stride_ak,
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        b = tl.load(
            B + offs_n[:, None] * stride_bn + (kb * BLOCK_K + offs_k)[None, :] * stride_bk,
            mask=offs_n[:, None] < N,
            other=0.0,
        )
        c_local = tl.dot(a, b.T, out_dtype=tl.float32)
        sa = tl.load(Sa + offs_m * stride_sam + kb * stride_sak, mask=offs_m < M, other=0.0).to(tl.float32)
        sb = tl.load(Sb + pid_n * stride_sbn + kb * stride_sbk).to(tl.float32)
        acc += c_local * (sa[:, None] * sb)
    c = acc.to(tl.bfloat16)
    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _sparse_attn_triton_kernel(
    Q,
    KV,
    Out,
    SINK,
    TOPK,
    b,
    m,
    n,
    h,
    d,
    topk,
    sq_b,
    sq_m,
    sq_h,
    sq_d,
    skv_b,
    skv_n,
    skv_d,
    so_b,
    so_m,
    so_h,
    so_d,
    st_b,
    st_m,
    st_k,
    scale,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sparse attention via topk-gather + online softmax + attn_sink.

    Translated from the checkpoint's tilelang ``sparse_attn_kernel`` but tiled over
    heads: one program per (seq_pos, batch, head-block of ``BLOCK_H``). For each
    KV block (``BLOCK`` positions) gather rows by ``topk_idxs`` (idx == -1 is a
    masked/zero row), run FlashAttention-style running max/sum, and add the
    learnable ``attn_sink`` bias to the denominator. Head-tiling keeps the per-
    program tile small enough for the real DSV4 shape (128 heads x 512 head_dim).
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    offs_blk = tl.arange(0, BLOCK)
    h_mask = offs_h < h

    q_ptr = Q + pid_b * sq_b + pid_m * sq_m + offs_h[:, None] * sq_h + offs_d[None, :] * sq_d
    q = tl.load(q_ptr, mask=h_mask[:, None] & (offs_d[None, :] < d), other=0.0).to(tl.bfloat16)

    acc_o = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_H,), dtype=tl.float32)
    scores_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)

    for t in range(tl.cdiv(topk, BLOCK)):
        kpos = t * BLOCK + offs_blk
        valid = kpos < topk
        idx = tl.load(TOPK + pid_b * st_b + pid_m * st_m + kpos * st_k, mask=valid, other=-1)
        idx_valid = idx != -1
        kv_ptr = KV + pid_b * skv_b + idx[:, None] * skv_n + offs_d[None, :] * skv_d
        kv = tl.load(kv_ptr, mask=idx_valid[:, None] & (offs_d[None, :] < d), other=0.0).to(tl.bfloat16)
        acc_s = tl.dot(q, kv.T, out_dtype=tl.float32) * scale
        acc_s = tl.where(idx_valid[None, :], acc_s, -float("inf"))

        scores_max_prev = scores_max
        scores_max = tl.maximum(scores_max_prev, tl.max(acc_s, axis=1))
        scores_scale = tl.exp(scores_max_prev - scores_max)
        acc_s = tl.exp(acc_s - scores_max[:, None])
        sum_exp = sum_exp * scores_scale + tl.sum(acc_s, axis=1)
        acc_o = acc_o * scores_scale[:, None]
        acc_o += tl.dot(acc_s.to(kv.dtype), kv, out_dtype=tl.float32)

    sink = tl.load(SINK + offs_h, mask=h_mask, other=0.0)
    sum_exp += tl.exp(sink - scores_max)
    o = acc_o / sum_exp[:, None]
    o_ptr = Out + pid_b * so_b + pid_m * so_m + offs_h[:, None] * so_h + offs_d[None, :] * so_d
    tl.store(o_ptr, o.to(tl.bfloat16), mask=h_mask[:, None] & (offs_d[None, :] < d))


# ---------------------------------------------------------------------------
# Kernel fallbacks
# ---------------------------------------------------------------------------


def _hadamard_transform_pt(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    orig_dtype = x.dtype
    n = x.shape[-1]
    assert n & (n - 1) == 0
    h = x.float()
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            a = h[..., i : i + step]
            b = h[..., i + step : i + step * 2]
            h[..., i : i + step] = a + b
            h[..., i + step : i + step * 2] = a - b
        step *= 2
    return (h * scale).to(orig_dtype)


# FP4 (e2m1) code points, indexed by the 4-bit nibble value (sign bit = high bit).
_FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# Non-negative FP4 (e2m1) magnitudes and whether each is "even" (mantissa bit 0),
# used for round-half-to-even tie-breaking to match the tilelang FP4 cast.
_FP4_E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_FP4_E2M1_POS_IS_EVEN = torch.tensor([1, 0, 1, 0, 1, 0, 1, 1], dtype=torch.bool)


def _round_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` to the nearest FP4 (e2m1) value, ties to even (IEEE default).

    Mirrors the tilelang ``T.Cast(FP4, ...)`` rounding (verified against the kernel:
    0.25->0, 0.75->1.0, 1.25->1.0, 1.75->2.0, 2.5->2.0, 3.5->4.0). The FP4 grid is
    non-uniform, so this rounds magnitudes against the positive code points with an
    explicit ties-to-even rule, then restores the sign.

    :param torch.Tensor x: Input values (already divided by the block scale).

    :return: Tensor of FP4 grid values (same shape as ``x``).
    """
    pos = _FP4_E2M1_POS.to(x.device)
    is_even = _FP4_E2M1_POS_IS_EVEN.to(x.device)
    mag = x.abs()
    dist = (mag.unsqueeze(-1) - pos).abs()
    # Nudge non-even code points by a tiny epsilon so that, on an exact tie, the
    # even code point wins (round-half-to-even).
    dist = dist + (~is_even).float() * 1e-6
    nearest = dist.argmin(dim=-1)
    return torch.sign(x) * pos[nearest]


def _pow2_round_scale(amax: torch.Tensor, fmt_max_inv: float) -> torch.Tensor:
    """Round a block amax to a power-of-2 (e8m0) scale: ``2 ** ceil(log2(amax * max_inv))``.

    Matches the checkpoint's ``fast_round_scale`` (IEEE-754 bit-trick) used by the
    tilelang act/fp4 quant kernels when ``scale_fmt`` is set (MXFP power-of-2 scales).

    :param torch.Tensor amax: Per-block absolute max (float32).
    :param float fmt_max_inv: Reciprocal of the target format max (1/448 FP8, 1/6 FP4).

    :return: Power-of-2 scale tensor (float32).
    """
    return torch.exp2(torch.ceil(torch.log2(amax * fmt_max_inv)))


def _quantize_fp8_blockwise(x: torch.Tensor, block_size: int, round_scale: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8-E4M3 quantization along the last dim.

    Faithfully mirrors the checkpoint's tilelang ``act_quant``: per-block
    ``amax = max(|x|, 1e-4)``; scale is ``amax/448`` (or the power-of-2 rounded
    form when ``round_scale``); values are cast to float8_e4m3fn.

    :param torch.Tensor x: Input activation.
    :param int block_size: Elements per quantization block along the last dim.
    :param bool round_scale: Round the block scale to a power of 2 (MXFP) when True.

    :return: Tuple ``(q_fp8, scale)`` where ``scale`` is one FP32 value per block.
    """
    fp8_max = 448.0
    n = x.size(-1)
    blocks = x.float().unflatten(-1, (n // block_size, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = _pow2_round_scale(amax, 1.0 / fp8_max) if round_scale else amax / fp8_max
    q = torch.clamp(blocks / scale, -fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q.flatten(-2), scale.squeeze(-1)


def _dequantize_fp8_blockwise(q_fp8: torch.Tensor, scale: torch.Tensor, block_size: int) -> torch.Tensor:
    """Dequantize a block-wise FP8 tensor back to float32.

    :param torch.Tensor q_fp8: FP8-E4M3 quantized values.
    :param torch.Tensor scale: One FP32 scale per block (last dim = num blocks).
    :param int block_size: Elements per block along the last dim.

    :return: Float32 dequantized tensor of the same shape as ``q_fp8``.
    """
    n = q_fp8.size(-1)
    blocks = q_fp8.float().unflatten(-1, (n // block_size, block_size))
    return (blocks * scale.unsqueeze(-1)).flatten(-2)


def _dequantize_e8m0(scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Decode an E8M0 (uint8 exponent) block scale to float32: ``2 ** (byte - 127)``."""
    return torch.pow(2.0, scale_e8m0.view(torch.uint8).float() - 127.0)


def act_quant(x, block_size, scale_fmt=None, scale_dtype=None, inplace=False):
    """Block-wise FP8-E4M3 activation quantization (pure-PyTorch ``act_quant``).

    Two call conventions match the checkpoint's tilelang kernel:

    * ``inplace=False`` (expert ``linear`` path): returns ``(q_fp8, scale)`` to be
      consumed by :func:`fp8_gemm` / :func:`fp4_gemm`.
    * ``inplace=True`` (attention KV path): a fused quant+dequant write-back.

    .. note::
       The tilelang ``act_quant`` kernel's ``inplace=True`` + power-of-2 scale
       (``scale_fmt`` set) path is a no-op in the checkpoint's tilelang build: it
       computes the block scale but writes the *unquantized* input straight back
       (verified by direct kernel comparison -- output is bit-identical to the
       input). To stay numerically aligned with that real behaviour we mirror it:
       the inplace e8m0 path leaves ``x`` unchanged. The non-inplace expert path
       (``scale_fmt=None``) does perform real FP8 quantization, matching tilelang.

    ``scale_fmt`` non-None selects power-of-2 (e8m0) block scales.
    """
    round_scale = scale_fmt is not None
    n = x.size(-1)
    if n % block_size != 0:
        return (x, None) if not inplace else x

    if inplace:
        # Mirror the tilelang inplace+e8m0 no-op (see note above).
        return x

    q, scale = _quantize_fp8_blockwise(x, block_size, round_scale)
    return q, scale


def fp4_act_quant(x, block_size, inplace=False):
    """Block-wise FP4 (e2m1) activation quantization with a power-of-2 block scale.

    Pure-PyTorch mirror of the checkpoint's tilelang ``fp4_act_quant``. Only the
    ``inplace=True`` fused quant+dequant-to-input-dtype path is used by the model.
    """
    if not inplace:
        return None

    fp4_max = 6.0
    n = x.size(-1)
    if n % block_size != 0:
        return None
    blocks = x.float().unflatten(-1, (n // block_size, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(6 * (2.0**-126))
    scale = _pow2_round_scale(amax, 1.0 / fp4_max)
    scaled = torch.clamp(blocks / scale, -fp4_max, fp4_max)
    deq = (_round_to_fp4_e2m1(scaled) * scale).flatten(-2).to(x.dtype)
    x.copy_(deq)
    return x


def quant_dequant_nvfp4_act(x: torch.Tensor, input_scale: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Fused quant+dequant of an activation to NVFP4 (returns the same dtype as ``x``).

    NVFP4 activation quantization is two-level and symmetric to the weight side
    (:func:`dequant_nvfp4_weight`, where ``eff_scale = wscale_fp8 * wscale2``):

    * a **dynamic per-group** (``group_size`` = 16) micro-scale stored in FP8-E4M3
      — the activation counterpart of ``wscale_fp8``; and
    * a **static per-tensor** F32 global scale ``input_scale`` (calibrated in
      Stage 2) — the counterpart of ``wscale2``.

    For each group the effective scale is ``fp8(group_amax / 6 / input_scale) *
    input_scale``; values are rounded onto the FP4 (e2m1) grid against it and
    dequantized back. This is what an actual NVFP4 inference does to activations.

    :param torch.Tensor x: Activation, ``[..., in]`` with ``in % group_size == 0``.
    :param torch.Tensor input_scale: F32 per-tensor scale (scalar).
    :param int group_size: FP4 group size (16 for NVFP4).

    :return: ``x`` fake-quantized to NVFP4, cast back to ``x.dtype``.
    """
    fp4_max = 6.0
    fp8_max = 448.0
    n = x.size(-1)
    if n % group_size != 0:
        raise ValueError(f"activation inner dim {n} not divisible by group_size {group_size}")
    s2 = input_scale.float().to(x.device).clamp_min(1e-30)
    blocks = x.float().unflatten(-1, (n // group_size, group_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    # Dynamic per-group micro-scale, expressed relative to the global scale and
    # quantized to FP8-E4M3 (clamp before the cast: e4m3fn has no inf, max 448).
    group_scale = torch.clamp(amax / fp4_max / s2, max=fp8_max)
    group_scale_fp8 = group_scale.to(torch.float8_e4m3fn).float()
    eff = (group_scale_fp8 * s2).clamp_min(1e-30)  # [..., in/16, 1]
    code = _round_to_fp4_e2m1(torch.clamp(blocks / eff, -fp4_max, fp4_max))
    return (code * eff).flatten(-2).to(x.dtype)


def fp8_gemm(x, x_scale, weight, w_scale, scale_dtype):
    """C = A_fp8 @ B_fp8^T with per-128 block scaling on both, FP32 accumulation.

    Triton mirror of the checkpoint's tilelang ``fp8_gemm``: the activation is
    already FP8 (from :func:`act_quant`); a per-128-block-scaled FP8 tensor-core
    matmul produces a bf16 output matching the kernel.
    """
    if x_scale is None or weight.dtype != torch.float8_e4m3fn:
        # Activation was not pre-quantized (e.g. attention BF16 path): fall back to
        # a blockwise-dequantized weight matmul in BF16.
        w = weight.to(torch.bfloat16)
        if w_scale is not None and not (hasattr(w_scale, "is_meta") and w_scale.is_meta):
            s = _dequantize_e8m0(w_scale).to(w.device)
            out_b, in_b = s.shape
            s = s.unsqueeze(-1).unsqueeze(-1).expand(out_b, in_b, 128, 128)
            s = s.permute(0, 2, 1, 3).reshape(out_b * 128, in_b * 128)[: w.shape[0], : w.shape[1]]
            w = (w * s).to(torch.bfloat16)
        return F.linear(x.to(torch.bfloat16), w)

    k = x.size(-1)
    a = x.reshape(-1, k).contiguous()  # [M, K] fp8
    m = a.shape[0]
    n = weight.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    sa = x_scale.reshape(m, -1).contiguous()  # [M, K/128] fp32 power-of-2
    sb = _dequantize_e8m0(w_scale).contiguous()  # [N/128, K/128] fp32
    block_m, block_n, block_k = 32, 128, 128
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fp8_gemm_triton_kernel[grid](
        a,
        weight,
        out,
        sa,
        sb,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        sa.stride(0),
        sa.stride(1),
        sb.stride(0),
        sb.stride(1),
        block_m,
        block_n,
        block_k,
    )
    return out.view(*x.shape[:-1], n)


def fp4_gemm(x, x_scale, weight, w_scale, scale_dtype):
    """C = A_fp8 @ B_fp4^T: FP8 activation (per-128 scale) x FP4 weight (per-32 e8m0).

    Pure-PyTorch mirror of the tilelang ``fp4_gemm``: dequantize the FP8 activation
    and the packed FP4 weight blockwise to float32 and matmul in float32.
    """
    block_size = 128
    fp4_lut = _FP4_E2M1_LUT.to(weight.device)
    raw = weight.view(torch.uint8)
    lo = (raw & 0x0F).long()
    hi = ((raw >> 4) & 0x0F).long()
    w = torch.stack([fp4_lut[lo], fp4_lut[hi]], dim=-1).flatten(-2)  # [N, K] float32
    if w_scale is not None:
        s = _dequantize_e8m0(w_scale).to(w.device)
        s = s.repeat_interleave(32, dim=-1)
        if s.shape[1] > w.shape[1]:
            s = s[:, : w.shape[1]]
        w = w * s

    if x_scale is None:
        # Activation not pre-quantized: BF16 matmul fallback.
        return F.linear(x.to(torch.bfloat16), w.to(torch.bfloat16))

    k = x.size(-1)
    a_deq = _dequantize_fp8_blockwise(x, x_scale, block_size)
    out = torch.matmul(a_deq, w[:, :k].t())
    return out.to(torch.bfloat16).view(*x.shape[:-1], w.shape[0])


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Sparse attention (triton), mirroring the checkpoint's tilelang ``sparse_attn``."""
    b, s, h, d = q.shape
    n = kv.shape[1]
    topk = topk_idxs.shape[-1]
    out_dtype = q.dtype
    q = q.to(torch.bfloat16).contiguous()
    kv = kv.to(torch.bfloat16).contiguous()
    topk_idxs = topk_idxs.to(q.device).contiguous().int()
    attn_sink = attn_sink.to(q.device).contiguous().float()
    o = torch.empty_like(q)
    block_h = 16
    grid = (s, b, triton.cdiv(h, block_h))
    _sparse_attn_triton_kernel[grid](
        q,
        kv,
        o,
        attn_sink,
        topk_idxs,
        b,
        s,
        n,
        h,
        d,
        topk,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        topk_idxs.stride(2),
        softmax_scale,
        block_h,
        d,
        64,
    )
    return o.to(out_dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    b, s, _ = mixes.shape
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc].float() * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc : 2 * hc].float() * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :].float() * hc_scale[2] + hc_base[2 * hc :]).view(b, s, hc, hc)
    comb_max = comb.max(dim=-1, keepdim=True).values
    comb = torch.exp(comb - comb_max)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre.to(mixes.dtype), post.to(mixes.dtype), comb.to(mixes.dtype)


# ---------------------------------------------------------------------------
# Weight dequant helpers (compact on-disk quant format -> BF16), shared by the
# input-scale calibration and the PPL eval scripts.
# ---------------------------------------------------------------------------


def dequant_nvfp4_weight(
    packed_u8: torch.Tensor,
    wscale_fp8: torch.Tensor,
    wscale2: torch.Tensor,
    device: torch.device,
    group_size: int = 16,
) -> torch.Tensor:
    """NVFP4 -> BF16.

    :param torch.Tensor packed_u8: ``[out, in/2]`` packed FP4 (2 codes per byte).
    :param torch.Tensor wscale_fp8: ``[out, in/16]`` per-group scale stored in FP8-E4M3.
    :param torch.Tensor wscale2: F32 per-tensor global scale.
    :param torch.device device: Target device for the dequantized BF16 weight.
    :param int group_size: FP4 group size (16 for NVFP4).

    :return: ``[out, in]`` BF16 weight.
    """
    raw = packed_u8.to(device)
    lut = _FP4_E2M1_LUT.to(device)
    lo = (raw & 0x0F).long()
    hi = ((raw >> 4) & 0x0F).long()
    w = torch.stack([lut[lo], lut[hi]], dim=-1).flatten(-2)  # [out, in] float32
    eff_scale = wscale_fp8.to(device).float() * wscale2.to(device).float()  # [out, in/16]
    eff_scale = eff_scale.repeat_interleave(group_size, dim=1)
    if eff_scale.shape[1] > w.shape[1]:
        eff_scale = eff_scale[:, : w.shape[1]]
    return (w * eff_scale).to(torch.bfloat16)


def dequant_mxfp4_weight(
    packed_i8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    device: torch.device,
    group_size: int = 32,
) -> torch.Tensor:
    """MXFP4 -> BF16 (the original checkpoint's routed-expert format).

    :param torch.Tensor packed_i8: ``[out, in/2]`` packed FP4 (2 codes per byte).
    :param torch.Tensor scale_e8m0: ``[out, in/32]`` per-group power-of-two block
        scale stored as UE8M0 (uint8 biased by 127).
    :param torch.device device: Target device for the dequantized BF16 weight.
    :param int group_size: FP4 group size along the (unpacked) inner dim (32).

    :return: ``[out, in]`` BF16 weight.
    """
    raw = packed_i8.view(torch.uint8).to(device)
    lut = _FP4_E2M1_LUT.to(device)
    lo = (raw & 0x0F).long()
    hi = ((raw >> 4) & 0x0F).long()
    w = torch.stack([lut[lo], lut[hi]], dim=-1).flatten(-2)  # [out, in] float32
    s = torch.pow(2.0, scale_e8m0.view(torch.uint8).to(device).float() - 127.0)
    s = s.repeat_interleave(group_size, dim=1)
    if s.shape[1] > w.shape[1]:
        s = s[:, : w.shape[1]]
    return (w * s).to(torch.bfloat16)


def dequant_fp8_block_weight(
    weight_fp8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    device: torch.device,
    block: int = 128,
) -> torch.Tensor:
    """FP8-E4M3 block-quant -> BF16.

    :param torch.Tensor weight_fp8: ``[out, in]`` FP8-E4M3 weight.
    :param torch.Tensor scale_e8m0: ``[out/128, in/128]`` power-of-two exponent
        stored as UE8M0 (uint8 biased by 127).
    :param torch.device device: Target device for the dequantized BF16 weight.
    :param int block: Block size (128).

    :return: ``[out, in]`` BF16 weight.
    """
    wf = weight_fp8.to(device).float()
    out_b, in_b = scale_e8m0.shape
    s = torch.pow(2.0, scale_e8m0.view(torch.uint8).to(device).float() - 127.0)
    s = s.unsqueeze(-1).unsqueeze(-1).expand(out_b, in_b, block, block)
    s = s.permute(0, 2, 1, 3).reshape(out_b * block, in_b * block)
    s = s[: wf.shape[0], : wf.shape[1]]
    return (wf * s).to(torch.bfloat16)


def register_kernels() -> None:
    """Register the synthetic ``kernel`` and ``fast_hadamard_transform`` modules.

    The checkpoint's ``model.py`` does ``from kernel import ...`` and
    ``import fast_hadamard_transform``; installing these into ``sys.modules`` makes
    it pick up the triton/PyTorch implementations here instead of tilelang.
    """
    kernel_module = _types.ModuleType("kernel")
    kernel_module.act_quant = act_quant
    kernel_module.fp4_act_quant = fp4_act_quant
    kernel_module.fp8_gemm = fp8_gemm
    kernel_module.fp4_gemm = fp4_gemm
    kernel_module.sparse_attn = sparse_attn
    kernel_module.hc_split_sinkhorn = hc_split_sinkhorn
    sys.modules["kernel"] = kernel_module

    fht_module = _types.ModuleType("fast_hadamard_transform")
    fht_module.hadamard_transform = _hadamard_transform_pt
    fht_module.__spec__ = _ilu.spec_from_loader("fast_hadamard_transform", loader=None)
    sys.modules["fast_hadamard_transform"] = fht_module


# Register on import so the checkpoint's model.py picks up these kernels.
register_kernels()
