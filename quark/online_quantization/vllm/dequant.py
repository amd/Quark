"""Dequant dispatch for the online-requant path.

Given a layer whose parameters were allocated and loaded by an offline
``LinearMethodBase`` (e.g. vLLM's ``Fp8LinearMethod`` for DeepSeek-R1's
block-quantized FP8), reconstruct a high-precision ``(N, K)`` tensor that
can be fed into an online quantization method's
``process_weights_after_loading``.
"""

from typing import Any

import torch


def dequant_layer(layer: torch.nn.Module, offline_cfg: dict[str, Any]) -> torch.Tensor:
    """Reconstruct the original-dtype ``(N, K)`` weight from ``layer``."""
    method = offline_cfg.get("quant_method", "")
    if method == "fp8":
        block = offline_cfg.get("weight_block_size")
        if block is not None:
            return _dequant_fp8_block(
                layer.weight.data,
                layer.weight_scale_inv.data,
                block_shape=tuple(block),
                out_dtype=layer.orig_dtype,
            )
        return _dequant_fp8_per_channel(
            layer.weight.data,
            layer.weight_scale.data,
            out_dtype=layer.orig_dtype,
        )
    raise NotImplementedError(f"online requant: offline scheme '{method}' not supported")


def _dequant_fp8_block(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_shape: tuple[int, int],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Block-wise FP8 dequant (e.g. DeepSeek-V3/R1, block=128x128)."""
    n, k = weight_fp8.shape
    b0, b1 = block_shape
    w = weight_fp8.to(torch.float32)
    sc = scale_inv.to(torch.float32).repeat_interleave(b0, dim=0)[:n].repeat_interleave(b1, dim=1)[:, :k]
    return (w * sc).to(out_dtype)


def _dequant_fp8_per_channel(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Per-(output)channel FP8 dequant; ``weight_scale`` has shape (N,) or (N, 1)."""
    if weight_scale.dim() == 1:
        weight_scale = weight_scale.view(-1, 1)
    return (weight_fp8.to(torch.float32) * weight_scale.to(torch.float32)).to(out_dtype)


# --------------------------------------------------------------------------
# Per-expert variants used by the MoE re-quant path.
# --------------------------------------------------------------------------


def dequant_fp8_block_per_expert(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_shape: tuple[int, int],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Block-FP8 dequant for ``(num_experts, N, K)`` MoE weights.

    ``weight_fp8`` is ``(E, N, K) float8_e4m3fn``;
    ``scale_inv``  is ``(E, ceil(N/B0), ceil(K/B1)) float32`` (DeepSeek-V3
    layout). Returns ``(E, N, K)`` in ``out_dtype``.
    """
    e = weight_fp8.shape[0]
    out = torch.empty_like(weight_fp8, dtype=out_dtype)
    for i in range(e):
        out[i] = _dequant_fp8_block(weight_fp8[i], scale_inv[i], block_shape, out_dtype)
    return out
