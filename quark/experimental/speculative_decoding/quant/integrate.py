#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark quantization integration.

The strongest real-world setup is *quantized target (MXFP4/FP8) + speculative
draft*. This module loads a Quark-quantized target as the verifier / hidden-state
source, and (optionally) provides the hook to PTQ/QAT-quantize the draft itself
via ``quark.torch``.
"""

from __future__ import annotations

from typing import Any

import torch

from quark.experimental.speculative_decoding.utils.logging import get_logger

logger = get_logger(__name__)


def load_target_verifier(
    model_path: str,
    quant: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> Any:
    """Load the target model to act as verifier + hidden-state source.

    If ``quant`` names a Quark-quantized checkpoint (e.g. ``"mxfp4"``/``"fp8"``) the
    model is rebuilt with its quantized modules via
    :func:`quark.torch.import_model_from_safetensors`; that path derives the dtype
    from the checkpoint config, so ``dtype`` applies to the HF fallback only.
    Otherwise -- and if that import fails -- we fall back to plain HF loading,
    which works for exported checkpoints that carry HF-readable quant metadata.
    The fallback is always logged: a quantized target loaded as plain BF16 is a
    silent accuracy/memory regression, so it must never pass unnoticed.
    """
    if quant and quant != "none":
        # device_map="auto" spreads a large MoE target over every visible GPU;
        # anything else is treated as an explicit single-device placement.
        multi_device = device_map in ("auto", "balanced")
        try:
            from quark.torch import import_model_from_safetensors

            return import_model_from_safetensors(
                None,
                model_path,
                multi_device=multi_device,
                trust_remote_code=trust_remote_code,
                device="cuda" if multi_device else (device_map or "cuda"),
            )
        except ImportError as e:
            logger.warning(
                "quark.torch.import_model_from_safetensors unavailable (%s); loading quant=%r "
                "target %s with plain transformers instead.",
                e,
                quant,
                model_path,
            )
        except Exception as e:  # noqa: BLE001 - fall back, but never silently
            logger.warning(
                "Quark import of quant=%r target %s failed (%s: %s); falling back to "
                "transformers. Verify the checkpoint really loads quantized.",
                quant,
                model_path,
                type(e).__name__,
                e,
            )
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust_remote_code
    )


def maybe_quantize_draft(draft: Any, quant: str | None) -> Any:
    """Optional draft PTQ/QAT via Quark. No-op unless a scheme is requested."""
    if not quant or quant == "none":
        return draft
    raise NotImplementedError(
        f"draft quant='{quant}' requested. Wire quark.torch PTQ/QAT here; the online/offline "
        "training path exports a BF16 draft by default, which is the validated Qwen3-8B recipe."
    )
