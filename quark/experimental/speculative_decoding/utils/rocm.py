#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""AMD/ROCm environment helpers.

Centralizes optional vLLM-ROCm environment wiring for AITER MoE and attention
backends. Apply these settings before launching a compatible vLLM serve for
extraction or deployment.
"""

from __future__ import annotations

import os

# Conservative defaults shared with the packaged runner. MoE-specific fast
# paths vary by target and quantization, so a model adapter (or ``extra``)
# supplies those rather than this module asserting one model's tuning.
ROCM_DEFAULTS: dict[str, str] = {
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_MOE": "0",
    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "0",
}


def is_rocm() -> bool:
    try:
        import torch

        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def apply_rocm_env(extra: dict[str, str] | None = None, attention_backend: str | None = None) -> dict[str, str]:
    """Set (and return) the ROCm env vars for a vLLM serve. No-op keys aren't overridden."""
    env = dict(ROCM_DEFAULTS)
    if attention_backend:
        env["VLLM_ATTENTION_BACKEND"] = attention_backend
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env
