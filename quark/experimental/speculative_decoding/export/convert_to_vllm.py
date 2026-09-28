#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Fold target metadata into an exported draft for one-argument vLLM deploy.

vLLM's EAGLE-3 speculative decoding needs to know the verifier (target) the
draft was trained against. This writes a ``vllm_speculative.json`` next to the
draft and (optionally) copies the draft into an ``out`` dir, so deployment is a
single ``--speculative-config`` pointing at one directory.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any


def convert_to_vllm(
    draft_hf: str,
    verifier: str | Any,
    out: str | None = None,
    num_speculative_tokens: int = 3,
) -> str:
    """Prepare a vLLM-ready draft directory.

    Args:
        draft_hf: path to the HF draft (output of :func:`export_hf`).
        verifier: target model path/repo id, or a loaded target with
            ``name_or_path``/``config._name_or_path``.
        out: optional destination dir (defaults to editing ``draft_hf`` in place).
        num_speculative_tokens: default NST to record for serving.
    """
    target_path: str | None
    if isinstance(verifier, str):
        target_path = verifier
    else:
        target_path = getattr(verifier, "name_or_path", None) or getattr(
            getattr(verifier, "config", None), "_name_or_path", None
        )
    if not target_path:
        raise ValueError("could not resolve the verifier (target) path from `verifier`.")

    dst = out or draft_hf
    if out and os.path.abspath(out) != os.path.abspath(draft_hf):
        shutil.copytree(draft_hf, out, dirs_exist_ok=True)

    spec = {
        "method": "eagle3",
        "model": os.path.abspath(dst),
        "target_model": target_path,
        "num_speculative_tokens": num_speculative_tokens,
        "draft_tensor_parallel_size": 1,
    }
    with open(os.path.join(dst, "vllm_speculative.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print(f"[convert_to_vllm] wrote vLLM speculative config -> {dst}/vllm_speculative.json")
    return dst
