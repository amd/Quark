#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Streaming extraction: a live vLLM serve streams hidden states to the trainer.

This regime is intended for targets that cannot be co-located with the draft
trainer. A ``vllm serve`` produces target hidden states on the fly and streams
them to a distributed trainer, avoiding a large offline feature dump.

Streaming requires environment support that is *not* pure-pip:

* a vLLM-ROCm build whose worker exposes the hidden-state extraction hook
  (``extract_hidden_states``), and
* a transport (Mooncake, or a ROCm-native RCCL/shared-memory transport).

Rather than silently importing an unavailable transport, this class validates
the environment and raises an actionable error. The Qwen3-8B quick-start uses
``online`` extraction and does not need this path.
"""

from __future__ import annotations

import os
from typing import Any

import torch


class StreamingExtractor:
    def __init__(self, spec_model: Any, train_cfg: Any) -> None:
        self.spec = spec_model
        self.cfg = train_cfg
        self.endpoint = train_cfg.target_endpoint
        self._check_env()

    def _check_env(self) -> None:
        missing = []
        if not self.endpoint:
            missing.append("train_cfg.target_endpoint (a running vLLM-ROCm serve with the extraction hook)")
        try:
            import mooncake  # type: ignore[import-not-found]  # noqa: F401
        except Exception:
            if os.environ.get("QSD_STREAM_TRANSPORT", "mooncake") == "mooncake":
                missing.append(
                    "a hidden-state transport: `pip`/build `mooncake` (or set QSD_STREAM_TRANSPORT and "
                    "provide an RCCL/shared-memory transport). See docs/streaming.md."
                )
        if missing:
            raise RuntimeError(
                "Streaming extraction is not available in this environment. Missing:\n  - "
                + "\n  - ".join(missing)
                + "\nUse extraction='online' (small/medium targets) or 'offline' instead."
            )

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # A concrete transport implementation plugs in here; kept explicit so the
        # import never fails for users on the online/offline paths.
        raise NotImplementedError(
            "Wire a concrete streaming transport (Mooncake or RCCL/shared-mem) to pull aux hidden "
            "states from the vLLM-ROCm serve. Tracked in docs/streaming.md."
        )
