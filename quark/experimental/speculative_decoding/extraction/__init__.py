#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Hidden-state extraction regimes: online / offline / streaming.

All three deliver the same training signal (the target's aux hidden states for
each position); they differ only in *where* the target runs and *how* the
features reach the trainer.
"""

from typing import Any

from quark.experimental.speculative_decoding.extraction.online import OnlineExtractor

__all__ = ["OnlineExtractor", "get_extractor"]


def get_extractor(mode: str, spec_model: Any, train_cfg: Any) -> Any:
    """Factory for the extractor selected by ``train_cfg.extraction``."""
    if mode == "online":
        return OnlineExtractor(spec_model)
    if mode == "offline":
        from quark.experimental.speculative_decoding.extraction.offline import OfflineExtractor

        return OfflineExtractor(spec_model, train_cfg)
    if mode == "streaming":
        from quark.experimental.speculative_decoding.extraction.streaming import StreamingExtractor

        return StreamingExtractor(spec_model, train_cfg)
    raise ValueError(f"unknown extraction mode: {mode!r} (use online|offline|streaming)")
