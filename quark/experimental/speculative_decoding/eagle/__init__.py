#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""EAGLE-3 draft-head modeling, HF config, and training losses."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quark.experimental.speculative_decoding.eagle.config import Eagle3Config
    from quark.experimental.speculative_decoding.eagle.losses import ttt_loss
    from quark.experimental.speculative_decoding.eagle.modeling_eagle3 import Eagle3DraftModel

_LAZY_ATTRS = {
    "Eagle3Config": ("quark.experimental.speculative_decoding.eagle.config", "Eagle3Config"),
    "Eagle3DraftModel": (
        "quark.experimental.speculative_decoding.eagle.modeling_eagle3",
        "Eagle3DraftModel",
    ),
    "ttt_loss": ("quark.experimental.speculative_decoding.eagle.losses", "ttt_loss"),
}


def __getattr__(name: str) -> Any:
    """Load transformer-backed EAGLE APIs only when callers request them."""
    if name not in _LAZY_ATTRS:
        raise AttributeError(name)
    module_name, attribute = _LAZY_ATTRS[name]
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "Eagle3Config",
    "Eagle3DraftModel",
    "ttt_loss",
]
