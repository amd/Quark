#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Data preparation: on-policy synthesis, dataset loading, draft-vocab calibration."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from quark.experimental.speculative_decoding.data.domain_manifest import (
    SUGGESTED_DOMAIN_NAMES,
    DomainManifest,
    DomainSource,
    build_domain_records,
    load_domain_manifest,
    normalized_sha256,
    parse_domain_manifest,
    provenance_license_summary,
    write_jsonl,
)

if TYPE_CHECKING:
    from quark.experimental.speculative_decoding.data.datasets import (
        ConversationDataset,
        load_conversations,
    )
    from quark.experimental.speculative_decoding.data.synth import synthesize
    from quark.experimental.speculative_decoding.data.vocab import calibrate_draft_vocab

_LAZY_ATTRS = {
    "ConversationDataset": (
        "quark.experimental.speculative_decoding.data.datasets",
        "ConversationDataset",
    ),
    "load_conversations": (
        "quark.experimental.speculative_decoding.data.datasets",
        "load_conversations",
    ),
    "synthesize": ("quark.experimental.speculative_decoding.data.synth", "synthesize"),
    "calibrate_draft_vocab": (
        "quark.experimental.speculative_decoding.data.vocab",
        "calibrate_draft_vocab",
    ),
}


def __getattr__(name: str) -> Any:
    """Load torch-backed data APIs only when callers request them."""
    if name not in _LAZY_ATTRS:
        raise AttributeError(name)
    module_name, attribute = _LAZY_ATTRS[name]
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "SUGGESTED_DOMAIN_NAMES",
    "ConversationDataset",
    "DomainManifest",
    "DomainSource",
    "build_domain_records",
    "calibrate_draft_vocab",
    "load_conversations",
    "load_domain_manifest",
    "normalized_sha256",
    "parse_domain_manifest",
    "provenance_license_summary",
    "synthesize",
    "write_jsonl",
]
