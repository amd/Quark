#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Intake: turns CLI args into a fully-populated Spec.

Design ref: IMPL_SPEC §4.5 (compute_arch_fingerprint) and §5.1 (arch_fingerprint
is a "generated at runtime" field, not user input).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.orchestration.spec_factory import build_spec_from_args
from quark.experimental.torch.quant_perf.session.spec import Spec


def compute_arch_fingerprint(config_json: dict[str, Any]) -> str:
    """A versioned architecture fingerprint from the effective text config.

    Uses ratios rather than absolute values, so different sizes of the same
    family (8B/70B) get properly distinct fingerprints.
    """
    text_config = config_json.get("text_config") if isinstance(config_json.get("text_config"), dict) else {}
    effective = text_config or config_json
    hidden = max(effective.get("hidden_size", 1), 1)
    num_attention_heads = max(
        effective.get("num_attention_heads", 1),
        1,
    )
    fields = {
        "model_type": config_json.get("model_type", "unknown"),
        "text_model_type": effective.get("model_type", ""),
        "num_layers": effective.get("num_hidden_layers", 0),
        "hidden_size": hidden,
        # MLP ratio (more stable than an absolute intermediate_size)
        "mlp_ratio": round(
            effective.get("intermediate_size", hidden) / hidden,
            2,
        ),
        # GQA ratio (num_kv_heads / num_heads)
        "gqa_ratio": round(
            effective.get("num_key_value_heads", num_attention_heads) / num_attention_heads,
            2,
        ),
        # MoE
        "is_moe": bool(effective.get("num_experts") or effective.get("num_local_experts")),
        "num_experts": effective.get(
            "num_experts",
            effective.get("num_local_experts", 0),
        ),
    }
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]
    return f"v2-{digest}"


def load_model_config(model_dir: str) -> dict[str, Any]:
    """Loads a model's config.json without requiring a full `transformers`
    model load -- just the JSON dict compute_arch_fingerprint needs.

    `model_dir` is either a local directory (read config.json directly) or a
    HF hub repo id (downloaded via huggingface_hub, cached locally).
    """
    local_config = Path(model_dir) / "config.json"
    if local_config.exists():
        return json.loads(local_config.read_text())

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=model_dir, filename="config.json")
    return json.loads(Path(path).read_text())


def build_spec(args: argparse.Namespace) -> Spec:
    """The real entry point cli.py's main() should call: parses args into a
    Spec (via cli.build_spec) then fills in arch_fingerprint from the base
    model's real config.json."""
    spec = build_spec_from_args(args)
    config_json = load_model_config(spec.base_model)
    raw_text_config = config_json.get("text_config")
    text_config = raw_text_config if isinstance(raw_text_config, dict) else {}
    # The profile is resolved and frozen by the orchestrator after the session
    # directory exists, so evidence and optional LLM extraction are auditable.
    return replace(
        spec,
        model_arch=str(config_json.get("model_type") or text_config.get("model_type") or "unknown"),
        arch_fingerprint=compute_arch_fingerprint(config_json),
        eval_profile=None,
    )
