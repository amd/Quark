#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for quark.experimental.torch.quant_perf.orchestration.intake: compute_arch_fingerprint and load_model_config."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quark.experimental.torch.quant_perf.cli import build_parser
from quark.experimental.torch.quant_perf.orchestration.intake import (
    build_spec,
    compute_arch_fingerprint,
    load_model_config,
)

QWEN_LIKE_CONFIG = {
    "model_type": "qwen2",
    "num_hidden_layers": 28,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
}


def test_fingerprint_is_deterministic():
    a = compute_arch_fingerprint(QWEN_LIKE_CONFIG)
    b = compute_arch_fingerprint(dict(QWEN_LIKE_CONFIG))
    assert a == b
    assert a.startswith("v2-")
    assert len(a) == 19


def test_fingerprint_differs_by_model_type():
    other = dict(QWEN_LIKE_CONFIG, model_type="llama")
    assert compute_arch_fingerprint(QWEN_LIKE_CONFIG) != compute_arch_fingerprint(other)


def test_fingerprint_same_family_different_size_differs():
    """8B vs 70B of the same family should not collide."""
    small = QWEN_LIKE_CONFIG
    large = dict(QWEN_LIKE_CONFIG, num_hidden_layers=80, hidden_size=8192, intermediate_size=28672)
    assert compute_arch_fingerprint(small) != compute_arch_fingerprint(large)


def test_fingerprint_moe_flag():
    dense = QWEN_LIKE_CONFIG
    moe = dict(QWEN_LIKE_CONFIG, num_experts=8)
    assert compute_arch_fingerprint(dense) != compute_arch_fingerprint(moe)


def test_load_model_config_from_local_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(QWEN_LIKE_CONFIG))
    loaded = load_model_config(str(tmp_path))
    assert loaded == QWEN_LIKE_CONFIG


def test_build_spec_records_model_arch_family(tmp_path):
    config = dict(QWEN_LIKE_CONFIG, model_type="qwen3_5_moe")
    (tmp_path / "config.json").write_text(json.dumps(config))

    spec = build_spec(build_parser().parse_args(["--model", str(tmp_path)]))

    assert spec.model_arch == "qwen3_5_moe"


def test_build_spec_prefers_top_level_model_family(tmp_path):
    config = dict(
        QWEN_LIKE_CONFIG,
        model_type="qwen3_5_moe",
        text_config={"model_type": "qwen3_5_moe_text"},
    )
    (tmp_path / "config.json").write_text(json.dumps(config))

    spec = build_spec(build_parser().parse_args(["--model", str(tmp_path)]))

    assert spec.model_arch == "qwen3_5_moe"


def test_fingerprint_uses_nested_text_config_and_distinguishes_model_sizes():
    small = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "num_experts": 256,
        },
    }
    large = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 4096,
            "num_hidden_layers": 60,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "num_experts": 512,
        },
    }

    small_fingerprint = compute_arch_fingerprint(small)
    large_fingerprint = compute_arch_fingerprint(large)

    assert small_fingerprint.startswith("v2-")
    assert large_fingerprint.startswith("v2-")
    assert small_fingerprint != large_fingerprint
