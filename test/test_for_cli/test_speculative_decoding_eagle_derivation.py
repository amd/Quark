#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""CPU-only tests for public-config EAGLE-3 draft derivation.

Derivation lives in ``make_draft_config.py`` rather than in the library: the
runner invokes it inside a container that has vLLM and the trainer but not
Quark, so it cannot import from this package. These tests drive that script the
way the runner does.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "experimental"
    / "speculative_decoding"
    / "eagle3"
    / "common"
    / "scripts"
    / "make_draft_config.py"
)

MINIMAL_TEXT_CONFIG: dict[str, Any] = {
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 32_000,
}


def _derive(tmp_path: Path, config: dict[str, Any], *, name: str = "target") -> dict[str, Any]:
    target = tmp_path / name
    target.mkdir()
    (target / "config.json").write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / f"{name}.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--target", str(target), "--out", str(output)],
        check=True,
        capture_output=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def _aux_for_depth(tmp_path: Path, depth: int, *, name: str) -> list[int]:
    config = {**MINIMAL_TEXT_CONFIG, "num_hidden_layers": depth}
    return _derive(tmp_path, config, name=name)["eagle_aux_hidden_state_layer_ids"]


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (12, [2, 6, 9]),
        (32, [2, 16, 29]),
        (36, [2, 18, 33]),
        (60, [2, 30, 57]),
        (80, [2, 40, 77]),
    ],
)
def test_aux_layers_follow_the_published_eagle3_triple(tmp_path: Path, depth: int, expected: list[int]) -> None:
    """``[2, N // 2, N - 3]`` is the reference low/middle/high selection."""
    assert _aux_for_depth(tmp_path, depth, name=f"d{depth}") == expected


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (3, [0, 1, 2]),
        (4, [1, 2, 3]),
        (6, [1, 3, 4]),
    ],
)
def test_shallow_targets_fall_back_to_evenly_spaced_layers(tmp_path: Path, depth: int, expected: list[int]) -> None:
    """The reference triple collides below seven layers, so spacing takes over."""
    assert _aux_for_depth(tmp_path, depth, name=f"d{depth}") == expected


def test_aux_layers_reach_the_top_of_the_target(tmp_path: Path) -> None:
    """The high state must sit near the output for EAGLE-3 to fuse it usefully.

    An evenly spaced rule stops at three quarters of the depth and never
    contributes a near-final hidden state, which cost measurable acceptance
    length on a 60-layer target.
    """
    for depth in (12, 32, 36, 48, 60, 80, 126):
        aux = _aux_for_depth(tmp_path, depth, name=f"top{depth}")
        assert aux == sorted(set(aux))
        assert max(aux) < depth
        assert max(aux) >= depth - 3


def test_draft_keeps_target_width_geometry_and_vocabulary(tmp_path: Path) -> None:
    derived = _derive(tmp_path, {"text_config": MINIMAL_TEXT_CONFIG})

    assert derived["num_hidden_layers"] == 1
    assert derived["hidden_size"] == 2048
    assert derived["intermediate_size"] == 8192
    assert derived["num_attention_heads"] == 16
    assert derived["num_key_value_heads"] == 4
    assert derived["head_dim"] == 128
    assert derived["vocab_size"] == 32_000
    # No vocabulary compression, so the draft loads in vLLM without a mapping.
    assert derived["draft_vocab_size"] == 32_000
    assert derived["target_hidden_size"] == 2048
    assert derived["target_num_hidden_layers"] == 24
    assert derived["architectures"] == ["LlamaForCausalLMEagle3"]
    assert derived["fc_norm"] is True
    assert derived["norm_output"] is True


def test_moe_expert_width_and_activation_use_dense_draft_defaults(tmp_path: Path) -> None:
    """An MoE ``intermediate_size`` describes one expert, not a dense draft MLP."""
    derived = _derive(
        tmp_path,
        {
            "text_config": {
                "hidden_size": 6144,
                "intermediate_size": 3072,
                "num_hidden_layers": 60,
                "num_attention_heads": 64,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 200_064,
                "hidden_act": "swigluoai",
            }
        },
    )

    assert derived["intermediate_size"] == 18_432
    assert derived["hidden_act"] == "silu"
    assert derived["num_key_value_heads"] == 4
    assert derived["eagle_aux_hidden_state_layer_ids"] == [2, 30, 57]


@pytest.mark.parametrize("nested_key", ["text_config", "language_config", "llm_config"])
def test_nested_text_towers_are_resolved(tmp_path: Path, nested_key: str) -> None:
    derived = _derive(tmp_path, {nested_key: MINIMAL_TEXT_CONFIG}, name=nested_key)

    assert derived["hidden_size"] == 2048
    assert derived["target_num_hidden_layers"] == 24
    assert derived["eagle_aux_hidden_state_layer_ids"] == [2, 12, 21]


def test_head_dim_falls_back_to_hidden_over_heads(tmp_path: Path) -> None:
    config = {key: value for key, value in MINIMAL_TEXT_CONFIG.items() if key != "head_dim"}
    derived = _derive(tmp_path, config)

    assert derived["head_dim"] == 2048 // 16
