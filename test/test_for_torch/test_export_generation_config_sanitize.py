#
# Copyright (C) 2023 - 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Tests for the generation_config sanitization in export_hf_model.

Background: transformers v5 strictly validates generation_config: setting
``top_p`` / ``top_k`` / ``typical_p`` without ``do_sample=True`` raises at
``save_pretrained`` time. Upstream HF checkpoints (e.g. zai-org/GLM-5) ship
configs that fail this check, which would otherwise discard a multi-hour
calibration. ``export_hf_model`` flips ``do_sample=True`` in that case.
"""

import pytest

from quark.torch.export import safetensors as safetensors_mod


class _StubGenerationConfig:
    """Mimics the small slice of transformers.GenerationConfig we touch."""

    def __init__(self, **kwargs):
        self.top_p = kwargs.get("top_p")
        self.top_k = kwargs.get("top_k")
        self.typical_p = kwargs.get("typical_p")
        self.do_sample = kwargs.get("do_sample", False)


class _StubModel:
    def __init__(self, generation_config):
        self.generation_config = generation_config
        self.saved_to = None

    def save_pretrained(self, export_dir, state_dict=None):
        self.saved_to = (str(export_dir), state_dict)


@pytest.fixture
def stub_export_helpers(monkeypatch):
    """Stub out get_state_dict_for_export so we can run without a real model."""
    monkeypatch.setattr(safetensors_mod, "get_state_dict_for_export", lambda model: {})
    yield


@pytest.mark.parametrize(
    "kwargs,expected_do_sample,expect_warning",
    [
        # Sampling flag + do_sample=False  →  flipped to True, warning emitted.
        ({"top_p": 0.9, "do_sample": False}, True, True),
        ({"top_k": 50, "do_sample": False}, True, True),
        ({"typical_p": 0.95, "do_sample": False}, True, True),
        # Sampling flag already with do_sample=True  →  left alone, no warning.
        ({"top_p": 0.9, "do_sample": True}, True, False),
        # No sampling flags  →  left alone, no warning.
        ({"do_sample": False}, False, False),
        # top_k=0 is the HF "disabled" sentinel — should NOT count as set.
        ({"top_k": 0, "do_sample": False}, False, False),
    ],
    ids=[
        "top_p_flips_do_sample",
        "top_k_flips_do_sample",
        "typical_p_flips_do_sample",
        "do_sample_already_true_unchanged",
        "no_sampling_fields_unchanged",
        "top_k_zero_is_disabled_sentinel",
    ],
)
def test_generation_config_sanitization(
    stub_export_helpers, capfd, tmp_path, kwargs, expected_do_sample, expect_warning
):
    gen_cfg = _StubGenerationConfig(**kwargs)
    model = _StubModel(gen_cfg)

    safetensors_mod.export_hf_model(model, tmp_path)

    assert gen_cfg.do_sample is expected_do_sample
    assert model.saved_to is not None and model.saved_to[0] == str(tmp_path)

    warning_text = capfd.readouterr().err
    if expect_warning:
        assert "do_sample=True" in warning_text
    else:
        assert "do_sample=True" not in warning_text


def test_no_generation_config_attribute_is_safe(stub_export_helpers, tmp_path):
    """Models without ``generation_config`` (older transformers) shouldn't crash."""

    class _ModelNoGenCfg:
        def __init__(self):
            self.saved_to = None

        def save_pretrained(self, export_dir, state_dict=None):
            self.saved_to = str(export_dir)

    model = _ModelNoGenCfg()
    safetensors_mod.export_hf_model(model, tmp_path)
    assert model.saved_to == str(tmp_path)
