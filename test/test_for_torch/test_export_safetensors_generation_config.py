#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Unit tests for export_hf_model's generation_config sanitization.

transformers v5 strictly validates sampling fields (top_p / top_k / typical_p) and
rejects a config that sets them while do_sample=False. Some upstream HF checkpoints
ship configs that fail this check, which would discard hours of calibration on an
export-time error. export_hf_model patches this in-place at save time.
"""

import types

import pytest

from quark.torch.export import safetensors as safetensors_module


@pytest.fixture
def fake_save(monkeypatch):
    """Stub out the heavy state-dict + save path so we exercise only the sanitization."""
    monkeypatch.setattr(safetensors_module, "get_state_dict_for_export", lambda model: {})

    saved: dict[str, object] = {}

    def _fake_save_pretrained(self, export_dir, state_dict=None, **kwargs):  # noqa: ARG001
        saved["export_dir"] = export_dir

    return _fake_save_pretrained, saved


def _build_model(generation_config):
    model = types.SimpleNamespace()
    model.generation_config = generation_config
    model.config = None
    # bind save_pretrained as a no-op method
    model.save_pretrained = lambda export_dir, state_dict=None, **kwargs: None  # noqa: ARG005
    return model


def test_sanitizes_top_p_with_do_sample_false(fake_save, tmp_path):
    _, _ = fake_save
    gen_cfg = types.SimpleNamespace(top_p=0.95, top_k=None, typical_p=None, do_sample=False)
    model = _build_model(gen_cfg)

    safetensors_module.export_hf_model(model, tmp_path)

    assert gen_cfg.do_sample is True


def test_sanitizes_top_k_with_do_sample_false(fake_save, tmp_path):
    gen_cfg = types.SimpleNamespace(top_p=None, top_k=50, typical_p=None, do_sample=False)
    model = _build_model(gen_cfg)

    safetensors_module.export_hf_model(model, tmp_path)

    assert gen_cfg.do_sample is True


def test_sanitizes_typical_p_with_do_sample_false(fake_save, tmp_path):
    gen_cfg = types.SimpleNamespace(top_p=None, top_k=None, typical_p=0.9, do_sample=False)
    model = _build_model(gen_cfg)

    safetensors_module.export_hf_model(model, tmp_path)

    assert gen_cfg.do_sample is True


def test_leaves_do_sample_true_alone(fake_save, tmp_path):
    """When do_sample is already True, sanitization must be a no-op."""
    gen_cfg = types.SimpleNamespace(top_p=0.95, top_k=50, typical_p=None, do_sample=True)
    model = _build_model(gen_cfg)

    safetensors_module.export_hf_model(model, tmp_path)

    assert gen_cfg.do_sample is True


def test_top_k_zero_is_not_treated_as_sampling_flag(fake_save, tmp_path):
    """top_k=0 means "disabled" in transformers — must not trip do_sample sanitization."""
    gen_cfg = types.SimpleNamespace(top_p=None, top_k=0, typical_p=None, do_sample=False)
    model = _build_model(gen_cfg)

    safetensors_module.export_hf_model(model, tmp_path)

    assert gen_cfg.do_sample is False


def test_no_sampling_fields_set_keeps_do_sample_false(fake_save, tmp_path):
    gen_cfg = types.SimpleNamespace(top_p=None, top_k=None, typical_p=None, do_sample=False)
    model = _build_model(gen_cfg)

    safetensors_module.export_hf_model(model, tmp_path)

    assert gen_cfg.do_sample is False


def test_missing_generation_config_is_tolerated(fake_save, tmp_path):
    model = _build_model(None)

    # Must not raise even though there's no generation_config to inspect.
    safetensors_module.export_hf_model(model, tmp_path)
