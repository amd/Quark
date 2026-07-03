#
# Copyright (C) 2023 - 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Unit tests for the huge-MoE HF loader throttle helpers in model_preparation."""

import pytest

from quark.torch.utils.llm import model_preparation as mp


class _Cfg:
    """Minimal stub HF config; attributes are set via kwargs."""

    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


# =============================================================================
# _estimate_quantized_linears
# =============================================================================
class TestEstimateQuantizedLinears:
    def test_none_config_returns_zero(self):
        assert mp._estimate_quantized_linears(None) == 0

    def test_missing_num_hidden_layers_returns_zero(self):
        assert mp._estimate_quantized_linears(_Cfg()) == 0

    def test_dense_model_uses_seven_linears_per_layer(self):
        # 4 attn + 3 MLP per dense layer.
        assert mp._estimate_quantized_linears(_Cfg(num_hidden_layers=10)) == 70

    def test_moe_model_with_num_local_experts(self):
        # 4 attn + router + experts*3 per MoE layer.
        cfg = _Cfg(num_hidden_layers=4, num_local_experts=8)
        assert mp._estimate_quantized_linears(cfg) == 4 * (5 + 8 * 3)

    def test_first_k_dense_replace_splits_dense_and_moe(self):
        cfg = _Cfg(num_hidden_layers=10, n_routed_experts=16, first_k_dense_replace=3)
        # 3 dense (×7) + 7 moe (×(5 + 16×3))
        assert mp._estimate_quantized_linears(cfg) == 3 * 7 + 7 * (5 + 16 * 3)

    def test_text_config_wrapper_is_unwrapped(self):
        # VLM/multimodal wrappers nest LM hyperparameters under .text_config.
        cfg = _Cfg(text_config=_Cfg(num_hidden_layers=4, num_experts=8))
        assert mp._estimate_quantized_linears(cfg) == 4 * (5 + 8 * 3)

    def test_negative_or_non_int_values_are_treated_as_zero(self):
        cfg = _Cfg(num_hidden_layers=4, num_local_experts=-1, first_k_dense_replace="oops")
        # Bad expert count → dense path; bad first_k_dense → 0.
        assert mp._estimate_quantized_linears(cfg) == 4 * 7


# =============================================================================
# _hf_loader_target_workers
# =============================================================================
class TestHfLoaderTargetWorkers:
    def test_env_override_always_wins(self, monkeypatch):
        monkeypatch.setenv("QUARK_HF_LOADER_WORKERS", "4")
        # Even with single-GPU + tiny model, env override applies.
        assert mp._hf_loader_target_workers(is_multi_gpu=False, config=None) == 4

    def test_env_override_clamped_to_one(self, monkeypatch):
        monkeypatch.setenv("QUARK_HF_LOADER_WORKERS", "0")
        assert mp._hf_loader_target_workers(is_multi_gpu=True, config=None) == 1

    def test_single_gpu_returns_none(self, monkeypatch):
        monkeypatch.delenv("QUARK_HF_LOADER_WORKERS", raising=False)
        assert mp._hf_loader_target_workers(is_multi_gpu=False, config=None) is None

    def test_multi_gpu_small_model_returns_none(self, monkeypatch):
        monkeypatch.delenv("QUARK_HF_LOADER_WORKERS", raising=False)
        # Qwen3-30B-ish: well under the 30k threshold.
        cfg = _Cfg(num_hidden_layers=48, num_local_experts=8)
        assert mp._estimate_quantized_linears(cfg) < mp._HUGE_MOE_LINEAR_THRESHOLD
        assert mp._hf_loader_target_workers(is_multi_gpu=True, config=cfg) is None

    def test_multi_gpu_huge_moe_throttles_to_one(self, monkeypatch):
        monkeypatch.delenv("QUARK_HF_LOADER_WORKERS", raising=False)
        # GLM-5-ish: well over the 30k threshold.
        cfg = _Cfg(num_hidden_layers=92, n_routed_experts=160, first_k_dense_replace=3)
        assert mp._estimate_quantized_linears(cfg) > mp._HUGE_MOE_LINEAR_THRESHOLD
        assert mp._hf_loader_target_workers(is_multi_gpu=True, config=cfg) == 1


# =============================================================================
# _set / _restore_hf_loader_workers
# =============================================================================
class TestSetRestoreHfLoaderWorkers:
    def test_set_with_none_target_is_noop(self):
        assert mp._set_hf_loader_workers(None) is None

    def test_set_when_loader_module_missing_is_noop(self, monkeypatch):
        monkeypatch.setattr(mp, "_hf_loader_module", None)
        assert mp._set_hf_loader_workers(1) is None

    def test_set_returns_previous_and_applies_new(self, monkeypatch):
        fake_loader = type("L", (), {"GLOBAL_WORKERS": 8})()
        monkeypatch.setattr(mp, "_hf_loader_module", fake_loader)

        saved = mp._set_hf_loader_workers(1)
        assert saved == 8
        assert fake_loader.GLOBAL_WORKERS == 1

        mp._restore_hf_loader_workers(saved)
        assert fake_loader.GLOBAL_WORKERS == 8

    def test_restore_with_none_is_noop(self, monkeypatch):
        fake_loader = type("L", (), {"GLOBAL_WORKERS": 8})()
        monkeypatch.setattr(mp, "_hf_loader_module", fake_loader)
        mp._restore_hf_loader_workers(None)
        assert fake_loader.GLOBAL_WORKERS == 8


# =============================================================================
# _warn_vm_max_map_count
# =============================================================================
class TestWarnVmMaxMapCount:
    @pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="linux-only path")
    def test_warns_when_below_safe_threshold(self, monkeypatch, capfd):
        import builtins

        real_open = builtins.open

        def fake_open(path, *args, **kwargs):
            if path == "/proc/sys/vm/max_map_count":
                from io import StringIO

                return StringIO("65530\n")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", fake_open)
        mp._warn_vm_max_map_count(num_linears=50000)
        assert "vm.max_map_count" in capfd.readouterr().err

    def test_no_warning_when_already_safe(self, monkeypatch, capfd):
        import builtins

        real_open = builtins.open

        def fake_open(path, *args, **kwargs):
            if path == "/proc/sys/vm/max_map_count":
                from io import StringIO

                return StringIO(f"{mp._VM_MAX_MAP_COUNT_SAFE}\n")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", fake_open)
        mp._warn_vm_max_map_count(num_linears=50000)
        assert "vm.max_map_count" not in capfd.readouterr().err
