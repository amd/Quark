#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for the mix_precision module (config, searcher, eval utilities, switcher,
vllm_inverse_quantizer, vllm_plugin)."""

from __future__ import annotations

import dataclasses
import fnmatch
import json
import math
import sys
import types
from copy import deepcopy
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# =============================================================================
# config.py
# =============================================================================


class TestNormalizeQuantMode:
    def test_native_aliases(self):
        from quark.experimental.torch.mix_precision.config import normalize_quant_mode

        for alias in ("native", "original", "bf16"):
            assert normalize_quant_mode(alias) == "native"

    def test_passthrough_modes(self):
        from quark.experimental.torch.mix_precision.config import normalize_quant_mode

        for mode in ("fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8", "mxfp6_e2m3"):
            assert normalize_quant_mode(mode) == mode

    def test_none_returns_empty_string(self):
        from quark.experimental.torch.mix_precision.config import normalize_quant_mode

        assert normalize_quant_mode(None) == ""


class TestIsNativeMode:
    def test_native_variants_are_native(self):
        from quark.experimental.torch.mix_precision.config import is_native_mode

        for mode in ("native", "original", "bf16"):
            assert is_native_mode(mode), f"{mode!r} should be native"

    def test_none_is_not_native(self):
        from quark.experimental.torch.mix_precision.config import is_native_mode

        # None normalizes to "" (empty string), not "native"
        assert not is_native_mode(None)

    def test_quant_modes_not_native(self):
        from quark.experimental.torch.mix_precision.config import is_native_mode

        for mode in ("fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8", "mxfp6_e2m3"):
            assert not is_native_mode(mode), f"{mode!r} should not be native"


class TestGetSupportedSchemes:
    def test_mi300_excludes_mxfp4(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget, get_supported_schemes

        schemes = get_supported_schemes(HardwareTarget.MI300)
        assert "fp8" in schemes
        assert "ptpc_fp8" in schemes
        assert "mxfp4" not in schemes
        assert "mxfp4_fp8" not in schemes

    def test_mi355_includes_mxfp4(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget, get_supported_schemes

        schemes = get_supported_schemes(HardwareTarget.MI355)
        assert "mxfp4" in schemes
        assert "mxfp4_fp8" in schemes
        assert "mxfp6_e2m3" in schemes

    def test_string_hardware_accepted(self):
        from quark.experimental.torch.mix_precision.config import get_supported_schemes

        schemes = get_supported_schemes("mi300")
        assert "fp8" in schemes

    def test_none_returns_all_modes(self):
        from quark.experimental.torch.mix_precision.config import ALL_QUANT_MODES, get_supported_schemes

        schemes = get_supported_schemes(None)
        assert set(schemes) == set(ALL_QUANT_MODES)


class TestGetLayerConfig:
    def test_native_returns_none(self):
        from quark.experimental.torch.mix_precision.config import get_layer_config

        assert get_layer_config("native") is None

    def test_known_modes_return_config(self):
        from quark.experimental.torch.mix_precision.config import get_layer_config

        for mode in ("fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8", "mxfp6_e2m3"):
            result = get_layer_config(mode)
            assert result is not None, f"get_layer_config({mode!r}) should return a QLayerConfig"

    def test_unknown_mode_returns_none(self):
        from quark.experimental.torch.mix_precision.config import get_layer_config

        assert get_layer_config("int8") is None


class TestMixPrecisionConfigValidate:
    def test_early_stop_is_disabled_by_default(self):
        from quark.experimental.torch.mix_precision.config import MixPrecisionConfig

        assert MixPrecisionConfig().early_stop is False

    def test_valid_config_passes(self):
        from quark.experimental.torch.mix_precision.config import MixPrecisionConfig

        config = MixPrecisionConfig(eval_metrics=["gsm8k"], eval_threshold=1.02)
        config.validate()  # should not raise

    def test_invalid_metric_raises(self):
        from quark.experimental.torch.mix_precision.config import MixPrecisionConfig

        config = MixPrecisionConfig(eval_metrics=["rouge"])
        with pytest.raises(ValueError, match="Invalid metric"):
            config.validate()

    def test_threshold_below_one_raises(self):
        from quark.experimental.torch.mix_precision.config import MixPrecisionConfig

        config = MixPrecisionConfig(eval_threshold=0.95)
        with pytest.raises(ValueError, match="eval_threshold"):
            config.validate()

    @pytest.mark.parametrize("max_configs", [0, -1])
    def test_max_configs_must_be_positive(self, max_configs):
        from quark.experimental.torch.mix_precision.config import MixPrecisionConfig

        config = MixPrecisionConfig(max_configs=max_configs)
        with pytest.raises(ValueError, match="max_configs"):
            config.validate()

    def test_negative_min_kv_scale_raises(self):
        from quark.experimental.torch.mix_precision.config import MixPrecisionConfig

        config = MixPrecisionConfig(min_kv_scale=-0.1)
        with pytest.raises(ValueError, match="min_kv_scale"):
            config.validate()

    def test_decoder_layer_granularity_not_implemented(self):
        from quark.experimental.torch.mix_precision.config import MixPrecisionConfig, SearchGranularity

        config = MixPrecisionConfig(granularity=SearchGranularity.DECODER_LAYER)
        with pytest.raises(NotImplementedError):
            config.validate()

    def test_user_facing_strings_are_normalized(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget, MixPrecisionConfig, SearchGranularity

        config = MixPrecisionConfig(hardware="mi325", granularity="module")

        assert config.hardware_target is HardwareTarget.MI325
        assert config.search_granularity is SearchGranularity.MODULE

    def test_search_modes_must_be_supported_by_hardware(self):
        from quark.experimental.torch.mix_precision.config import MixPrecisionConfig

        config = MixPrecisionConfig(hardware="mi300", search_modes=["native", "mxfp4"])

        with pytest.raises(ValueError, match="Unsupported search modes"):
            config.validate()


class TestMixPrecisionConfigBuildSearchConfig:
    def test_builds_module_search_config_by_default(self):
        from quark.experimental.torch.mix_precision.config import MixPrecisionConfig, ModuleSearchConfig

        config = MixPrecisionConfig()
        search_config = config._build_search_config()
        assert isinstance(search_config, ModuleSearchConfig)

    def test_builds_search_scope_without_exposing_module_config(self):
        from quark.experimental.torch.mix_precision.config import MixPrecisionConfig, ModuleSearchConfig

        config = MixPrecisionConfig(search_modes=["native", "ptpc_fp8"], kv_cache_quant=False)
        search_config = config._build_search_config()

        assert isinstance(search_config, ModuleSearchConfig)
        assert search_config.layer_modes == ["native", "ptpc_fp8"]
        assert search_config.kv_cache_modes == ["native"]

    def test_file_to_file_search_defaults_off_and_can_be_enabled(self):
        from quark.experimental.torch.mix_precision.config import MixPrecisionConfig

        assert MixPrecisionConfig().file2file_quantization is False
        assert MixPrecisionConfig(file2file_quantization=True).file2file_quantization is True

    def test_legacy_mlp_partition_expands_to_dense_and_routed(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config

        config = create_quant_config(layer_partitions={"mlp": "ptpc_fp8"})

        assert "mlp_mode" not in config
        assert config["dense_mlp_mode"] == "ptpc_fp8"
        assert config["routed_moe_mode"] == "ptpc_fp8"


class TestMixPrecisionPublicApi:
    def test_package_exports_only_high_level_api(self):
        import quark.experimental.torch.mix_precision as mix_precision

        assert mix_precision.__all__ == ["MixPrecisionConfig", "MixPrecisionQuantizer"]

    def test_quantizer_accepts_user_config(self):
        from quark.experimental.torch.mix_precision import MixPrecisionConfig, MixPrecisionQuantizer

        quantizer = MixPrecisionQuantizer(MixPrecisionConfig(hardware="mi300"))

        assert quantizer.config.hardware_target.value == "mi300"
        assert quantizer.result is None

    def test_all_failed_candidate_evaluations_raise(self):
        from quark.experimental.torch.mix_precision.quantizer import _require_successful_evaluation

        with pytest.raises(RuntimeError, match="All 3.*disk full"):
            _require_successful_evaluation([], ["config idx=0: disk full"], 3)

    def test_at_least_one_successful_evaluation_is_accepted(self):
        from quark.experimental.torch.mix_precision.config import ConfigEvalResult
        from quark.experimental.torch.mix_precision.quantizer import _require_successful_evaluation

        result = ConfigEvalResult(config={}, metrics={}, relative_change={}, is_valid=True, rank=1)

        _require_successful_evaluation([result], ["another config failed"], 2)

    @pytest.mark.parametrize(
        "failure_phase,error,cleanup_fails,aborts",
        [
            (None, None, False, False),
            ("calibration", torch.OutOfMemoryError("allocation failed"), False, True),
            ("calibration", RuntimeError("worker: HIP out of memory"), True, True),
            ("evaluation", RuntimeError("worker: OutOfMemoryError"), False, True),
            ("evaluation", TimeoutError("sample_tokens timed out"), True, True),
            ("evaluation", RuntimeError("unsupported candidate"), False, False),
            (None, None, True, True),
        ],
    )
    def test_search_progress_and_failures(self, monkeypatch, failure_phase, error, cleanup_fails, aborts):
        from quark.experimental.torch.mix_precision import quantizer as quantizer_module
        from quark.experimental.torch.mix_precision.config import MixPrecisionConfig

        configs = [
            {"mlp_mode": "ptpc_fp8", "kv_cache_mode": "native", "attention_mode": "native"},
            {"mlp_mode": "fp8", "kv_cache_mode": "native", "attention_mode": "native"},
            {"mlp_mode": "mxfp4", "kv_cache_mode": "native", "attention_mode": "native"},
        ]
        roofline_inputs = []
        evaluated_modes = []
        progress = []

        class _FakeSearcher:
            def __init__(self, **_kwargs):
                pass

            def generate_sorted_configs(self):
                return configs

        class _FakeQConfig:
            def __init__(self, config):
                self.config = config

            def to_dict(self):
                return dict(self.config)

        class _FakeLLM:
            def __init__(self, **_kwargs):
                pass

            def reset_prefix_cache(self):
                pass

            def reset_mm_cache(self):
                if cleanup_fails and evaluated_modes and (failure_phase != "evaluation" or len(evaluated_modes) == 2):
                    raise torch.OutOfMemoryError("cleanup failed")

            def collective_rpc(self, method, args=()):
                if method == "quark_search_moe_backend_report":
                    return [{"selected": "triton", "records": [], "layers": [], "probes": []}]
                if method == "requantize_with_config":
                    evaluated_modes.append(args[0]["mlp_mode"])
                    if failure_phase == "calibration":
                        raise error
                    return [True]
                if method == "reset_to_original":
                    return [True]
                raise AssertionError(f"Unexpected RPC method: {method}")

        def _compute_roofline(candidate_configs, **_kwargs):
            roofline_inputs.extend(candidate_configs)
            return [0.1, 0.6, 1.0]

        vllm_stub = types.ModuleType("vllm")
        vllm_stub.LLM = _FakeLLM
        monkeypatch.setitem(sys.modules, "vllm", vllm_stub)
        monkeypatch.setattr(quantizer_module, "ConfigSearcher", _FakeSearcher)
        monkeypatch.setattr(quantizer_module, "load_transformers_model", lambda *_args, **_kwargs: object())
        monkeypatch.setattr(quantizer_module, "_file_to_file_memory_reason", lambda _: None)
        monkeypatch.setattr(quantizer_module, "_preprocess_qconfig_model", lambda _model: None)
        monkeypatch.setattr(quantizer_module, "categorize_layers", lambda *_args, **_kwargs: {"mlp": set()})
        monkeypatch.setattr(
            quantizer_module,
            "_resolve_partition_source_weight_bitwidths",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(quantizer_module, "compute_and_display_roofline", _compute_roofline)
        monkeypatch.setattr(quantizer_module, "extract_tp_from_vllm_args", lambda _args: 1)
        monkeypatch.setattr(quantizer_module, "build_vllm_engine_kwargs", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(
            quantizer_module,
            "create_qconfig_from_quant_config",
            lambda *, config, **_kwargs: _FakeQConfig(config),
        )
        monkeypatch.setattr(quantizer_module, "display_results", lambda *_args, **_kwargs: None)

        def evaluate(*_args):
            if failure_phase == "evaluation" and len(evaluated_modes) == 2:
                raise error
            return 1.0

        monkeypatch.setattr(quantizer_module.MixPrecisionQuantizer, "_evaluate", evaluate)

        quantizer = quantizer_module.MixPrecisionQuantizer(
            MixPrecisionConfig(hardware="mi355", max_configs=2, early_stop=False, skip_baseline_eval=True)
        )
        if aborts:
            with pytest.raises(RuntimeError) as caught:
                quantizer.search("model", progress_callback=progress.append)
            assert str(error or "cleanup failed") in str(caught.value)
            if error is not None:
                assert caught.value.__cause__ is error
            if isinstance(error, TimeoutError):
                from quark.experimental.torch.mix_precision.run_helpers import is_out_of_memory

                assert not is_out_of_memory(caught.value)
            assert len(evaluated_modes) == (2 if failure_phase == "evaluation" else 1)
            return

        result = quantizer.search("model", progress_callback=progress.append)

        assert roofline_inputs == configs
        assert evaluated_modes == ["mxfp4", "fp8"]
        assert result.total_configs_available == 3
        if error is not None:
            assert result.total_configs_evaluated == 1
            return
        assert result.total_configs_evaluated == 2
        assert [snapshot.total_configs_evaluated for snapshot in progress] == [1, 2]
        assert [len(snapshot.all_results) for snapshot in progress] == [1, 2]
        assert progress[0].best_config == configs[2]
        assert progress[0].all_results is not result.all_results


class TestMixPrecisionExportBest:
    @staticmethod
    def _make_quantizer(best_config):
        from quark.experimental.torch.mix_precision import MixPrecisionConfig, MixPrecisionQuantizer
        from quark.experimental.torch.mix_precision.config import SearchResult

        quantizer = MixPrecisionQuantizer(MixPrecisionConfig())
        quantizer.model_path = "/models/test-model"
        quantizer.result = SearchResult(
            best_config=best_config,
            all_results=[],
            baseline_metrics={},
            total_configs_evaluated=1,
            total_configs_available=1,
            search_time_seconds=0.0,
            granularity=quantizer.config.search_granularity,
            hardware=quantizer.config.hardware_target,
        )
        return quantizer

    def test_file_to_file_export_is_used_for_calibration_free_config(self, monkeypatch, tmp_path):
        best_config = {
            "self_attn_mode": "native",
            "mlp_mode": "mxfp4",
            "kv_cache_mode": "native",
            "attention_mode": "native",
        }
        quantizer = self._make_quantizer(best_config)
        file_to_file_export = MagicMock()
        standard_export = MagicMock()
        monkeypatch.setattr(quantizer, "_export_best_file_to_file", file_to_file_export)
        monkeypatch.setattr(quantizer, "_export_best_standard", standard_export)

        destination = tmp_path / "export"
        result = quantizer.export_best(str(destination), file2file_quantization=True)

        assert result == destination
        file_to_file_export.assert_called_once_with("/models/test-model", best_config, destination)
        standard_export.assert_not_called()

    def test_config_enables_file_to_file_export_without_repeating_flag(self, monkeypatch, tmp_path):
        best_config = {
            "self_attn_mode": "ptpc_fp8",
            "mlp_mode": "mxfp4",
            "kv_cache_mode": "native",
            "attention_mode": "native",
        }
        quantizer = self._make_quantizer(best_config)
        quantizer.config.file2file_quantization = True
        file_to_file_export = MagicMock()
        standard_export = MagicMock()
        monkeypatch.setattr(quantizer, "_export_best_file_to_file", file_to_file_export)
        monkeypatch.setattr(quantizer, "_export_best_standard", standard_export)

        destination = tmp_path / "export"
        quantizer.export_best(str(destination))

        file_to_file_export.assert_called_once_with("/models/test-model", best_config, destination)
        standard_export.assert_not_called()

    @pytest.mark.parametrize("has_compatible_result", [False, True])
    def test_required_file_to_file_export_reuses_only_compatible_results(
        self, monkeypatch, tmp_path, caplog, has_compatible_result
    ):
        best_config = {
            "self_attn_mode": "native",
            "mlp_mode": "mxfp4_fp8",
            "kv_cache_mode": "native",
            "attention_mode": "native",
        }
        quantizer = self._make_quantizer(best_config)
        compatible = {**best_config, "mlp_mode": "mxfp4"}
        if has_compatible_result:
            quantizer.result.all_results = [SimpleNamespace(config=compatible, is_valid=True)]
        file_to_file_export = MagicMock()
        standard_export = MagicMock()
        monkeypatch.setattr(quantizer, "_export_best_file_to_file", file_to_file_export)
        monkeypatch.setattr(quantizer, "_export_best_standard", standard_export)
        caplog.set_level("WARNING", logger="quark.experimental.torch.mix_precision.quantizer")

        destination = tmp_path / "export"
        if has_compatible_result:
            assert quantizer.export_best(str(destination), file2file_quantization=True) == destination
            file_to_file_export.assert_called_once_with("/models/test-model", compatible, destination)
            assert quantizer.result.best_config == compatible
        else:
            with pytest.raises(RuntimeError, match="compatible"):
                quantizer.export_best(str(destination), file2file_quantization=True)
            file_to_file_export.assert_not_called()
        standard_export.assert_not_called()
        assert "requires calibration" in caplog.text

    def test_traditional_export_remains_the_default(self, monkeypatch, tmp_path):
        from quark.experimental.torch.mix_precision import quantizer as quantizer_module

        best_config = {
            "self_attn_mode": "native",
            "mlp_mode": "mxfp4",
            "kv_cache_mode": "native",
            "attention_mode": "native",
        }
        quantizer = self._make_quantizer(best_config)
        file_to_file_export = MagicMock()
        model = object()
        load_model = MagicMock(return_value=model)
        export_order = []
        tokenizer = SimpleNamespace(
            save_pretrained=MagicMock(side_effect=lambda *_args: export_order.append("tokenizer"))
        )
        load_tokenizer = MagicMock(return_value=tokenizer)
        quantized_model = object()
        frozen_model = object()
        apply_config = MagicMock(return_value=quantized_model)
        freeze = MagicMock(return_value=frozen_model)
        export_safetensors = MagicMock(side_effect=lambda *_args: export_order.append("model"))
        monkeypatch.setattr(quantizer, "_export_best_file_to_file", file_to_file_export)
        monkeypatch.setattr(quantizer, "_load_preprocessed_export_model", load_model)
        monkeypatch.setattr(quantizer_module.AutoTokenizer, "from_pretrained", load_tokenizer)
        monkeypatch.setattr(quantizer_module, "apply_quant_config", apply_config)
        monkeypatch.setattr(quantizer_module.ModelQuantizer, "freeze", freeze)
        monkeypatch.setattr(quantizer_module, "export_safetensors", export_safetensors)

        destination = tmp_path / "export"
        quantizer.export_best(str(destination))

        file_to_file_export.assert_not_called()
        load_model.assert_called_once_with("/models/test-model", device_map="auto")
        load_tokenizer.assert_called_once_with("/models/test-model", trust_remote_code=True)
        apply_config.assert_called_once_with(
            model=model,
            config=best_config,
            tokenizer=tokenizer,
            num_calib_samples=quantizer.config.num_calib_samples,
            calib_seq_len=quantizer.config.calib_seq_len,
            hardware=quantizer.config.hardware_target,
            exclude_patterns=quantizer.config.exclude_patterns,
            min_kv_scale=quantizer.config.min_kv_scale,
        )
        freeze.assert_called_once_with(quantized_model)
        export_safetensors.assert_called_once_with(frozen_model, str(destination))
        tokenizer.save_pretrained.assert_called_once_with(str(destination))
        assert export_order == ["tokenizer", "model"]

    def test_preprocessed_export_model_warns_when_preprocessing_is_skipped(self, monkeypatch, caplog):
        from quark.experimental.torch.mix_precision import quantizer as quantizer_module

        model = object()
        load_model = MagicMock(return_value=model)
        preprocess = MagicMock(side_effect=ValueError("unsupported model"))
        monkeypatch.setattr(quantizer_module, "load_transformers_model", load_model)
        monkeypatch.setattr(quantizer_module, "preprocess_for_quantization", preprocess)
        caplog.set_level("WARNING", logger="quark.experimental.torch.mix_precision.quantizer")

        result = quantizer_module.MixPrecisionQuantizer._load_preprocessed_export_model(
            "/models/test-model", device_map="meta"
        )

        assert result is model
        load_model.assert_called_once_with("/models/test-model", torch_dtype="auto", device_map="meta")
        preprocess.assert_called_once_with(model)
        assert "preprocess_for_quantization skipped: unsupported model" in caplog.text

    def test_file_to_file_export_builds_qconfig_and_uses_model_weight_converters(self, monkeypatch, tmp_path):
        from quark.experimental.torch.mix_precision import quantizer as quantizer_module

        best_config = {
            "self_attn_mode": "native",
            "mlp_mode": "mxfp4",
            "kv_cache_mode": "native",
            "attention_mode": "native",
        }
        quantizer = self._make_quantizer(best_config)
        model = nn.Module()
        model.config = SimpleNamespace(model_type="qwen3_5_moe")
        load_model = MagicMock(return_value=model)
        qconfig = SimpleNamespace(exclude=["model.layers.*.mlp.experts.*.gate_proj"])
        create_qconfig = MagicMock(return_value=qconfig)
        converters = [object(), object()]
        template = SimpleNamespace(f2f_weight_converters=converters)
        direct_quantize_checkpoint = MagicMock()
        model_quantizer = MagicMock(return_value=SimpleNamespace(direct_quantize_checkpoint=direct_quantize_checkpoint))
        resolve_source = MagicMock(return_value=("/resolved/test-model", {"model_type": "qwen3_5_moe"}))
        reconcile_excludes = MagicMock()
        monkeypatch.setattr(quantizer_module, "_resolve_file_to_file_source", resolve_source)
        monkeypatch.setattr(quantizer_module, "_reconcile_converted_file_to_file_excludes", reconcile_excludes)
        monkeypatch.setattr(quantizer, "_load_preprocessed_export_model", load_model)
        monkeypatch.setattr(quantizer_module, "create_qconfig_from_quant_config", create_qconfig)
        monkeypatch.setattr(quantizer_module.LLMTemplate, "list_available", MagicMock(return_value=["qwen3_5_moe"]))
        monkeypatch.setattr(quantizer_module.LLMTemplate, "get", MagicMock(return_value=template))
        monkeypatch.setattr(quantizer_module, "ModelQuantizer", model_quantizer)

        destination = tmp_path / "export"
        result = quantizer.export_best(str(destination), file2file_quantization=True)

        assert result == destination
        resolve_source.assert_called_once_with("/models/test-model")
        load_model.assert_called_once_with("/resolved/test-model", device_map="meta")
        create_qconfig.assert_called_once_with(
            model=model,
            config=best_config,
            exclude_patterns=quantizer.config.exclude_patterns,
            min_kv_scale=quantizer.config.min_kv_scale,
        )
        model_quantizer.assert_called_once_with(qconfig)
        direct_quantize_checkpoint.assert_called_once_with(
            pretrained_model_path="/resolved/test-model",
            save_path=str(destination),
            weight_converters=converters,
            keep_excluded_layers_as_original_model_state=False,
        )
        reconcile_excludes.assert_called_once_with(
            source_dir="/resolved/test-model",
            destination=destination,
            exclude_patterns=qconfig.exclude,
            keep_original_quantized_state=False,
        )

    def test_file_to_file_export_preserves_excluded_state_for_quantized_source(self, monkeypatch, tmp_path):
        from quark.experimental.torch.mix_precision import quantizer as quantizer_module

        best_config = {"self_attn_mode": "native", "mlp_mode": "mxfp4"}
        quantizer = self._make_quantizer(best_config)
        model = nn.Module()
        model.config = SimpleNamespace(model_type="llama")
        direct_quantize_checkpoint = MagicMock()
        monkeypatch.setattr(
            quantizer_module,
            "_resolve_file_to_file_source",
            MagicMock(
                return_value=(
                    "/resolved/test-model",
                    {"quantization_config": {"quant_method": "quark"}},
                )
            ),
        )
        monkeypatch.setattr(quantizer, "_load_preprocessed_export_model", MagicMock(return_value=model))
        monkeypatch.setattr(quantizer_module, "create_qconfig_from_quant_config", MagicMock(return_value=object()))
        monkeypatch.setattr(quantizer_module.LLMTemplate, "list_available", MagicMock(return_value=[]))
        monkeypatch.setattr(
            quantizer_module,
            "ModelQuantizer",
            MagicMock(return_value=SimpleNamespace(direct_quantize_checkpoint=direct_quantize_checkpoint)),
        )

        quantizer.export_best(str(tmp_path / "export"), file2file_quantization=True)

        assert direct_quantize_checkpoint.call_args.kwargs["keep_excluded_layers_as_original_model_state"] is True

    def test_file_to_file_source_resolves_hub_id_and_validates_safetensors(self, monkeypatch, tmp_path):
        from quark.experimental.torch.mix_precision import quantizer as quantizer_module

        cached_model = tmp_path / "cached-model"
        cached_model.mkdir()
        (cached_model / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
        (cached_model / "model.safetensors").touch()
        snapshot_download = MagicMock(return_value=str(cached_model))
        monkeypatch.setattr(quantizer_module, "snapshot_download", snapshot_download)

        resolved_path, config = quantizer_module._resolve_file_to_file_source("org/model")

        snapshot_download.assert_called_once_with(repo_id="org/model")
        assert resolved_path == str(cached_model.resolve())
        assert config == {"model_type": "llama"}

    def test_file_to_file_source_rejects_checkpoint_without_safetensors(self, tmp_path):
        from quark.experimental.torch.mix_precision.quantizer import _resolve_file_to_file_source

        (tmp_path / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
        (tmp_path / "pytorch_model.bin").touch()

        with pytest.raises(ValueError, match="at least one .safetensors"):
            _resolve_file_to_file_source(str(tmp_path))

    def test_file_to_file_source_requires_model_directory_and_config(self, tmp_path):
        from quark.experimental.torch.mix_precision.quantizer import _resolve_file_to_file_source

        model_file = tmp_path / "model.safetensors"
        model_file.touch()
        with pytest.raises(ValueError, match="requires a model directory"):
            _resolve_file_to_file_source(str(model_file))

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").touch()
        with pytest.raises(ValueError, match="requires config.json"):
            _resolve_file_to_file_source(str(model_dir))

    def test_file_to_file_source_requires_json_object_config(self, tmp_path):
        from quark.experimental.torch.mix_precision.quantizer import _resolve_file_to_file_source

        (tmp_path / "config.json").write_text("[]", encoding="utf-8")
        (tmp_path / "model.safetensors").touch()

        with pytest.raises(ValueError, match="JSON object"):
            _resolve_file_to_file_source(str(tmp_path))

    def test_file_to_file_tensor_names_fall_back_to_safetensors_headers(self, tmp_path):
        from safetensors.torch import save_file

        from quark.experimental.torch.mix_precision.quantizer import _read_safetensors_tensor_names

        save_file(
            {"model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2)},
            str(tmp_path / "model.safetensors"),
        )

        assert _read_safetensors_tensor_names(tmp_path) == {"model.layers.0.self_attn.q_proj.weight"}

    @pytest.mark.parametrize("keep_original_state", [False, True])
    def test_file_to_file_reconciles_converter_output_excludes(self, tmp_path, keep_original_state):
        from quark.experimental.torch.mix_precision.quantizer import _reconcile_converted_file_to_file_excludes

        source_dir = tmp_path / "source"
        destination = tmp_path / "output"
        source_dir.mkdir()
        destination.mkdir()

        source_weight_map = {
            "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
            "model.layers.0.mlp.experts.gate_up_proj": "model.safetensors",
        }
        output_weight_map = {
            "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
            "model.layers.0.mlp.experts.0.gate_proj.weight": "model.safetensors",
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale": "model.safetensors",
        }
        (source_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": source_weight_map}), encoding="utf-8"
        )
        (destination / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": output_weight_map}), encoding="utf-8"
        )
        (destination / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5_moe",
                    "quantization_config": {
                        "exclude": ["lm_head"],
                        "layer_quant_config": {"sentinel": {}},
                    },
                }
            ),
            encoding="utf-8",
        )

        _reconcile_converted_file_to_file_excludes(
            source_dir=source_dir,
            destination=destination,
            exclude_patterns=[
                "model.layers.*.self_attn.q_proj",
                "model.layers.*.mlp.experts.*.gate_proj",
            ],
            keep_original_quantized_state=keep_original_state,
        )

        exported_config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
        expected_excludes = (
            ["lm_head", "model.layers.0.mlp.experts.0.gate_proj"]
            if keep_original_state
            else ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.self_attn.q_proj"]
        )
        assert exported_config["quantization_config"]["exclude"] == expected_excludes
        assert exported_config["quantization_config"]["layer_quant_config"] == {"sentinel": {}}
        assert not (destination / "config.json.tmp").exists()


class TestFileToFileSearchSpace:
    def test_rejects_reduction_to_native_only(self):
        from quark.experimental.torch.mix_precision.quantizer import _filter_file_to_file_compatible_configs

        with pytest.raises(RuntimeError, match="quantized candidate"):
            _filter_file_to_file_compatible_configs([{"mlp_mode": "fp8"}, {"mlp_mode": "native"}])

    def test_removes_calibration_dependent_configs_and_warns(self, caplog):
        from quark.experimental.torch.mix_precision.quantizer import _filter_file_to_file_compatible_configs

        compatible = {
            "self_attn_mode": "ptpc_fp8",
            "mlp_mode": "mxfp4",
            "kv_cache_mode": "native",
        }
        w4a8 = {
            "self_attn_mode": "ptpc_fp8",
            "mlp_mode": "mxfp4_fp8",
            "kv_cache_mode": "native",
        }
        static_kv_cache = {
            "self_attn_mode": "mxfp4",
            "mlp_mode": "mxfp4",
            "kv_cache_mode": "fp8",
        }
        caplog.set_level("WARNING", logger="quark.experimental.torch.mix_precision.quantizer")

        filtered = _filter_file_to_file_compatible_configs([w4a8, compatible, static_kv_cache])

        assert filtered == [compatible]
        assert "2/3 generated candidate configs require calibration" in caplog.text
        assert "mxfp4_fp8 (W4A8)" in caplog.text
        assert "fp8" in caplog.text
        assert "1 calibration-free configs remain" in caplog.text

    def test_keeps_calibration_free_search_space_without_warning(self, caplog):
        from quark.experimental.torch.mix_precision.quantizer import _filter_file_to_file_compatible_configs

        configs = [
            {"self_attn_mode": "ptpc_fp8", "mlp_mode": "mxfp4"},
            {"self_attn_mode": "mxfp4", "mlp_mode": "mxfp6_e2m3"},
        ]

        assert _filter_file_to_file_compatible_configs(configs) is configs
        assert not caplog.records

    def test_rejects_search_space_with_only_calibration_configs(self, caplog):
        from quark.experimental.torch.mix_precision.quantizer import _filter_file_to_file_compatible_configs

        caplog.set_level("WARNING", logger="quark.experimental.torch.mix_precision.quantizer")

        with pytest.raises(RuntimeError, match="removed every generated candidate"):
            _filter_file_to_file_compatible_configs([{"mlp_mode": "mxfp4_fp8"}])

        assert "1/1 generated candidate configs require calibration" in caplog.text


# =============================================================================
# searcher.py
# =============================================================================


class TestConfigSearcher:
    def test_generates_configs_for_mi300(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(hardware=HardwareTarget.MI300)
        configs = searcher.generate_sorted_configs()
        assert len(configs) > 0

    def test_no_all_native_config(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(hardware=HardwareTarget.MI300)
        configs = searcher.generate_sorted_configs()
        for config in configs:
            layer_modes = [config.get(f"{p}_mode") for p in searcher.layer_sensitivity]
            assert any(m != "native" for m in layer_modes), f"All-native config must not appear: {config}"

    def test_sorted_conservative_to_aggressive(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(hardware=HardwareTarget.MI300)
        configs = searcher.generate_sorted_configs()
        scores = [searcher.compute_score(c) for c in configs]
        assert scores == sorted(scores, reverse=True), "Configs must be sorted high-score first"

    def test_precision_hierarchy_respected(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        # self_attn (sensitivity=3) must have precision >= dense_mlp (sensitivity=1)
        search_config = ModuleSearchConfig(layer_sensitivity={"self_attn": 3, "dense_mlp": 1})
        searcher = ConfigSearcher(search_config=search_config, hardware=HardwareTarget.MI300)
        pw = searcher.precision_weights
        for config in searcher.generate_sorted_configs():
            attn_score = pw.get(config.get("self_attn_mode", "native"), 0)
            mlp_score = pw.get(config.get("dense_mlp_mode", "native"), 0)
            assert attn_score >= mlp_score, f"self_attn precision must be >= dense_mlp: {config}"

    def test_mi355_includes_mxfp4_configs(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(hardware=HardwareTarget.MI355)
        configs = searcher.generate_sorted_configs()
        all_modes = (
            {config.get("self_attn_mode") for config in configs}
            | {config.get("dense_mlp_mode") for config in configs}
            | {config.get("routed_moe_mode") for config in configs}
        )
        assert "mxfp4" in all_modes, "MI355 should produce mxfp4 configs"

    def test_available_partitions_filter(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        # Only self_attn and dense_mlp — linear_attn should be excluded
        searcher = ConfigSearcher(
            hardware=HardwareTarget.MI300,
            available_partitions={"self_attn", "dense_mlp"},
        )
        assert "linear_attn" not in searcher.layer_sensitivity

    def test_layer_modes_override(self):
        from quark.experimental.torch.mix_precision.config import ModuleSearchConfig
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        search_config = ModuleSearchConfig(layer_modes=["native", "ptpc_fp8"])
        searcher = ConfigSearcher(search_config=search_config)
        configs = searcher.generate_sorted_configs()
        for config in configs:
            for p in searcher.layer_sensitivity:
                mode = config.get(f"{p}_mode", "native")
                assert mode in ("native", "ptpc_fp8"), f"Unexpected mode {mode!r} in config: {config}"

    def test_kv_cache_modes_override(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        search_config = ModuleSearchConfig(kv_cache_modes=["native"])
        searcher = ConfigSearcher(search_config=search_config, hardware=HardwareTarget.MI300)
        configs = searcher.generate_sorted_configs()
        for config in configs:
            assert config.get("kv_cache_mode") == "native", f"kv_cache must stay native: {config}"

    def test_compute_score_ordering(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(hardware=HardwareTarget.MI300)
        conservative = {
            "self_attn_mode": "ptpc_fp8",
            "dense_mlp_mode": "ptpc_fp8",
            "kv_cache_mode": "native",
            "attention_mode": "native",
        }
        aggressive = {
            "self_attn_mode": "fp8",
            "dense_mlp_mode": "fp8",
            "kv_cache_mode": "native",
            "attention_mode": "native",
        }
        assert searcher.compute_score(conservative) > searcher.compute_score(aggressive)


class TestSourceAwareSearchSpace:
    def _layer_modes(self, searcher):
        modes = set()
        for config in searcher.generate_sorted_configs():
            for p in searcher.layer_sensitivity:
                modes.add(config.get(f"{p}_mode", "native"))
        return modes

    def test_mxfp4_source_drops_higher_precision_weight_modes(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(
            hardware=HardwareTarget.MI355,
            available_partitions={"self_attn", "routed_moe"},
            source_weight_bitwidth={"self_attn": 4, "routed_moe": 4},
        )
        modes = self._layer_modes(searcher)
        assert modes <= {"native", "mxfp4", "mxfp4_fp8"}
        assert not ({"fp8", "ptpc_fp8", "mxfp6_e2m3"} & modes)
        assert searcher.partition_modes["kv_cache"] == ["native", "fp8"]

    def test_per_partition_only_filters_quantized_partitions(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        # routed_moe source is MXFP4 (w4); self_attn source is unquantized (bf16) so it is
        # absent from the map and must keep its higher-precision targets.
        searcher = ConfigSearcher(
            hardware=HardwareTarget.MI355,
            available_partitions={"self_attn", "routed_moe"},
            source_weight_bitwidth={"routed_moe": 4},
        )
        routed_modes = set(searcher.partition_modes["routed_moe"])
        attn_modes = set(searcher.partition_modes["self_attn"])
        assert not ({"fp8", "ptpc_fp8", "mxfp6_e2m3"} & routed_modes)
        assert {"fp8", "ptpc_fp8", "mxfp6_e2m3"} <= attn_modes

    def test_shared_expert_respects_its_own_source_bitwidth(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(
            search_config=ModuleSearchConfig(layer_modes=["native", "fp8", "mxfp4"]),
            hardware=HardwareTarget.MI355,
            available_partitions={"routed_moe", "shared_expert"},
            source_weight_bitwidth={"shared_expert": 4},
        )
        configs = searcher.generate_sorted_configs()

        assert any(config["routed_moe_mode"] == "fp8" for config in configs)
        assert all(config["shared_expert_mode"] != "fp8" for config in configs if config["routed_moe_mode"] == "fp8")
        assert any(config["shared_expert_mode"] == "mxfp4" for config in configs)

    def test_fp8_source_keeps_all_modes(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        unfiltered = ConfigSearcher(
            hardware=HardwareTarget.MI355,
            available_partitions={"self_attn", "routed_moe"},
        )
        filtered = ConfigSearcher(
            hardware=HardwareTarget.MI355,
            available_partitions={"self_attn", "routed_moe"},
            source_weight_bitwidth={"self_attn": 8, "routed_moe": 8},
        )
        assert self._layer_modes(filtered) == self._layer_modes(unfiltered)

    def test_float_source_is_a_noop(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        baseline = ConfigSearcher(hardware=HardwareTarget.MI355, available_partitions={"self_attn", "dense_mlp"})
        none_source = ConfigSearcher(
            hardware=HardwareTarget.MI355,
            available_partitions={"self_attn", "dense_mlp"},
            source_weight_bitwidth=None,
        )
        assert self._layer_modes(none_source) == self._layer_modes(baseline)

    def test_kimi_split_keeps_dense_fp8_and_drops_routed_fp8(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(
            search_config=ModuleSearchConfig(layer_modes=["native", "ptpc_fp8", "mxfp4"]),
            hardware=HardwareTarget.MI355,
            available_partitions={"self_attn", "dense_mlp", "routed_moe"},
            source_weight_bitwidth={"routed_moe": 4},
        )

        assert searcher.partition_modes["dense_mlp"] == ["native", "ptpc_fp8", "mxfp4"]
        assert searcher.partition_modes["routed_moe"] == ["native", "mxfp4"]
        assert all("mlp_mode" not in config for config in searcher.generate_sorted_configs())


class TestResolvePartitionSourceWeightBitwidths:
    def test_uniform_source_filters_all_partitions(self):
        from quark.experimental.torch.mix_precision.quantizer import _resolve_partition_source_weight_bitwidths

        model = SimpleNamespace(config=SimpleNamespace(quantization_config={"quant_method": "mxfp4"}))
        partition_layers = {
            "self_attn": {"model.layers.0.self_attn.q_proj"},
            "routed_moe": {"model.layers.0.mlp.experts.0.gate_proj"},
        }
        assert _resolve_partition_source_weight_bitwidths(model, None, partition_layers) == {
            "self_attn": 4,
            "routed_moe": 4,
        }

    def test_unquantized_partition_is_left_unfiltered(self):
        from quark.experimental.torch.mix_precision.quantizer import _resolve_partition_source_weight_bitwidths

        model = SimpleNamespace(
            config=SimpleNamespace(
                quantization_config={"quant_method": "mxfp4", "modules_to_not_convert": ["self_attn"]}
            )
        )
        partition_layers = {
            "self_attn": {"model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.o_proj"},
            "routed_moe": {"model.layers.0.mlp.experts.0.gate_proj"},
        }
        assert _resolve_partition_source_weight_bitwidths(model, None, partition_layers) == {"routed_moe": 4}

    def test_globbed_unquantized_partition_is_left_unfiltered(self):
        from quark.experimental.torch.mix_precision.quantizer import _resolve_partition_source_weight_bitwidths

        model = SimpleNamespace(
            config=SimpleNamespace(
                quantization_config={"quant_method": "mxfp4", "modules_to_not_convert": ["*.self_attn.*"]}
            )
        )
        partition_layers = {
            "self_attn": {"model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.o_proj"},
            "routed_moe": {"model.layers.0.mlp.experts.0.gate_proj"},
        }
        assert _resolve_partition_source_weight_bitwidths(model, None, partition_layers) == {"routed_moe": 4}

    def test_modules_to_not_convert_falls_back_to_disk(self, tmp_path):
        from quark.experimental.torch.mix_precision.quantizer import _resolve_partition_source_weight_bitwidths

        (tmp_path / "config.json").write_text(
            json.dumps({"quantization_config": {"quant_method": "mxfp4", "modules_to_not_convert": ["self_attn"]}})
        )
        model = SimpleNamespace(config=SimpleNamespace(quantization_config=SimpleNamespace(quant_method="mxfp4")))
        partition_layers = {
            "self_attn": {"model.layers.0.self_attn.q_proj"},
            "routed_moe": {"model.layers.0.mlp.experts.0.gate_proj"},
        }
        assert _resolve_partition_source_weight_bitwidths(model, str(tmp_path), partition_layers) == {"routed_moe": 4}

    def test_kimi_k3_compressed_tensors_filters_only_packed_experts(self):
        from quark.experimental.torch.mix_precision.quantizer import (
            _resolve_partition_source_weight_bitwidths,
            _resolve_source_weight_bitwidths_by_layer,
        )

        quantization_config = {
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "config_groups": {"group_0": {"weights": {"num_bits": 4}}},
            "ignore": [
                "re:.*self_attn.*",
                "re:.*shared_experts.*",
                "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
            ],
        }
        model = SimpleNamespace(
            config=SimpleNamespace(
                quantization_config=None,
                text_config=SimpleNamespace(quantization_config=quantization_config),
            )
        )
        partition_layers = {
            "linear_attn": {"language_model.model.layers.0.self_attn.in_proj_qkv"},
            "self_attn": {"language_model.model.layers.3.self_attn.q_proj"},
            "dense_mlp": {"language_model.model.layers.0.mlp.gate_proj"},
            "routed_moe": {"language_model.model.layers.1.block_sparse_moe.experts.0.w1"},
            "shared_expert": {"language_model.model.layers.1.block_sparse_moe.shared_experts.gate_proj"},
        }

        layer_bitwidths = _resolve_source_weight_bitwidths_by_layer(model, None, partition_layers)
        assert layer_bitwidths == {"language_model.model.layers.1.block_sparse_moe.experts.0.w1": 4}
        # Separating dense MLP and routed MoE gives the routed partition a
        # uniform W4 floor without constraining the BF16 dense partition.
        assert _resolve_partition_source_weight_bitwidths(
            model,
            None,
            partition_layers,
            layer_bitwidths,
        ) == {"routed_moe": 4}

    def test_float_model_resolves_to_none(self):
        from quark.experimental.torch.mix_precision.quantizer import _resolve_partition_source_weight_bitwidths

        model = SimpleNamespace(config=SimpleNamespace(quantization_config=None))
        assert _resolve_partition_source_weight_bitwidths(model, None, {"dense_mlp": {"x"}}) is None


class TestRooflineNavigation:
    def test_valid_config_moves_to_immediate_higher_neighbor(self):
        from quark.experimental.torch.mix_precision.run_helpers import _next_roofline_neighbor

        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        assert (
            _next_roofline_neighbor(
                scores=scores,
                visited={1},
                current_idx=1,
                move_higher=True,
            )
            == 2
        )

    def test_invalid_config_moves_to_immediate_lower_neighbor(self):
        from quark.experimental.torch.mix_precision.run_helpers import _next_roofline_neighbor

        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        assert (
            _next_roofline_neighbor(
                scores=scores,
                visited={1},
                current_idx=1,
                move_higher=False,
            )
            == 0
        )

    def test_equal_scores_use_original_rank_as_tiebreaker(self):
        from quark.experimental.torch.mix_precision.run_helpers import _next_roofline_neighbor

        assert (
            _next_roofline_neighbor(
                scores=[0.1, 0.2, 0.2, 0.3],
                visited={1},
                current_idx=1,
                move_higher=True,
            )
            == 2
        )

    def test_returns_none_when_preferred_direction_is_exhausted(self):
        from quark.experimental.torch.mix_precision.run_helpers import _next_roofline_neighbor

        assert (
            _next_roofline_neighbor(
                scores=[0.1, 0.2, 0.3],
                visited={0, 1, 2},
                current_idx=2,
                move_higher=True,
            )
            is None
        )

    @pytest.mark.parametrize(
        ("current_idx", "move_higher", "expected_idx"),
        [(3, True, 2), (0, False, 1)],
    )
    def test_full_sweep_falls_back_to_nearest_neighbor_in_opposite_direction(
        self,
        current_idx,
        move_higher,
        expected_idx,
    ):
        from quark.experimental.torch.mix_precision.run_helpers import _next_roofline_candidate

        assert (
            _next_roofline_candidate(
                scores=[0.1, 0.2, 0.3, 0.4],
                visited={current_idx},
                current_idx=current_idx,
                move_higher=move_higher,
                allow_direction_fallback=True,
            )
            == expected_idx
        )

    def test_early_stop_does_not_fall_back_to_opposite_direction(self):
        from quark.experimental.torch.mix_precision.run_helpers import _next_roofline_candidate

        assert (
            _next_roofline_candidate(
                scores=[0.1, 0.2, 0.3, 0.4],
                visited={3},
                current_idx=3,
                move_higher=True,
                allow_direction_fallback=False,
            )
            is None
        )


class TestHardwareRooflinePolicy:
    @pytest.mark.parametrize(
        ("hardware", "gpu_type", "anchor_mode"),
        [
            ("mi300", "mi300x", "ptpc_fp8"),
            ("mi325", "mi325x", "ptpc_fp8"),
            ("mi355", "mi355x", "mxfp4"),
        ],
    )
    def test_policy_selects_hardware_gpu_and_anchor(self, hardware, gpu_type, anchor_mode):
        from quark.experimental.torch.mix_precision.run_helpers import _get_hardware_search_policy

        policy = _get_hardware_search_policy(hardware)

        assert policy.gpu_type == gpu_type
        assert policy.anchor_mode == anchor_mode

    @pytest.mark.parametrize(("anchor_mode", "expected_idx"), [("ptpc_fp8", 0), ("mxfp4", 1)])
    def test_finds_hardware_mlp_only_anchor(self, anchor_mode, expected_idx):
        from quark.experimental.torch.mix_precision.run_helpers import _find_roofline_start_config

        configs = [
            {
                "self_attn_mode": "native",
                "mlp_mode": "ptpc_fp8",
                "kv_cache_mode": "native",
                "attention_mode": "native",
            },
            {
                "self_attn_mode": "native",
                "mlp_mode": "mxfp4",
                "kv_cache_mode": "native",
                "attention_mode": "native",
            },
            {
                "self_attn_mode": "ptpc_fp8",
                "mlp_mode": "mxfp4",
                "kv_cache_mode": "native",
                "attention_mode": "native",
            },
        ]

        assert _find_roofline_start_config(configs, [0.2, 0.6, 1.0], anchor_mode) == expected_idx

    def test_missing_preferred_mode_uses_fastest_allowed_mlp_only_candidate(self):
        from quark.experimental.torch.mix_precision.run_helpers import _find_roofline_start_config

        configs = [
            {"self_attn_mode": "native", "mlp_mode": "fp8"},
            {"self_attn_mode": "native", "mlp_mode": "ptpc_fp8"},
            {"self_attn_mode": "fp8", "mlp_mode": "fp8"},
        ]

        assert _find_roofline_start_config(configs, [0.4, 0.7, 1.0], "mxfp4") == 1

    def test_missing_mlp_candidate_uses_lowest_roofline_score(self):
        from quark.experimental.torch.mix_precision.run_helpers import _find_roofline_start_config

        configs = [
            {"self_attn_mode": "fp8", "kv_cache_mode": "native"},
            {"self_attn_mode": "ptpc_fp8", "kv_cache_mode": "native"},
        ]

        assert _find_roofline_start_config(configs, [0.9, 0.1], "mxfp4") == 1


class TestRooflineEarlyStop:
    def test_invalid_anchor_descends_when_lower_neighbor_is_unvisited(self):
        from quark.experimental.torch.mix_precision.run_helpers import _should_stop_search_early

        assert not _should_stop_search_early(
            early_stop=True,
            is_valid=False,
            current_idx=1,
            scores=[0.0, 0.5, 1.0],
            visited={1},
            has_valid_config=False,
        )

    def test_invalid_anchor_stops_after_lower_neighbors_are_exhausted(self):
        from quark.experimental.torch.mix_precision.run_helpers import _should_stop_search_early

        assert _should_stop_search_early(
            early_stop=True,
            is_valid=False,
            current_idx=0,
            scores=[0.0, 0.5, 1.0],
            visited={0, 1},
            has_valid_config=False,
        )

    def test_invalid_config_above_valid_frontier_stops_immediately(self):
        from quark.experimental.torch.mix_precision.run_helpers import _should_stop_search_early

        assert _should_stop_search_early(
            early_stop=True,
            is_valid=False,
            current_idx=4,
            scores=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            visited={1, 2, 3, 4},
            has_valid_config=True,
        )

    @pytest.mark.parametrize(("early_stop", "is_valid"), [(False, False), (True, True)])
    def test_search_does_not_stop_when_disabled_or_valid(self, early_stop, is_valid):
        from quark.experimental.torch.mix_precision.run_helpers import _should_stop_search_early

        assert not _should_stop_search_early(
            early_stop=early_stop,
            is_valid=is_valid,
            current_idx=1,
            scores=[1.0, 0.5],
            visited={1},
            has_valid_config=is_valid,
        )


class TestVllmMoeBackendArgs:
    @pytest.mark.parametrize(
        "args,expected",
        [([], 0.75), (["--gpu-memory-utilization", "0.8"], 0.8), (["--gpu_memory_utilization=0.65"], 0.65)],
    )
    def test_search_memory_default_and_explicit_value(self, args, expected):
        from quark.experimental.torch.mix_precision.run_helpers import build_vllm_engine_kwargs

        assert build_vllm_engine_kwargs(args, object)["gpu_memory_utilization"] == expected

    @pytest.mark.parametrize("value", ["0", "-0.1", "1.01", "nan", "inf", "invalid", ""])
    def test_search_rejects_invalid_memory_utilization(self, value):
        from quark.experimental.torch.mix_precision.run_helpers import build_vllm_engine_kwargs

        with pytest.raises(ValueError, match="gpu.memory.utilization"):
            build_vllm_engine_kwargs([f"--gpu-memory-utilization={value}"], object)

    def test_keeps_backend_omitted_for_vllm_auto_selection(self):
        from quark.experimental.torch.mix_precision.run_helpers import _validate_moe_backend

        original = ["--tensor-parallel-size", "2"]
        result = _validate_moe_backend(original)

        assert result == ["--tensor-parallel-size", "2"]
        assert original == ["--tensor-parallel-size", "2"]

    @pytest.mark.parametrize(
        "args",
        [
            ["--moe-backend", "auto"],
            ["--moe-backend", "aiter"],
            ["--moe-backend", "aiter-mxfp4-bf16"],
            ["--moe-backend", "triton"],
            ["--moe-backend=triton"],
            ["--moe-backend=triton_unfused"],
            ["--moe-backend", "triton-unfused"],
            ["--moe-backend", "emulation"],
        ],
    )
    def test_accepts_supported_backend(self, args):
        from quark.experimental.torch.mix_precision.run_helpers import _validate_moe_backend

        assert _validate_moe_backend(args) == args

    @pytest.mark.parametrize("backend", ["cuda", "cpu"])
    def test_rejects_unsupported_backend(self, backend):
        from quark.experimental.torch.mix_precision.run_helpers import _validate_moe_backend

        with pytest.raises(ValueError, match="Mixed-precision search supports"):
            _validate_moe_backend(["--moe-backend", backend])

    def test_build_kwargs_leaves_moe_backend_at_vllm_default(self):
        from quark.experimental.torch.mix_precision.run_helpers import build_vllm_engine_kwargs

        assert "moe_backend" not in build_vllm_engine_kwargs([], object)

    def test_missing_backend_value_is_rejected(self):
        from quark.experimental.torch.mix_precision.run_helpers import _validate_moe_backend

        with pytest.raises(ValueError, match="requires a backend value"):
            _validate_moe_backend(["--moe-backend"])

    def test_server_command_leaves_moe_backend_at_vllm_default(self):
        from quark.experimental.torch.mix_precision.run_helpers import extend_vllm_server_cmd

        command = ["vllm", "serve", "model"]
        extend_vllm_server_cmd(command, [])

        assert command == ["vllm", "serve", "model", "--enforce-eager"]

    def test_fp8_to_mxfp4_moe_search_uses_canonical_triton_layout(self):
        from quark.experimental.torch.mix_precision.moe_backend import resolve_search_moe_backend

        args, resolution = resolve_search_moe_backend(
            ["--tensor-parallel-size=8"],
            [{"routed_moe_mode": "mxfp4"}],
            {"routed_moe": 8},
            source_weight_mode="fp8",
            model_config=None,
        )

        assert args == ["--tensor-parallel-size=8", "--moe-backend=auto"]
        assert resolution.selected == "pending"
        assert resolution.requires_weight_requantization

    def test_ptpc_fp8_to_fp8_moe_search_uses_canonical_triton_layout(self):
        from quark.experimental.torch.mix_precision.moe_backend import resolve_search_moe_backend

        args, resolution = resolve_search_moe_backend(
            ["--tensor-parallel-size=8", "--moe-backend=auto"],
            [{"routed_moe_mode": "fp8"}],
            {"routed_moe": 8},
            source_weight_mode="ptpc_fp8",
            model_config=None,
        )

        assert args == ["--tensor-parallel-size=8", "--moe-backend=auto"]
        assert resolution.selected == "pending"
        assert resolution.requires_weight_requantization

    def test_source_compatible_mxfp4_search_preserves_aiter_backend(self):
        from quark.experimental.torch.mix_precision.moe_backend import resolve_search_moe_backend

        args = ["--tensor-parallel-size=8", "--moe-backend=aiter"]

        assert (
            resolve_search_moe_backend(
                args,
                [{"routed_moe_mode": "mxfp4"}],
                {"routed_moe": 4},
                source_weight_mode="mxfp4",
                model_config=None,
            )[0]
            == args
        )

    def test_quark_ptpc_fp8_source_weight_mode_is_detected(self):
        from quark.experimental.torch.mix_precision.quantizer import _resolve_source_weight_mode

        model = SimpleNamespace(
            config=SimpleNamespace(
                quantization_config={
                    "quant_method": "quark",
                    "global_quant_config": {
                        "weight": {
                            "dtype": "fp8_e4m3",
                            "qscheme": "per_channel",
                            "ch_axis": 0,
                            "is_dynamic": False,
                        }
                    },
                }
            )
        )

        assert _resolve_source_weight_mode(model, None) == "ptpc_fp8"


# =============================================================================
# eval.py — pure computation functions
# =============================================================================


class TestExtractStrict:
    def test_extracts_after_hash(self):
        from quark.experimental.torch.mix_precision.eval import extract_strict

        assert extract_strict("Let me calculate. #### 42") == "42"

    def test_negative_number(self):
        from quark.experimental.torch.mix_precision.eval import extract_strict

        assert extract_strict("The answer is #### -7") == "-7"

    def test_no_hash_returns_none(self):
        from quark.experimental.torch.mix_precision.eval import extract_strict

        assert extract_strict("The answer is 42") is None

    def test_empty_string_returns_none(self):
        from quark.experimental.torch.mix_precision.eval import extract_strict

        assert extract_strict("") is None


class TestExtractFlexible:
    def test_extracts_last_number(self):
        from quark.experimental.torch.mix_precision.eval import extract_flexible

        result = extract_flexible("First 3 apples, then 5 more, total 8")
        assert result == "8"

    def test_no_number_returns_none(self):
        from quark.experimental.torch.mix_precision.eval import extract_flexible

        assert extract_flexible("no numbers here") is None

    def test_empty_string_returns_none(self):
        from quark.experimental.torch.mix_precision.eval import extract_flexible

        assert extract_flexible("") is None

    def test_dollar_amount(self):
        from quark.experimental.torch.mix_precision.eval import extract_flexible

        result = extract_flexible("She has $23 left")
        assert result is not None


class TestCalculateExactMatch:
    def test_correct_answer_matches(self):
        from quark.experimental.torch.mix_precision.eval import calculate_exact_match

        assert calculate_exact_match("42", "#### 42") is True

    def test_wrong_answer_no_match(self):
        from quark.experimental.torch.mix_precision.eval import calculate_exact_match

        assert calculate_exact_match("41", "#### 42") is False

    def test_none_prediction_no_match(self):
        from quark.experimental.torch.mix_precision.eval import calculate_exact_match

        assert calculate_exact_match(None, "#### 42") is False

    def test_comma_stripped_from_reference(self):
        from quark.experimental.torch.mix_precision.eval import calculate_exact_match

        # "1,000" in reference should match "1000"
        assert calculate_exact_match("1000", "#### 1,000") is True


class TestEvaluateGsm8kEntry:
    def test_correct_flexible(self):
        from quark.experimental.torch.mix_precision.eval import evaluate_gsm8k_entry

        is_correct, extracted = evaluate_gsm8k_entry("So the total is 8 apples.", "#### 8", strategy="flexible")
        assert is_correct is True
        assert extracted is not None

    def test_wrong_answer(self):
        from quark.experimental.torch.mix_precision.eval import evaluate_gsm8k_entry

        is_correct, _ = evaluate_gsm8k_entry("The answer is 7.", "#### 8", strategy="flexible")
        assert is_correct is False

    def test_strict_strategy(self):
        from quark.experimental.torch.mix_precision.eval import evaluate_gsm8k_entry

        is_correct, _ = evaluate_gsm8k_entry("#### 42", "#### 42", strategy="strict")
        assert is_correct is True

    def test_hybrid_falls_back_to_flexible(self):
        from quark.experimental.torch.mix_precision.eval import evaluate_gsm8k_entry

        # No "####" in output, so strict fails; flexible picks up 42
        is_correct, _ = evaluate_gsm8k_entry("The answer is 42.", "#### 42", strategy="hybrid")
        assert is_correct is True


class TestTruncateAtStopStrings:
    def test_truncates_at_first_stop(self):
        from quark.experimental.torch.mix_precision.eval import _truncate_at_stop_strings

        result = _truncate_at_stop_strings("Answer: 8\nQuestion: next one", ["Question:"])
        assert "Question:" not in result
        assert "8" in result

    def test_no_stop_string_returns_full(self):
        from quark.experimental.torch.mix_precision.eval import _truncate_at_stop_strings

        text = "The answer is 42."
        assert _truncate_at_stop_strings(text, ["STOP"]) == text

    def test_picks_earliest_stop(self):
        from quark.experimental.torch.mix_precision.eval import _truncate_at_stop_strings

        result = _truncate_at_stop_strings("A Q: B Human: C", ["Human:", "Q:"])
        assert result == "A"


class TestGsm8kEvaluateOutputs:
    def test_all_correct(self):
        from quark.experimental.torch.mix_precision.eval import _gsm8k_evaluate_outputs

        generated = ["The answer is 5.", "The answer is 10."]
        references = ["#### 5", "#### 10"]
        correct, total = _gsm8k_evaluate_outputs(generated, references, verbose=False)
        assert correct == 2
        assert total == 2

    def test_all_wrong(self):
        from quark.experimental.torch.mix_precision.eval import _gsm8k_evaluate_outputs

        generated = ["The answer is 99.", "The answer is 99."]
        references = ["#### 5", "#### 10"]
        correct, total = _gsm8k_evaluate_outputs(generated, references, verbose=False)
        assert correct == 0
        assert total == 2

    def test_mixed_correct(self):
        from quark.experimental.torch.mix_precision.eval import _gsm8k_evaluate_outputs

        generated = ["The answer is 5.", "The answer is 99."]
        references = ["#### 5", "#### 10"]
        correct, total = _gsm8k_evaluate_outputs(generated, references, verbose=False)
        assert correct == 1
        assert total == 2


# =============================================================================
# switcher.py
# =============================================================================


class TestNeedsCalibration:
    def test_fp8_requires_calib(self):
        from quark.experimental.torch.mix_precision.switcher import needs_calibration

        assert needs_calibration({"self_attn_mode": "fp8", "mlp_mode": "native"}) is True

    def test_mxfp4_fp8_requires_calib(self):
        from quark.experimental.torch.mix_precision.switcher import needs_calibration

        assert needs_calibration({"self_attn_mode": "mxfp4_fp8"}) is True

    def test_ptpc_fp8_no_calib(self):
        from quark.experimental.torch.mix_precision.switcher import needs_calibration

        assert needs_calibration({"self_attn_mode": "ptpc_fp8", "mlp_mode": "native"}) is False

    def test_mxfp4_no_calib(self):
        from quark.experimental.torch.mix_precision.switcher import needs_calibration

        assert needs_calibration({"self_attn_mode": "mxfp4"}) is False

    def test_native_only_no_calib(self):
        from quark.experimental.torch.mix_precision.switcher import needs_calibration

        assert needs_calibration({"self_attn_mode": "native", "mlp_mode": "native"}) is False

    def test_mixed_fp8_and_ptpc_requires_calib(self):
        from quark.experimental.torch.mix_precision.switcher import needs_calibration

        assert needs_calibration({"self_attn_mode": "ptpc_fp8", "mlp_mode": "fp8"}) is True


# =============================================================================
# vllm_inverse_quantizer.py — pure helpers (no vLLM install needed)
# =============================================================================

# ---------------------------------------------------------------------------
# vLLM stub helpers
# vllm_plugin.py wraps its vLLM import in try/except; inject a minimal stub
# package so VLLM_AVAILABLE=True and module-level constants are populated.
# ---------------------------------------------------------------------------


def _make_vllm_stub() -> dict[str, types.ModuleType]:
    """Build a complete vLLM stub sufficient for vllm_plugin.py module-level init."""
    import torch.nn as _nn

    vllm = types.ModuleType("vllm")
    vllm.__version__ = "0.99.0"

    vllm_config = types.ModuleType("vllm.config")
    vllm_config.get_current_vllm_config_or_none = lambda: None

    fused_moe = types.ModuleType("vllm.model_executor.layers.fused_moe.fused_moe")
    fused_moe.invoke_fused_moe_triton_kernel = MagicMock()

    # FusedMoE must be a real nn.Module subclass so QuantVLLMFusedMoE(QuantMixin) can
    # reference it in type annotations without blowing up at class definition time.
    class _FakeFusedMoE(_nn.Module):
        moe_config = None

        def __init__(self, *a, **kw):
            super().__init__()

    class _FakeSharedFusedMoE(_nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()

    class _FakeMoERunner(_nn.Module):
        def __init__(self, routed_experts=None, *a, **kw):
            super().__init__()
            if routed_experts is not None:
                self.routed_experts = routed_experts

    fused_moe_layer = types.ModuleType("vllm.model_executor.layers.fused_moe.layer")
    fused_moe_layer.FusedMoE = _FakeFusedMoE
    fused_moe_layer.MoERunner = _FakeMoERunner

    shared_moe = types.ModuleType("vllm.model_executor.layers.fused_moe.shared_fused_moe")
    shared_moe.SharedFusedMoE = _FakeSharedFusedMoE

    # vllm_plugin.py defines QuantVLLM* as subclasses of these linear types.
    # They must be real nn.Module subclasses.
    class _FakeLinearBase(_nn.Module):
        input_size = 4
        output_size = 4
        bias = None
        skip_bias_add = False

        def __init__(self, *a, **kw):
            super().__init__()

    class _FakeRowParallelLinear(_FakeLinearBase):
        pass

    class _FakeReplicatedLinear(_FakeLinearBase):
        pass

    class _FakeColumnParallelLinear(_FakeLinearBase):
        pass

    class _FakeMergedColumnParallelLinear(_FakeLinearBase):
        pass

    class _FakeQKVParallelLinear(_FakeLinearBase):
        pass

    class _FakeUnquantizedLinearMethod:
        pass

    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.ReplicatedLinear = _FakeReplicatedLinear
    linear.RowParallelLinear = _FakeRowParallelLinear
    linear.ColumnParallelLinear = _FakeColumnParallelLinear
    linear.MergedColumnParallelLinear = _FakeMergedColumnParallelLinear
    linear.QKVParallelLinear = _FakeQKVParallelLinear
    linear.UnquantizedLinearMethod = _FakeUnquantizedLinearMethod

    sampling = types.ModuleType("vllm.sampling_params")
    sampling.SamplingParams = MagicMock

    v1_core_sched_output = types.ModuleType("vllm.v1.core.sched.output")
    v1_core_sched_output.CachedRequestData = MagicMock
    v1_core_sched_output.NewRequestData = MagicMock
    v1_core_sched_output.SchedulerOutput = MagicMock

    v1_worker_gpu = types.ModuleType("vllm.v1.worker.gpu_worker")
    v1_worker_gpu.Worker = MagicMock

    return {
        "vllm": vllm,
        "vllm.config": vllm_config,
        "vllm.sampling_params": sampling,
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.fused_moe": types.ModuleType("vllm.model_executor.layers.fused_moe"),
        "vllm.model_executor.layers.fused_moe.fused_moe": fused_moe,
        "vllm.model_executor.layers.fused_moe.layer": fused_moe_layer,
        "vllm.model_executor.layers.fused_moe.shared_fused_moe": shared_moe,
        "vllm.model_executor.layers.linear": linear,
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.core": types.ModuleType("vllm.v1.core"),
        "vllm.v1.core.sched": types.ModuleType("vllm.v1.core.sched"),
        "vllm.v1.core.sched.output": v1_core_sched_output,
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.gpu_worker": v1_worker_gpu,
    }


def _inject_vllm_stubs() -> dict[str, Any]:
    stubs = _make_vllm_stub()
    previous: dict[str, Any] = {}
    for name, mod in stubs.items():
        is_package = any(other.startswith(name + ".") for other in stubs)
        mod.__spec__ = ModuleSpec(
            name,
            loader=None,
            is_package=is_package,
        )
        if is_package:
            mod.__path__ = []
        previous[name] = sys.modules.get(name)
        sys.modules[name] = mod
    return previous


def _remove_vllm_stubs(previous: dict[str, Any]) -> None:
    for name, old in previous.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


# Inject stubs and import vllm_plugin once at module load time so every test
# that does `from quark.experimental.torch.plugin.vllm_plugin import ...` gets the
# stub-backed version from sys.modules cache.
_vllm_stub_previous = _inject_vllm_stubs()
for _k in list(sys.modules.keys()):
    if _k.startswith("quark.experimental.torch.plugin"):
        sys.modules.pop(_k, None)
import quark.experimental.torch.plugin.vllm_plugin as _vllm_plugin_mod  # noqa: E402

# Remove vllm stubs from sys.modules after import. The module-level bindings
# in vllm_plugin already captured the stub objects, so removing them here
# does not break functionality. This prevents importlib.util.find_spec("vllm")
# from raising ValueError when other test files are collected (Python raises
# ValueError when a module in sys.modules has __spec__ = None).
_remove_vllm_stubs(_vllm_stub_previous)


@pytest.fixture(scope="module")
def vllm_plugin():
    """Return the already-imported (stub-backed) vllm_plugin module."""
    return _vllm_plugin_mod


# ---------------------------------------------------------------------------
# Module factory helpers
# ---------------------------------------------------------------------------


def _make_fp8_linear_module(class_name: str = "RowParallelLinear", use_scale_inv: bool = True) -> nn.Module:
    class Fp8LinearMethod:
        pass

    DynamicCls = type(class_name, (nn.Module,), {"__init__": lambda self: super(DynamicCls, self).__init__()})
    module = DynamicCls()
    module.quant_method = Fp8LinearMethod()
    module.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.float8_e4m3fn))
    if use_scale_inv:
        module.weight_scale_inv = torch.ones(4, 1)
    else:
        module.weight_scale = torch.ones(4, 1)
    return module


def _make_fp8_moe_module(class_name: str = "FusedMoE", use_scale_inv: bool = True) -> nn.Module:
    class Fp8MoEMethod:
        pass

    DynamicCls = type(class_name, (nn.Module,), {"__init__": lambda self: super(DynamicCls, self).__init__()})
    module = DynamicCls()
    module.quant_method = Fp8MoEMethod()
    module.w13_weight = torch.zeros(2, 4, 4, dtype=torch.float8_e4m3fn)
    module.w2_weight = torch.zeros(2, 4, 4, dtype=torch.float8_e4m3fn)
    if use_scale_inv:
        module.w13_weight_scale_inv = torch.ones(2, 1)
        module.w2_weight_scale_inv = torch.ones(2, 1)
    else:
        module.w13_weight_scale = torch.ones(2, 1)
        module.w2_weight_scale = torch.ones(2, 1)
    return module


def _make_compressed_tensors_fp8_module(module_kind: str, strategy: Any) -> nn.Module:
    """Mimic vLLM's linear scheme or MoE method without compressed-tensors installed."""
    if module_kind == "linear":
        module = _make_fp8_linear_module("ReplicatedLinear")
        scheme = type("CompressedTensorsW8A8Fp8", (), {})()
        scheme.strategy = strategy
        scheme.is_static_input_scheme = True
        module.scheme = scheme
    else:
        module = _make_fp8_moe_module()
        scheme = type("CompressedTensorsW8A8Fp8MoEMethod", (), {})()
        scheme.static_input_scales = getattr(strategy, "value", strategy) == "tensor"
        module.quant_method = scheme
    scheme.weight_quant = SimpleNamespace(strategy=strategy)
    scheme.weight_block_size = None
    return module


def _make_mxfp4_moe_module() -> nn.Module:
    class Backend:
        value = "TRITON"

    class Mxfp4MoEMethod:
        weight_dtype = "gpt_oss_mxfp4"
        mxfp4_backend = Backend()

    DynamicCls = type("FusedMoE", (nn.Module,), {"__init__": lambda self: super(DynamicCls, self).__init__()})
    module = DynamicCls()
    module.quant_method = Mxfp4MoEMethod()
    module.w13_weight = torch.zeros(2, 4, dtype=torch.uint8)
    module.w2_weight = torch.zeros(2, 4, dtype=torch.uint8)
    module.w13_weight_scale = torch.ones(2, 1)
    module.w2_weight_scale = torch.ones(2, 1)
    return module


def _make_w4a8_moe_module() -> nn.Module:
    class QuarkW4A8Fp8MoEMethod:
        pass

    DynamicCls = type("RoutedExperts", (nn.Module,), {"__init__": lambda self: super(DynamicCls, self).__init__()})
    module = DynamicCls()
    module.quant_method = QuarkW4A8Fp8MoEMethod()
    module.w13_weight = torch.zeros(2, 4, 1, dtype=torch.int32)
    module.w2_weight = torch.zeros(2, 4, 1, dtype=torch.int32)
    module.w13_weight_scale_2 = torch.ones(2, 4)
    module.w2_weight_scale_2 = torch.ones(2, 4)
    return module


# ---------------------------------------------------------------------------
# _tensor_is_fp8
# ---------------------------------------------------------------------------


class TestTensorIsFp8:
    def test_fp8_e4m3_returns_true(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _tensor_is_fp8

        assert _tensor_is_fp8(torch.zeros(4, dtype=torch.float8_e4m3fn)) is True

    def test_bf16_returns_false(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _tensor_is_fp8

        assert _tensor_is_fp8(torch.zeros(4, dtype=torch.bfloat16)) is False

    def test_none_returns_false(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _tensor_is_fp8

        assert _tensor_is_fp8(None) is False


# ---------------------------------------------------------------------------
# _is_fp8_quant_method / _is_mxfp4_quant_method
# ---------------------------------------------------------------------------


class TestIsFp8QuantMethod:
    def test_fp8_class_name_detected(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _is_fp8_quant_method

        class Fp8LinearMethod:
            pass

        module = nn.Linear(4, 4)
        module.quant_method = Fp8LinearMethod()
        assert _is_fp8_quant_method(module) is True

    def test_no_quant_method_returns_false(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _is_fp8_quant_method

        assert _is_fp8_quant_method(nn.Linear(4, 4)) is False

    def test_non_fp8_method_returns_false(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _is_fp8_quant_method

        class Int8LinearMethod:
            pass

        module = nn.Linear(4, 4)
        module.quant_method = Int8LinearMethod()
        assert _is_fp8_quant_method(module) is False


class TestIsMxfp4QuantMethod:
    def test_mxfp4_class_name_detected(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _is_mxfp4_quant_method

        class Mxfp4LinearMethod:
            pass

        module = nn.Linear(4, 4)
        module.quant_method = Mxfp4LinearMethod()
        assert _is_mxfp4_quant_method(module) is True

    def test_weight_dtype_mxfp4_detected(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _is_mxfp4_quant_method

        class SomeMoEMethod:
            weight_dtype = "mxfp4"

        module = nn.Linear(4, 4)
        module.quant_method = SomeMoEMethod()
        assert _is_mxfp4_quant_method(module) is True

    def test_gpt_oss_mxfp4_weight_dtype_detected(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _is_mxfp4_quant_method

        class GptOssMoEMethod:
            weight_dtype = "gpt_oss_mxfp4"

        module = nn.Linear(4, 4)
        module.quant_method = GptOssMoEMethod()
        assert _is_mxfp4_quant_method(module) is True

    def test_fp8_method_not_mxfp4(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _is_mxfp4_quant_method

        class Fp8LinearMethod:
            pass

        module = nn.Linear(4, 4)
        module.quant_method = Fp8LinearMethod()
        assert _is_mxfp4_quant_method(module) is False


# ---------------------------------------------------------------------------
# _mxfp4_backend_name
# ---------------------------------------------------------------------------


class TestMxfp4BackendName:
    def test_enum_backend_returns_string(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _mxfp4_backend_name

        class Backend:
            value = "TRITON"

        class MockMethod:
            mxfp4_backend = Backend()

        module = nn.Linear(4, 4)
        module.quant_method = MockMethod()
        assert _mxfp4_backend_name(module) == "TRITON"

    def test_no_quant_method_returns_none(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _mxfp4_backend_name

        assert _mxfp4_backend_name(nn.Linear(4, 4)) is None

    def test_no_backend_attr_returns_none(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _mxfp4_backend_name

        class MockMethod:
            pass

        module = nn.Linear(4, 4)
        module.quant_method = MockMethod()
        assert _mxfp4_backend_name(module) is None


# ---------------------------------------------------------------------------
# _tensor_is_uint8_or_triton_mxfp4
# ---------------------------------------------------------------------------


class TestTensorIsUint8OrTritonMxfp4:
    def test_uint8_tensor_returns_true(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _tensor_is_uint8_or_triton_mxfp4

        assert _tensor_is_uint8_or_triton_mxfp4(torch.zeros(4, dtype=torch.uint8)) is True

    def test_non_uint8_tensor_returns_false(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _tensor_is_uint8_or_triton_mxfp4

        assert _tensor_is_uint8_or_triton_mxfp4(torch.zeros(4, dtype=torch.bfloat16)) is False

    def test_triton_tensor_with_uint8_storage_returns_true(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _tensor_is_uint8_or_triton_mxfp4

        storage = SimpleNamespace(data=torch.zeros(4, dtype=torch.uint8))
        assert _tensor_is_uint8_or_triton_mxfp4(SimpleNamespace(storage=storage)) is True

    def test_triton_tensor_with_non_uint8_storage_returns_false(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import _tensor_is_uint8_or_triton_mxfp4

        storage = SimpleNamespace(data=torch.zeros(4, dtype=torch.float32))
        assert _tensor_is_uint8_or_triton_mxfp4(SimpleNamespace(storage=storage)) is False


# ---------------------------------------------------------------------------
# is_prequantized_vllm_linear
# ---------------------------------------------------------------------------


class TestIsPrequantizedVllmLinear:
    def test_valid_fp8_linear_detected(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_linear

        assert is_prequantized_vllm_linear(_make_fp8_linear_module("RowParallelLinear")) is True

    def test_valid_fp8_linear_with_weight_scale(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_linear

        assert is_prequantized_vllm_linear(_make_fp8_linear_module("ColumnParallelLinear", use_scale_inv=False)) is True

    def test_replicated_fp8_linear_is_detected(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_linear

        assert is_prequantized_vllm_linear(_make_fp8_linear_module("ReplicatedLinear")) is True

    def test_scheme_backed_fp8_linear_is_detected(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_linear

        class QuarkLinearMethod:
            pass

        class QuarkW8A8Fp8:
            pass

        module = _make_fp8_linear_module("RowParallelLinear")
        module.quant_method = QuarkLinearMethod()
        module.scheme = QuarkW8A8Fp8()

        assert is_prequantized_vllm_linear(module) is True

    def test_scheme_backed_fp8_linear_builds_inverse_quantizer(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import (
            VLLMFp8LinearInverseQuantizer,
            create_inverse_quantizer_for_vllm_linear,
        )

        class QuarkLinearMethod:
            pass

        class QuarkW8A8Fp8:
            pass

        module = _make_fp8_linear_module("RowParallelLinear")
        module.quant_method = QuarkLinearMethod()
        module.scheme = QuarkW8A8Fp8()

        assert isinstance(create_inverse_quantizer_for_vllm_linear(module), VLLMFp8LinearInverseQuantizer)

    def test_unknown_class_name_returns_false(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_linear

        assert is_prequantized_vllm_linear(_make_fp8_linear_module("UnknownLinear")) is False

    def test_non_fp8_weight_returns_false(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_linear

        module = _make_fp8_linear_module("RowParallelLinear")
        module.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
        assert is_prequantized_vllm_linear(module) is False

    def test_missing_scale_returns_false(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_linear

        class Fp8LinearMethod:
            pass

        DynamicCls = type(
            "RowParallelLinear", (nn.Module,), {"__init__": lambda self: super(DynamicCls, self).__init__()}
        )
        module = DynamicCls()
        module.quant_method = Fp8LinearMethod()
        module.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.float8_e4m3fn))
        assert is_prequantized_vllm_linear(module) is False


# ---------------------------------------------------------------------------
# is_prequantized_vllm_fp8_moe / mxfp4_moe / moe
# ---------------------------------------------------------------------------


class TestIsPrequantizedVllmFp8Moe:
    def test_valid_fp8_moe_detected(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_fp8_moe

        assert is_prequantized_vllm_fp8_moe(_make_fp8_moe_module()) is True

    def test_fp8_moe_with_weight_scale_detected(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_fp8_moe

        assert is_prequantized_vllm_fp8_moe(_make_fp8_moe_module(use_scale_inv=False)) is True

    def test_unknown_class_returns_false(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_fp8_moe

        assert is_prequantized_vllm_fp8_moe(_make_fp8_moe_module("UnknownMoE")) is False

    def test_non_fp8_weights_returns_false(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_fp8_moe

        module = _make_fp8_moe_module()
        module.w13_weight = torch.zeros(2, 4, 4, dtype=torch.bfloat16)
        assert is_prequantized_vllm_fp8_moe(module) is False


class TestIsPrequantizedVllmMxfp4Moe:
    def test_valid_mxfp4_moe_detected(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_mxfp4_moe

        assert is_prequantized_vllm_mxfp4_moe(_make_mxfp4_moe_module()) is True

    @pytest.mark.parametrize("backend_name", ["CUDA_CUSTOM", "TRITON_UNFUSED", "AITER_MXFP4_BF16", "XPU"])
    def test_unsupported_backend_returns_false(self, backend_name):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_mxfp4_moe

        class UnsupportedBackend:
            value = backend_name

        class Mxfp4Method:
            weight_dtype = "gpt_oss_mxfp4"
            mxfp4_backend = UnsupportedBackend()

        DynamicCls = type("FusedMoE", (nn.Module,), {"__init__": lambda self: super(DynamicCls, self).__init__()})
        module = DynamicCls()
        module.quant_method = Mxfp4Method()
        module.w13_weight = torch.zeros(2, 4, dtype=torch.uint8)
        module.w2_weight = torch.zeros(2, 4, dtype=torch.uint8)
        module.w13_weight_scale = torch.ones(2, 1)
        module.w2_weight_scale = torch.ones(2, 1)
        assert is_prequantized_vllm_mxfp4_moe(module) is False

    def test_missing_scales_returns_false(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_mxfp4_moe

        module = _make_mxfp4_moe_module()
        del module.w13_weight_scale
        assert is_prequantized_vllm_mxfp4_moe(module) is False


class TestIsPrequantizedVllmMoe:
    def test_fp8_moe_returns_true(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_moe

        assert is_prequantized_vllm_moe(_make_fp8_moe_module()) is True

    def test_mxfp4_moe_returns_true(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_moe

        assert is_prequantized_vllm_moe(_make_mxfp4_moe_module()) is True

    def test_plain_module_returns_false(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_moe

        assert is_prequantized_vllm_moe(nn.Linear(4, 4)) is False

    def test_unsupported_packed_source_is_still_routed_as_prequantized(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import is_prequantized_vllm_moe

        assert is_prequantized_vllm_moe(_make_w4a8_moe_module()) is True


# ---------------------------------------------------------------------------
# VLLMFp8LinearInverseQuantizer — constructor guards
# ---------------------------------------------------------------------------


class TestVLLMFp8LinearInverseQuantizerGuards:
    def test_raises_without_fp8_quant_method(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        with pytest.raises(ValueError, match="Unsupported vLLM module"):
            VLLMFp8LinearInverseQuantizer(nn.Linear(4, 4))

    def test_raises_when_use_deep_gemm(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        class Fp8MethodDeepGemm:
            use_deep_gemm = True
            use_marlin = False

        module = _make_fp8_linear_module("RowParallelLinear")
        module.quant_method = Fp8MethodDeepGemm()
        with pytest.raises(NotImplementedError, match="use_deep_gemm=True"):
            VLLMFp8LinearInverseQuantizer(module)

    def test_raises_when_use_marlin(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        class Fp8MethodMarlin:
            use_deep_gemm = False
            use_marlin = True

        module = _make_fp8_linear_module("RowParallelLinear")
        module.quant_method = Fp8MethodMarlin()
        with pytest.raises(NotImplementedError, match="use_marlin=True"):
            VLLMFp8LinearInverseQuantizer(module)

    def test_raises_when_no_scale(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        class Fp8LinearMethod:
            pass

        DynamicCls = type(
            "RowParallelLinear", (nn.Module,), {"__init__": lambda self: super(DynamicCls, self).__init__()}
        )
        module = DynamicCls()
        module.quant_method = Fp8LinearMethod()
        module.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.float8_e4m3fn))
        with pytest.raises(ValueError, match="weight_scale"):
            VLLMFp8LinearInverseQuantizer(module)

    def test_constructed_with_scale_inv(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        inv_q = VLLMFp8LinearInverseQuantizer(_make_fp8_linear_module("RowParallelLinear"))
        assert isinstance(inv_q.scale, torch.Tensor)

    def test_block_quant_detected_from_weight_block_size(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        class Fp8BlockMethod:
            use_deep_gemm = False
            use_marlin = False
            block_quant = True

        module = _make_fp8_linear_module("RowParallelLinear")
        module.weight_block_size = (128, 128)
        module.quant_method = Fp8BlockMethod()
        assert VLLMFp8LinearInverseQuantizer(module).block_quant is True


# ---------------------------------------------------------------------------
# VLLMFp8MoEWeightInverseQuantizer — constructor guards
# ---------------------------------------------------------------------------


class TestVLLMFp8MoEWeightInverseQuantizerGuards:
    def test_raises_without_fp8_quant_method(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        with pytest.raises(ValueError, match="Unsupported vLLM MoE"):
            VLLMFp8MoEWeightInverseQuantizer(nn.Linear(4, 4), "w13_weight_scale_inv")

    def test_raises_when_use_deep_gemm(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        class Fp8MoEMethodDeepGemm:
            use_deep_gemm = True

        module = _make_fp8_moe_module()
        module.quant_method = Fp8MoEMethodDeepGemm()
        with pytest.raises(NotImplementedError, match="use_deep_gemm=True"):
            VLLMFp8MoEWeightInverseQuantizer(module, "w13_weight_scale_inv")

    def test_raises_when_scale_attr_missing(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        with pytest.raises(ValueError, match="must have nonexistent_scale"):
            VLLMFp8MoEWeightInverseQuantizer(_make_fp8_moe_module(), "nonexistent_scale")

    def test_constructed_with_scale_inv(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        inv_q = VLLMFp8MoEWeightInverseQuantizer(_make_fp8_moe_module(use_scale_inv=True), "w13_weight_scale_inv")
        assert isinstance(inv_q.scale, torch.Tensor)


# ---------------------------------------------------------------------------
# VLLMMxfp4MoEWeightInverseQuantizer — constructor guards
# ---------------------------------------------------------------------------


class TestVLLMMxfp4MoEWeightInverseQuantizerGuards:
    def test_raises_without_mxfp4_quant_method(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        with pytest.raises(ValueError, match="Unsupported vLLM MoE"):
            VLLMMxfp4MoEWeightInverseQuantizer(nn.Linear(4, 4), "w13_weight_scale")

    @pytest.mark.parametrize("backend_name", ["CUDA_ONLY", "TRITON_UNFUSED", "AITER_MXFP4_BF16", "XPU"])
    def test_raises_unsupported_backend(self, backend_name):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        class BadBackend:
            value = backend_name

        class Mxfp4Method:
            weight_dtype = "gpt_oss_mxfp4"
            mxfp4_backend = BadBackend()

        DynamicCls = type("FusedMoE", (nn.Module,), {"__init__": lambda self: super(DynamicCls, self).__init__()})
        module = DynamicCls()
        module.quant_method = Mxfp4Method()
        module.w13_weight_scale = torch.ones(2, 1)
        with pytest.raises(NotImplementedError, match="backend="):
            VLLMMxfp4MoEWeightInverseQuantizer(module, "w13_weight_scale")

    def test_raises_when_scale_not_tensor(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        module = _make_mxfp4_moe_module()
        object.__getattribute__(module, "__dict__")["w13_weight_scale"] = "not_a_tensor"
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            VLLMMxfp4MoEWeightInverseQuantizer(module, "w13_weight_scale")

    def test_constructed_successfully(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        inv_q = VLLMMxfp4MoEWeightInverseQuantizer(_make_mxfp4_moe_module(), "w13_weight_scale")
        assert inv_q.backend_name == "TRITON"
        assert isinstance(inv_q.scale, torch.Tensor)


# ---------------------------------------------------------------------------
# create_inverse_quantizer_for_vllm_linear / create_vllm_moe_inverse_quantizers
# ---------------------------------------------------------------------------


class TestCreateInverseQuantizerFunctions:
    def test_create_linear_raises_if_not_prequantized(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import create_inverse_quantizer_for_vllm_linear

        with pytest.raises(ValueError, match="not a pre-quantized"):
            create_inverse_quantizer_for_vllm_linear(nn.Linear(4, 4))

    def test_create_moe_raises_if_not_prequantized(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import create_vllm_moe_inverse_quantizers

        with pytest.raises(ValueError, match="not a supported"):
            create_vllm_moe_inverse_quantizers(nn.Linear(4, 4))

    def test_create_linear_returns_fp8_inverse_quantizer(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import (
            VLLMFp8LinearInverseQuantizer,
            create_inverse_quantizer_for_vllm_linear,
        )

        assert isinstance(
            create_inverse_quantizer_for_vllm_linear(_make_fp8_linear_module("RowParallelLinear")),
            VLLMFp8LinearInverseQuantizer,
        )

    def test_create_moe_returns_fp8_pair(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import (
            VLLMFp8MoEWeightInverseQuantizer,
            create_vllm_moe_inverse_quantizers,
        )

        w13_q, w2_q = create_vllm_moe_inverse_quantizers(_make_fp8_moe_module(use_scale_inv=True))
        assert isinstance(w13_q, VLLMFp8MoEWeightInverseQuantizer)
        assert isinstance(w2_q, VLLMFp8MoEWeightInverseQuantizer)

    def test_create_moe_returns_mxfp4_pair(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import (
            VLLMMxfp4MoEWeightInverseQuantizer,
            create_vllm_moe_inverse_quantizers,
        )

        w13_q, w2_q = create_vllm_moe_inverse_quantizers(_make_mxfp4_moe_module())
        assert isinstance(w13_q, VLLMMxfp4MoEWeightInverseQuantizer)
        assert isinstance(w2_q, VLLMMxfp4MoEWeightInverseQuantizer)

    def test_create_moe_selects_scale_inv_when_available(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import create_vllm_moe_inverse_quantizers

        w13_q, w2_q = create_vllm_moe_inverse_quantizers(_make_fp8_moe_module(use_scale_inv=True))
        assert w13_q.scale_attr_name == "w13_weight_scale_inv"
        assert w2_q.scale_attr_name == "w2_weight_scale_inv"

    def test_create_moe_falls_back_to_weight_scale(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import create_vllm_moe_inverse_quantizers

        w13_q, w2_q = create_vllm_moe_inverse_quantizers(_make_fp8_moe_module(use_scale_inv=False))
        assert w13_q.scale_attr_name == "w13_weight_scale"
        assert w2_q.scale_attr_name == "w2_weight_scale"

    def test_create_moe_rejects_w4a8_source_explicitly(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import create_vllm_moe_inverse_quantizers

        with pytest.raises(NotImplementedError, match="W4A8 MoE sources.*inverse-conversion"):
            create_vllm_moe_inverse_quantizers(_make_w4a8_moe_module())


class TestVllmSourceTargetMatching:
    @pytest.mark.parametrize(
        "source_weight,target_mode,expected",
        [
            ({"dtype": "fp8_e4m3", "qscheme": "per_tensor"}, "fp8", True),
            ({"dtype": "fp8_e4m3", "qscheme": "per_channel", "ch_axis": 0}, "ptpc_fp8", True),
            (
                {
                    "dtype": "fp8_e4m3",
                    "qscheme": "per_tensor",
                    "block_size": [128, 128],
                },
                "fp8",
                False,
            ),
            (
                {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "ch_axis": -1,
                    "scale_format": "e8m0",
                    "scale_calculation_mode": "even",
                },
                "mxfp4",
                True,
            ),
        ],
    )
    def test_weight_format_comparison_normalizes_source_metadata(
        self,
        source_weight,
        target_mode,
        expected,
    ):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import weight_quantization_formats_match

        target = get_layer_config(target_mode)

        assert weight_quantization_formats_match(source_weight, target.weight) is expected

    @pytest.mark.parametrize("module_factory", [_make_fp8_linear_module, _make_fp8_moe_module])
    def test_fp8_source_weight_matches_fp8_weight_target(self, module_factory):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import vllm_source_weight_matches_target

        module = module_factory()
        assert vllm_source_weight_matches_target(module, get_layer_config("fp8")) is True
        assert vllm_source_weight_matches_target(module, get_layer_config("mxfp4")) is False

    @pytest.mark.parametrize("use_enum", [False, True])
    @pytest.mark.parametrize(
        "source_qscheme,target_mode,expected",
        [
            ("per_tensor", "fp8", True),
            ("per_tensor", "ptpc_fp8", False),
            ("per_channel", "fp8", False),
            ("per_channel", "ptpc_fp8", True),
        ],
    )
    def test_fp8_source_weight_matching_respects_granularity(self, source_qscheme, target_mode, expected, use_enum):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import (
            vllm_source_matches_target,
            vllm_source_weight_matches_target,
        )
        from quark.torch.quantization.config.type import QSchemeType

        module = _make_fp8_linear_module()
        target = get_layer_config(target_mode)
        module.scheme = SimpleNamespace(
            weight_dtype="fp8",
            weight_qscheme=QSchemeType(source_qscheme) if use_enum else source_qscheme,
            input_qscheme=target.input_tensors.qscheme,
            is_static_input_scheme=not target.input_tensors.is_dynamic,
        )

        assert vllm_source_weight_matches_target(module, target) is expected
        assert vllm_source_matches_target(module, target) is expected

    @pytest.mark.parametrize("block_owner", ["module", "scheme", "quant_method"])
    @pytest.mark.parametrize("target_mode", ["fp8", "ptpc_fp8"])
    def test_fp8_source_weight_matching_rejects_blocks(self, block_owner, target_mode):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import (
            vllm_source_matches_target,
            vllm_source_weight_matches_target,
        )

        module = _make_fp8_linear_module()
        target = get_layer_config(target_mode)
        module.scheme = SimpleNamespace(
            weight_dtype="fp8",
            weight_qscheme=target.weight.qscheme,
            input_qscheme=target.input_tensors.qscheme,
            is_static_input_scheme=not target.input_tensors.is_dynamic,
        )
        owner = module if block_owner == "module" else getattr(module, block_owner)
        owner.weight_block_size = (128, 128)

        assert vllm_source_weight_matches_target(module, target) is False
        assert vllm_source_matches_target(module, target) is False

    @pytest.mark.parametrize("module_kind", ["linear", "moe"])
    @pytest.mark.parametrize("enum_like", [False, True])
    @pytest.mark.parametrize(
        "strategy,target_mode,expected",
        [
            ("tensor", "fp8", True),
            ("tensor", "ptpc_fp8", False),
            ("channel", "fp8", False),
            ("channel", "ptpc_fp8", True),
            ("group", "fp8", False),
            ("block", "fp8", False),
            ("unknown", "fp8", False),
            (None, "fp8", False),
        ],
    )
    def test_compressed_tensors_fp8_weight_strategy(self, strategy, target_mode, expected, enum_like, module_kind):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import (
            vllm_source_matches_target,
            vllm_source_weight_matches_target,
        )

        source_strategy = SimpleNamespace(value=strategy) if enum_like and strategy is not None else strategy
        module = _make_compressed_tensors_fp8_module(module_kind, source_strategy)

        assert vllm_source_weight_matches_target(module, get_layer_config(target_mode)) is expected
        assert vllm_source_matches_target(module, get_layer_config("fp8")) is (strategy == "tensor")

    def test_unknown_fp8_scheme_does_not_assume_per_tensor(self):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import (
            vllm_source_matches_target,
            vllm_source_weight_matches_target,
        )

        module = _make_fp8_linear_module()
        module.scheme = SimpleNamespace(weight_dtype="fp8", is_static_input_scheme=True)
        target = get_layer_config("fp8")

        assert vllm_source_weight_matches_target(module, target) is False
        assert vllm_source_matches_target(module, target) is False

    def test_mxfp4_source_weight_matches_mxfp4_weight_targets(self):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import vllm_source_weight_matches_target
        from quark.torch.quantization.config.type import Dtype, QSchemeType

        module = _make_mxfp4_moe_module()
        assert vllm_source_weight_matches_target(module, get_layer_config("mxfp4")) is True
        assert vllm_source_weight_matches_target(module, get_layer_config("mxfp4_fp8")) is True
        assert vllm_source_weight_matches_target(module, get_layer_config("fp8")) is False
        different_group_size = SimpleNamespace(
            weight=SimpleNamespace(
                dtype=Dtype.fp4,
                qscheme=QSchemeType.per_group,
                group_size=16,
                ch_axis=-1,
                scale_format="e8m0",
                scale_calculation_mode="even",
            )
        )
        assert vllm_source_weight_matches_target(module, different_group_size) is False

    def test_activation_only_target_preserves_prequantized_source_weight(self):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import vllm_source_weight_matches_target
        from quark.torch.quantization.config.config import QLayerConfig

        module = _make_mxfp4_moe_module()
        target = QLayerConfig(weight=None, input_tensors=deepcopy(get_layer_config("fp8").input_tensors))

        assert vllm_source_weight_matches_target(module, target) is True

    def test_static_fp8_source_matches_fp8_target(self):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import vllm_source_matches_target

        module = _make_fp8_linear_module("RowParallelLinear")
        module.quant_method.quant_config = SimpleNamespace(activation_scheme="static")
        assert vllm_source_matches_target(module, get_layer_config("fp8")) is True

    def test_fp8_source_does_not_match_mxfp4_target(self):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import vllm_source_matches_target

        module = _make_fp8_linear_module("RowParallelLinear")
        module.quant_method.quant_config = SimpleNamespace(activation_scheme="static")
        assert vllm_source_matches_target(module, get_layer_config("mxfp4")) is False

    def test_output_quantization_keeps_weight_input_source_match(self):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.mix_precision.utils import _get_fp8_output_spec
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import vllm_source_matches_target

        module = _make_fp8_linear_module("QKVParallelLinear")
        module.quant_method.quant_config = SimpleNamespace(activation_scheme="static")
        target = deepcopy(get_layer_config("fp8"))
        target.output_tensors = _get_fp8_output_spec()

        assert vllm_source_matches_target(module, target) is True

    def test_bias_quantization_prevents_source_match(self):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import vllm_source_matches_target

        module = _make_fp8_linear_module("RowParallelLinear")
        module.quant_method.quant_config = SimpleNamespace(activation_scheme="static")
        target = deepcopy(get_layer_config("fp8"))
        target.bias = deepcopy(target.weight)

        assert vllm_source_matches_target(module, target) is False


# =============================================================================
# vllm_plugin.py — pure utility functions (vLLM stubs injected via fixture)
# =============================================================================


class TestEnvEnabled:
    def test_env_var_1_returns_true(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_plugin import _env_enabled

        monkeypatch.setenv("TEST_QUARK_FLAG", "1")
        assert _env_enabled("TEST_QUARK_FLAG") is True

    def test_env_var_true_returns_true(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_plugin import _env_enabled

        monkeypatch.setenv("TEST_QUARK_FLAG", "true")
        assert _env_enabled("TEST_QUARK_FLAG") is True

    def test_env_var_0_returns_false(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_plugin import _env_enabled

        monkeypatch.setenv("TEST_QUARK_FLAG", "0")
        assert _env_enabled("TEST_QUARK_FLAG") is False

    def test_unset_var_returns_false(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_plugin import _env_enabled

        monkeypatch.delenv("TEST_QUARK_FLAG", raising=False)
        assert _env_enabled("TEST_QUARK_FLAG") is False

    def test_custom_default_used_when_unset(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_plugin import _env_enabled

        monkeypatch.delenv("TEST_QUARK_FLAG2", raising=False)
        assert _env_enabled("TEST_QUARK_FLAG2", default="1") is True


class TestKvCacheDtypeForCalib:
    def test_fp8_dtype_becomes_auto_in_calib_phase(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_plugin import _kv_cache_dtype_for_calib

        monkeypatch.setenv("QUARK_CALIB_PHASE", "1")
        assert _kv_cache_dtype_for_calib("fp8_e4m3") == "auto"

    def test_fp8_dtype_unchanged_outside_calib_phase(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_plugin import _kv_cache_dtype_for_calib

        monkeypatch.setenv("QUARK_CALIB_PHASE", "0")
        assert _kv_cache_dtype_for_calib("fp8_e4m3") == "fp8_e4m3"

    def test_non_fp8_dtype_unchanged_in_calib_phase(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_plugin import _kv_cache_dtype_for_calib

        monkeypatch.setenv("QUARK_CALIB_PHASE", "1")
        assert _kv_cache_dtype_for_calib("auto") == "auto"

    def test_fp8_dtype_unchanged_without_env_var(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_plugin import _kv_cache_dtype_for_calib

        monkeypatch.delenv("QUARK_CALIB_PHASE", raising=False)
        assert _kv_cache_dtype_for_calib("fp8_e5m2") == "fp8_e5m2"


class TestUnwrapFakeQuantMethod:
    def test_non_fake_quant_method_returned_as_is(self):
        from quark.experimental.torch.plugin.vllm_plugin import _unwrap_fake_quant_method

        class SomeMethod:
            pass

        m = SomeMethod()
        assert _unwrap_fake_quant_method(m) is m

    def test_single_layer_unwrapped(self):
        from quark.experimental.torch.plugin.vllm_plugin import FakeQuantLinearMethod, _unwrap_fake_quant_method

        class Original:
            pass

        orig = Original()
        wrapped = MagicMock(spec=FakeQuantLinearMethod)
        wrapped.original_quant_method = orig
        wrapped.__class__ = FakeQuantLinearMethod
        assert _unwrap_fake_quant_method(wrapped) is orig

    def test_double_wrapped_unwrapped_fully(self):
        from quark.experimental.torch.plugin.vllm_plugin import FakeQuantLinearMethod, _unwrap_fake_quant_method

        class Inner:
            pass

        inner = Inner()
        mid = FakeQuantLinearMethod.__new__(FakeQuantLinearMethod)
        mid.original_quant_method = inner
        outer = FakeQuantLinearMethod.__new__(FakeQuantLinearMethod)
        outer.original_quant_method = mid
        assert _unwrap_fake_quant_method(outer) is inner


class TestFilterSupportedInitKwargs:
    def test_filters_unsupported_params(self):
        from quark.experimental.torch.plugin.vllm_plugin import _filter_supported_init_kwargs

        class Foo:
            def __init__(self, a: int, b: str) -> None:
                pass

        assert _filter_supported_init_kwargs(Foo, {"a": 1, "b": "x", "c": 99}) == {"a": 1, "b": "x"}

    def test_self_always_excluded(self):
        from quark.experimental.torch.plugin.vllm_plugin import _filter_supported_init_kwargs

        class Bar:
            def __init__(self, x: int) -> None:
                pass

        result = _filter_supported_init_kwargs(Bar, {"self": "bad", "x": 5})
        assert "self" not in result
        assert result == {"x": 5}

    def test_empty_kwargs_returns_empty(self):
        from quark.experimental.torch.plugin.vllm_plugin import _filter_supported_init_kwargs

        class Baz:
            def __init__(self, x: int) -> None:
                pass

        assert _filter_supported_init_kwargs(Baz, {}) == {}


class TestAdaptLayerPatternsForVllm:
    def test_q_proj_maps_to_qkv_proj(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("qkv_proj" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj"))

    def test_k_proj_maps_to_qkv_proj(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("qkv_proj" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.k_proj"))

    def test_v_proj_maps_to_qkv_proj(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("qkv_proj" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.v_proj"))

    def test_gate_proj_maps_to_gate_up_proj_and_experts(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        result = adapt_layer_patterns_for_vllm("model.layers.*.mlp.gate_proj")
        assert any("gate_up_proj" in p for p in result)
        assert any("*experts*" in p for p in result)

    def test_up_proj_maps_to_gate_up_proj_and_experts(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        result = adapt_layer_patterns_for_vllm("model.layers.*.mlp.up_proj")
        assert any("gate_up_proj" in p for p in result)
        assert any("*experts*" in p for p in result)

    def test_plural_shared_expert_exclude_maps_to_vllm(self):
        from quark.experimental.torch.plugin.fakequant_worker import _adapt_exclude_patterns_for_vllm

        config = SimpleNamespace(
            exclude=["model.layers.*.mlp.shared_experts.gate_proj"],
            kv_cache_quant_config={},
        )
        _adapt_exclude_patterns_for_vllm(config)
        assert "*shared_expert*" in config.exclude

    def test_routed_moe_gate_exclude_maps_to_vllm_runtime_path(self):
        from quark.experimental.torch.plugin.fakequant_worker import _adapt_exclude_patterns_for_vllm

        config = SimpleNamespace(
            exclude=["model.language_model.layers.0.mlp.gate"],
            kv_cache_quant_config={},
        )
        _adapt_exclude_patterns_for_vllm(config)
        assert "language_model.model.layers.0.mlp.experts.gate" in config.exclude

    def test_routed_source_floor_config_wins_over_dense_mlp_runtime_alias(self, monkeypatch, vllm_plugin):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.fakequant_worker import _adapt_config_for_vllm
        from quark.torch.quantization import model_transformation
        from quark.torch.quantization.config.config import QConfig

        dense_config = get_layer_config("fp8")
        routed_config = deepcopy(dense_config)
        routed_config.weight = None
        config = QConfig(
            global_quant_config=dense_config,
            layer_quant_config={
                "model.layers.0.mlp.gate_proj": dense_config,
                "model.layers.*.mlp.experts.*.w1": routed_config,
            },
            exclude=[],
        )

        model = nn.Module()
        model.mlp = nn.Module()
        model.mlp.experts = vllm_plugin.vllm_moe_runner()
        monkeypatch.setitem(
            model_transformation.LAYER_TO_QUANT_LAYER_MAP,
            vllm_plugin.vllm_moe_runner,
            vllm_plugin.QuantVLLMMoERunner,
        )

        _adapt_config_for_vllm(model, config)

        assert config.layer_quant_config["*experts*"] is routed_config

    def test_shared_expert_gate_exclude_maps_to_alias_and_nested_runtime_path(self):
        from quark.experimental.torch.plugin.fakequant_worker import _adapt_exclude_patterns_for_vllm

        config = SimpleNamespace(
            exclude=["model.language_model.layers.0.mlp.shared_expert_gate"],
            kv_cache_quant_config={},
        )
        _adapt_exclude_patterns_for_vllm(config)
        assert "language_model.model.layers.0.mlp.shared_expert_gate" in config.exclude
        assert "*shared_expert*.expert_gate" in config.exclude

    def test_shared_expert_maps_without_broad_routed_expert_pattern(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        result = adapt_layer_patterns_for_vllm("model.layers.*.mlp.shared_experts.gate_proj")
        assert "*shared_expert*" in result
        assert "*experts*" not in result

    def test_in_proj_qkv_maps_to_in_proj_qkvz(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("in_proj_qkvz" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.linear_attn.in_proj_qkv"))

    def test_in_proj_b_maps_to_in_proj_ba(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("in_proj_ba" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.linear_attn.in_proj_b"))

    def test_in_proj_a_maps_to_in_proj_ba(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("in_proj_ba" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.linear_attn.in_proj_a"))

    def test_no_double_replacement_for_gate_up_proj(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert not any(
            "gate_gate_up_proj" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.mlp.gate_up_proj")
        )

    def test_self_attn_not_expanded_to_linear_attn(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert not any(".linear_attn." in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj"))

    def test_linear_attn_not_expanded_to_self_attn(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert not any(
            ".self_attn." in p for p in adapt_layer_patterns_for_vllm("model.layers.*.linear_attn.in_proj_qkv")
        )

    def test_self_attn_adds_attn_variant(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any(".attn." in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj"))

    def test_prefix_alias_expansion_model_to_language_model(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("language_model." in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj"))

    def test_mla_variant_added_for_self_attn(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any(
            ".self_attn.mla_attn." in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj")
        )

    def test_returns_tuple(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert isinstance(adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj"), tuple)

    def test_unmatched_pattern_returned_unchanged(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        result = adapt_layer_patterns_for_vllm("model.layers.*.lm_head")
        assert isinstance(result, tuple) and len(result) >= 1


class TestRefreshMlaAbsorbedWeights:
    class _FakeQuantLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 2))
            self.quant_method = object()
            self._weight_quantizer = object()
            self._source_matches_target = False
            self._source_weight_matches_target = False

        def get_quant_weight(self, weight):
            return weight.detach() + 1

    class _FakeMla(nn.Module):
        def __init__(self, kv_b_proj):
            super().__init__()
            self.kv_b_proj = kv_b_proj
            self.impl = SimpleNamespace(kv_b_proj=kv_b_proj)
            self.absorbed_weights: list[torch.Tensor] = []

        def process_weights_after_loading(self, act_dtype):
            del act_dtype
            self.absorbed_weights.append(self.kv_b_proj.weight.detach().clone())

    def test_native_candidate_does_not_rebuild_unchanged_absorbed_weights(self, monkeypatch, vllm_plugin):
        monkeypatch.setattr(vllm_plugin, "vllm_mla_attention", self._FakeMla)
        mla = self._FakeMla(nn.Linear(2, 2, bias=False))

        assert vllm_plugin.refresh_mla_absorbed_weights(mla, torch.bfloat16, quantize=True) == 0
        assert vllm_plugin.refresh_mla_absorbed_weights(mla, torch.bfloat16, quantize=False) == 0
        assert mla.absorbed_weights == []

    def test_quantized_absorbed_weights_are_restored_once(self, monkeypatch, vllm_plugin):
        monkeypatch.setattr(vllm_plugin, "vllm_mla_attention", self._FakeMla)
        monkeypatch.setattr(vllm_plugin, "QuantVLLMParallelLinearBase", self._FakeQuantLinear)
        quant_linear = self._FakeQuantLinear()
        mla = self._FakeMla(quant_linear)
        mla.impl.kv_b_proj = object()

        assert vllm_plugin.refresh_mla_absorbed_weights(mla, torch.bfloat16, quantize=True) == 1
        assert mla.impl.kv_b_proj is quant_linear
        torch.testing.assert_close(mla.absorbed_weights[0], torch.ones(2, 2))
        assert mla._quark_mla_absorbed_weight_is_quantized is True

        original = nn.Linear(2, 2, bias=False)
        mla.kv_b_proj = original
        assert vllm_plugin.refresh_mla_absorbed_weights(mla, torch.bfloat16, quantize=False) == 1
        assert mla.impl.kv_b_proj is original
        assert mla._quark_mla_absorbed_weight_is_quantized is False
        assert vllm_plugin.refresh_mla_absorbed_weights(mla, torch.bfloat16, quantize=False) == 0
        assert len(mla.absorbed_weights) == 2

    def test_refreshed_aiter_mla_tensors_keep_their_original_storage(self, vllm_plugin):
        module = nn.Module()
        module.W_K = torch.zeros(2, 3)
        module.W_K_scale = torch.zeros(2, 1)
        old_wk = module.W_K
        old_scale = module.W_K_scale
        previous = {name: getattr(module, name, None) for name in vllm_plugin._MLA_ABSORBED_WEIGHT_ATTRS}

        module.W_K = torch.ones(2, 3)
        module.W_K_scale = torch.full((2, 1), 2.0)
        preserved = vllm_plugin._restore_mla_absorbed_tensor_storage(module, previous)

        assert preserved == 2
        assert module.W_K is old_wk
        assert module.W_K_scale is old_scale
        torch.testing.assert_close(module.W_K, torch.ones(2, 3))
        torch.testing.assert_close(module.W_K_scale, torch.full((2, 1), 2.0))

    def test_refreshed_aiter_mla_tensor_shape_change_fails_closed(self, vllm_plugin):
        module = nn.Module()
        old_wk = torch.zeros(2, 3)
        module.W_K = torch.zeros(4, 3)

        with pytest.raises(RuntimeError, match="Cannot update MLA W_K in place"):
            vllm_plugin._restore_mla_absorbed_tensor_storage(module, {"W_K": old_wk})


class TestFakeQuantWorkerCalibration:
    @pytest.mark.parametrize("needs_zeroing", [False, True, None])
    def test_calibration_request_respects_cache_zeroing_policy(self, monkeypatch, needs_zeroing):
        from quark.experimental.torch.plugin import fakequant_worker

        cache_config = SimpleNamespace(
            num_blocks=16,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))],
        )
        if needs_zeroing is not None:
            cache_config.needs_kv_cache_zeroing = needs_zeroing
        monkeypatch.setattr(fakequant_worker, "_create_new_data_cls", lambda cls, **kwargs: SimpleNamespace(**kwargs))
        monkeypatch.setattr(fakequant_worker, "SamplingParams", SimpleNamespace)
        monkeypatch.setattr(
            fakequant_worker, "CachedRequestData", SimpleNamespace(make_empty=lambda: SimpleNamespace())
        )
        requests = []
        zeroer = MagicMock() if needs_zeroing else None
        output = object()

        def execute(scheduler_output):
            requests.append(scheduler_output)
            blocks = getattr(scheduler_output, "new_block_ids_to_zero", None)
            if blocks:
                # vLLM V2 enforces this contract before executing the model.
                assert zeroer is not None
                zeroer.zero_block_ids(blocks)
            return output

        worker = SimpleNamespace(
            model_runner=SimpleNamespace(kv_cache_config=cache_config),
            _calib_step_idx=0,
            execute_model=execute,
        )
        assert fakequant_worker.QuarkFakeQuantWorker._execute_calibration_step(worker, [1] * 20) is output
        assert requests[0].scheduled_new_reqs[0].block_ids == ([1, 2],)
        assert requests[1].finished_req_ids == {"calib-0"}
        if needs_zeroing:
            zeroer.zero_block_ids.assert_called_once_with([1, 2])
        else:
            assert requests[0].new_block_ids_to_zero is None

    def test_model_mutation_barrier_synchronizes_available_accelerator(self, monkeypatch):
        from quark.experimental.torch.plugin.fakequant_worker import QuarkFakeQuantWorker

        calls = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append(True))

        QuarkFakeQuantWorker._synchronize_before_or_after_model_mutation()

        assert calls == [True]

    def test_build_calibration_block_ids_reserves_non_null_pages_per_group(self):
        from quark.experimental.torch.plugin.fakequant_worker import _build_calibration_block_ids

        kv_cache_config = SimpleNamespace(
            num_blocks=16,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=528)),
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=256)),
            ],
        )
        block_ids, new_block_ids = _build_calibration_block_ids(kv_cache_config, num_tokens=544)
        assert block_ids == ([1, 2], [3, 4, 5])
        assert new_block_ids == [1, 2, 3, 4, 5]
        assert all(block_id != 0 for block_id in new_block_ids)

    def test_build_calibration_block_ids_rejects_insufficient_capacity(self):
        from quark.experimental.torch.plugin.fakequant_worker import _build_calibration_block_ids

        kv_cache_config = SimpleNamespace(
            num_blocks=2,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=528))],
        )
        with pytest.raises(RuntimeError, match="Not enough KV cache blocks"):
            _build_calibration_block_ids(kv_cache_config, num_tokens=1056)

    def test_validate_static_activation_scales_accepts_finite_scale(self):
        from quark.experimental.torch.plugin.fakequant_worker import _validate_static_activation_scales
        from quark.torch.quantization.config.template import FP8Scheme
        from quark.torch.quantization.tensor_quantize import StaticScaledFakeQuantize

        model = nn.Module()
        model.add_module(
            "_input_quantizer",
            StaticScaledFakeQuantize(FP8Scheme().config.input_tensors, device=torch.device("cpu")),
        )
        assert _validate_static_activation_scales(model) == 1

    def test_validate_static_activation_scales_rejects_nan(self):
        from quark.experimental.torch.plugin.fakequant_worker import _validate_static_activation_scales
        from quark.torch.quantization.config.template import FP8Scheme
        from quark.torch.quantization.tensor_quantize import StaticScaledFakeQuantize

        model = nn.Module()
        quantizer = StaticScaledFakeQuantize(FP8Scheme().config.input_tensors, device=torch.device("cpu"))
        quantizer.scale.fill_(float("nan"))
        model.add_module("_input_quantizer", quantizer)
        with pytest.raises(RuntimeError, match="non-finite scales"):
            _validate_static_activation_scales(model)


class TestAdaptKvCachePatternForVllm:
    def test_k_proj_maps_to_qkv_proj_pattern(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.self_attn.k_proj") == "*qkv_proj"

    def test_v_proj_maps_to_qkv_proj_pattern(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.self_attn.v_proj") == "*qkv_proj"

    def test_qkv_proj_maps_to_qkv_proj_pattern(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.self_attn.qkv_proj") == "*qkv_proj"

    def test_in_proj_qkv_maps_to_in_proj_qkvz_pattern(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.linear_attn.in_proj_qkv") == "*in_proj_qkvz"

    def test_in_proj_qkvz_maps_to_in_proj_qkvz_pattern(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.linear_attn.in_proj_qkvz") == "*in_proj_qkvz"

    def test_o_proj_returns_none(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.self_attn.o_proj") is None

    def test_mlp_pattern_returns_none(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.mlp.gate_proj") is None


class TestVllmAvailableFlag:
    def test_vllm_available_is_bool(self):
        from quark.experimental.torch.plugin.vllm_plugin import VLLM_AVAILABLE

        assert isinstance(VLLM_AVAILABLE, bool)

    def test_vllm_available_true_with_stubs(self, vllm_plugin):
        assert vllm_plugin.VLLM_AVAILABLE is True

    def test_new_vllm_selects_moe_runner(self, vllm_plugin):
        assert vllm_plugin.vllm_moe_runner is vllm_plugin.vllm_fused_moe_layer.MoERunner

    def test_registration_maps_moe_runner_to_quant_wrapper(self, monkeypatch, vllm_plugin):
        layer_map = {}
        monkeypatch.setattr(vllm_plugin.model_transformation, "LAYER_TO_QUANT_LAYER_MAP", layer_map)
        monkeypatch.setattr(vllm_plugin, "_install_alias_preserving_layer_replacement_patch", lambda: None)
        monkeypatch.setattr(vllm_plugin, "_install_quark_moe_a2_patch", lambda: None)
        monkeypatch.setattr(vllm_plugin, "_install_quark_kv_cache_calib_patch", lambda: None)

        vllm_plugin.register_vllm_quantization_plugins()

        assert layer_map[vllm_plugin.vllm_moe_runner] is vllm_plugin.QuantVLLMMoERunner
        assert vllm_plugin.vllm_fused_moe_layer.FusedMoE not in layer_map


class TestAliasPreservingLayerReplacement:
    @staticmethod
    def _model():
        class AliasedLinearModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                shared = nn.Linear(8, 8, bias=False)
                self.first = shared
                self.second = shared

        return AliasedLinearModel()

    @staticmethod
    def _layer_config():
        from quark.torch.quantization.config.config import QLayerConfig, QTensorConfig
        from quark.torch.quantization.config.type import Dtype
        from quark.torch.quantization.observer.observer import PlaceholderObserver

        tensor_config = QTensorConfig(dtype=Dtype.float16, observer_cls=PlaceholderObserver)
        return QLayerConfig(input_tensors=tensor_config, weight=tensor_config)

    def test_rebinds_unconfigured_runtime_alias_when_not_excluded(self, vllm_plugin):
        from quark.torch.quantization import model_transformation
        from quark.torch.quantization.config.config import QConfig
        from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear

        vllm_plugin._install_alias_preserving_layer_replacement_patch()
        model = self._model()
        layer_config = self._layer_config()
        model_transformation.in_place_replace_layer(
            model,
            QConfig(global_quant_config=layer_config),
            dict(model.named_modules(remove_duplicate=False)),
            {"first": layer_config},
        )

        assert isinstance(model.first, QuantLinear)
        assert model.first is model.second

    def test_explicit_exclusion_wins_across_aliases(self, vllm_plugin):
        from quark.torch.quantization import model_transformation
        from quark.torch.quantization.config.config import QConfig

        vllm_plugin._install_alias_preserving_layer_replacement_patch()
        model = self._model()
        layer_config = self._layer_config()
        config = QConfig(global_quant_config=layer_config, exclude=["second"])
        module_configs = {"first": layer_config}

        model_transformation.in_place_replace_layer(
            model,
            config,
            dict(model.named_modules(remove_duplicate=False)),
            module_configs,
        )

        assert model.first is model.second
        assert isinstance(model.first, nn.Linear)
        assert module_configs == {}
        assert {"first", "second"}.issubset(config.exclude)

    def test_qwen_router_public_exclude_suppresses_internal_experts_alias(self, vllm_plugin):
        from quark.torch.quantization import model_transformation
        from quark.torch.quantization.config.config import QConfig

        class QwenMlp(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                gate = nn.Linear(8, 8, bias=False)
                self.gate = gate
                self.experts = nn.Module()
                self.experts.add_module("_gate", gate)

        class QwenModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = QwenMlp()

        vllm_plugin._install_alias_preserving_layer_replacement_patch()
        model = QwenModel()
        layer_config = self._layer_config()
        config = QConfig(global_quant_config=layer_config, exclude=["*mlp.gate"])
        module_configs = {"mlp.experts._gate": layer_config}

        model_transformation.in_place_replace_layer(
            model,
            config,
            dict(model.named_modules(remove_duplicate=False)),
            module_configs,
        )

        assert model.mlp.gate is model.mlp.experts._gate
        assert isinstance(model.mlp.gate, nn.Linear)
        assert module_configs == {}
        assert "mlp.experts._gate" in config.exclude

    def test_explicit_target_keeps_all_runtime_aliases_out_of_exclude(self):
        from quark.experimental.torch.plugin.fakequant_worker import _restrict_to_explicit_vllm_layers
        from quark.torch.quantization.config.config import QConfig

        shared_gate = nn.Linear(8, 8, bias=False)
        named_modules = {
            "language_model.model.layers.0.mlp.experts.gate": shared_gate,
            "language_model.model.layers.0.mlp.gate": shared_gate,
        }
        layer_config = self._layer_config()
        config = QConfig(
            global_quant_config=layer_config,
            layer_quant_config={"language_model.model.layers.0.mlp.experts.gate": layer_config},
        )

        _restrict_to_explicit_vllm_layers(config, named_modules)

        assert "language_model.model.layers.0.mlp.gate" not in config.exclude

    def test_routed_source_floor_config_respects_excluded_alias(self, monkeypatch, vllm_plugin):
        from quark.torch.quantization import model_transformation
        from quark.torch.quantization.config.config import QConfig

        vllm_plugin._install_alias_preserving_layer_replacement_patch()
        shared = nn.Linear(8, 8, bias=False)
        configured = [
            "language_model.model.layers.1.block_sparse_moe.routed_expert_down_proj",
            "language_model.model.layers.1.block_sparse_moe.experts.routed_input_transform",
        ]
        excluded = "language_model.model.layers.1.mlp.routed_expert_down_proj"
        named_modules = dict.fromkeys([*configured, excluded], shared)
        layer_config = self._layer_config()
        original_calls = []
        rebound = []

        monkeypatch.setattr(
            vllm_plugin,
            "_orig_in_place_replace_layer",
            lambda model, config, modules, configs: original_calls.append((modules, configs)),
        )
        monkeypatch.setattr(vllm_plugin, "getattr_recursive", lambda model, name: shared)
        monkeypatch.setattr(
            vllm_plugin,
            "setattr_recursive",
            lambda model, name, value: rebound.append((name, value)),
        )

        model_transformation.in_place_replace_layer(
            nn.Module(),
            QConfig(global_quant_config=layer_config, exclude=[excluded]),
            named_modules,
            dict.fromkeys(configured, layer_config),
        )

        assert len(original_calls) == 1
        assert original_calls[0][1] == {}
        assert rebound == []

    def test_routed_output_transform_config_respects_excluded_alias(self, monkeypatch, vllm_plugin):
        from quark.torch.quantization import model_transformation
        from quark.torch.quantization.config.config import QConfig

        vllm_plugin._install_alias_preserving_layer_replacement_patch()
        shared = nn.Linear(8, 8, bias=False)
        configured = [
            "language_model.model.layers.1.block_sparse_moe.routed_expert_up_proj",
            "language_model.model.layers.1.block_sparse_moe.experts.routed_output_transform.up_proj",
        ]
        excluded = "language_model.model.layers.1.mlp.routed_output_transform.up_proj"
        named_modules = dict.fromkeys([*configured, excluded], shared)
        layer_config = self._layer_config()
        original_calls = []
        rebound = []

        monkeypatch.setattr(
            vllm_plugin,
            "_orig_in_place_replace_layer",
            lambda model, config, modules, configs: original_calls.append((modules, configs)),
        )
        monkeypatch.setattr(vllm_plugin, "getattr_recursive", lambda model, name: shared)
        monkeypatch.setattr(
            vllm_plugin,
            "setattr_recursive",
            lambda model, name, value: rebound.append((name, value)),
        )

        model_transformation.in_place_replace_layer(
            nn.Module(),
            QConfig(global_quant_config=layer_config, exclude=[excluded]),
            named_modules,
            dict.fromkeys(configured, layer_config),
        )

        assert len(original_calls) == 1
        assert original_calls[0][1] == {}
        assert rebound == []


class TestVllmAliasConfigResolution:
    @staticmethod
    def _runtime_model():
        model = nn.Module()
        model.block_sparse_moe = nn.Module()
        model.mlp = model.block_sparse_moe
        projection = nn.Linear(8, 8, bias=False)
        model.block_sparse_moe.routed_expert_down_proj = projection
        model.block_sparse_moe.experts = nn.Module()
        model.block_sparse_moe.experts.routed_input_transform = projection
        model.other = nn.Linear(8, 8, bias=False)
        return model

    @pytest.mark.parametrize("native", [False, True])
    @pytest.mark.parametrize("projection_mode", ["fp8", "mxfp4"])
    @pytest.mark.parametrize("reverse_alias_order", [False, True])
    def test_source_projection_controls_all_runtime_aliases(
        self, native, projection_mode, reverse_alias_order, vllm_plugin
    ):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.fakequant_worker import _adapt_config_for_vllm
        from quark.torch.quantization.config.config import QConfig
        from quark.torch.quantization.model_transformation import setup_config_per_layer

        model = self._runtime_model()
        projection_name = "block_sparse_moe.routed_expert_down_proj"
        moe_config = get_layer_config("mxfp4" if projection_mode == "fp8" else "fp8")
        projection_config = get_layer_config(projection_mode)
        config = QConfig(
            global_quant_config=moe_config,
            layer_quant_config={
                "block_sparse_moe.experts.*.w1": moe_config,
                **({} if native else {projection_name: projection_config}),
            },
            exclude=[projection_name] if native else [],
        )

        _adapt_config_for_vllm(model, config)
        module_configs = {}
        named_modules = dict(model.named_modules(remove_duplicate=False))
        if reverse_alias_order:
            named_modules = dict(reversed(named_modules.items()))
        setup_config_per_layer(config, named_modules, module_configs)

        aliases = [name for name, module in named_modules.items() if module is model.mlp.routed_expert_down_proj]
        assert len(aliases) == 4
        assert "other" not in module_configs

        # Exercise actual replacement, including exclusion propagation and rebinding.
        from quark.torch.quantization import model_transformation
        from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear

        original = model.get_submodule(projection_name)
        vllm_plugin._install_alias_preserving_layer_replacement_patch()
        model_transformation.in_place_replace_layer(model, config, named_modules, module_configs)
        replaced = model.get_submodule(projection_name)
        assert all(model.get_submodule(alias) is replaced for alias in aliases)
        if native:
            assert replaced is original
            assert all(alias not in module_configs for alias in aliases)
        else:
            assert isinstance(replaced, QuantLinear)
            assert all(module_configs[alias] == projection_config for alias in aliases)

    def test_conflicting_explicit_alias_targets_are_rejected(self, vllm_plugin):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.fakequant_worker import _adapt_config_for_vllm
        from quark.torch.quantization import model_transformation
        from quark.torch.quantization.config.config import QConfig

        config = QConfig(
            global_quant_config=get_layer_config("mxfp4"),
            layer_quant_config={
                "block_sparse_moe.routed_expert_down_proj": get_layer_config("fp8"),
                "mlp.routed_expert_down_proj": get_layer_config("mxfp4"),
            },
        )
        model = self._runtime_model()
        _adapt_config_for_vllm(model, config)
        named_modules = dict(model.named_modules(remove_duplicate=False))
        module_configs = {}
        model_transformation.setup_config_per_layer(config, named_modules, module_configs)
        vllm_plugin._install_alias_preserving_layer_replacement_patch()
        with pytest.raises(ValueError, match="Aliased vLLM module paths request different quantization configs"):
            model_transformation.in_place_replace_layer(model, config, named_modules, module_configs)

    @pytest.mark.parametrize("adapt", [False, True])
    def test_user_experts_wildcard_is_not_a_generated_fallback(self, adapt, vllm_plugin):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.fakequant_worker import _adapt_config_for_vllm
        from quark.torch.quantization import model_transformation
        from quark.torch.quantization.config.config import QConfig

        model = self._runtime_model()
        config = QConfig(
            global_quant_config=get_layer_config("fp8"),
            layer_quant_config={
                "*experts*": get_layer_config("mxfp4"),
                "block_sparse_moe.routed_expert_down_proj": get_layer_config("fp8"),
            },
        )
        if adapt:
            _adapt_config_for_vllm(model, config)
        named_modules = dict(model.named_modules(remove_duplicate=False))
        module_configs = {}
        model_transformation.setup_config_per_layer(config, named_modules, module_configs)
        vllm_plugin._install_alias_preserving_layer_replacement_patch()
        with pytest.raises(ValueError, match="Aliased vLLM module paths request different quantization configs"):
            model_transformation.in_place_replace_layer(model, config, named_modules, module_configs)

    @pytest.mark.parametrize("shared", [False, True])
    @pytest.mark.parametrize("reverse_alias_order", [False, True])
    def test_generated_fallbacks_respect_explicit_targets_and_override_global(
        self, shared, reverse_alias_order, vllm_plugin
    ):
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.plugin.fakequant_worker import _adapt_config_for_vllm
        from quark.torch.quantization import model_transformation
        from quark.torch.quantization.config.config import QConfig
        from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear

        fp8, mxfp4 = get_layer_config("fp8"), get_layer_config("mxfp4")
        if shared:
            model = nn.Module()
            model.mlp = nn.Module()
            model.mlp.shared_experts = nn.Module()
            model.mlp.shared_experts.gate_up_proj = nn.Linear(8, 8)
            model.mlp.shared_experts.down_proj = nn.Linear(8, 8)
            model.mlp.experts = nn.Module()
            model.mlp.experts._shared_experts = model.mlp.shared_experts
            config = QConfig(
                global_quant_config=mxfp4,
                layer_quant_config={
                    "mlp.shared_experts.gate_up_proj": mxfp4,
                    "mlp.shared_experts.down_proj": fp8,
                },
            )
            target, expected = model.mlp.shared_experts.down_proj, fp8
        else:
            model = self._runtime_model()
            config = QConfig(global_quant_config=fp8, layer_quant_config={"block_sparse_moe.experts.*.w1": mxfp4})
            target, expected = model.mlp.routed_expert_down_proj, mxfp4
        # Re-adaptation must retain origin information without serializing it.
        _adapt_config_for_vllm(model, config)
        _adapt_config_for_vllm(model, config)
        assert "_quark_vllm_fallback_patterns" not in config.to_dict()
        named_modules = dict(model.named_modules(remove_duplicate=False))
        if reverse_alias_order:
            named_modules = dict(reversed(named_modules.items()))
        aliases = [name for name, module in named_modules.items() if module is target]
        module_configs = {}
        model_transformation.setup_config_per_layer(config, named_modules, module_configs)
        vllm_plugin._install_alias_preserving_layer_replacement_patch()
        model_transformation.in_place_replace_layer(model, config, named_modules, module_configs)
        wrapper = model.get_submodule(aliases[0])
        assert isinstance(wrapper, QuantLinear)
        assert all(model.get_submodule(alias) is wrapper for alias in aliases)
        assert all(module_configs[alias] == expected for alias in aliases)


# =============================================================================
# utils.py — categorize_layers, pattern helpers
# =============================================================================


class TestLinearAttnPatternMatching:
    def test_linear_attn_names_match(self):
        from quark.experimental.torch.mix_precision.utils import _LINEAR_ATTN_STANDARD, _match_patterns

        linear_attn_names = [
            "model.layers.0.linear_attn.in_proj.weight",
            "model.layers.1.linear_attn.out_proj.weight",
            "model.layers.2.linear_attn.q_proj.weight",
            "model.layers.3.linear_attn.kv_proj.weight",
            "model.layers.4.linear_attn.qkv_proj.weight",
        ]
        for name in linear_attn_names:
            assert _match_patterns(name, _LINEAR_ATTN_STANDARD), f"{name} should match linear_attn patterns"

    def test_self_attn_and_mlp_do_not_match(self):
        from quark.experimental.torch.mix_precision.utils import _LINEAR_ATTN_STANDARD, _match_patterns

        for name in ("model.layers.0.self_attn.q_proj.weight", "model.layers.1.mlp.gate_proj.weight"):
            assert not _match_patterns(name, _LINEAR_ATTN_STANDARD), f"{name} should NOT match linear_attn patterns"


class TestLayerPatternsThreePartitions:
    def test_qwen3_5_moe_has_all_partitions(self):
        from quark.experimental.torch.mix_precision.utils import LAYER_PATTERNS

        assert "qwen3_5_moe" in LAYER_PATTERNS
        qwen = LAYER_PATTERNS["qwen3_5_moe"]
        for part in ("linear_attn", "self_attn", "dense_mlp", "routed_moe", "shared_expert"):
            assert part in qwen and len(qwen[part]) > 0

    def test_default_entry_includes_linear_attn(self):
        from quark.experimental.torch.mix_precision.utils import LAYER_PATTERNS

        assert "linear_attn" in LAYER_PATTERNS["default"]


class TestCompactGeneratedExcludeNames:
    def test_compacts_layer_and_expert_numeric_path_segments(self):
        from quark.experimental.torch.mix_precision.utils import _compact_generated_exclude_names

        names = [
            "model.layers.0.mlp.experts.0.gate_proj",
            "model.layers.0.mlp.experts.1.gate_proj",
            "model.layers.1.mlp.experts.0.gate_proj",
        ]
        assert _compact_generated_exclude_names(names) == [
            "model.layers.*.mlp.experts.*.gate_proj",
        ]

    def test_does_not_wildcard_digits_inside_module_tokens(self):
        from quark.experimental.torch.mix_precision.utils import _compact_generated_exclude_names

        names = [
            "model.layers.0.mlp.experts.0.fc1",
            "model.layers.1.mlp.experts.1.fc1",
            "model.layers.0.mlp.experts.0.w2",
            "model.layers.1.mlp.experts.1.w2",
        ]
        assert set(_compact_generated_exclude_names(names)) == {
            "model.layers.*.mlp.experts.*.fc1",
            "model.layers.*.mlp.experts.*.w2",
        }

    def test_preserves_non_expert_layer_names_for_runtime_compatibility(self):
        from quark.experimental.torch.mix_precision.utils import _compact_generated_exclude_names

        names = [
            "model.language_model.layers.3.self_attn.q_proj",
            "model.language_model.layers.3.self_attn.k_proj",
            "model.language_model.layers.7.self_attn.q_proj",
            "model.language_model.layers.7.self_attn.k_proj",
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
            "model.language_model.layers.1.linear_attn.in_proj_qkv",
        ]

        assert _compact_generated_exclude_names(names) == names


class _MockConfig:
    def __init__(self, layer_types):
        self.layer_types = layer_types


class _MockNestedConfig:
    def __init__(self, layer_types):
        self.text_config = _MockConfig(layer_types)


class _MockModel:
    def __init__(self, config):
        self.config = config


class _MockFP8ExpertLinear(nn.Module):
    _is_fp8_block_quantized_linear = True

    def __init__(self):
        super().__init__()
        self.in_features = 8
        self.out_features = 8
        self.weight = nn.Parameter(torch.empty(8, 8), requires_grad=False)


class _MockExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = _MockFP8ExpertLinear()
        self.up_proj = _MockFP8ExpertLinear()
        self.down_proj = _MockFP8ExpertLinear()


class _MockPreprocessedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.add_module("0", _MockExpert())
        self.add_module("1", _MockExpert())


class _MockPreprocessedMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([self._layer(), self._layer()])

    @staticmethod
    def _layer():
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.experts = _MockPreprocessedExperts()
        return layer


class TestGetLayerPartitionFromConfig:
    def test_linear_attention_layer_detected(self):
        from quark.experimental.torch.mix_precision.utils import _get_layer_partition_from_config

        model = _MockModel(_MockNestedConfig(["linear_attention", "linear_attention", "full_attention"]))
        assert _get_layer_partition_from_config(model, "model.layers.0.self_attn.q_proj") == "linear_attn"
        assert _get_layer_partition_from_config(model, "model.layers.2.self_attn.q_proj") == "self_attn"

    def test_no_layer_types_returns_none(self):
        from quark.experimental.torch.mix_precision.utils import _get_layer_partition_from_config

        cfg = _MockConfig([])
        del cfg.layer_types
        assert _get_layer_partition_from_config(_MockModel(cfg), "model.layers.0.self_attn.q_proj") is None

    def test_non_layer_name_returns_none(self):
        from quark.experimental.torch.mix_precision.utils import _get_layer_partition_from_config

        model = _MockModel(_MockConfig(["linear_attention", "full_attention"]))
        assert _get_layer_partition_from_config(model, "model.embed_tokens.weight") is None


class TestCategorizeLayers:
    def test_three_partitions_for_hybrid_model(self):
        from quark.experimental.torch.mix_precision.utils import categorize_layers

        class MockHybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_0_linear_attn_q_proj = nn.Linear(2048, 2048)
                self.layers_0_linear_attn_out_proj = nn.Linear(2048, 2048)
                self.layers_3_self_attn_q_proj = nn.Linear(2048, 2048)
                self.layers_3_self_attn_k_proj = nn.Linear(2048, 2048)
                self.layers_0_mlp_gate_proj = nn.Linear(2048, 8192)
                self.layers_3_mlp_gate_proj = nn.Linear(2048, 8192)

        cats = categorize_layers(MockHybridModel(), model_type="qwen3_5_moe")
        for part in ("linear_attn", "self_attn", "dense_mlp"):
            assert part in cats
        assert len(cats["linear_attn"]) == 2
        assert len(cats["self_attn"]) == 2
        assert len(cats["dense_mlp"]) == 2

    def test_two_partitions_for_traditional_model(self):
        from quark.experimental.torch.mix_precision.utils import categorize_layers

        class MockTraditionalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_0_self_attn_q_proj = nn.Linear(4096, 4096)
                self.layers_0_mlp_gate_proj = nn.Linear(4096, 11008)

        cats = categorize_layers(MockTraditionalModel(), model_type="llama")
        assert "linear_attn" not in cats
        assert "self_attn" in cats
        assert "dense_mlp" in cats
        assert "routed_moe" not in cats

    def test_shared_expert_is_separate_from_routed_experts(self):
        from quark.experimental.torch.mix_precision.utils import categorize_layers

        class MockMoEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_3_mlp_experts_0_gate_proj = nn.Linear(2048, 8192)
                self.layers_3_mlp_shared_experts_gate_proj = nn.Linear(2048, 8192)

        cats = categorize_layers(MockMoEModel(), model_type="qwen3_5_moe", exclude_patterns=[])
        assert cats["routed_moe"] == {"layers_3_mlp_experts_0_gate_proj"}
        assert "dense_mlp" not in cats
        assert cats["shared_expert"] == {"layers_3_mlp_shared_experts_gate_proj"}


class _KVProjectionBlock(nn.Module):
    def __init__(self, *, include_v_proj: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.k_proj = nn.Linear(16, 16)
        if include_v_proj:
            self.v_proj = nn.Linear(16, 16)
        self.o_proj = nn.Linear(16, 16)


class _LinearAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(16, 48)
        self.in_proj_z = nn.Linear(16, 16)
        self.in_proj_a = nn.Linear(16, 16)
        self.in_proj_b = nn.Linear(16, 16)
        self.out_proj = nn.Linear(16, 16)


class _MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(16, 32)
        self.up_proj = nn.Linear(16, 32)
        self.down_proj = nn.Linear(32, 16)


class _DecoderLayer(nn.Module):
    def __init__(self, *, attention: str = "self", include_v_proj: bool = True):
        super().__init__()
        if attention == "linear":
            self.linear_attn = _LinearAttentionBlock()
        else:
            self.self_attn = _KVProjectionBlock(include_v_proj=include_v_proj)
        self.mlp = _MLPBlock()


class _WrappedProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)


class _TowerAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _WrappedProjection()
        self.k_proj = _WrappedProjection()
        self.v_proj = _WrappedProjection()
        self.o_proj = _WrappedProjection()


class _TowerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _TowerAttention()


class _Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([_TowerLayer()])


class _KVTestModel(nn.Module):
    def __init__(
        self,
        layers: list[nn.Module],
        *,
        model_type: str = "gemma",
        include_multimodal_towers: bool = False,
    ):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(layers)
        if include_multimodal_towers:
            self.model.vision_tower = _Tower()
            self.model.audio_tower = _Tower()
        self.config = SimpleNamespace(model_type=model_type)


def _matching_layer_configs(qconfig, layer_name: str):
    return [
        layer_config
        for pattern, layer_config in qconfig.layer_quant_config.items()
        if fnmatch.fnmatch(layer_name, pattern)
    ]


def _self_attention_projection_names(model: nn.Module, projections: tuple[str, ...]) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and ".self_attn." in name
        and any(name.endswith(projection) for projection in projections)
    ]


class TestCreateQconfigThreePartitions:
    def test_three_partition_qconfig_structure(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

        class MockThreePartModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_0_linear_attn_q_proj = nn.Linear(2048, 2048)
                self.layers_0_linear_attn_out_proj = nn.Linear(2048, 2048)
                self.layers_1_self_attn_q_proj = nn.Linear(2048, 2048)
                self.layers_1_self_attn_k_proj = nn.Linear(2048, 2048)
                self.layers_0_mlp_gate_proj = nn.Linear(2048, 8192)
                self.layers_1_mlp_gate_proj = nn.Linear(2048, 8192)

        config = create_quant_config(
            layer_partitions={"linear_attn": "fp8", "self_attn": "ptpc_fp8", "dense_mlp": "native"},
            kv_cache_mode="native",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(MockThreePartModel(), config)
        assert qconfig.global_quant_config is not None
        assert isinstance(qconfig.layer_quant_config, dict)
        assert any("linear_attn" in k for k in qconfig.layer_quant_config)
        assert any("self_attn" in k for k in qconfig.layer_quant_config)
        assert any("mlp" in k for k in qconfig.exclude)

    def test_two_partition_backward_compat(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

        class MockTwoPartModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_0_self_attn_q_proj = nn.Linear(4096, 4096)
                self.layers_0_mlp_gate_proj = nn.Linear(4096, 11008)

        config = create_quant_config(
            layer_partitions={"self_attn": "fp8", "mlp": "native"},
            kv_cache_mode="native",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(MockTwoPartModel(), config)
        assert qconfig.global_quant_config is not None

    def test_original_exclude_patterns_are_preserved_for_runtime_only_modules(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

        class MockMoEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_0_mlp_experts_0_gate_proj = nn.Linear(32, 64)
                self.layers_0_mlp_gate = nn.Identity()

        config = create_quant_config(
            layer_partitions={"routed_moe": "mxfp4"},
            kv_cache_mode="native",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(
            MockMoEModel(),
            config,
            exclude_patterns=["*mlp.gate", "*shared_expert_gate*"],
        )
        assert "*mlp.gate" in qconfig.exclude
        assert "*shared_expert_gate*" in qconfig.exclude

    def test_shared_expert_native_is_excluded_from_quantization(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

        class MockMoEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_3_mlp_experts_0_gate_proj = nn.Linear(2048, 8192)
                self.layers_3_mlp_shared_experts_gate_proj = nn.Linear(2048, 8192)

        config = create_quant_config(
            layer_partitions={"routed_moe": "fp8", "shared_expert": "native"},
            kv_cache_mode="native",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(MockMoEModel(), config, exclude_patterns=[])
        assert any("shared_experts" in name for name in qconfig.exclude)
        assert not any("shared_experts" in name for name in qconfig.layer_quant_config)

    def test_shared_expert_matching_mlp_is_quantized(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

        class MockMoEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_3_mlp_experts_0_gate_proj = nn.Linear(2048, 8192)
                self.layers_3_mlp_shared_experts_gate_proj = nn.Linear(2048, 8192)

        config = create_quant_config(
            layer_partitions={"routed_moe": "fp8", "shared_expert": "fp8"},
            kv_cache_mode="native",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(MockMoEModel(), config, exclude_patterns=[])
        assert any("shared_experts" in name for name in qconfig.layer_quant_config)

    def test_shared_expert_different_from_mlp_is_rejected(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

        class MockMoEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_3_mlp_experts_0_gate_proj = nn.Linear(2048, 8192)
                self.layers_3_mlp_shared_experts_gate_proj = nn.Linear(2048, 8192)

        config = create_quant_config(
            layer_partitions={"routed_moe": "fp8", "shared_expert": "mxfp4"},
            kv_cache_mode="native",
            attention_mode="native",
        )
        with pytest.raises(ValueError, match="shared_expert_mode must be native or match routed_moe_mode"):
            create_qconfig_from_quant_config(MockMoEModel(), config, exclude_patterns=[])

    def test_mixed_source_floor_keeps_packed_weight_and_target_activation(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

        class MockMixedMLPModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_0_mlp_gate_proj = nn.Linear(32, 64)
                self.layers_1_mlp_experts_0_gate_proj = nn.Linear(32, 64)

        model = MockMixedMLPModel()
        dense_name = "layers_0_mlp_gate_proj"
        routed_name = "layers_1_mlp_experts_0_gate_proj"
        config = create_quant_config(
            layer_partitions={"dense_mlp": "fp8", "routed_moe": "fp8"},
            kv_cache_mode="native",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(
            model,
            config,
            exclude_patterns=[],
            source_weight_bitwidth_by_layer={routed_name: 4},
        )

        assert _matching_layer_configs(qconfig, dense_name)
        routed_configs = _matching_layer_configs(qconfig, routed_name)
        assert routed_configs
        assert all(layer_config.weight is None for layer_config in routed_configs)
        assert all(layer_config.input_tensors is not None for layer_config in routed_configs)
        assert not any(fnmatch.fnmatch(routed_name, pattern) for pattern in qconfig.exclude)

    def test_mixed_source_floor_allows_same_width_target_on_packed_layer(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

        class MockMixedMLPModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_0_mlp_gate_proj = nn.Linear(32, 64)
                self.layers_1_mlp_experts_0_gate_proj = nn.Linear(32, 64)

        model = MockMixedMLPModel()
        dense_name = "layers_0_mlp_gate_proj"
        routed_name = "layers_1_mlp_experts_0_gate_proj"
        config = create_quant_config(
            layer_partitions={"dense_mlp": "mxfp4", "routed_moe": "mxfp4"},
            kv_cache_mode="native",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(
            model,
            config,
            exclude_patterns=[],
            source_weight_bitwidth_by_layer={routed_name: 4},
        )

        assert _matching_layer_configs(qconfig, dense_name)
        assert _matching_layer_configs(qconfig, routed_name)
        assert not any(fnmatch.fnmatch(routed_name, pattern) for pattern in qconfig.exclude)

    def test_dense_mlp_and_routed_moe_receive_independent_modes(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config
        from quark.torch.quantization.config.type import Dtype

        class MockSplitMLPModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_0_mlp_gate_proj = nn.Linear(32, 64)
                self.layers_1_mlp_experts_0_gate_proj = nn.Linear(32, 64)

        model = MockSplitMLPModel()
        dense_name = "layers_0_mlp_gate_proj"
        routed_name = "layers_1_mlp_experts_0_gate_proj"
        config = create_quant_config(
            layer_partitions={"dense_mlp": "ptpc_fp8", "routed_moe": "mxfp4"},
            kv_cache_mode="native",
            attention_mode="native",
        )

        qconfig = create_qconfig_from_quant_config(model, config, exclude_patterns=[])
        dense_configs = _matching_layer_configs(qconfig, dense_name)
        routed_configs = _matching_layer_configs(qconfig, routed_name)

        assert dense_configs and all(layer_config.weight.dtype is Dtype.fp8_e4m3 for layer_config in dense_configs)
        assert routed_configs and all(layer_config.weight.dtype is Dtype.fp4 for layer_config in routed_configs)

    def test_preprocessed_fp8_experts_emit_leaf_and_fused_runtime_configs(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config
        from quark.torch.quantization.config.type import Dtype

        config = create_quant_config(
            layer_partitions={"routed_moe": "mxfp4"},
            kv_cache_mode="native",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(_MockPreprocessedMoEModel(), config, exclude_patterns=[])

        expected_patterns = {
            "model.layers.*.mlp.experts",
            "model.layers.*.mlp.experts.*.gate_proj",
            "model.layers.*.mlp.experts.*.up_proj",
            "model.layers.*.mlp.experts.*.down_proj",
        }
        assert expected_patterns.issubset(qconfig.layer_quant_config)
        for pattern in expected_patterns:
            assert qconfig.layer_quant_config[pattern].weight.dtype is Dtype.fp4


class TestCreateQconfigFp8KvCache:
    @pytest.mark.parametrize("self_attn_mode", ["fp8", "ptpc_fp8", "native"])
    def test_generic_and_concrete_self_attention_kv_configs_are_equivalent(self, self_attn_mode):
        from quark.experimental.torch.mix_precision.config import create_quant_config, get_layer_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config
        from quark.torch.quantization.config.type import Dtype
        from quark.torch.quantization.model_transformation import setup_config_per_layer, setup_kv_cache_config

        model = _KVTestModel([_DecoderLayer(), _DecoderLayer()])
        config = create_quant_config(
            layer_partitions={"self_attn": self_attn_mode, "mlp": "native"},
            kv_cache_mode="fp8",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(model, config)

        assert "*k_proj" in qconfig.layer_quant_config
        assert "*v_proj" in qconfig.layer_quant_config
        assert qconfig.kv_cache_group == ["*k_proj", "*v_proj"]
        assert set(qconfig.kv_cache_quant_config) == {"*k_proj", "*v_proj"}
        assert qconfig.global_quant_config.output_tensors is None

        base_config = get_layer_config(self_attn_mode)
        for name in _self_attention_projection_names(model, ("k_proj", "v_proj")):
            matches = _matching_layer_configs(qconfig, name)
            assert len(matches) == 2
            assert all(layer_config == matches[0] for layer_config in matches)
            for layer_config in matches:
                assert layer_config.output_tensors is not None
                assert layer_config.output_tensors.dtype is Dtype.fp8_e4m3
                if base_config is None:
                    assert layer_config.weight is None
                    assert layer_config.input_tensors is None
                    assert layer_config.bias is None
                else:
                    assert layer_config is not base_config
                    assert layer_config.weight == base_config.weight
                    assert layer_config.input_tensors == base_config.input_tensors
                    assert layer_config.bias == base_config.bias
                    assert base_config.output_tensors is None

        for name in _self_attention_projection_names(model, ("q_proj", "o_proj")):
            matches = _matching_layer_configs(qconfig, name)
            if base_config is None:
                assert matches == []
                assert any(fnmatch.fnmatch(name, pattern) for pattern in qconfig.exclude)
            else:
                assert len(matches) == 1
                assert matches[0].output_tensors is None

        runtime_qconfig = deepcopy(qconfig)
        module_configs = {}
        named_modules = dict(model.named_modules())
        setup_config_per_layer(runtime_qconfig, named_modules, module_configs)
        setup_kv_cache_config(runtime_qconfig, named_modules, module_configs)
        for name in _self_attention_projection_names(model, ("k_proj", "v_proj")):
            assert module_configs[name].output_tensors is not None
            assert module_configs[name].output_tensors.dtype is Dtype.fp8_e4m3

    def test_native_kv_cache_does_not_change_existing_layer_configs(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

        model = _KVTestModel([_DecoderLayer(), _DecoderLayer()])
        config = create_quant_config(
            layer_partitions={"self_attn": "fp8", "mlp": "native"},
            kv_cache_mode="native",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(model, config)

        assert qconfig.kv_cache_quant_config == {}
        assert qconfig.kv_cache_group == []
        assert "*k_proj" not in qconfig.layer_quant_config
        assert "*v_proj" not in qconfig.layer_quant_config
        assert all(layer_config.output_tensors is None for layer_config in qconfig.layer_quant_config.values())

    def test_qwen35_linear_attention_is_not_selected_for_kv_cache(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

        model = _KVTestModel(
            [_DecoderLayer(attention="linear"), _DecoderLayer(attention="self")],
            model_type="qwen3_5_moe",
        )
        config = create_quant_config(
            layer_partitions={"linear_attn": "fp8", "self_attn": "fp8", "mlp": "native"},
            kv_cache_mode="fp8",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(model, config)

        linear_attention_names = [
            name for name, module in model.named_modules() if isinstance(module, nn.Linear) and ".linear_attn." in name
        ]
        assert linear_attention_names
        for name in linear_attention_names:
            matches = _matching_layer_configs(qconfig, name)
            assert len(matches) == 1
            assert matches[0].output_tensors is None
            assert not any(fnmatch.fnmatch(name, pattern) for pattern in qconfig.kv_cache_group)

    def test_multimodal_towers_remain_excluded_from_kv_cache(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config
        from quark.torch.quantization.model_transformation import setup_config_per_layer, setup_kv_cache_config

        model = _KVTestModel([_DecoderLayer()], include_multimodal_towers=True)
        config = create_quant_config(
            layer_partitions={"self_attn": "fp8", "mlp": "native"},
            kv_cache_mode="fp8",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(
            model,
            config,
            exclude_patterns=["*vision_tower*", "*audio_tower*"],
        )

        tower_names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and ("vision_tower" in name or "audio_tower" in name)
        ]
        assert tower_names
        assert all(any(fnmatch.fnmatch(name, pattern) for pattern in qconfig.exclude) for name in tower_names)
        assert all(_matching_layer_configs(qconfig, name) == [] for name in tower_names)

        runtime_qconfig = deepcopy(qconfig)
        module_configs = {}
        named_modules = dict(model.named_modules())
        setup_config_per_layer(runtime_qconfig, named_modules, module_configs)
        setup_kv_cache_config(runtime_qconfig, named_modules, module_configs)
        assert all(name not in module_configs for name in tower_names)

    def test_gemma4_k_equals_v_layers_only_configure_existing_v_projections(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

        model = _KVTestModel(
            [
                _DecoderLayer(),
                _DecoderLayer(),
                _DecoderLayer(),
                _DecoderLayer(),
                _DecoderLayer(),
                _DecoderLayer(include_v_proj=False),
            ]
        )
        config = create_quant_config(
            layer_partitions={"self_attn": "ptpc_fp8", "mlp": "native"},
            kv_cache_mode="fp8",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(model, config)

        k_names = _self_attention_projection_names(model, ("k_proj",))
        v_names = _self_attention_projection_names(model, ("v_proj",))
        assert len(k_names) == 6
        assert len(v_names) == 5
        assert all(len(_matching_layer_configs(qconfig, name)) == 2 for name in k_names + v_names)
        assert all(
            all(layer_config.output_tensors is not None for layer_config in _matching_layer_configs(qconfig, name))
            for name in k_names + v_names
        )

    def test_serialized_kv_configs_have_one_vllm_compatible_output_scheme(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

        model = _KVTestModel([_DecoderLayer(), _DecoderLayer()])
        config = create_quant_config(
            layer_partitions={"self_attn": "ptpc_fp8", "mlp": "fp8"},
            kv_cache_mode="fp8",
            attention_mode="native",
        )
        serialized = create_qconfig_from_quant_config(model, config).to_dict()
        matched_configs = [
            layer_config
            for name, layer_config in serialized["layer_quant_config"].items()
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in ("*k_proj", "*v_proj"))
        ]

        assert matched_configs
        assert matched_configs[0]["output_tensors"] is not None
        assert all(
            layer_config["output_tensors"] == matched_configs[0]["output_tensors"] for layer_config in matched_configs
        )

    def test_qconfig_serialization_round_trip_preserves_canonical_kv_configs(self):
        from quark.experimental.torch.mix_precision.config import create_quant_config
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config
        from quark.torch.quantization.config.config import QConfig

        model = _KVTestModel([_DecoderLayer(), _DecoderLayer()])
        config = create_quant_config(
            layer_partitions={"self_attn": "ptpc_fp8", "mlp": "fp8"},
            kv_cache_mode="fp8",
            attention_mode="native",
        )
        qconfig = create_qconfig_from_quant_config(model, config)

        assert QConfig.from_dict(qconfig.to_dict()).to_dict() == qconfig.to_dict()


class TestConfigSearcherPartitions:
    def test_split_mlp_partition_search_generates_configs(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(
            search_config=ModuleSearchConfig(
                layer_sensitivity={"linear_attn": 3, "self_attn": 3, "dense_mlp": 1, "routed_moe": 1}
            ),
            hardware=HardwareTarget.MI300,
        )
        assert "linear_attn" in searcher.layer_sensitivity
        assert "self_attn" in searcher.layer_sensitivity
        assert "dense_mlp" in searcher.layer_sensitivity
        assert "routed_moe" in searcher.layer_sensitivity
        configs = searcher.generate_sorted_configs()
        assert len(configs) > 0
        for cfg in configs[:5]:
            for key in (
                "linear_attn_mode",
                "self_attn_mode",
                "dense_mlp_mode",
                "routed_moe_mode",
                "kv_cache_mode",
                "attention_mode",
            ):
                assert key in cfg
        # No all-native config
        for cfg in configs:
            modes = [cfg.get(f"{p}_mode") for p in ("linear_attn", "self_attn", "dense_mlp", "routed_moe")]
            assert any(m != "native" for m in modes if m is not None)

    def test_two_partition_search_still_works(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(
            search_config=ModuleSearchConfig(layer_sensitivity={"self_attn": 3, "dense_mlp": 1}),
            hardware=HardwareTarget.MI300,
        )
        configs = searcher.generate_sorted_configs()
        assert len(configs) > 0
        for cfg in configs[:3]:
            assert "linear_attn_mode" not in cfg

    def test_shared_expert_is_native_or_matches_routed_moe(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(
            search_config=ModuleSearchConfig(layer_modes=["native", "fp8"]),
            hardware=HardwareTarget.MI300,
            available_partitions={"self_attn", "routed_moe", "shared_expert"},
        )
        for config in searcher.generate_sorted_configs():
            shared_mode = config["shared_expert_mode"]
            assert shared_mode == "native" or shared_mode == config["routed_moe_mode"]

    def test_routed_moe_only_search_keeps_shared_expert_choices(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(
            search_config=ModuleSearchConfig(
                layer_sensitivity={"routed_moe": 1},
                layer_modes=["native", "fp8"],
            ),
            hardware=HardwareTarget.MI300,
            available_partitions={"routed_moe", "shared_expert"},
        )
        configs = searcher.generate_sorted_configs()
        assert any(cfg["shared_expert_mode"] == "native" for cfg in configs)
        assert any(cfg["shared_expert_mode"] == cfg["routed_moe_mode"] == "fp8" for cfg in configs)


class TestDefaultPartitionSensitivity:
    def test_all_layer_partitions_defined(self):
        from quark.experimental.torch.mix_precision.config import DEFAULT_PARTITION_SENSITIVITY

        assert DEFAULT_PARTITION_SENSITIVITY["linear_attn"] == 3
        assert DEFAULT_PARTITION_SENSITIVITY["self_attn"] == 3
        assert DEFAULT_PARTITION_SENSITIVITY["dense_mlp"] == 1
        assert DEFAULT_PARTITION_SENSITIVITY["routed_moe"] == 1
        assert DEFAULT_PARTITION_SENSITIVITY["shared_expert"] == 1
        assert DEFAULT_PARTITION_SENSITIVITY["linear_attn"] > DEFAULT_PARTITION_SENSITIVITY["dense_mlp"]


class TestSharedExpertRoofline:
    def test_shared_expert_mode_changes_only_shared_expert_bytes(self):
        from quark.experimental.torch.mix_precision.perf_roofline import (
            ModelPerfMeta,
            compute_effective_weight_bytes,
        )

        meta = ModelPerfMeta(
            native_dtype_bytes=2.0,
            num_layers=1,
            hidden_size=4,
            num_kv_heads=1,
            head_dim=4,
            full_attn_layers=1,
            self_attn_numel=0,
            linear_attn_numel=0,
            mlp_dense_numel=0,
            moe_expert_numel=100,
            shared_expert_numel=20,
            self_attn_bytes_native=0,
            linear_attn_bytes_native=0,
            mlp_dense_bytes_native=0,
            moe_expert_bytes_native=200,
            shared_expert_bytes_native=40,
            other_bytes_native=0,
            num_experts=1,
            experts_per_tok=1,
        )
        native_total, native_breakdown = compute_effective_weight_bytes(
            meta,
            {"routed_moe_mode": "fp8", "shared_expert_mode": "native"},
        )
        matched_total, matched_breakdown = compute_effective_weight_bytes(
            meta,
            {"routed_moe_mode": "fp8", "shared_expert_mode": "fp8"},
        )
        assert native_breakdown["moe_expert"] == matched_breakdown["moe_expert"] == 100
        assert native_breakdown["shared_expert"] == 40
        assert matched_breakdown["shared_expert"] == 20
        assert native_total - matched_total == 20

    def test_dense_mlp_and_routed_moe_modes_change_separate_weight_buckets(self):
        from quark.experimental.torch.mix_precision.perf_roofline import (
            ModelPerfMeta,
            compute_effective_weight_bytes,
        )

        meta = ModelPerfMeta(
            native_dtype_bytes=2.0,
            num_layers=1,
            hidden_size=4,
            num_kv_heads=1,
            head_dim=4,
            full_attn_layers=1,
            self_attn_numel=0,
            linear_attn_numel=0,
            mlp_dense_numel=100,
            moe_expert_numel=100,
            self_attn_bytes_native=0,
            linear_attn_bytes_native=0,
            mlp_dense_bytes_native=200,
            moe_expert_bytes_native=200,
            other_bytes_native=0,
            num_experts=1,
            experts_per_tok=1,
        )

        dense_fp8_total, dense_fp8 = compute_effective_weight_bytes(
            meta,
            {"dense_mlp_mode": "fp8", "routed_moe_mode": "native"},
        )
        routed_fp8_total, routed_fp8 = compute_effective_weight_bytes(
            meta,
            {"dense_mlp_mode": "native", "routed_moe_mode": "fp8"},
        )

        assert dense_fp8["mlp_dense"] == 100
        assert dense_fp8["moe_expert"] == 200
        assert routed_fp8["mlp_dense"] == 200
        assert routed_fp8["moe_expert"] == 100
        assert dense_fp8_total == routed_fp8_total


class TestSharedExpertSearchToExport:
    """End-to-end regression: configs emitted by the searcher must survive the
    export-time QConfig build, with shared_expert routed correctly."""

    @staticmethod
    def _moe_model() -> nn.Module:
        class MockMoEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_0_self_attn_q_proj = nn.Linear(32, 32)
                self.layers_0_self_attn_o_proj = nn.Linear(32, 32)
                self.layers_0_mlp_experts_0_gate_proj = nn.Linear(32, 64)
                self.layers_0_mlp_experts_0_down_proj = nn.Linear(64, 32)
                self.layers_0_mlp_shared_experts_gate_proj = nn.Linear(32, 64)
                self.layers_0_mlp_shared_experts_down_proj = nn.Linear(64, 32)

        return MockMoEModel()

    def test_every_searched_config_builds_a_valid_export_qconfig(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher
        from quark.experimental.torch.mix_precision.utils import create_qconfig_from_quant_config

        searcher = ConfigSearcher(
            search_config=ModuleSearchConfig(layer_modes=["native", "fp8"]),
            hardware=HardwareTarget.MI300,
            available_partitions={"self_attn", "routed_moe", "shared_expert"},
        )
        configs = searcher.generate_sorted_configs()
        assert configs, "searcher produced no configs"

        saw_native = False
        saw_quantized = False
        for config in configs:
            assert "shared_expert_mode" in config
            # The searcher must never emit a shared_expert mode the export path
            # rejects; this call raises ValueError on an invalid combination.
            qconfig = create_qconfig_from_quant_config(self._moe_model(), config, exclude_patterns=[])

            shared_in_excluded = any("shared_experts" in name for name in qconfig.exclude)
            shared_in_quantized = any("shared_experts" in name for name in qconfig.layer_quant_config)

            if config["shared_expert_mode"] == "native":
                saw_native = True
                assert shared_in_excluded
                assert not shared_in_quantized
            else:
                saw_quantized = True
                assert config["shared_expert_mode"] == config["routed_moe_mode"]
                assert shared_in_quantized

        assert saw_native, "no config exercised the native shared_expert path"
        assert saw_quantized, "no config exercised the quantized shared_expert path"


class TestVllmPluginLinearAttnIndependence:
    def test_self_attn_not_expanded_to_linear_attn(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        result = adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj")
        assert any(".self_attn." in p for p in result)
        assert not any(".linear_attn." in p for p in result)

    def test_explicit_linear_attn_pattern_preserved(self):
        from quark.experimental.torch.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        result = adapt_layer_patterns_for_vllm("model.layers.*.linear_attn.q_proj")
        assert any(".linear_attn." in p for p in result)
        assert not any(".self_attn." in p for p in result)


# =============================================================================
# Backward compatibility — two-partition models (LLaMA / Mistral)
# =============================================================================


class MockLLaMAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_layers_0_self_attn_q_proj = nn.Linear(4096, 4096)
        self.model_layers_0_self_attn_k_proj = nn.Linear(4096, 4096)
        self.model_layers_0_self_attn_v_proj = nn.Linear(4096, 4096)
        self.model_layers_0_self_attn_o_proj = nn.Linear(4096, 4096)
        self.model_layers_0_mlp_gate_proj = nn.Linear(4096, 11008)
        self.model_layers_0_mlp_up_proj = nn.Linear(4096, 11008)
        self.model_layers_0_mlp_down_proj = nn.Linear(11008, 4096)


class TestBackwardCompatCategorization:
    def test_two_partition_model_has_no_linear_attn(self):
        from quark.experimental.torch.mix_precision.utils import categorize_layers

        cats = categorize_layers(MockLLaMAModel(), model_type="llama")
        assert "linear_attn" not in cats
        assert "self_attn" in cats
        assert "dense_mlp" in cats
        assert "routed_moe" not in cats
        assert len(cats["self_attn"]) == 4
        assert len(cats["dense_mlp"]) == 3


class TestBackwardCompatSearch:
    def test_two_partition_search_generates_expected_config_count(self):
        from quark.experimental.torch.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.mix_precision.searcher import ConfigSearcher
        from quark.experimental.torch.mix_precision.utils import categorize_layers

        cats = categorize_layers(MockLLaMAModel(), model_type="llama")
        searcher = ConfigSearcher(
            search_config=ModuleSearchConfig(),
            hardware=HardwareTarget.MI300,
            available_partitions=set(cats.keys()),
        )
        configs = searcher.generate_sorted_configs()
        assert len(configs) > 0
        assert "linear_attn" not in searcher.layer_sensitivity
        assert searcher.layer_sensitivity["self_attn"] == 3
        assert searcher.layer_sensitivity["dense_mlp"] == 1
        assert "routed_moe" not in searcher.layer_sensitivity


# =============================================================================
# vllm_plugin.py — plugin classes (FakeQuantLinearMethod / QuantVLLMParallelLinearBase /
# QuantVLLMFusedMoE / reset_vllm_fake_quant_model / calibrate_moe_weight_params)
# Tested via mocked QuantMixin internals; vLLM is provided by the module-level stubs.
# =============================================================================


class _RecordingApply:
    """Stand-in for an ``original_quant_method`` exposing ``.apply``."""

    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.calls: list[tuple[torch.nn.Module, torch.Tensor, torch.Tensor | None]] = []

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        # Snapshot weight value at call time (not the parameter object) for assertions.
        self.calls.append((layer, x.clone(), None if bias is None else bias.clone()))
        return self.output


class _MockQuantLayer:
    """Minimal stand-in for a QuantMixin instance used by FakeQuantLinearMethod."""

    def __init__(
        self,
        weight_quantizer: Any = None,
        weight_quantizer_inv: Any = None,
    ) -> None:
        self.weight_quantizer = weight_quantizer
        self._weight_quantizer_inv = weight_quantizer_inv
        self.input_calls: list[torch.Tensor] = []
        self.bias_calls: list[torch.Tensor] = []
        self.output_calls: list[torch.Tensor] = []
        self.weight_calls: list[torch.Tensor] = []

    def get_quant_input(self, x: torch.Tensor) -> torch.Tensor:
        self.input_calls.append(x.clone())
        return x * 2.0

    def get_quant_bias(self, b: torch.Tensor) -> torch.Tensor:
        self.bias_calls.append(b.clone())
        return b + 1.0

    def get_quant_output(self, y: torch.Tensor) -> torch.Tensor:
        self.output_calls.append(y.clone())
        return y - 0.5

    def get_quant_weight(self, w: torch.Tensor) -> torch.Tensor:
        self.weight_calls.append(w.clone())
        return torch.full_like(w, 7.0)


class _FrozenQuantizerStub:
    """Stand-in quantizer with frozen_params=True (no runtime weight QDQ)."""

    frozen_params = True


class _DynamicQuantizerStub:
    """Stand-in quantizer with frozen_params=False (runtime weight QDQ)."""

    frozen_params = False


class TestFakeQuantLinearMethodApply:
    def test_no_runtime_weight_override_when_frozen_and_no_inv(self):
        from quark.experimental.torch.plugin.vllm_plugin import FakeQuantLinearMethod

        layer = nn.Linear(4, 4)
        original_weight = layer.weight
        original = _RecordingApply(output=torch.zeros(2, 4))
        quant_layer = _MockQuantLayer(weight_quantizer=_FrozenQuantizerStub())
        method = FakeQuantLinearMethod(original, quant_layer)

        x = torch.ones(2, 4)
        out = method.apply(layer, x)

        # Frozen + no inverse quantizer: weight is NOT replaced before apply.
        assert layer.weight is original_weight
        assert quant_layer.weight_calls == []
        # input goes through get_quant_input first
        assert len(quant_layer.input_calls) == 1
        torch.testing.assert_close(original.calls[0][1], x * 2.0)
        # output goes through get_quant_output last
        assert len(quant_layer.output_calls) == 1
        torch.testing.assert_close(out, torch.zeros(2, 4) - 0.5)

    def test_runtime_weight_override_when_inv_present(self):
        from quark.experimental.torch.plugin.vllm_plugin import FakeQuantLinearMethod

        layer = nn.Linear(4, 4)
        original_weight = layer.weight
        original = _RecordingApply(output=torch.zeros(2, 4))
        quant_layer = _MockQuantLayer(
            weight_quantizer=_FrozenQuantizerStub(),
            weight_quantizer_inv=object(),  # truthy → triggers override path
        )
        method = FakeQuantLinearMethod(original, quant_layer)

        out = method.apply(layer, torch.ones(2, 4))

        # get_quant_weight was invoked; original weight restored after apply
        assert len(quant_layer.weight_calls) == 1
        assert layer.weight is original_weight
        # original.apply saw the *quantized* weight (all 7s). Verify by checking
        # the layer's weight was a Parameter wrapping 7s during the call window —
        # we can't intercept that directly, but we can confirm the original
        # parameter is restored as a Parameter object (not the temp one).
        assert isinstance(layer.weight, nn.Parameter)
        assert torch.equal(layer.weight, original_weight)
        torch.testing.assert_close(out, torch.zeros(2, 4) - 0.5)

    def test_runtime_weight_override_when_dynamic_quantizer(self):
        from quark.experimental.torch.plugin.vllm_plugin import FakeQuantLinearMethod

        layer = nn.Linear(4, 4)
        original = _RecordingApply(output=torch.zeros(2, 4))
        quant_layer = _MockQuantLayer(weight_quantizer=_DynamicQuantizerStub())
        method = FakeQuantLinearMethod(original, quant_layer)

        method.apply(layer, torch.ones(2, 4))

        # Dynamic (non-frozen) weight quantizer triggers runtime override too
        assert len(quant_layer.weight_calls) == 1

    def test_bias_passes_through_get_quant_bias(self):
        from quark.experimental.torch.plugin.vllm_plugin import FakeQuantLinearMethod

        layer = nn.Linear(4, 4)
        original = _RecordingApply(output=torch.zeros(2, 4))
        quant_layer = _MockQuantLayer(weight_quantizer=_FrozenQuantizerStub())
        method = FakeQuantLinearMethod(original, quant_layer)

        bias = torch.full((4,), 3.0)
        method.apply(layer, torch.ones(2, 4), bias=bias)

        assert len(quant_layer.bias_calls) == 1
        # original.apply received bias + 1 (from get_quant_bias)
        torch.testing.assert_close(original.calls[0][2], bias + 1.0)

    def test_non_contiguous_input_made_contiguous(self):
        from quark.experimental.torch.plugin.vllm_plugin import FakeQuantLinearMethod

        layer = nn.Linear(4, 4)
        original = _RecordingApply(output=torch.zeros(4, 2))
        quant_layer = _MockQuantLayer(weight_quantizer=_FrozenQuantizerStub())
        method = FakeQuantLinearMethod(original, quant_layer)

        x = torch.ones(2, 4).t()  # transpose makes it non-contiguous
        assert not x.is_contiguous()
        method.apply(layer, x)

        # The tensor passed to get_quant_input must be contiguous
        assert quant_layer.input_calls[0].is_contiguous()


class TestMoeA2PatchState:
    @dataclasses.dataclass
    class _AiterMetadata:
        stage2: Any
        run_1stage: bool = False

    class _AiterStage2:
        func = object()

        def __init__(self):
            self.calls: list[torch.Tensor] = []

        def __call__(self, a2: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            self.calls.append(a2.clone())
            return a2

    def test_top_one_routing_quantizes_only_second_triton_gemm(self, monkeypatch):
        import quark.experimental.torch.plugin.vllm_plugin as vp

        calls = []

        def invoke(a, *args, **kwargs):
            calls.append(a.clone())

        monkeypatch.setattr(vp, "_orig_invoke_fused_moe_triton_kernel", invoke)
        state = vp._MoeA2PatchState(quantizer=lambda value: value + 10.0)
        token = vp._quark_moe_a2_ctx.set(state)
        common_args = (
            torch.ones(1),
            torch.ones(1),
            None,
            None,
            None,
            None,
            torch.ones(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            False,
            1,
            {},
            None,
            False,
            False,
            False,
            False,
            False,
        )
        try:
            vp._patched_invoke_fused_moe_triton_kernel(torch.ones(1), *common_args)
            vp._patched_invoke_fused_moe_triton_kernel(torch.full((1,), 2.0), *common_args)
        finally:
            vp._quark_moe_a2_ctx.reset(token)

        torch.testing.assert_close(calls[0], torch.ones(1))
        torch.testing.assert_close(calls[1], torch.full((1,), 12.0))
        assert state.applied is True

    def test_pre_source_hook_applies_target_only_before_second_source_quant(self, monkeypatch):
        import quark.experimental.torch.plugin.vllm_plugin as vp

        source_inputs: list[torch.Tensor] = []

        def source_quantize(a, *args, **kwargs):
            source_inputs.append(a.clone())
            return a, None

        monkeypatch.setattr(vp, "_orig_moe_kernel_quantize_input", source_quantize)
        state = vp._MoeA2PatchState(quantizer=lambda value: value + 10.0, source_is_prequantized=True)
        token = vp._quark_moe_a2_ctx.set(state)
        try:
            vp._patched_moe_kernel_quantize_input(torch.ones(2, 4), None, None, False)
            vp._patched_moe_kernel_quantize_input(torch.full((2, 4), 2.0), None, None, False)
        finally:
            vp._quark_moe_a2_ctx.reset(token)

        torch.testing.assert_close(source_inputs[0], torch.ones(2, 4))
        torch.testing.assert_close(source_inputs[1], torch.full((2, 4), 12.0))
        assert state.applied is True
        assert state.a2_ready_for_invoke is True

    def test_prequantized_low_level_fallback_fails_before_wrong_order(self, monkeypatch):
        import quark.experimental.torch.plugin.vllm_plugin as vp

        monkeypatch.setattr(vp, "_orig_invoke_fused_moe_triton_kernel", lambda *args, **kwargs: None)
        state = vp._MoeA2PatchState(quantizer=lambda value: value + 10.0, source_is_prequantized=True)
        token = vp._quark_moe_a2_ctx.set(state)
        common_args = (
            torch.ones(1),
            torch.ones(1),
            None,
            None,
            None,
            None,
            torch.ones(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            False,
            1,
            {},
            None,
            False,
            False,
            False,
            False,
            False,
        )
        try:
            vp._patched_invoke_fused_moe_triton_kernel(torch.ones(1), *common_args)
            with pytest.raises(RuntimeError, match="without applying target QDQ"):
                vp._patched_invoke_fused_moe_triton_kernel(torch.ones(1), *common_args)
        finally:
            vp._quark_moe_a2_ctx.reset(token)

    def test_native_triton_activation_hook_quantizes_a2_in_place(self):
        import quark.experimental.torch.plugin.vllm_plugin as vp

        output = torch.full((2, 4), 2.0)
        state = vp._MoeA2PatchState(quantizer=lambda value: value + 10.0, source_is_prequantized=True)
        token = vp._quark_moe_a2_ctx.set(state)
        try:
            vp._apply_target_moe_a2_qdq_in_place(output)
            vp._apply_target_moe_a2_qdq_in_place(output)
        finally:
            vp._quark_moe_a2_ctx.reset(token)

        torch.testing.assert_close(output, torch.full((2, 4), 22.0))
        assert state.applied is True

    def test_aiter_two_stage_metadata_applies_target_qdq_before_stage2(self, monkeypatch):
        import quark.experimental.torch.plugin.vllm_plugin as vp

        stage2 = self._AiterStage2()
        metadata = self._AiterMetadata(stage2=stage2)
        monkeypatch.setattr(vp, "_orig_aiter_get_2stage_cfgs", lambda *args, **kwargs: metadata)
        state = vp._MoeA2PatchState(
            quantizer=lambda value: value + 10,
            source_is_prequantized=True,
            source_backend="AITER_MXFP4_BF16",
        )
        token = vp._quark_moe_a2_ctx.set(state)
        try:
            patched = vp._patched_aiter_get_2stage_cfgs("shape-key")
            result = patched.stage2(torch.full((2, 4), 2, dtype=torch.bfloat16), a2_scale=None)
        finally:
            vp._quark_moe_a2_ctx.reset(token)

        assert patched is not metadata
        assert patched.stage2.func is stage2.func
        torch.testing.assert_close(stage2.calls[0], torch.full((2, 4), 12, dtype=torch.bfloat16))
        torch.testing.assert_close(result, stage2.calls[0])
        assert state.applied is True
        assert state.aiter_stage2_calls == 1

    def test_aiter_metadata_is_untouched_without_active_target_qdq(self, monkeypatch):
        import quark.experimental.torch.plugin.vllm_plugin as vp

        metadata = self._AiterMetadata(stage2=self._AiterStage2())
        monkeypatch.setattr(vp, "_orig_aiter_get_2stage_cfgs", lambda *args, **kwargs: metadata)

        assert vp._patched_aiter_get_2stage_cfgs("shape-key") is metadata

    def test_aiter_one_stage_fails_closed_when_target_a2_qdq_is_active(self, monkeypatch):
        import quark.experimental.torch.plugin.vllm_plugin as vp

        metadata = self._AiterMetadata(stage2=None, run_1stage=True)
        monkeypatch.setattr(vp, "_orig_aiter_get_2stage_cfgs", lambda *args, **kwargs: metadata)
        state = vp._MoeA2PatchState(
            quantizer=lambda value: value,
            source_is_prequantized=True,
            source_backend="AITER_MXFP4_BF16",
        )
        token = vp._quark_moe_a2_ctx.set(state)
        try:
            with pytest.raises(RuntimeError, match="two-stage kernel is required"):
                vp._patched_aiter_get_2stage_cfgs("shape-key")
        finally:
            vp._quark_moe_a2_ctx.reset(token)


class TestSetModuleAttrAllowNonParameter:
    def test_non_parameter_value_removes_from_parameters_dict(self):
        from quark.experimental.torch.plugin.vllm_plugin import _set_module_attr_allow_non_parameter

        module = nn.Linear(4, 4)
        # Pretend "weight" was previously registered as a Parameter (it is)
        assert "weight" in module._parameters

        replacement = torch.zeros(4, 4)  # plain Tensor, not a Parameter
        _set_module_attr_allow_non_parameter(module, "weight", replacement)

        assert "weight" not in module._parameters
        assert torch.equal(module.weight, replacement)
        assert not isinstance(module.weight, nn.Parameter)

    def test_parameter_value_left_in_parameters_dict(self):
        from quark.experimental.torch.plugin.vllm_plugin import _set_module_attr_allow_non_parameter

        module = nn.Linear(4, 4)
        new_param = nn.Parameter(torch.ones(4, 4))
        _set_module_attr_allow_non_parameter(module, "weight", new_param)

        # Parameter-valued sets should leave the parameter dict entry
        assert "weight" in module._parameters
        assert torch.equal(module.weight, torch.ones(4, 4))


class TestSetQuantMethodAttr:
    def test_non_module_assignment_clears_modules_dict(self):
        from quark.experimental.torch.plugin.vllm_plugin import _set_quant_method_attr

        host = nn.Module()
        # First register quant_method as a child Module, then swap to a plain object.
        prior = nn.Linear(4, 4)
        host.add_module("quant_method", prior)
        assert "quant_method" in host._modules

        plain = SimpleNamespace(name="plain")
        _set_quant_method_attr(host, plain)

        assert "quant_method" not in host._modules
        assert host.__dict__["quant_method"] is plain
        assert host.quant_method is plain

    def test_module_assignment_clears_dict_entry(self):
        from quark.experimental.torch.plugin.vllm_plugin import _set_quant_method_attr

        host = nn.Module()
        host.__dict__["quant_method"] = SimpleNamespace(name="plain")

        new_method = nn.Linear(4, 4)
        _set_quant_method_attr(host, new_method)

        assert host._modules.get("quant_method") is new_method
        assert "quant_method" not in host.__dict__


class TestLogVllmPrequantWrap:
    def test_does_not_raise_with_full_module(self):
        from quark.experimental.torch.plugin.vllm_plugin import _log_vllm_prequant_wrap

        module = _make_fp8_linear_module("RowParallelLinear", use_scale_inv=True)
        _log_vllm_prequant_wrap(module, "QuantVLLMRowParallelLinear")  # should not raise

    def test_falls_back_through_scale_attrs(self):
        from quark.experimental.torch.plugin.vllm_plugin import _log_vllm_prequant_wrap

        module = _make_fp8_moe_module(use_scale_inv=False)
        _log_vllm_prequant_wrap(module, "QuantVLLMFusedMoE")  # should not raise

    def test_handles_module_without_quant_method(self):
        from quark.experimental.torch.plugin.vllm_plugin import _log_vllm_prequant_wrap

        module = nn.Linear(4, 4)
        _log_vllm_prequant_wrap(module, "Wrapper")  # should not raise


# ---------------------------------------------------------------------------
# QuantVLLMParallelLinearBase
# ---------------------------------------------------------------------------


def _make_qlayer_config_empty() -> Any:
    """Build a QLayerConfig with all specs unset (init_quantizer → all None)."""
    from quark.torch.quantization.config.config import QLayerConfig

    return QLayerConfig()


class TestQuantVLLMParallelLinearBaseInit:
    def test_default_state_after_init(self):
        from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(quant_config=None)
        assert wrapper._quant_config is None
        assert wrapper._device.type in ("cuda", "cpu")  # default cuda but cpu-only CI ok
        assert wrapper._quantizer_initialized is False
        assert wrapper._float_module_cls is None
        assert wrapper._float_init_kwargs is None
        assert wrapper._weight_quantizer_inv is None
        assert wrapper._source_module is None

    def test_init_quantizers_is_noop_when_no_config(self):
        from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(quant_config=None)
        wrapper._init_quantizers()  # must not raise
        assert wrapper._quantizer_initialized is False

    def test_init_quantizers_wraps_quant_method_in_fake_quant(self):
        from quark.experimental.torch.plugin.vllm_plugin import FakeQuantLinearMethod, QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(
            quant_config=_make_qlayer_config_empty(),
            device=torch.device("cpu"),
        )
        original_method = SimpleNamespace(apply=lambda layer, x, bias=None: x)
        # Plain attribute (not registered child module)
        wrapper.__dict__["quant_method"] = original_method

        wrapper._init_quantizers()

        assert wrapper._quantizer_initialized is True
        assert isinstance(wrapper.quant_method, FakeQuantLinearMethod)
        assert wrapper.quant_method.original_quant_method is original_method
        assert wrapper.quant_method.quant_layer is wrapper
        assert wrapper._original_quant_method is original_method

    def test_init_quantizers_idempotent(self):
        from quark.experimental.torch.plugin.vllm_plugin import FakeQuantLinearMethod, QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(
            quant_config=_make_qlayer_config_empty(),
            device=torch.device("cpu"),
        )
        wrapper.__dict__["quant_method"] = SimpleNamespace(apply=lambda *a, **kw: None)
        wrapper._init_quantizers()
        wrapped_once = wrapper.quant_method
        wrapper._init_quantizers()
        # Second call must not double-wrap
        assert wrapper.quant_method is wrapped_once
        assert isinstance(wrapper.quant_method, FakeQuantLinearMethod)
        assert not isinstance(wrapper.quant_method.original_quant_method, FakeQuantLinearMethod)


class TestQuantVLLMParallelLinearBaseGetQuantWeight:
    def test_inverse_quantizer_path_used_when_set(self):
        from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(
            quant_config=_make_qlayer_config_empty(),
            device=torch.device("cpu"),
        )
        wrapper._init_quantizers()  # makes _weight_quantizer = None (empty config)

        # Inject an inverse quantizer that returns a known dequant tensor
        dequant_value = torch.full((4, 4), 9.0)
        inv = SimpleNamespace(dequantize=lambda w: dequant_value)
        wrapper._weight_quantizer_inv = inv

        result = wrapper.get_quant_weight(torch.zeros(4, 4))
        # No fwd quantizer → returns inverse-quantized tensor as is
        torch.testing.assert_close(result, dequant_value)

    def test_passthrough_when_no_inverse_and_no_quantizer(self):
        from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(
            quant_config=_make_qlayer_config_empty(),
            device=torch.device("cpu"),
        )
        wrapper._init_quantizers()
        x = torch.arange(16.0).reshape(4, 4)
        # Empty config + no inverse → falls back to QuantMixin.get_quant_weight which is identity
        torch.testing.assert_close(wrapper.get_quant_weight(x), x)

    def test_weight_quantizer_skipped_when_source_weight_matches_target(self):
        from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(
            quant_config=_make_qlayer_config_empty(),
            device=torch.device("cpu"),
        )
        wrapper._init_quantizers()
        dequant_value = torch.full((4, 4), 9.0)
        wrapper._weight_quantizer_inv = SimpleNamespace(dequantize=lambda w: dequant_value)
        # A target weight quantizer that would perturb the value if applied.
        wrapper._weight_quantizer = lambda w: w + 100.0

        # Source weight == target weight → skip the target weight QDQ (return source dequant).
        wrapper._source_weight_matches_target = True
        torch.testing.assert_close(wrapper.get_quant_weight(torch.zeros(4, 4)), dequant_value)

        # Source weight differs → apply the target weight QDQ.
        wrapper._source_weight_matches_target = False
        torch.testing.assert_close(wrapper.get_quant_weight(torch.zeros(4, 4)), dequant_value + 100.0)


class TestVLLMOnlineQuantizationState:
    def test_enables_rocm_contiguous_safeguard_only_for_new_mxfp4_weight_qdq(self, monkeypatch, vllm_plugin):
        from quark.torch.quantization.config.type import Dtype

        wrapper = vllm_plugin.QuantVLLMParallelLinearBase(quant_config=None, device=torch.device("cpu"))
        wrapper._weight_quantizer = SimpleNamespace(dtype=Dtype.fp4)
        model = nn.Sequential(wrapper)
        monkeypatch.setattr(torch.version, "hip", "7.2")

        try:
            assert vllm_plugin.set_vllm_online_quantization_state(model, active=True) is True

            wrapper._source_weight_matches_target = True
            assert vllm_plugin.set_vllm_online_quantization_state(model, active=True) is False

            wrapper._source_weight_matches_target = False
            assert vllm_plugin.set_vllm_online_quantization_state(model, active=False) is False
        finally:
            vllm_plugin.set_vllm_online_quantization_state(model, active=False)


class TestQuantVLLMPrequantizedLinear:
    @staticmethod
    def _populate_linear_attrs(module):
        module.input_size = 4
        module.output_size = 4
        module.bias = None
        module.skip_bias_add = False
        module.params_dtype = torch.bfloat16
        module.prefix = "model.layers.0.proj"
        module.return_bias = True
        module.disable_tp = False

    def test_replicated_linear_from_float_uses_exact_match_path(self):
        import quark.experimental.torch.plugin.vllm_plugin as vp
        from quark.experimental.torch.mix_precision.config import get_layer_config

        source = _make_fp8_linear_module("ReplicatedLinear")
        source.quant_method.quant_config = SimpleNamespace(activation_scheme="static")
        self._populate_linear_attrs(source)

        wrapper = vp.QuantVLLMReplicatedLinear.from_float(
            source,
            get_layer_config("fp8"),
            device=torch.device("cpu"),
        )

        assert wrapper._source_matches_target is True
        assert wrapper._source_module is source
        assert wrapper.is_prequantized is True
        assert wrapper.weight_quantizer is None
        assert wrapper.input_quantizer is None

        from quark.torch.quantization.api import ModelQuantizer

        before_weight = source.weight.detach().clone()
        ModelQuantizer.freeze(nn.Sequential(wrapper))
        torch.testing.assert_close(source.weight, before_weight)

    @pytest.mark.parametrize("source_format", ["block", "per_channel", "compressed_channel"])
    def test_mismatched_fp8_granularity_applies_target_weight_qdq(self, monkeypatch, source_format):
        import quark.experimental.torch.plugin.vllm_plugin as vp
        from quark.experimental.torch.mix_precision.config import get_layer_config

        if source_format == "compressed_channel":
            source = _make_compressed_tensors_fp8_module("linear", "channel")
        else:
            source = _make_fp8_linear_module("ReplicatedLinear")
            source.quant_method.quant_config = SimpleNamespace(activation_scheme="static")
            if source_format == "block":
                source.quant_method.weight_block_size = (2, 2)
            else:
                source.quant_method.weight_qscheme = "per_channel"
        self._populate_linear_attrs(source)
        dequant_weight = torch.full((4, 4), 9.0)
        inverse = SimpleNamespace(dequantize=lambda weight: dequant_weight)
        monkeypatch.setattr(vp, "_create_prequantized_vllm_linear_inverse_quantizer", lambda layer: inverse)

        wrapper = vp.QuantVLLMReplicatedLinear.from_float(
            source,
            get_layer_config("fp8"),
            device=torch.device("cpu"),
        )

        assert wrapper._source_matches_target is False
        assert wrapper._source_weight_matches_target is False
        assert wrapper._weight_quantizer_inv is inverse
        assert wrapper.weight_quantizer is not None
        observed = []

        class _TargetWeightQuantizer(nn.Module):
            def forward(self, weight):
                observed.append(weight.clone())
                return weight + 1.0

        wrapper._weight_quantizer = _TargetWeightQuantizer()
        torch.testing.assert_close(wrapper.get_quant_weight(source.weight), dequant_weight + 1.0)
        assert len(observed) == 1
        torch.testing.assert_close(observed[0], dequant_weight)

    def test_qkv_exact_match_still_runs_output_observer(self, monkeypatch):
        import quark.experimental.torch.plugin.vllm_plugin as vp
        from quark.experimental.torch.mix_precision.config import get_layer_config
        from quark.experimental.torch.mix_precision.utils import _get_fp8_output_spec

        monkeypatch.setattr(vp.vllm_linear.QKVParallelLinear, "output_sizes", [4, 2, 2], raising=False)
        source = _make_fp8_linear_module("QKVParallelLinear")
        source.quant_method.quant_config = SimpleNamespace(activation_scheme="static")
        self._populate_linear_attrs(source)
        source.hidden_size = 4
        source.head_size = 2
        source.total_num_heads = 2
        source.total_num_kv_heads = 2
        source.output_sizes = [4, 2, 2]
        source.forward = lambda x: (x + 1.0, None)

        target = deepcopy(get_layer_config("fp8"))
        target.output_tensors = _get_fp8_output_spec()
        wrapper = vp.QuantVLLMQKVParallelLinear.from_prequantized(
            source,
            target,
            device=torch.device("cpu"),
        )

        assert wrapper._source_matches_target is True
        assert wrapper._source_weight_matches_target is True
        assert wrapper.is_prequantized is True
        assert isinstance(wrapper._output_quantizer, vp.QKVOutputObserverQuantizer)

        observed = []

        class _OutputObserver(nn.Module):
            def forward(self, value):
                observed.append(value.clone())
                return value

        wrapper._output_quantizer = _OutputObserver()
        result, bias = wrapper(torch.ones(1, 4))

        assert bias is None
        torch.testing.assert_close(result, torch.full((1, 4), 2.0))
        assert len(observed) == 1


class TestQuantVLLMParallelLinearBaseToFloatModule:
    def test_uses_source_module_when_present(self):
        from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(quant_config=None, device=torch.device("cpu"))
        sentinel = nn.Linear(4, 4)
        wrapper._source_module = sentinel

        assert wrapper.to_float_module() is sentinel

    def test_raises_when_no_metadata(self):
        from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(quant_config=None, device=torch.device("cpu"))
        with pytest.raises(ValueError, match="float-module metadata"):
            wrapper.to_float_module()

    def test_rebuilds_float_module_from_metadata(self):
        from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        # Use a Linear subclass that pre-declares quant_method so the
        # `hasattr(float_module, "quant_method")` guard in to_float_module passes.
        class LinearWithQuantMethod(nn.Linear):
            quant_method = None  # default; will be overwritten by to_float_module

        wrapper = QuantVLLMParallelLinearBase(quant_config=None, device=torch.device("cpu"))
        wrapper._float_module_cls = LinearWithQuantMethod
        wrapper._float_init_kwargs = {"in_features": 4, "out_features": 4, "extra_unused": "ignored"}

        wrapper.weight = nn.Parameter(torch.ones(4, 4))
        wrapper.bias = nn.Parameter(torch.zeros(4))
        wrapper._original_quant_method = SimpleNamespace(name="orig")

        rebuilt = wrapper.to_float_module()

        assert isinstance(rebuilt, LinearWithQuantMethod)
        assert rebuilt.in_features == 4
        assert rebuilt.out_features == 4
        torch.testing.assert_close(rebuilt.weight, torch.ones(4, 4))
        # quant_method gets restored on the rebuilt float module
        assert rebuilt.quant_method.name == "orig"

    def test_rebuilds_float_module_skips_quant_method_when_not_supported(self):
        from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        # Plain nn.Linear has no quant_method attribute → guard skips assignment.
        wrapper = QuantVLLMParallelLinearBase(quant_config=None, device=torch.device("cpu"))
        wrapper._float_module_cls = nn.Linear
        wrapper._float_init_kwargs = {"in_features": 4, "out_features": 4}
        wrapper.weight = nn.Parameter(torch.ones(4, 4))
        wrapper.bias = nn.Parameter(torch.zeros(4))
        wrapper._original_quant_method = SimpleNamespace(name="orig")

        rebuilt = wrapper.to_float_module()

        assert isinstance(rebuilt, nn.Linear)
        assert not hasattr(rebuilt, "quant_method")


# ---------------------------------------------------------------------------
# QuantVLLMFusedMoE
# ---------------------------------------------------------------------------


class _FakeMoEInner(nn.Module):
    """Minimal stand-in for vLLM FusedMoE. Tracks calls to forward/forward_impl."""

    def __init__(self) -> None:
        super().__init__()
        self.w13_weight = nn.Parameter(torch.ones(2, 4, 4))
        self.w2_weight = nn.Parameter(torch.ones(2, 4, 4))
        self.forward_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.forward_impl_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        # Record what w13/w2 looked like at call time (clones to detach from later restore)
        self.forward_calls.append(
            (
                hidden_states.clone(),
                router_logits.clone(),
                self.w13_weight.detach().clone(),
                self.w2_weight.detach().clone(),
            )
        )
        return hidden_states + 1.0

    def forward_impl(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        self.forward_impl_calls.append(
            (
                hidden_states.clone(),
                router_logits.clone(),
                self.w13_weight.detach().clone(),
                self.w2_weight.detach().clone(),
            )
        )
        return hidden_states - 1.0


def _make_quant_moe_wrapper() -> Any:
    """Create a QuantVLLMFusedMoE wrapping a minimal FakeMoEInner with empty QLayerConfig."""
    from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMFusedMoE

    return QuantVLLMFusedMoE(
        inner=_FakeMoEInner(),
        layer_quant_config=_make_qlayer_config_empty(),
        device=torch.device("cpu"),
    )


class TestQuantVLLMFusedMoEInit:
    def test_state_after_init_with_empty_qlayer_config(self):
        wrapper = _make_quant_moe_wrapper()
        assert wrapper._quantizer_initialized is True  # _init_moe_quantizers ran in __init__
        # Empty QLayerConfig → all quantizers None
        a1, a2, w13, w2 = wrapper._get_moe_quantizers()
        assert a1 is None and a2 is None and w13 is None and w2 is None
        # Inverse quantizers default to None
        assert wrapper._w13_weight_quantizer_inv is None
        assert wrapper._w2_weight_quantizer_inv is None

    def test_inner_is_registered_as_child_module(self):
        wrapper = _make_quant_moe_wrapper()
        assert "_moe_inner" in wrapper._modules
        assert wrapper._inner is wrapper._modules["_moe_inner"]

    def test_getattr_delegates_to_inner_for_unknown_attributes(self):
        wrapper = _make_quant_moe_wrapper()
        # w13_weight is on inner, not directly on wrapper.__dict__; __getattr__ delegates
        assert torch.equal(wrapper.w13_weight, wrapper._inner.w13_weight)

    def test_getattr_returns_none_for_special_calibration_attrs(self):
        wrapper = _make_quant_moe_wrapper()
        # api.py expects these; MoE has none, so __getattr__ returns None
        assert wrapper._weight_quantizer is None
        assert wrapper._bias_quantizer is None


class TestQuantVLLMMoERunner:
    def test_weight_holder_redirects_to_routed_experts(self, vllm_plugin):
        routed_experts = _FakeMoEInner()
        runner = vllm_plugin.vllm_moe_runner(routed_experts=routed_experts)

        wrapper = vllm_plugin.QuantVLLMMoERunner.from_float(
            float_module=runner,
            layer_quant_config=_make_qlayer_config_empty(),
            device=torch.device("cpu"),
        )

        assert wrapper._inner is runner
        assert wrapper._weight_holder is routed_experts
        assert wrapper._weight_holder.w13_weight is routed_experts.w13_weight
        assert wrapper._weight_holder.w2_weight is routed_experts.w2_weight

    def test_a1_quantizes_only_routed_input_not_shared_or_router(self, vllm_plugin):
        class _RoutedExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.w13_weight = nn.Parameter(torch.ones(2, 4, 4))
                self.w2_weight = nn.Parameter(torch.ones(2, 4, 4))

        class _Gate(nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor):
                self.inputs.append(x.clone())
                return x.sum(dim=-1, keepdim=True), None

        class _Runner(nn.Module):
            def __init__(self):
                super().__init__()
                self.routed_experts = _RoutedExperts()
                self.gate = _Gate()
                self._fse_fuse_gate = False
                self.observed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

            def apply_routed_input_transform(self, hidden_states: torch.Tensor):
                return hidden_states, hidden_states

            def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
                routed, shared = self.apply_routed_input_transform(hidden_states)
                if self.gate is not None:
                    router_logits, _ = self.gate(routed)
                assert shared is not None
                self.observed = (routed.clone(), shared.clone(), router_logits.clone())
                return routed + shared

        runner = _Runner()
        original_gate = runner.gate
        original_transform = runner.apply_routed_input_transform
        wrapper = vllm_plugin.QuantVLLMMoERunner(
            inner=runner,
            layer_quant_config=_make_qlayer_config_empty(),
            device=torch.device("cpu"),
        )

        class _A1:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * 10.0

        wrapper._a1_input_quantizer = _A1()
        hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        out = wrapper._apply_fake_quant_and_forward("forward", hidden, hidden.clone())

        assert runner.observed is not None
        routed, shared, router_logits = runner.observed
        torch.testing.assert_close(routed, hidden * 10.0)
        torch.testing.assert_close(shared, hidden)
        torch.testing.assert_close(original_gate.inputs[0], hidden)
        torch.testing.assert_close(router_logits, hidden.sum(dim=-1, keepdim=True))
        torch.testing.assert_close(out, hidden * 10.0 + hidden)
        assert runner.gate is original_gate
        assert runner.apply_routed_input_transform == original_transform


class TestQuantVLLMFusedMoEApplyMoEWeightQuantizer:
    def test_none_quantizer_returns_weight_unchanged(self):
        wrapper = _make_quant_moe_wrapper()
        w = torch.arange(24.0).reshape(2, 3, 4)
        result = wrapper._apply_moe_weight_quantizer(None, w)
        assert result is w

    def test_per_expert_channel_path_reshapes_to_2d(self):
        """[E, C, K] weight is flattened to [E*C, K] for per-expert-channel QDQ."""
        wrapper = _make_quant_moe_wrapper()

        captured: list[torch.Tensor] = []

        class FakeQuantizer:
            _quark_moe_per_expert_channel = True

            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                captured.append(x.clone())
                return x * 2.0  # numel-preserving so reshape branch fires

        weight = torch.arange(24.0).reshape(2, 3, 4)  # [E=2, C=3, K=4]
        out = wrapper._apply_moe_weight_quantizer(FakeQuantizer(), weight)

        # Quantizer should see a [E*C, K] = [6, 4] flattened view
        assert captured[0].shape == (6, 4)
        # Output reshaped back to [E, C, K]
        assert out.shape == (2, 3, 4)
        torch.testing.assert_close(out, weight * 2.0)

    def test_per_expert_tensor_path_reshapes_to_2d(self):
        """[E, C, K] weight is flattened to [E, C*K] for per-expert-tensor QDQ."""
        wrapper = _make_quant_moe_wrapper()

        captured: list[torch.Tensor] = []

        class FakeQuantizer:
            _quark_moe_per_expert_tensor = True

            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                captured.append(x.clone())
                return x + 0.5

        weight = torch.arange(24.0).reshape(2, 3, 4)
        out = wrapper._apply_moe_weight_quantizer(FakeQuantizer(), weight)

        assert captured[0].shape == (2, 12)
        assert out.shape == (2, 3, 4)
        torch.testing.assert_close(out, weight + 0.5)

    def test_per_expert_tensor_falls_back_for_non_3d_weight(self):
        wrapper = _make_quant_moe_wrapper()

        captured: list[torch.Tensor] = []

        class FakeQuantizer:
            _quark_moe_per_expert_tensor = True

            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                captured.append(x.clone())
                return x

        weight_2d = torch.ones(4, 4)
        wrapper._apply_moe_weight_quantizer(FakeQuantizer(), weight_2d)
        # Non-3D weight skips reshape and calls quantizer(weight) directly
        assert captured[0].shape == (4, 4)


class TestQuantVLLMFusedMoEApplyFakeQuantAndForward:
    def test_missing_a2_backend_hook_fails_closed(self):
        wrapper = _make_quant_moe_wrapper()
        wrapper.__dict__["_a2_input_quantizer"] = lambda value: value

        with pytest.raises(RuntimeError, match="did not expose a2"):
            wrapper._apply_fake_quant_and_forward(
                "forward",
                torch.zeros(2, 4),
                torch.zeros(2, 2),
            )

    def test_inner_forward_called_and_weights_restored(self):
        wrapper = _make_quant_moe_wrapper()
        orig_w13 = wrapper._inner.w13_weight
        orig_w2 = wrapper._inner.w2_weight

        hidden = torch.full((2, 4), 3.0)
        router = torch.zeros(2, 2)

        out = wrapper._apply_fake_quant_and_forward("forward", hidden, router)

        assert len(wrapper._inner.forward_calls) == 1
        # forward output preserved
        torch.testing.assert_close(out, hidden + 1.0)
        # No weight quantizer / no inverse → weights unchanged inside forward
        # but key invariant: original Parameter object restored after the call
        assert wrapper._inner.w13_weight is orig_w13
        assert wrapper._inner.w2_weight is orig_w2

    def test_inverse_quantizer_dequantizes_w13_for_inner_call(self):
        wrapper = _make_quant_moe_wrapper()
        orig_w13 = wrapper._inner.w13_weight

        # Inverse quantizer dequantizes orig_w13 into a known tensor; the rebind
        # branch fires because _w13_weight_quantizer_inv is not None.
        dequant_w13 = torch.full((2, 4, 4), 5.0)
        wrapper._w13_weight_quantizer_inv = SimpleNamespace(dequantize=lambda w: dequant_w13)

        hidden = torch.zeros(2, 4)
        router = torch.zeros(2, 2)
        wrapper._apply_fake_quant_and_forward("forward", hidden, router)

        # The forward call observed the dequantized w13 (5.0) on inner.w13_weight
        observed_w13 = wrapper._inner.forward_calls[0][2]
        torch.testing.assert_close(observed_w13, dequant_w13)
        # Original Parameter object restored after the forward call returns
        assert wrapper._inner.w13_weight is orig_w13

    def test_a1_quantizer_applied_to_hidden_states(self):
        wrapper = _make_quant_moe_wrapper()

        captured: list[torch.Tensor] = []

        class _A1:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                captured.append(x.clone())
                return x * 10.0

        # Inject a1 quantizer directly
        wrapper._a1_input_quantizer = _A1()

        hidden = torch.full((2, 4), 1.5)
        router = torch.zeros(2, 2)
        wrapper._apply_fake_quant_and_forward("forward", hidden, router)

        assert len(captured) == 1
        torch.testing.assert_close(captured[0], hidden)
        # inner.forward saw the a1-quantized hidden_states (×10)
        torch.testing.assert_close(wrapper._inner.forward_calls[0][0], hidden * 10.0)

    def test_legacy_shared_expert_receives_original_input(self):
        from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMFusedMoE

        class _SharedExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.inputs.append(x.clone())
                return x

        class _LegacySharedInner(_FakeMoEInner):
            def __init__(self):
                super().__init__()
                self._shared_experts = _SharedExpert()

            def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
                self._shared_experts(hidden_states)
                return super().forward(hidden_states, router_logits)

        inner = _LegacySharedInner()
        original_shared_forward = inner._shared_experts.forward
        wrapper = QuantVLLMFusedMoE(
            inner=inner,
            layer_quant_config=_make_qlayer_config_empty(),
            device=torch.device("cpu"),
        )

        class _A1:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * 10.0

        wrapper._a1_input_quantizer = _A1()
        hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        wrapper._apply_fake_quant_and_forward("forward", hidden, torch.zeros(2, 2))

        torch.testing.assert_close(inner.forward_calls[0][0], hidden * 10.0)
        torch.testing.assert_close(inner._shared_experts.inputs[0], hidden)
        assert inner._shared_experts.forward == original_shared_forward


class TestQuantVLLMFusedMoEFromFloat:
    def test_non_prequant_module_routed_through_normal_init(self):
        from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMFusedMoE

        inner = _FakeMoEInner()  # has w13_weight/w2_weight but no quant_method
        result = QuantVLLMFusedMoE.from_float(
            float_module=inner,
            layer_quant_config=_make_qlayer_config_empty(),
            device=torch.device("cpu"),
        )
        assert isinstance(result, QuantVLLMFusedMoE)
        assert result._inner is inner

    def test_empty_target_preserves_prequantized_source(self, monkeypatch):
        import quark.experimental.torch.plugin.vllm_plugin as vp

        # Stub out the runtime unquantized MoE method builder which depends on
        # vLLM internals not present in our test stubs.
        runtime_method = SimpleNamespace(name="runtime")
        monkeypatch.setattr(
            vp,
            "_build_runtime_unquantized_moe_method",
            lambda layer, layer_quant_config=None: runtime_method,
        )

        prequant_module = _make_fp8_moe_module(use_scale_inv=True)
        source_method = prequant_module.quant_method

        result = vp.QuantVLLMFusedMoE.from_float(
            float_module=prequant_module,
            layer_quant_config=_make_qlayer_config_empty(),
            device=torch.device("cpu"),
        )

        assert isinstance(result, vp.QuantVLLMFusedMoE)
        assert result._inner is prequant_module
        assert result._w13_weight_quantizer_inv is None
        assert result._w2_weight_quantizer_inv is None
        assert prequant_module.quant_method is source_method

    def test_compressed_channel_fp8_keeps_target_weight_quantizers(self, monkeypatch):
        import quark.experimental.torch.plugin.vllm_plugin as vp
        from quark.experimental.torch.mix_precision.config import get_layer_config

        source = _make_compressed_tensors_fp8_module("moe", "channel")
        runtime_method = SimpleNamespace(name="runtime")
        w13_inv, w2_inv = object(), object()
        monkeypatch.setattr(vp, "create_vllm_moe_inverse_quantizers", lambda layer: (w13_inv, w2_inv))
        monkeypatch.setattr(
            vp,
            "_build_runtime_unquantized_moe_method",
            lambda layer, layer_quant_config: runtime_method,
        )

        wrapper = vp.QuantVLLMFusedMoE.from_prequantized(source, get_layer_config("fp8"), device=torch.device("cpu"))

        assert wrapper._source_matches_target is False
        assert wrapper._source_weight_matches_target is False
        assert wrapper._w13_weight_quantizer_inv is w13_inv
        assert wrapper._w2_weight_quantizer_inv is w2_inv
        assert wrapper._w13_weight_quantizer is not None
        assert wrapper._w2_weight_quantizer is not None
        assert source.quant_method is runtime_method

    @pytest.mark.parametrize("target_mode", ["mxfp4", "mxfp4_fp8"])
    @pytest.mark.parametrize("backend", ["TRITON", "TRITON_UNFUSED"])
    def test_matching_mxfp4_weight_preserves_packed_source_and_method(self, monkeypatch, target_mode, backend):
        import quark.experimental.torch.plugin.vllm_plugin as vp
        from quark.experimental.torch.mix_precision.config import get_layer_config

        source = _make_mxfp4_moe_module()
        source.quant_method.mxfp4_backend.value = backend
        source_method = source.quant_method
        source_w13 = source.w13_weight
        source_w2 = source.w2_weight

        def unexpected_inverse(*args, **kwargs):
            raise AssertionError("matching MXFP4 weights must not enter the inverse codec")

        monkeypatch.setattr(vp, "create_vllm_moe_inverse_quantizers", unexpected_inverse)
        wrapper = vp.QuantVLLMFusedMoE.from_prequantized(
            source,
            get_layer_config(target_mode),
            device=torch.device("cpu"),
        )

        assert wrapper._source_matches_target is False
        assert wrapper._source_weight_matches_target is True
        assert wrapper._w13_weight_quantizer_inv is None
        assert wrapper._w2_weight_quantizer_inv is None
        assert wrapper._w13_weight_quantizer is None
        assert wrapper._w2_weight_quantizer is None
        assert wrapper._a1_input_quantizer is not None
        assert wrapper._a2_input_quantizer is not None
        assert source.quant_method is source_method
        assert source.w13_weight is source_w13
        assert source.w2_weight is source_w2

    def test_matching_mxfp4_weight_allows_aiter_bf16_activation_override(self):
        import quark.experimental.torch.plugin.vllm_plugin as vp
        from quark.experimental.torch.mix_precision.config import get_layer_config

        source = _make_mxfp4_moe_module()
        source.quant_method.mxfp4_backend.value = "AITER_MXFP4_BF16"
        source_method = source.quant_method

        wrapper = vp.QuantVLLMFusedMoE.from_prequantized(
            source,
            get_layer_config("mxfp4"),
            device=torch.device("cpu"),
        )

        assert source.quant_method is source_method
        assert wrapper._source_moe_backend == "AITER_MXFP4_BF16"
        assert wrapper._source_weight_matches_target is True
        assert wrapper._w13_weight_quantizer_inv is None
        assert wrapper._w2_weight_quantizer_inv is None
        assert wrapper._a1_input_quantizer is not None
        assert wrapper._a2_input_quantizer is not None

    def test_matching_mxfp4_weight_rejects_backend_without_a2_hook(self):
        import quark.experimental.torch.plugin.vllm_plugin as vp
        from quark.experimental.torch.mix_precision.config import get_layer_config

        source = _make_mxfp4_moe_module()
        source.quant_method.mxfp4_backend.value = "CUDA_ONLY"

        with pytest.raises(NotImplementedError, match="supported .* a2 hook"):
            vp.QuantVLLMFusedMoE.from_prequantized(
                source,
                get_layer_config("mxfp4"),
                device=torch.device("cpu"),
            )

    def test_matching_mxfp4_weight_allows_triton_emulation_activation_override(self):
        import quark.experimental.torch.plugin.vllm_plugin as vp
        from quark.experimental.torch.mix_precision.config import get_layer_config

        source = _make_mxfp4_moe_module()
        source.quant_method.mxfp4_backend.value = "EMULATION"
        source_method = source.quant_method

        wrapper = vp.QuantVLLMFusedMoE.from_prequantized(
            source,
            get_layer_config("mxfp4"),
            device=torch.device("cpu"),
        )

        assert source.quant_method is source_method
        assert wrapper._source_weight_matches_target is True
        assert wrapper._w13_weight_quantizer_inv is None
        assert wrapper._w2_weight_quantizer_inv is None
        assert wrapper._a1_input_quantizer is not None
        assert wrapper._a2_input_quantizer is not None

    def test_activation_only_target_preserves_mxfp4_source(self, monkeypatch):
        import quark.experimental.torch.plugin.vllm_plugin as vp
        from quark.experimental.torch.mix_precision.config import get_layer_config

        source = _make_mxfp4_moe_module()
        source_method = source.quant_method
        target = deepcopy(get_layer_config("fp8"))
        target.weight = None

        def unexpected_inverse(*args, **kwargs):
            raise AssertionError("activation-only target must not enter the inverse codec")

        monkeypatch.setattr(vp, "create_vllm_moe_inverse_quantizers", unexpected_inverse)
        wrapper = vp.QuantVLLMFusedMoE.from_prequantized(source, target, device=torch.device("cpu"))

        assert wrapper._source_matches_target is False
        assert wrapper._source_weight_matches_target is True
        assert wrapper._w13_weight_quantizer_inv is None
        assert wrapper._w2_weight_quantizer_inv is None
        assert wrapper._a1_input_quantizer is not None
        assert wrapper._a2_input_quantizer is not None
        assert source.quant_method is source_method

    def test_exact_match_does_not_construct_moe_quantizers(self):
        import quark.experimental.torch.plugin.vllm_plugin as vp
        from quark.experimental.torch.mix_precision.config import get_layer_config

        source = _make_mxfp4_moe_module()
        source.quant_method.input_dtype = "mxfp4"
        source_method = source.quant_method

        wrapper = vp.QuantVLLMFusedMoE.from_prequantized(
            source,
            get_layer_config("mxfp4"),
            device=torch.device("cpu"),
        )

        assert wrapper._source_matches_target is True
        assert wrapper.is_prequantized is True
        assert wrapper._get_moe_quantizers() == (None, None, None, None)
        assert source.quant_method is source_method
        before_w13 = source.w13_weight.clone()
        wrapper.freeze(quantize=True)
        torch.testing.assert_close(source.w13_weight, before_w13)

        from quark.torch.quantization.api import ModelQuantizer

        ModelQuantizer.freeze(nn.Sequential(wrapper))
        torch.testing.assert_close(source.w13_weight, before_w13)


class TestQuantVLLMFusedMoEFreezeAndProperties:
    def test_freeze_with_no_quantizers_is_noop(self):
        wrapper = _make_quant_moe_wrapper()
        # No quantizers configured → freeze is a no-op (no exception)
        wrapper.freeze(quantize=True)
        wrapper.freeze_moe_quantizers()  # alias

    def test_freeze_calls_to_frozen_module_on_input_quantizers(self):
        wrapper = _make_quant_moe_wrapper()

        frozen_a1 = SimpleNamespace(name="frozen_a1")
        frozen_a2 = SimpleNamespace(name="frozen_a2")

        class _Q:
            def __init__(self, frozen):
                self._frozen = frozen
                self.is_dynamic = True
                self.calls: list[bool] = []

            def to_frozen_module(self, frozen_params: bool) -> Any:
                self.calls.append(frozen_params)
                return self._frozen

        a1 = _Q(frozen_a1)
        a2 = _Q(frozen_a2)
        wrapper._a1_input_quantizer = a1
        wrapper._a2_input_quantizer = a2

        wrapper.freeze(quantize=True)

        # is_dynamic=True → frozen_params arg is False
        assert a1.calls == [False]
        assert a2.calls == [False]
        assert wrapper._a1_input_quantizer is frozen_a1
        assert wrapper._a2_input_quantizer is frozen_a2

    def test_to_float_module_restores_source_quant_method(self):
        wrapper = _make_quant_moe_wrapper()
        original_quant_method = SimpleNamespace(name="orig_method")
        wrapper._source_quant_method = original_quant_method

        result = wrapper.to_float_module()

        assert result is wrapper._inner
        assert wrapper._inner.quant_method is original_quant_method

    def test_to_float_module_without_source_returns_inner_unchanged(self):
        wrapper = _make_quant_moe_wrapper()
        # Save inner reference before to_float_module (which doesn't modify it here)
        inner_before = wrapper._inner
        result = wrapper.to_float_module()
        assert result is inner_before

    def test_weight_property_routes_via_freeze_target(self):
        wrapper = _make_quant_moe_wrapper()

        wrapper._freeze_weight_target = "w13"
        assert wrapper.weight is wrapper._inner.w13_weight

        wrapper._freeze_weight_target = "w2"
        assert wrapper.weight is wrapper._inner.w2_weight

        wrapper._freeze_weight_target = None
        # When freeze target is unset, the property raises AttributeError; nn.Module
        # then falls back to __getattr__, which delegates to self._inner. Since
        # _FakeMoEInner has no `weight`, the final error mentions the inner class.
        with pytest.raises(AttributeError):
            _ = wrapper.weight

    def test_get_quant_weight_dispatches_by_identity(self):
        wrapper = _make_quant_moe_wrapper()
        # No quantizers and no inverses → returns input unchanged
        out_w13 = wrapper.get_quant_weight(wrapper._inner.w13_weight)
        out_w2 = wrapper.get_quant_weight(wrapper._inner.w2_weight)
        assert torch.equal(out_w13, wrapper._inner.w13_weight)
        assert torch.equal(out_w2, wrapper._inner.w2_weight)

        # Unknown tensor passed through as-is
        unknown = torch.zeros(2, 2)
        assert wrapper.get_quant_weight(unknown) is unknown


# ---------------------------------------------------------------------------
# reset_vllm_fake_quant_model
# ---------------------------------------------------------------------------


class TestResetVllmFakeQuantModel:
    def test_replaces_quant_wrappers_with_float_modules(self):
        from quark.experimental.torch.plugin.vllm_plugin import (
            QuantVLLMParallelLinearBase,
            reset_vllm_fake_quant_model,
        )

        # Build a model: top-level container holding a QuantVLLMParallelLinearBase
        # whose to_float_module() metadata produces a plain nn.Linear.
        class Container(nn.Module):
            def __init__(self, child: nn.Module) -> None:
                super().__init__()
                self.proj = child

        wrapper = QuantVLLMParallelLinearBase(quant_config=None, device=torch.device("cpu"))
        wrapper._float_module_cls = nn.Linear
        wrapper._float_init_kwargs = {"in_features": 4, "out_features": 4}
        wrapper.weight = nn.Parameter(torch.full((4, 4), 0.25))
        wrapper.bias = nn.Parameter(torch.zeros(4))

        model = Container(wrapper)
        assert isinstance(model.proj, QuantVLLMParallelLinearBase)

        result = reset_vllm_fake_quant_model(model)

        assert result is model
        # After reset, the wrapper at "proj" is replaced by a plain nn.Linear
        assert isinstance(model.proj, nn.Linear)
        assert not isinstance(model.proj, QuantVLLMParallelLinearBase)
        torch.testing.assert_close(model.proj.weight, torch.full((4, 4), 0.25))

    def test_preserves_alias_identity_when_resetting_linear_wrappers(self):
        from quark.experimental.torch.plugin.vllm_plugin import (
            QuantVLLMParallelLinearBase,
            reset_vllm_fake_quant_model,
        )

        class Container(nn.Module):
            def __init__(self, child: nn.Module) -> None:
                super().__init__()
                self.first = child
                self.second = child

        wrapper = QuantVLLMParallelLinearBase(quant_config=None, device=torch.device("cpu"))
        wrapper._float_module_cls = nn.Linear
        wrapper._float_init_kwargs = {"in_features": 4, "out_features": 4}
        wrapper.weight = nn.Parameter(torch.full((4, 4), 0.25))
        wrapper.bias = nn.Parameter(torch.zeros(4))
        model = Container(wrapper)

        reset_vllm_fake_quant_model(model)

        assert model.first is model.second

    def test_kimi_gdn_restore_metadata_keeps_special_parallel_layout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import quark.experimental.torch.plugin.vllm_plugin as vp

        class FakeKimiGDN(nn.Module):
            pass

        source = FakeKimiGDN()
        source.replicated_shard_id = 1
        source.tp_size = 8
        source.output_sizes = [4096, 512, 4096]
        init_kwargs = {
            "input_size": 7168,
            "output_sizes": source.output_sizes,
            "quark_quant_config": object(),
            "quant_config": None,
        }
        monkeypatch.setattr(vp, "vllm_kimi_gdn_merged_column_parallel_linear", FakeKimiGDN)

        restore_class, restore_kwargs = vp._merged_float_restore_metadata(source, init_kwargs)

        assert restore_class is FakeKimiGDN
        assert restore_kwargs["output_sizes"] == [4096, 64, 4096]
        assert restore_kwargs["replicated_shard_id"] == 1
        assert restore_kwargs["tp_size"] == 8
        assert "quark_quant_config" not in restore_kwargs

    def test_replaces_moe_wrapper_with_inner(self):
        from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMFusedMoE, reset_vllm_fake_quant_model

        class Container(nn.Module):
            def __init__(self, child: nn.Module) -> None:
                super().__init__()
                self.experts = child

        wrapper = _make_quant_moe_wrapper()
        inner_ref = wrapper._inner
        model = Container(wrapper)
        assert isinstance(model.experts, QuantVLLMFusedMoE)

        reset_vllm_fake_quant_model(model)

        # MoE wrapper.to_float_module() returns its inner FusedMoE
        assert model.experts is inner_ref

    def test_no_wrappers_is_noop(self):
        from quark.experimental.torch.plugin.vllm_plugin import reset_vllm_fake_quant_model

        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        # Should be a no-op and return the same model
        assert reset_vllm_fake_quant_model(model) is model
        # Children unchanged
        assert isinstance(model[0], nn.Linear)


# ---------------------------------------------------------------------------
# calibrate_moe_weight_params
# ---------------------------------------------------------------------------


class TestCalibrateMoeWeightParams:
    def test_runs_get_quant_weight_for_uncalibrated_quantizer(self, monkeypatch):
        import quark.experimental.torch.plugin.vllm_plugin as vp

        # Stub ScaledFakeQuantize for the isinstance check
        class _FakeScaledFakeQuantize:
            pass

        monkeypatch.setattr(vp, "ScaledFakeQuantize", _FakeScaledFakeQuantize)

        wrapper = _make_quant_moe_wrapper()

        # Inject quantizers that look uncalibrated (scale.numel()==1, scale.item()==1).
        # Must be callable: get_quant_weight → _apply_moe_weight_quantizer → quantizer(weight)
        class _UncalibratedQ(_FakeScaledFakeQuantize):
            def __init__(self) -> None:
                self.scale = torch.tensor([1.0])
                self.observer_disabled = False

            def __call__(self, weight: torch.Tensor) -> torch.Tensor:
                return weight

            def disable_observer(self) -> None:
                self.observer_disabled = True

        w13_q = _UncalibratedQ()
        w2_q = _UncalibratedQ()
        # Place quantizers in __dict__ so _get_moe_quantizers returns them
        wrapper.__dict__["_w13_weight_quantizer"] = w13_q
        wrapper.__dict__["_w2_weight_quantizer"] = w2_q

        # Wrap inside a model
        class Container(nn.Module):
            def __init__(self, child: nn.Module) -> None:
                super().__init__()
                self.experts = child

        model = Container(wrapper)

        # get_quant_weight call count tracker
        calls: list[torch.Tensor] = []
        original_gqw = wrapper.get_quant_weight

        def counted_gqw(x: torch.Tensor) -> torch.Tensor:
            calls.append(x)
            return original_gqw(x)

        wrapper.get_quant_weight = counted_gqw  # type: ignore[method-assign]

        vp.calibrate_moe_weight_params(model)

        # Both w13 and w2 should have been driven through get_quant_weight
        assert len(calls) == 2
        assert w13_q.observer_disabled is True
        assert w2_q.observer_disabled is True

    def test_skips_already_calibrated_quantizer(self, monkeypatch):
        import quark.experimental.torch.plugin.vllm_plugin as vp

        class _FakeScaledFakeQuantize:
            pass

        monkeypatch.setattr(vp, "ScaledFakeQuantize", _FakeScaledFakeQuantize)

        wrapper = _make_quant_moe_wrapper()

        class _CalibratedQ(_FakeScaledFakeQuantize):
            def __init__(self) -> None:
                self.scale = torch.tensor([0.5])  # not 1.0 → considered calibrated
                self.observer_disabled = False

            def disable_observer(self) -> None:
                self.observer_disabled = True

        w13_q = _CalibratedQ()
        wrapper.__dict__["_w13_weight_quantizer"] = w13_q
        # No w2 quantizer

        class Container(nn.Module):
            def __init__(self, child: nn.Module) -> None:
                super().__init__()
                self.experts = child

        model = Container(wrapper)

        calls: list[torch.Tensor] = []
        original_gqw = wrapper.get_quant_weight
        wrapper.get_quant_weight = lambda x: (calls.append(x), original_gqw(x))[1]  # type: ignore[method-assign]

        vp.calibrate_moe_weight_params(model)

        # Already-calibrated quantizer is NOT re-run, but observer is still disabled
        assert calls == []
        assert w13_q.observer_disabled is True

    def test_skips_modules_that_are_not_moe_wrappers(self, monkeypatch):
        import quark.experimental.torch.plugin.vllm_plugin as vp

        # Should be a no-op on a model containing no MoE wrappers
        model = nn.Sequential(nn.Linear(4, 4))
        vp.calibrate_moe_weight_params(model)  # no exception


# =============================================================================
# dequantize() — VLLMFp8LinearInverseQuantizer
# =============================================================================


class TestVLLMFp8LinearInverseQuantizerDequantize:
    def test_per_channel_scale_calls_dequantize_op(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        module = _make_fp8_linear_module("RowParallelLinear", use_scale_inv=True)
        module.weight_scale_inv = torch.ones(4, 1)  # per-channel: [out_features, 1]
        inv_q = VLLMFp8LinearInverseQuantizer(module)

        dequantize_calls: list = []

        def _fake_dequantize(dtype, weight, scale, zero_point, axis, group_size, qscheme):
            dequantize_calls.append({"scale_shape": tuple(scale.shape), "axis": axis, "qscheme": qscheme})
            return weight.to(torch.bfloat16)

        monkeypatch.setattr(torch.ops.quark, "dequantize", _fake_dequantize, raising=False)

        weight = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
        out = inv_q.dequantize(weight)

        assert len(dequantize_calls) == 1
        assert dequantize_calls[0]["axis"] == 0  # per-channel axis
        # transpose_output=True for non-block-quant: output is transposed
        assert out.shape == (4, 4)

    def test_per_tensor_scale_uses_per_tensor_qscheme(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        module = _make_fp8_linear_module("RowParallelLinear", use_scale_inv=True)
        # A 0-D or single-element 1-D scale: ndim==1, numel()==1 → per_tensor path (axis=-1)
        module.weight_scale_inv = torch.tensor([1.0])  # shape [1], numel==1
        inv_q = VLLMFp8LinearInverseQuantizer(module)

        calls: list = []

        def _fake_dequantize(dtype, weight, scale, zero_point, axis, group_size, qscheme):
            calls.append({"axis": axis, "qscheme": qscheme})
            return weight.to(torch.bfloat16)

        monkeypatch.setattr(torch.ops.quark, "dequantize", _fake_dequantize, raising=False)

        inv_q.dequantize(torch.zeros(4, 4, dtype=torch.float8_e4m3fn))

        # numel()==1 means the condition `numel() > 1` is False → per_tensor (axis=-1)
        assert calls[0]["axis"] == -1
        assert "per_tensor" in calls[0]["qscheme"]

    def test_block_quant_calls_dequantize_fp8_per_block(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        class Fp8BlockMethod:
            use_deep_gemm = False
            use_marlin = False
            block_quant = True

        module = _make_fp8_linear_module("RowParallelLinear", use_scale_inv=True)
        module.weight_block_size = (2, 2)
        module.quant_method = Fp8BlockMethod()
        inv_q = VLLMFp8LinearInverseQuantizer(module)

        block_calls: list = []

        def _fake_dequantize_per_block(weight, scale, block_size):
            block_calls.append(block_size)
            return weight.to(torch.bfloat16)

        monkeypatch.setattr(torch.ops.quark, "dequantize_fp8_per_block", _fake_dequantize_per_block, raising=False)

        inv_q.dequantize(torch.zeros(4, 4, dtype=torch.float8_e4m3fn))

        assert block_calls == [[2, 2]]

    def test_block_quant_missing_block_size_raises(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        class Fp8BlockMethod:
            use_deep_gemm = False
            use_marlin = False
            block_quant = True

        module = _make_fp8_linear_module("RowParallelLinear", use_scale_inv=True)
        module.quant_method = Fp8BlockMethod()
        inv_q = VLLMFp8LinearInverseQuantizer(module)
        inv_q.block_size = None  # force missing

        with pytest.raises(ValueError, match="weight_block_size"):
            inv_q.dequantize(torch.zeros(4, 4, dtype=torch.float8_e4m3fn))


# =============================================================================
# dequantize() — VLLMFp8MoEWeightInverseQuantizer
# =============================================================================


class TestVLLMFp8MoEWeightInverseQuantizerDequantize:
    def test_2d_input_raises(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        inv_q = VLLMFp8MoEWeightInverseQuantizer(_make_fp8_moe_module(), "w13_weight_scale_inv")
        with pytest.raises(ValueError, match="3D"):
            inv_q.dequantize(torch.zeros(4, 4, dtype=torch.float8_e4m3fn))

    def test_1d_scale_per_expert_dequantize(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        num_experts, c, h = 2, 4, 4
        module = _make_fp8_moe_module(use_scale_inv=True)
        module.w13_weight_scale_inv = torch.ones(num_experts)  # 1D: one scalar per expert
        inv_q = VLLMFp8MoEWeightInverseQuantizer(module, "w13_weight_scale_inv")

        call_args: list = []

        def _fake_dequantize(dtype, weight, scale, zero_point, axis, group_size, qscheme):
            call_args.append({"scale": scale, "axis": axis})
            return weight.to(torch.bfloat16)

        monkeypatch.setattr(torch.ops.quark, "dequantize", _fake_dequantize, raising=False)

        weight = torch.zeros(num_experts, c, h, dtype=torch.float8_e4m3fn)
        out = inv_q.dequantize(weight)

        assert len(call_args) == num_experts  # one call per expert
        assert out.shape == (num_experts, c, h)

    def test_2d_scale_squeezed_per_expert(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        num_experts, c, h = 3, 4, 4
        module = _make_fp8_moe_module(use_scale_inv=True)
        module.w13_weight_scale_inv = torch.ones(num_experts, 1)  # [E, 1] → squeeze to [E]
        inv_q = VLLMFp8MoEWeightInverseQuantizer(module, "w13_weight_scale_inv")

        def _fake_dequantize(dtype, weight, scale, zero_point, axis, group_size, qscheme):
            return weight.to(torch.bfloat16)

        monkeypatch.setattr(torch.ops.quark, "dequantize", _fake_dequantize, raising=False)

        weight = torch.zeros(num_experts, c, h, dtype=torch.float8_e4m3fn)
        out = inv_q.dequantize(weight)
        assert out.shape == (num_experts, c, h)

    def test_unsupported_scale_shape_raises(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        num_experts, c, h = 2, 4, 4
        module = _make_fp8_moe_module(use_scale_inv=True)
        module.w13_weight_scale_inv = torch.ones(num_experts, 2)  # [E, 2]: unsupported
        inv_q = VLLMFp8MoEWeightInverseQuantizer(module, "w13_weight_scale_inv")

        with pytest.raises(NotImplementedError, match="scale shape"):
            inv_q.dequantize(torch.zeros(num_experts, c, h, dtype=torch.float8_e4m3fn))

    def test_shuffled_aiter_weight_fails_closed(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        module = _make_fp8_moe_module(use_scale_inv=True)
        inv_q = VLLMFp8MoEWeightInverseQuantizer(module, "w13_weight_scale_inv")
        weight = module.w13_weight
        weight.is_shuffled = True
        monkeypatch.setattr(
            torch.ops.quark,
            "dequantize",
            lambda *args, **kwargs: weight.to(torch.bfloat16),
            raising=False,
        )

        with pytest.raises(NotImplementedError, match="shuffled"):
            inv_q.dequantize(weight)


# =============================================================================
# dequantize() — VLLMMxfp4MoEWeightInverseQuantizer
# =============================================================================


class TestVLLMMxfp4MoEWeightInverseQuantizerDequantize:
    def test_dequantize_calls_mx_dq_mxfp4(self, monkeypatch):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        num_experts, scale_groups = 2, 1
        module = _make_mxfp4_moe_module()
        # scale shape [E, G]: packed weight expected shape [E, G*16]
        module.w13_weight_scale = torch.ones(num_experts, scale_groups)
        module.w13_weight = torch.zeros(num_experts, scale_groups * 16, dtype=torch.uint8)
        inv_q = VLLMMxfp4MoEWeightInverseQuantizer(module, "w13_weight_scale")

        mx_calls: list = []

        import types as _types

        mx_stub = _types.ModuleType("quark.torch.kernel.mx")

        def _fake_dq_mxfp4(packed_weight, scale, float_dtype):
            mx_calls.append({"weight_shape": tuple(packed_weight.shape), "scale_shape": tuple(scale.shape)})
            return torch.zeros(*packed_weight.shape[:1], packed_weight.shape[-1] * 2, dtype=float_dtype)

        mx_stub.dq_mxfp4 = _fake_dq_mxfp4
        monkeypatch.setitem(sys.modules, "quark.torch.kernel.mx", mx_stub)
        monkeypatch.setitem(sys.modules, "quark.torch.kernel", _types.ModuleType("quark.torch.kernel"))

        weight = torch.zeros(num_experts, scale_groups * 16, dtype=torch.uint8)
        inv_q.dequantize(weight)

        assert len(mx_calls) == 1
        assert mx_calls[0]["scale_shape"] == (num_experts, scale_groups)

    def test_restore_packed_layout_contiguous_match(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        module = _make_mxfp4_moe_module()
        module.w13_weight_scale = torch.ones(2, 1)
        inv_q = VLLMMxfp4MoEWeightInverseQuantizer(module, "w13_weight_scale")

        # Expected shape from scale [2, 1]: (2, 1*16) = (2, 16)
        packed = torch.zeros(2, 16, dtype=torch.uint8)
        result = inv_q._restore_packed_layout(packed)
        assert result.shape == (2, 16)
        assert result.is_contiguous()

    def test_restore_packed_layout_transposed_match(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        module = _make_mxfp4_moe_module()
        module.w13_weight_scale = torch.ones(2, 1)
        inv_q = VLLMMxfp4MoEWeightInverseQuantizer(module, "w13_weight_scale")

        # Transpose of (2, 16) is (16, 2), which after transpose(-2,-1) gives (2, 16)
        packed = torch.zeros(16, 2, dtype=torch.uint8)
        result = inv_q._restore_packed_layout(packed)
        assert result.shape == (2, 16)

    def test_restore_packed_layout_bad_shape_raises(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        module = _make_mxfp4_moe_module()
        module.w13_weight_scale = torch.ones(2, 1)
        inv_q = VLLMMxfp4MoEWeightInverseQuantizer(module, "w13_weight_scale")

        with pytest.raises(ValueError, match="Cannot restore"):
            inv_q._restore_packed_layout(torch.zeros(3, 7, dtype=torch.uint8))

    def test_unwrap_triton_tensor_passthrough_for_plain_tensor(self):
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        t = torch.zeros(4, dtype=torch.uint8)
        assert VLLMMxfp4MoEWeightInverseQuantizer._unwrap_triton_tensor(t) is t

    def test_unwrap_triton_tensor_extracts_storage_data(self):
        from types import SimpleNamespace

        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        data = torch.zeros(4, dtype=torch.uint8)
        fake_tensor = SimpleNamespace(storage=SimpleNamespace(data=data))
        result = VLLMMxfp4MoEWeightInverseQuantizer._unwrap_triton_tensor(fake_tensor)
        assert result is data

    def test_unwrap_triton_tensor_bad_storage_raises(self):
        from types import SimpleNamespace

        from quark.experimental.torch.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        bad = SimpleNamespace(storage=SimpleNamespace(data="not_a_tensor"))
        with pytest.raises(ValueError, match="Expected torch.Tensor"):
            VLLMMxfp4MoEWeightInverseQuantizer._unwrap_triton_tensor(bad)


# =============================================================================
# evaluate_ppl_offline
# =============================================================================
from quark.experimental.torch.mix_precision import eval as _ppl_eval_module  # noqa: E402


def _make_fake_llm(input_ids, target_logprob=-2.0, with_logprobs=True):
    """Fake vLLM whose ``generate`` returns deterministic prompt_logprobs."""

    def tokenizer(text, return_tensors=None):
        return SimpleNamespace(input_ids=torch.tensor([input_ids], dtype=torch.long))

    def generate(prompts_arg, sampling_params=None, **kwargs):
        """Generate deterministic prompt_logprobs for testing.
        Args:
            prompts_arg: List of prompt dictionaries containing prompt_token_ids
            sampling_params: Sampling parameters with prompt_logprobs setting
            **kwargs: Additional keyword arguments (unused)
        Returns:
            List of SimpleNamespace objects with prompt_logprobs attribute
        """
        assert sampling_params.kw["prompt_logprobs"] == 1
        assert sampling_params.kw["prompt_logprobs"] == 1
        outs = []
        for p in prompts_arg:
            token_ids = p["prompt_token_ids"]
            plp = (
                None
                if not with_logprobs
                else ([None] + [{tid: SimpleNamespace(logprob=target_logprob)} for tid in token_ids[1:]])
            )
            outs.append(SimpleNamespace(prompt_logprobs=plp))
        return outs

    return SimpleNamespace(get_tokenizer=lambda: tokenizer, generate=generate)


@pytest.fixture
def patched_ppl_module(monkeypatch):
    """Patch the symbols bound at the top of eval.py with lightweight fakes."""
    monkeypatch.setattr(_ppl_eval_module, "load_dataset", lambda *a, **kw: {"text": ["hi x " * 50]}, raising=False)
    monkeypatch.setattr(_ppl_eval_module, "SamplingParams", lambda **kw: SimpleNamespace(kw=kw), raising=False)
    monkeypatch.setattr(
        _ppl_eval_module,
        "TokensPrompt",
        lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids},
        raising=False,
    )
    return _ppl_eval_module


class TestEvaluatePplOffline:
    """Verify the wikitext-2 PPL evaluator: tokenize, chunk, sum NLL, exp(mean)."""

    def test_ppl_matches_exp_mean_nll(self, patched_ppl_module):
        # 3 chunks * (seq_len - 1) = 9 scored positions, each at logprob = -2.
        result = patched_ppl_module.evaluate_ppl_offline(
            _make_fake_llm(list(range(1, 25)), target_logprob=-2.0),
            seq_len=4,
            max_chunks=3,
        )
        assert result["num_chunks"] == 3
        assert result["tokens_scored"] == 9
        assert math.isclose(result["nll_total"], 18.0)
        assert math.isclose(result["ppl"], math.exp(2.0))

    def test_ppl_raises_when_text_too_short(self, patched_ppl_module):
        with pytest.raises(RuntimeError, match="less than seq_len"):
            patched_ppl_module.evaluate_ppl_offline(_make_fake_llm([1, 2, 3]), seq_len=8)

    def test_ppl_raises_when_vllm_returns_no_prompt_logprobs(self, patched_ppl_module):
        with pytest.raises(RuntimeError, match="did not return prompt_logprobs"):
            patched_ppl_module.evaluate_ppl_offline(
                _make_fake_llm(list(range(1, 17)), with_logprobs=False),
                seq_len=4,
                max_chunks=2,
            )
