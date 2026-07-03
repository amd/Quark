#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for the mix_precision module (config, searcher, eval utilities, switcher,
vllm_inverse_quantizer, vllm_plugin)."""

from __future__ import annotations

import math
import sys
import types
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
        from quark.experimental.torch.llm.mix_precision.config import normalize_quant_mode

        for alias in ("native", "original", "bf16"):
            assert normalize_quant_mode(alias) == "native"

    def test_passthrough_modes(self):
        from quark.experimental.torch.llm.mix_precision.config import normalize_quant_mode

        for mode in ("fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8", "mxfp6_e2m3"):
            assert normalize_quant_mode(mode) == mode

    def test_none_returns_empty_string(self):
        from quark.experimental.torch.llm.mix_precision.config import normalize_quant_mode

        assert normalize_quant_mode(None) == ""


class TestIsNativeMode:
    def test_native_variants_are_native(self):
        from quark.experimental.torch.llm.mix_precision.config import is_native_mode

        for mode in ("native", "original", "bf16"):
            assert is_native_mode(mode), f"{mode!r} should be native"

    def test_none_is_not_native(self):
        from quark.experimental.torch.llm.mix_precision.config import is_native_mode

        # None normalizes to "" (empty string), not "native"
        assert not is_native_mode(None)

    def test_quant_modes_not_native(self):
        from quark.experimental.torch.llm.mix_precision.config import is_native_mode

        for mode in ("fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8", "mxfp6_e2m3"):
            assert not is_native_mode(mode), f"{mode!r} should not be native"


class TestGetSupportedSchemes:
    def test_mi300_excludes_mxfp4(self):
        from quark.experimental.torch.llm.mix_precision.config import HardwareTarget, get_supported_schemes

        schemes = get_supported_schemes(HardwareTarget.MI300)
        assert "fp8" in schemes
        assert "ptpc_fp8" in schemes
        assert "mxfp4" not in schemes
        assert "mxfp4_fp8" not in schemes

    def test_mi355_includes_mxfp4(self):
        from quark.experimental.torch.llm.mix_precision.config import HardwareTarget, get_supported_schemes

        schemes = get_supported_schemes(HardwareTarget.MI355)
        assert "mxfp4" in schemes
        assert "mxfp4_fp8" in schemes
        assert "mxfp6_e2m3" in schemes

    def test_string_hardware_accepted(self):
        from quark.experimental.torch.llm.mix_precision.config import get_supported_schemes

        schemes = get_supported_schemes("mi300")
        assert "fp8" in schemes

    def test_none_returns_all_modes(self):
        from quark.experimental.torch.llm.mix_precision.config import ALL_QUANT_MODES, get_supported_schemes

        schemes = get_supported_schemes(None)
        assert set(schemes) == set(ALL_QUANT_MODES)


class TestGetLayerConfig:
    def test_native_returns_none(self):
        from quark.experimental.torch.llm.mix_precision.config import get_layer_config

        assert get_layer_config("native") is None

    def test_known_modes_return_config(self):
        from quark.experimental.torch.llm.mix_precision.config import get_layer_config

        for mode in ("fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8", "mxfp6_e2m3"):
            result = get_layer_config(mode)
            assert result is not None, f"get_layer_config({mode!r}) should return a QLayerConfig"

    def test_unknown_mode_returns_none(self):
        from quark.experimental.torch.llm.mix_precision.config import get_layer_config

        assert get_layer_config("int8") is None


class TestMixPrecisionConfigValidate:
    def test_valid_config_passes(self):
        from quark.experimental.torch.llm.mix_precision.config import MixPrecisionConfig

        config = MixPrecisionConfig(eval_metrics=["gsm8k"], eval_threshold=1.02)
        config.validate()  # should not raise

    def test_invalid_metric_raises(self):
        from quark.experimental.torch.llm.mix_precision.config import MixPrecisionConfig

        config = MixPrecisionConfig(eval_metrics=["rouge"])
        with pytest.raises(ValueError, match="Invalid metric"):
            config.validate()

    def test_threshold_below_one_raises(self):
        from quark.experimental.torch.llm.mix_precision.config import MixPrecisionConfig

        config = MixPrecisionConfig(eval_threshold=0.95)
        with pytest.raises(ValueError, match="eval_threshold"):
            config.validate()

    def test_negative_min_kv_scale_raises(self):
        from quark.experimental.torch.llm.mix_precision.config import MixPrecisionConfig

        config = MixPrecisionConfig(min_kv_scale=-0.1)
        with pytest.raises(ValueError, match="min_kv_scale"):
            config.validate()

    def test_decoder_layer_granularity_not_implemented(self):
        from quark.experimental.torch.llm.mix_precision.config import MixPrecisionConfig, SearchGranularity

        config = MixPrecisionConfig(granularity=SearchGranularity.DECODER_LAYER)
        with pytest.raises(NotImplementedError):
            config.validate()


class TestMixPrecisionConfigGetSearchConfig:
    def test_returns_module_search_config_by_default(self):
        from quark.experimental.torch.llm.mix_precision.config import MixPrecisionConfig, ModuleSearchConfig

        config = MixPrecisionConfig()
        search_config = config.get_search_config()
        assert isinstance(search_config, ModuleSearchConfig)

    def test_returns_provided_module_search_config(self):
        from quark.experimental.torch.llm.mix_precision.config import MixPrecisionConfig, ModuleSearchConfig

        custom = ModuleSearchConfig(layer_sensitivity={"self_attn": 5, "mlp": 2})
        config = MixPrecisionConfig(module_search_config=custom)
        assert config.get_search_config() is custom


# =============================================================================
# searcher.py
# =============================================================================


class TestConfigSearcher:
    def test_generates_configs_for_mi300(self):
        from quark.experimental.torch.llm.mix_precision.config import HardwareTarget
        from quark.experimental.torch.llm.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(hardware=HardwareTarget.MI300)
        configs = searcher.generate_sorted_configs()
        assert len(configs) > 0

    def test_no_all_native_config(self):
        from quark.experimental.torch.llm.mix_precision.config import HardwareTarget
        from quark.experimental.torch.llm.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(hardware=HardwareTarget.MI300)
        configs = searcher.generate_sorted_configs()
        for config in configs:
            layer_modes = [config.get(f"{p}_mode") for p in searcher.layer_sensitivity]
            assert any(m != "native" for m in layer_modes), f"All-native config must not appear: {config}"

    def test_sorted_conservative_to_aggressive(self):
        from quark.experimental.torch.llm.mix_precision.config import HardwareTarget
        from quark.experimental.torch.llm.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(hardware=HardwareTarget.MI300)
        configs = searcher.generate_sorted_configs()
        scores = [searcher.compute_score(c) for c in configs]
        assert scores == sorted(scores, reverse=True), "Configs must be sorted high-score first"

    def test_precision_hierarchy_respected(self):
        from quark.experimental.torch.llm.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.llm.mix_precision.searcher import ConfigSearcher

        # self_attn (sensitivity=3) must have precision >= mlp (sensitivity=1)
        search_config = ModuleSearchConfig(layer_sensitivity={"self_attn": 3, "mlp": 1})
        searcher = ConfigSearcher(search_config=search_config, hardware=HardwareTarget.MI300)
        pw = searcher.precision_weights
        for config in searcher.generate_sorted_configs():
            attn_score = pw.get(config.get("self_attn_mode", "native"), 0)
            mlp_score = pw.get(config.get("mlp_mode", "native"), 0)
            assert attn_score >= mlp_score, f"self_attn precision must be >= mlp: {config}"

    def test_mi355_includes_mxfp4_configs(self):
        from quark.experimental.torch.llm.mix_precision.config import HardwareTarget
        from quark.experimental.torch.llm.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(hardware=HardwareTarget.MI355)
        configs = searcher.generate_sorted_configs()
        all_modes = {config.get("self_attn_mode") for config in configs} | {
            config.get("mlp_mode") for config in configs
        }
        assert "mxfp4" in all_modes, "MI355 should produce mxfp4 configs"

    def test_available_partitions_filter(self):
        from quark.experimental.torch.llm.mix_precision.config import HardwareTarget
        from quark.experimental.torch.llm.mix_precision.searcher import ConfigSearcher

        # Only self_attn and mlp — linear_attn should be excluded
        searcher = ConfigSearcher(
            hardware=HardwareTarget.MI300,
            available_partitions={"self_attn", "mlp"},
        )
        assert "linear_attn" not in searcher.layer_sensitivity

    def test_layer_modes_override(self):
        from quark.experimental.torch.llm.mix_precision.config import ModuleSearchConfig
        from quark.experimental.torch.llm.mix_precision.searcher import ConfigSearcher

        search_config = ModuleSearchConfig(layer_modes=["native", "ptpc_fp8"])
        searcher = ConfigSearcher(search_config=search_config)
        configs = searcher.generate_sorted_configs()
        for config in configs:
            for p in searcher.layer_sensitivity:
                mode = config.get(f"{p}_mode", "native")
                assert mode in ("native", "ptpc_fp8"), f"Unexpected mode {mode!r} in config: {config}"

    def test_kv_cache_modes_override(self):
        from quark.experimental.torch.llm.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.llm.mix_precision.searcher import ConfigSearcher

        search_config = ModuleSearchConfig(kv_cache_modes=["native"])
        searcher = ConfigSearcher(search_config=search_config, hardware=HardwareTarget.MI300)
        configs = searcher.generate_sorted_configs()
        for config in configs:
            assert config.get("kv_cache_mode") == "native", f"kv_cache must stay native: {config}"

    def test_compute_score_ordering(self):
        from quark.experimental.torch.llm.mix_precision.config import HardwareTarget
        from quark.experimental.torch.llm.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(hardware=HardwareTarget.MI300)
        conservative = {
            "self_attn_mode": "ptpc_fp8",
            "mlp_mode": "ptpc_fp8",
            "kv_cache_mode": "native",
            "attention_mode": "native",
        }
        aggressive = {"self_attn_mode": "fp8", "mlp_mode": "fp8", "kv_cache_mode": "native", "attention_mode": "native"}
        assert searcher.compute_score(conservative) > searcher.compute_score(aggressive)


# =============================================================================
# eval.py — pure computation functions
# =============================================================================


class TestExtractStrict:
    def test_extracts_after_hash(self):
        from quark.experimental.torch.llm.mix_precision.eval import extract_strict

        assert extract_strict("Let me calculate. #### 42") == "42"

    def test_negative_number(self):
        from quark.experimental.torch.llm.mix_precision.eval import extract_strict

        assert extract_strict("The answer is #### -7") == "-7"

    def test_no_hash_returns_none(self):
        from quark.experimental.torch.llm.mix_precision.eval import extract_strict

        assert extract_strict("The answer is 42") is None

    def test_empty_string_returns_none(self):
        from quark.experimental.torch.llm.mix_precision.eval import extract_strict

        assert extract_strict("") is None


class TestExtractFlexible:
    def test_extracts_last_number(self):
        from quark.experimental.torch.llm.mix_precision.eval import extract_flexible

        result = extract_flexible("First 3 apples, then 5 more, total 8")
        assert result == "8"

    def test_no_number_returns_none(self):
        from quark.experimental.torch.llm.mix_precision.eval import extract_flexible

        assert extract_flexible("no numbers here") is None

    def test_empty_string_returns_none(self):
        from quark.experimental.torch.llm.mix_precision.eval import extract_flexible

        assert extract_flexible("") is None

    def test_dollar_amount(self):
        from quark.experimental.torch.llm.mix_precision.eval import extract_flexible

        result = extract_flexible("She has $23 left")
        assert result is not None


class TestCalculateExactMatch:
    def test_correct_answer_matches(self):
        from quark.experimental.torch.llm.mix_precision.eval import calculate_exact_match

        assert calculate_exact_match("42", "#### 42") is True

    def test_wrong_answer_no_match(self):
        from quark.experimental.torch.llm.mix_precision.eval import calculate_exact_match

        assert calculate_exact_match("41", "#### 42") is False

    def test_none_prediction_no_match(self):
        from quark.experimental.torch.llm.mix_precision.eval import calculate_exact_match

        assert calculate_exact_match(None, "#### 42") is False

    def test_comma_stripped_from_reference(self):
        from quark.experimental.torch.llm.mix_precision.eval import calculate_exact_match

        # "1,000" in reference should match "1000"
        assert calculate_exact_match("1000", "#### 1,000") is True


class TestEvaluateGsm8kEntry:
    def test_correct_flexible(self):
        from quark.experimental.torch.llm.mix_precision.eval import evaluate_gsm8k_entry

        is_correct, extracted = evaluate_gsm8k_entry("So the total is 8 apples.", "#### 8", strategy="flexible")
        assert is_correct is True
        assert extracted is not None

    def test_wrong_answer(self):
        from quark.experimental.torch.llm.mix_precision.eval import evaluate_gsm8k_entry

        is_correct, _ = evaluate_gsm8k_entry("The answer is 7.", "#### 8", strategy="flexible")
        assert is_correct is False

    def test_strict_strategy(self):
        from quark.experimental.torch.llm.mix_precision.eval import evaluate_gsm8k_entry

        is_correct, _ = evaluate_gsm8k_entry("#### 42", "#### 42", strategy="strict")
        assert is_correct is True

    def test_hybrid_falls_back_to_flexible(self):
        from quark.experimental.torch.llm.mix_precision.eval import evaluate_gsm8k_entry

        # No "####" in output, so strict fails; flexible picks up 42
        is_correct, _ = evaluate_gsm8k_entry("The answer is 42.", "#### 42", strategy="hybrid")
        assert is_correct is True


class TestTruncateAtStopStrings:
    def test_truncates_at_first_stop(self):
        from quark.experimental.torch.llm.mix_precision.eval import _truncate_at_stop_strings

        result = _truncate_at_stop_strings("Answer: 8\nQuestion: next one", ["Question:"])
        assert "Question:" not in result
        assert "8" in result

    def test_no_stop_string_returns_full(self):
        from quark.experimental.torch.llm.mix_precision.eval import _truncate_at_stop_strings

        text = "The answer is 42."
        assert _truncate_at_stop_strings(text, ["STOP"]) == text

    def test_picks_earliest_stop(self):
        from quark.experimental.torch.llm.mix_precision.eval import _truncate_at_stop_strings

        result = _truncate_at_stop_strings("A Q: B Human: C", ["Human:", "Q:"])
        assert result == "A"


class TestGsm8kEvaluateOutputs:
    def test_all_correct(self):
        from quark.experimental.torch.llm.mix_precision.eval import _gsm8k_evaluate_outputs

        generated = ["The answer is 5.", "The answer is 10."]
        references = ["#### 5", "#### 10"]
        correct, total = _gsm8k_evaluate_outputs(generated, references, verbose=False)
        assert correct == 2
        assert total == 2

    def test_all_wrong(self):
        from quark.experimental.torch.llm.mix_precision.eval import _gsm8k_evaluate_outputs

        generated = ["The answer is 99.", "The answer is 99."]
        references = ["#### 5", "#### 10"]
        correct, total = _gsm8k_evaluate_outputs(generated, references, verbose=False)
        assert correct == 0
        assert total == 2

    def test_mixed_correct(self):
        from quark.experimental.torch.llm.mix_precision.eval import _gsm8k_evaluate_outputs

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
        from quark.experimental.torch.llm.mix_precision.switcher import needs_calibration

        assert needs_calibration({"self_attn_mode": "fp8", "mlp_mode": "native"}) is True

    def test_mxfp4_fp8_requires_calib(self):
        from quark.experimental.torch.llm.mix_precision.switcher import needs_calibration

        assert needs_calibration({"self_attn_mode": "mxfp4_fp8"}) is True

    def test_ptpc_fp8_no_calib(self):
        from quark.experimental.torch.llm.mix_precision.switcher import needs_calibration

        assert needs_calibration({"self_attn_mode": "ptpc_fp8", "mlp_mode": "native"}) is False

    def test_mxfp4_no_calib(self):
        from quark.experimental.torch.llm.mix_precision.switcher import needs_calibration

        assert needs_calibration({"self_attn_mode": "mxfp4"}) is False

    def test_native_only_no_calib(self):
        from quark.experimental.torch.llm.mix_precision.switcher import needs_calibration

        assert needs_calibration({"self_attn_mode": "native", "mlp_mode": "native"}) is False

    def test_mixed_fp8_and_ptpc_requires_calib(self):
        from quark.experimental.torch.llm.mix_precision.switcher import needs_calibration

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

    fused_moe_layer = types.ModuleType("vllm.model_executor.layers.fused_moe.layer")
    fused_moe_layer.FusedMoE = _FakeFusedMoE

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

    class _FakeColumnParallelLinear(_FakeLinearBase):
        pass

    class _FakeMergedColumnParallelLinear(_FakeLinearBase):
        pass

    class _FakeQKVParallelLinear(_FakeLinearBase):
        pass

    class _FakeUnquantizedLinearMethod:
        pass

    linear = types.ModuleType("vllm.model_executor.layers.linear")
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
# that does `from quark.experimental.plugin.vllm_plugin import ...` gets the
# stub-backed version from sys.modules cache.
_vllm_stub_previous = _inject_vllm_stubs()
for _k in list(sys.modules.keys()):
    if _k.startswith("quark.experimental.plugin"):
        sys.modules.pop(_k, None)
import quark.experimental.plugin.vllm_plugin as _vllm_plugin_mod  # noqa: E402

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


# ---------------------------------------------------------------------------
# _tensor_is_fp8
# ---------------------------------------------------------------------------


class TestTensorIsFp8:
    def test_fp8_e4m3_returns_true(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _tensor_is_fp8

        assert _tensor_is_fp8(torch.zeros(4, dtype=torch.float8_e4m3fn)) is True

    def test_bf16_returns_false(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _tensor_is_fp8

        assert _tensor_is_fp8(torch.zeros(4, dtype=torch.bfloat16)) is False

    def test_none_returns_false(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _tensor_is_fp8

        assert _tensor_is_fp8(None) is False


# ---------------------------------------------------------------------------
# _is_fp8_quant_method / _is_mxfp4_quant_method
# ---------------------------------------------------------------------------


class TestIsFp8QuantMethod:
    def test_fp8_class_name_detected(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _is_fp8_quant_method

        class Fp8LinearMethod:
            pass

        module = nn.Linear(4, 4)
        module.quant_method = Fp8LinearMethod()
        assert _is_fp8_quant_method(module) is True

    def test_no_quant_method_returns_false(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _is_fp8_quant_method

        assert _is_fp8_quant_method(nn.Linear(4, 4)) is False

    def test_non_fp8_method_returns_false(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _is_fp8_quant_method

        class Int8LinearMethod:
            pass

        module = nn.Linear(4, 4)
        module.quant_method = Int8LinearMethod()
        assert _is_fp8_quant_method(module) is False


class TestIsMxfp4QuantMethod:
    def test_mxfp4_class_name_detected(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _is_mxfp4_quant_method

        class Mxfp4LinearMethod:
            pass

        module = nn.Linear(4, 4)
        module.quant_method = Mxfp4LinearMethod()
        assert _is_mxfp4_quant_method(module) is True

    def test_weight_dtype_mxfp4_detected(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _is_mxfp4_quant_method

        class SomeMoEMethod:
            weight_dtype = "mxfp4"

        module = nn.Linear(4, 4)
        module.quant_method = SomeMoEMethod()
        assert _is_mxfp4_quant_method(module) is True

    def test_gpt_oss_mxfp4_weight_dtype_detected(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _is_mxfp4_quant_method

        class GptOssMoEMethod:
            weight_dtype = "gpt_oss_mxfp4"

        module = nn.Linear(4, 4)
        module.quant_method = GptOssMoEMethod()
        assert _is_mxfp4_quant_method(module) is True

    def test_fp8_method_not_mxfp4(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _is_mxfp4_quant_method

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
        from quark.experimental.plugin.vllm_inverse_quantizer import _mxfp4_backend_name

        class Backend:
            value = "TRITON"

        class MockMethod:
            mxfp4_backend = Backend()

        module = nn.Linear(4, 4)
        module.quant_method = MockMethod()
        assert _mxfp4_backend_name(module) == "TRITON"

    def test_no_quant_method_returns_none(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _mxfp4_backend_name

        assert _mxfp4_backend_name(nn.Linear(4, 4)) is None

    def test_no_backend_attr_returns_none(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _mxfp4_backend_name

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
        from quark.experimental.plugin.vllm_inverse_quantizer import _tensor_is_uint8_or_triton_mxfp4

        assert _tensor_is_uint8_or_triton_mxfp4(torch.zeros(4, dtype=torch.uint8)) is True

    def test_non_uint8_tensor_returns_false(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _tensor_is_uint8_or_triton_mxfp4

        assert _tensor_is_uint8_or_triton_mxfp4(torch.zeros(4, dtype=torch.bfloat16)) is False

    def test_triton_tensor_with_uint8_storage_returns_true(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _tensor_is_uint8_or_triton_mxfp4

        storage = SimpleNamespace(data=torch.zeros(4, dtype=torch.uint8))
        assert _tensor_is_uint8_or_triton_mxfp4(SimpleNamespace(storage=storage)) is True

    def test_triton_tensor_with_non_uint8_storage_returns_false(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import _tensor_is_uint8_or_triton_mxfp4

        storage = SimpleNamespace(data=torch.zeros(4, dtype=torch.float32))
        assert _tensor_is_uint8_or_triton_mxfp4(SimpleNamespace(storage=storage)) is False


# ---------------------------------------------------------------------------
# is_prequantized_vllm_linear
# ---------------------------------------------------------------------------


class TestIsPrequantizedVllmLinear:
    def test_valid_fp8_linear_detected(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_linear

        assert is_prequantized_vllm_linear(_make_fp8_linear_module("RowParallelLinear")) is True

    def test_valid_fp8_linear_with_weight_scale(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_linear

        assert is_prequantized_vllm_linear(_make_fp8_linear_module("ColumnParallelLinear", use_scale_inv=False)) is True

    def test_unknown_class_name_returns_false(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_linear

        assert is_prequantized_vllm_linear(_make_fp8_linear_module("UnknownLinear")) is False

    def test_non_fp8_weight_returns_false(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_linear

        module = _make_fp8_linear_module("RowParallelLinear")
        module.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
        assert is_prequantized_vllm_linear(module) is False

    def test_missing_scale_returns_false(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_linear

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
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_fp8_moe

        assert is_prequantized_vllm_fp8_moe(_make_fp8_moe_module()) is True

    def test_fp8_moe_with_weight_scale_detected(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_fp8_moe

        assert is_prequantized_vllm_fp8_moe(_make_fp8_moe_module(use_scale_inv=False)) is True

    def test_unknown_class_returns_false(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_fp8_moe

        assert is_prequantized_vllm_fp8_moe(_make_fp8_moe_module("UnknownMoE")) is False

    def test_non_fp8_weights_returns_false(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_fp8_moe

        module = _make_fp8_moe_module()
        module.w13_weight = torch.zeros(2, 4, 4, dtype=torch.bfloat16)
        assert is_prequantized_vllm_fp8_moe(module) is False


class TestIsPrequantizedVllmMxfp4Moe:
    def test_valid_mxfp4_moe_detected(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_mxfp4_moe

        assert is_prequantized_vllm_mxfp4_moe(_make_mxfp4_moe_module()) is True

    def test_unsupported_backend_returns_false(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_mxfp4_moe

        class UnsupportedBackend:
            value = "CUDA_CUSTOM"

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
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_mxfp4_moe

        module = _make_mxfp4_moe_module()
        del module.w13_weight_scale
        assert is_prequantized_vllm_mxfp4_moe(module) is False


class TestIsPrequantizedVllmMoe:
    def test_fp8_moe_returns_true(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_moe

        assert is_prequantized_vllm_moe(_make_fp8_moe_module()) is True

    def test_mxfp4_moe_returns_true(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_moe

        assert is_prequantized_vllm_moe(_make_mxfp4_moe_module()) is True

    def test_plain_module_returns_false(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import is_prequantized_vllm_moe

        assert is_prequantized_vllm_moe(nn.Linear(4, 4)) is False


# ---------------------------------------------------------------------------
# VLLMFp8LinearInverseQuantizer — constructor guards
# ---------------------------------------------------------------------------


class TestVLLMFp8LinearInverseQuantizerGuards:
    def test_raises_without_fp8_quant_method(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        with pytest.raises(ValueError, match="Unsupported vLLM module"):
            VLLMFp8LinearInverseQuantizer(nn.Linear(4, 4))

    def test_raises_when_use_deep_gemm(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        class Fp8MethodDeepGemm:
            use_deep_gemm = True
            use_marlin = False

        module = _make_fp8_linear_module("RowParallelLinear")
        module.quant_method = Fp8MethodDeepGemm()
        with pytest.raises(NotImplementedError, match="use_deep_gemm=True"):
            VLLMFp8LinearInverseQuantizer(module)

    def test_raises_when_use_marlin(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        class Fp8MethodMarlin:
            use_deep_gemm = False
            use_marlin = True

        module = _make_fp8_linear_module("RowParallelLinear")
        module.quant_method = Fp8MethodMarlin()
        with pytest.raises(NotImplementedError, match="use_marlin=True"):
            VLLMFp8LinearInverseQuantizer(module)

    def test_raises_when_no_scale(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

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
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

        inv_q = VLLMFp8LinearInverseQuantizer(_make_fp8_linear_module("RowParallelLinear"))
        assert isinstance(inv_q.scale, torch.Tensor)

    def test_block_quant_detected_from_weight_block_size(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

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
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        with pytest.raises(ValueError, match="Unsupported vLLM MoE"):
            VLLMFp8MoEWeightInverseQuantizer(nn.Linear(4, 4), "w13_weight_scale_inv")

    def test_raises_when_use_deep_gemm(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        class Fp8MoEMethodDeepGemm:
            use_deep_gemm = True

        module = _make_fp8_moe_module()
        module.quant_method = Fp8MoEMethodDeepGemm()
        with pytest.raises(NotImplementedError, match="use_deep_gemm=True"):
            VLLMFp8MoEWeightInverseQuantizer(module, "w13_weight_scale_inv")

    def test_raises_when_scale_attr_missing(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        with pytest.raises(ValueError, match="must have nonexistent_scale"):
            VLLMFp8MoEWeightInverseQuantizer(_make_fp8_moe_module(), "nonexistent_scale")

    def test_constructed_with_scale_inv(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        inv_q = VLLMFp8MoEWeightInverseQuantizer(_make_fp8_moe_module(use_scale_inv=True), "w13_weight_scale_inv")
        assert isinstance(inv_q.scale, torch.Tensor)


# ---------------------------------------------------------------------------
# VLLMMxfp4MoEWeightInverseQuantizer — constructor guards
# ---------------------------------------------------------------------------


class TestVLLMMxfp4MoEWeightInverseQuantizerGuards:
    def test_raises_without_mxfp4_quant_method(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        with pytest.raises(ValueError, match="Unsupported vLLM MoE"):
            VLLMMxfp4MoEWeightInverseQuantizer(nn.Linear(4, 4), "w13_weight_scale")

    def test_raises_unsupported_backend(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        class BadBackend:
            value = "CUDA_ONLY"

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
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        module = _make_mxfp4_moe_module()
        object.__getattribute__(module, "__dict__")["w13_weight_scale"] = "not_a_tensor"
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            VLLMMxfp4MoEWeightInverseQuantizer(module, "w13_weight_scale")

    def test_constructed_successfully(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        inv_q = VLLMMxfp4MoEWeightInverseQuantizer(_make_mxfp4_moe_module(), "w13_weight_scale")
        assert inv_q.backend_name == "TRITON"
        assert isinstance(inv_q.scale, torch.Tensor)


# ---------------------------------------------------------------------------
# create_inverse_quantizer_for_vllm_linear / create_vllm_moe_inverse_quantizers
# ---------------------------------------------------------------------------


class TestCreateInverseQuantizerFunctions:
    def test_create_linear_raises_if_not_prequantized(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import create_inverse_quantizer_for_vllm_linear

        with pytest.raises(ValueError, match="not a pre-quantized"):
            create_inverse_quantizer_for_vllm_linear(nn.Linear(4, 4))

    def test_create_moe_raises_if_not_prequantized(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import create_vllm_moe_inverse_quantizers

        with pytest.raises(ValueError, match="not a supported"):
            create_vllm_moe_inverse_quantizers(nn.Linear(4, 4))

    def test_create_linear_returns_fp8_inverse_quantizer(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import (
            VLLMFp8LinearInverseQuantizer,
            create_inverse_quantizer_for_vllm_linear,
        )

        assert isinstance(
            create_inverse_quantizer_for_vllm_linear(_make_fp8_linear_module("RowParallelLinear")),
            VLLMFp8LinearInverseQuantizer,
        )

    def test_create_moe_returns_fp8_pair(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import (
            VLLMFp8MoEWeightInverseQuantizer,
            create_vllm_moe_inverse_quantizers,
        )

        w13_q, w2_q = create_vllm_moe_inverse_quantizers(_make_fp8_moe_module(use_scale_inv=True))
        assert isinstance(w13_q, VLLMFp8MoEWeightInverseQuantizer)
        assert isinstance(w2_q, VLLMFp8MoEWeightInverseQuantizer)

    def test_create_moe_returns_mxfp4_pair(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import (
            VLLMMxfp4MoEWeightInverseQuantizer,
            create_vllm_moe_inverse_quantizers,
        )

        w13_q, w2_q = create_vllm_moe_inverse_quantizers(_make_mxfp4_moe_module())
        assert isinstance(w13_q, VLLMMxfp4MoEWeightInverseQuantizer)
        assert isinstance(w2_q, VLLMMxfp4MoEWeightInverseQuantizer)

    def test_create_moe_selects_scale_inv_when_available(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import create_vllm_moe_inverse_quantizers

        w13_q, w2_q = create_vllm_moe_inverse_quantizers(_make_fp8_moe_module(use_scale_inv=True))
        assert w13_q.scale_attr_name == "w13_weight_scale_inv"
        assert w2_q.scale_attr_name == "w2_weight_scale_inv"

    def test_create_moe_falls_back_to_weight_scale(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import create_vllm_moe_inverse_quantizers

        w13_q, w2_q = create_vllm_moe_inverse_quantizers(_make_fp8_moe_module(use_scale_inv=False))
        assert w13_q.scale_attr_name == "w13_weight_scale"
        assert w2_q.scale_attr_name == "w2_weight_scale"


# =============================================================================
# vllm_plugin.py — pure utility functions (vLLM stubs injected via fixture)
# =============================================================================


class TestEnvEnabled:
    def test_env_var_1_returns_true(self, monkeypatch):
        from quark.experimental.plugin.vllm_plugin import _env_enabled

        monkeypatch.setenv("TEST_QUARK_FLAG", "1")
        assert _env_enabled("TEST_QUARK_FLAG") is True

    def test_env_var_true_returns_true(self, monkeypatch):
        from quark.experimental.plugin.vllm_plugin import _env_enabled

        monkeypatch.setenv("TEST_QUARK_FLAG", "true")
        assert _env_enabled("TEST_QUARK_FLAG") is True

    def test_env_var_0_returns_false(self, monkeypatch):
        from quark.experimental.plugin.vllm_plugin import _env_enabled

        monkeypatch.setenv("TEST_QUARK_FLAG", "0")
        assert _env_enabled("TEST_QUARK_FLAG") is False

    def test_unset_var_returns_false(self, monkeypatch):
        from quark.experimental.plugin.vllm_plugin import _env_enabled

        monkeypatch.delenv("TEST_QUARK_FLAG", raising=False)
        assert _env_enabled("TEST_QUARK_FLAG") is False

    def test_custom_default_used_when_unset(self, monkeypatch):
        from quark.experimental.plugin.vllm_plugin import _env_enabled

        monkeypatch.delenv("TEST_QUARK_FLAG2", raising=False)
        assert _env_enabled("TEST_QUARK_FLAG2", default="1") is True


class TestKvCacheDtypeForCalib:
    def test_fp8_dtype_becomes_auto_in_calib_phase(self, monkeypatch):
        from quark.experimental.plugin.vllm_plugin import _kv_cache_dtype_for_calib

        monkeypatch.setenv("QUARK_CALIB_PHASE", "1")
        assert _kv_cache_dtype_for_calib("fp8_e4m3") == "auto"

    def test_fp8_dtype_unchanged_outside_calib_phase(self, monkeypatch):
        from quark.experimental.plugin.vllm_plugin import _kv_cache_dtype_for_calib

        monkeypatch.setenv("QUARK_CALIB_PHASE", "0")
        assert _kv_cache_dtype_for_calib("fp8_e4m3") == "fp8_e4m3"

    def test_non_fp8_dtype_unchanged_in_calib_phase(self, monkeypatch):
        from quark.experimental.plugin.vllm_plugin import _kv_cache_dtype_for_calib

        monkeypatch.setenv("QUARK_CALIB_PHASE", "1")
        assert _kv_cache_dtype_for_calib("auto") == "auto"

    def test_fp8_dtype_unchanged_without_env_var(self, monkeypatch):
        from quark.experimental.plugin.vllm_plugin import _kv_cache_dtype_for_calib

        monkeypatch.delenv("QUARK_CALIB_PHASE", raising=False)
        assert _kv_cache_dtype_for_calib("fp8_e5m2") == "fp8_e5m2"


class TestUnwrapFakeQuantMethod:
    def test_non_fake_quant_method_returned_as_is(self):
        from quark.experimental.plugin.vllm_plugin import _unwrap_fake_quant_method

        class SomeMethod:
            pass

        m = SomeMethod()
        assert _unwrap_fake_quant_method(m) is m

    def test_single_layer_unwrapped(self):
        from quark.experimental.plugin.vllm_plugin import FakeQuantLinearMethod, _unwrap_fake_quant_method

        class Original:
            pass

        orig = Original()
        wrapped = MagicMock(spec=FakeQuantLinearMethod)
        wrapped.original_quant_method = orig
        wrapped.__class__ = FakeQuantLinearMethod
        assert _unwrap_fake_quant_method(wrapped) is orig

    def test_double_wrapped_unwrapped_fully(self):
        from quark.experimental.plugin.vllm_plugin import FakeQuantLinearMethod, _unwrap_fake_quant_method

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
        from quark.experimental.plugin.vllm_plugin import _filter_supported_init_kwargs

        class Foo:
            def __init__(self, a: int, b: str) -> None:
                pass

        assert _filter_supported_init_kwargs(Foo, {"a": 1, "b": "x", "c": 99}) == {"a": 1, "b": "x"}

    def test_self_always_excluded(self):
        from quark.experimental.plugin.vllm_plugin import _filter_supported_init_kwargs

        class Bar:
            def __init__(self, x: int) -> None:
                pass

        result = _filter_supported_init_kwargs(Bar, {"self": "bad", "x": 5})
        assert "self" not in result
        assert result == {"x": 5}

    def test_empty_kwargs_returns_empty(self):
        from quark.experimental.plugin.vllm_plugin import _filter_supported_init_kwargs

        class Baz:
            def __init__(self, x: int) -> None:
                pass

        assert _filter_supported_init_kwargs(Baz, {}) == {}


class TestAdaptLayerPatternsForVllm:
    def test_q_proj_maps_to_qkv_proj(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("qkv_proj" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj"))

    def test_k_proj_maps_to_qkv_proj(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("qkv_proj" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.k_proj"))

    def test_v_proj_maps_to_qkv_proj(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("qkv_proj" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.v_proj"))

    def test_gate_proj_maps_to_gate_up_proj_and_experts(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        result = adapt_layer_patterns_for_vllm("model.layers.*.mlp.gate_proj")
        assert any("gate_up_proj" in p for p in result)
        assert any("*experts*" in p for p in result)

    def test_up_proj_maps_to_gate_up_proj_and_experts(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        result = adapt_layer_patterns_for_vllm("model.layers.*.mlp.up_proj")
        assert any("gate_up_proj" in p for p in result)
        assert any("*experts*" in p for p in result)

    def test_in_proj_qkv_maps_to_in_proj_qkvz(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("in_proj_qkvz" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.linear_attn.in_proj_qkv"))

    def test_in_proj_b_maps_to_in_proj_ba(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("in_proj_ba" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.linear_attn.in_proj_b"))

    def test_in_proj_a_maps_to_in_proj_ba(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("in_proj_ba" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.linear_attn.in_proj_a"))

    def test_no_double_replacement_for_gate_up_proj(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert not any(
            "gate_gate_up_proj" in p for p in adapt_layer_patterns_for_vllm("model.layers.*.mlp.gate_up_proj")
        )

    def test_self_attn_not_expanded_to_linear_attn(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert not any(".linear_attn." in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj"))

    def test_linear_attn_not_expanded_to_self_attn(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert not any(
            ".self_attn." in p for p in adapt_layer_patterns_for_vllm("model.layers.*.linear_attn.in_proj_qkv")
        )

    def test_self_attn_adds_attn_variant(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any(".attn." in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj"))

    def test_prefix_alias_expansion_model_to_language_model(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any("language_model." in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj"))

    def test_mla_variant_added_for_self_attn(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert any(
            ".self_attn.mla_attn." in p for p in adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj")
        )

    def test_returns_tuple(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        assert isinstance(adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj"), tuple)

    def test_unmatched_pattern_returned_unchanged(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        result = adapt_layer_patterns_for_vllm("model.layers.*.lm_head")
        assert isinstance(result, tuple) and len(result) >= 1


class TestAdaptKvCachePatternForVllm:
    def test_k_proj_maps_to_qkv_proj_pattern(self):
        from quark.experimental.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.self_attn.k_proj") == "*qkv_proj"

    def test_v_proj_maps_to_qkv_proj_pattern(self):
        from quark.experimental.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.self_attn.v_proj") == "*qkv_proj"

    def test_qkv_proj_maps_to_qkv_proj_pattern(self):
        from quark.experimental.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.self_attn.qkv_proj") == "*qkv_proj"

    def test_in_proj_qkv_maps_to_in_proj_qkvz_pattern(self):
        from quark.experimental.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.linear_attn.in_proj_qkv") == "*in_proj_qkvz"

    def test_in_proj_qkvz_maps_to_in_proj_qkvz_pattern(self):
        from quark.experimental.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.linear_attn.in_proj_qkvz") == "*in_proj_qkvz"

    def test_o_proj_returns_none(self):
        from quark.experimental.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.self_attn.o_proj") is None

    def test_mlp_pattern_returns_none(self):
        from quark.experimental.plugin.vllm_plugin import adapt_kv_cache_pattern_for_vllm

        assert adapt_kv_cache_pattern_for_vllm("model.layers.*.mlp.gate_proj") is None


class TestVllmAvailableFlag:
    def test_vllm_available_is_bool(self):
        from quark.experimental.plugin.vllm_plugin import VLLM_AVAILABLE

        assert isinstance(VLLM_AVAILABLE, bool)

    def test_vllm_available_true_with_stubs(self, vllm_plugin):
        assert vllm_plugin.VLLM_AVAILABLE is True


# =============================================================================
# utils.py — categorize_layers, pattern helpers (three-partition support)
# =============================================================================


class TestLinearAttnPatternMatching:
    def test_linear_attn_names_match(self):
        from quark.experimental.torch.llm.mix_precision.utils import _LINEAR_ATTN_STANDARD, _match_patterns

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
        from quark.experimental.torch.llm.mix_precision.utils import _LINEAR_ATTN_STANDARD, _match_patterns

        for name in ("model.layers.0.self_attn.q_proj.weight", "model.layers.1.mlp.gate_proj.weight"):
            assert not _match_patterns(name, _LINEAR_ATTN_STANDARD), f"{name} should NOT match linear_attn patterns"


class TestLayerPatternsThreePartitions:
    def test_qwen3_5_moe_has_all_three_partitions(self):
        from quark.experimental.torch.llm.mix_precision.utils import LAYER_PATTERNS

        assert "qwen3_5_moe" in LAYER_PATTERNS
        qwen = LAYER_PATTERNS["qwen3_5_moe"]
        for part in ("linear_attn", "self_attn", "mlp"):
            assert part in qwen and len(qwen[part]) > 0

    def test_default_entry_includes_linear_attn(self):
        from quark.experimental.torch.llm.mix_precision.utils import LAYER_PATTERNS

        assert "linear_attn" in LAYER_PATTERNS["default"]


class _MockConfig:
    def __init__(self, layer_types):
        self.layer_types = layer_types


class _MockNestedConfig:
    def __init__(self, layer_types):
        self.text_config = _MockConfig(layer_types)


class _MockModel:
    def __init__(self, config):
        self.config = config


class TestGetLayerPartitionFromConfig:
    def test_linear_attention_layer_detected(self):
        from quark.experimental.torch.llm.mix_precision.utils import _get_layer_partition_from_config

        model = _MockModel(_MockNestedConfig(["linear_attention", "linear_attention", "full_attention"]))
        assert _get_layer_partition_from_config(model, "model.layers.0.self_attn.q_proj") == "linear_attn"
        assert _get_layer_partition_from_config(model, "model.layers.2.self_attn.q_proj") == "self_attn"

    def test_no_layer_types_returns_none(self):
        from quark.experimental.torch.llm.mix_precision.utils import _get_layer_partition_from_config

        cfg = _MockConfig([])
        del cfg.layer_types
        assert _get_layer_partition_from_config(_MockModel(cfg), "model.layers.0.self_attn.q_proj") is None

    def test_non_layer_name_returns_none(self):
        from quark.experimental.torch.llm.mix_precision.utils import _get_layer_partition_from_config

        model = _MockModel(_MockConfig(["linear_attention", "full_attention"]))
        assert _get_layer_partition_from_config(model, "model.embed_tokens.weight") is None


class TestCategorizeLayers:
    def test_three_partitions_for_hybrid_model(self):
        from quark.experimental.torch.llm.mix_precision.utils import categorize_layers

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
        for part in ("linear_attn", "self_attn", "mlp"):
            assert part in cats
        assert len(cats["linear_attn"]) == 2
        assert len(cats["self_attn"]) == 2
        assert len(cats["mlp"]) == 2

    def test_two_partitions_for_traditional_model(self):
        from quark.experimental.torch.llm.mix_precision.utils import categorize_layers

        class MockTraditionalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_0_self_attn_q_proj = nn.Linear(4096, 4096)
                self.layers_0_mlp_gate_proj = nn.Linear(4096, 11008)

        cats = categorize_layers(MockTraditionalModel(), model_type="llama")
        assert "linear_attn" not in cats
        assert "self_attn" in cats
        assert "mlp" in cats


class TestCreateQconfigThreePartitions:
    def test_three_partition_qconfig_structure(self):
        from quark.experimental.torch.llm.mix_precision.config import create_quant_config
        from quark.experimental.torch.llm.mix_precision.utils import create_qconfig_from_quant_config

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
            layer_partitions={"linear_attn": "fp8", "self_attn": "ptpc_fp8", "mlp": "native"},
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
        from quark.experimental.torch.llm.mix_precision.config import create_quant_config
        from quark.experimental.torch.llm.mix_precision.utils import create_qconfig_from_quant_config

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


class TestConfigSearcherThreePartitions:
    def test_three_partition_search_generates_configs(self):
        from quark.experimental.torch.llm.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.llm.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(
            search_config=ModuleSearchConfig(layer_sensitivity={"linear_attn": 3, "self_attn": 3, "mlp": 1}),
            hardware=HardwareTarget.MI300,
        )
        assert "linear_attn" in searcher.layer_sensitivity
        assert "self_attn" in searcher.layer_sensitivity
        assert "mlp" in searcher.layer_sensitivity
        configs = searcher.generate_sorted_configs()
        assert len(configs) > 0
        for cfg in configs[:5]:
            for key in ("linear_attn_mode", "self_attn_mode", "mlp_mode", "kv_cache_mode", "attention_mode"):
                assert key in cfg
        # No all-native config
        for cfg in configs:
            modes = [cfg.get(f"{p}_mode") for p in ("linear_attn", "self_attn", "mlp")]
            assert any(m != "native" for m in modes if m is not None)

    def test_two_partition_search_still_works(self):
        from quark.experimental.torch.llm.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.llm.mix_precision.searcher import ConfigSearcher

        searcher = ConfigSearcher(
            search_config=ModuleSearchConfig(layer_sensitivity={"self_attn": 3, "mlp": 1}),
            hardware=HardwareTarget.MI300,
        )
        configs = searcher.generate_sorted_configs()
        assert len(configs) > 0
        for cfg in configs[:3]:
            assert "linear_attn_mode" not in cfg


class TestDefaultPartitionSensitivityThreePartitions:
    def test_all_three_partitions_defined(self):
        from quark.experimental.torch.llm.mix_precision.config import DEFAULT_PARTITION_SENSITIVITY

        assert DEFAULT_PARTITION_SENSITIVITY["linear_attn"] == 3
        assert DEFAULT_PARTITION_SENSITIVITY["self_attn"] == 3
        assert DEFAULT_PARTITION_SENSITIVITY["mlp"] == 1
        assert DEFAULT_PARTITION_SENSITIVITY["linear_attn"] > DEFAULT_PARTITION_SENSITIVITY["mlp"]


class TestVllmPluginLinearAttnIndependence:
    def test_self_attn_not_expanded_to_linear_attn(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

        result = adapt_layer_patterns_for_vllm("model.layers.*.self_attn.q_proj")
        assert any(".self_attn." in p for p in result)
        assert not any(".linear_attn." in p for p in result)

    def test_explicit_linear_attn_pattern_preserved(self):
        from quark.experimental.plugin.vllm_plugin import adapt_layer_patterns_for_vllm

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
        from quark.experimental.torch.llm.mix_precision.utils import categorize_layers

        cats = categorize_layers(MockLLaMAModel(), model_type="llama")
        assert "linear_attn" not in cats
        assert "self_attn" in cats
        assert "mlp" in cats
        assert len(cats["self_attn"]) == 4
        assert len(cats["mlp"]) == 3


class TestBackwardCompatSearch:
    def test_two_partition_search_generates_expected_config_count(self):
        from quark.experimental.torch.llm.mix_precision.config import HardwareTarget, ModuleSearchConfig
        from quark.experimental.torch.llm.mix_precision.searcher import ConfigSearcher
        from quark.experimental.torch.llm.mix_precision.utils import categorize_layers

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
        assert searcher.layer_sensitivity["mlp"] == 1


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
        from quark.experimental.plugin.vllm_plugin import FakeQuantLinearMethod

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
        from quark.experimental.plugin.vllm_plugin import FakeQuantLinearMethod

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
        from quark.experimental.plugin.vllm_plugin import FakeQuantLinearMethod

        layer = nn.Linear(4, 4)
        original = _RecordingApply(output=torch.zeros(2, 4))
        quant_layer = _MockQuantLayer(weight_quantizer=_DynamicQuantizerStub())
        method = FakeQuantLinearMethod(original, quant_layer)

        method.apply(layer, torch.ones(2, 4))

        # Dynamic (non-frozen) weight quantizer triggers runtime override too
        assert len(quant_layer.weight_calls) == 1

    def test_bias_passes_through_get_quant_bias(self):
        from quark.experimental.plugin.vllm_plugin import FakeQuantLinearMethod

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
        from quark.experimental.plugin.vllm_plugin import FakeQuantLinearMethod

        layer = nn.Linear(4, 4)
        original = _RecordingApply(output=torch.zeros(4, 2))
        quant_layer = _MockQuantLayer(weight_quantizer=_FrozenQuantizerStub())
        method = FakeQuantLinearMethod(original, quant_layer)

        x = torch.ones(2, 4).t()  # transpose makes it non-contiguous
        assert not x.is_contiguous()
        method.apply(layer, x)

        # The tensor passed to get_quant_input must be contiguous
        assert quant_layer.input_calls[0].is_contiguous()


class TestSetModuleAttrAllowNonParameter:
    def test_non_parameter_value_removes_from_parameters_dict(self):
        from quark.experimental.plugin.vllm_plugin import _set_module_attr_allow_non_parameter

        module = nn.Linear(4, 4)
        # Pretend "weight" was previously registered as a Parameter (it is)
        assert "weight" in module._parameters

        replacement = torch.zeros(4, 4)  # plain Tensor, not a Parameter
        _set_module_attr_allow_non_parameter(module, "weight", replacement)

        assert "weight" not in module._parameters
        assert torch.equal(module.weight, replacement)
        assert not isinstance(module.weight, nn.Parameter)

    def test_parameter_value_left_in_parameters_dict(self):
        from quark.experimental.plugin.vllm_plugin import _set_module_attr_allow_non_parameter

        module = nn.Linear(4, 4)
        new_param = nn.Parameter(torch.ones(4, 4))
        _set_module_attr_allow_non_parameter(module, "weight", new_param)

        # Parameter-valued sets should leave the parameter dict entry
        assert "weight" in module._parameters
        assert torch.equal(module.weight, torch.ones(4, 4))


class TestSetQuantMethodAttr:
    def test_non_module_assignment_clears_modules_dict(self):
        from quark.experimental.plugin.vllm_plugin import _set_quant_method_attr

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
        from quark.experimental.plugin.vllm_plugin import _set_quant_method_attr

        host = nn.Module()
        host.__dict__["quant_method"] = SimpleNamespace(name="plain")

        new_method = nn.Linear(4, 4)
        _set_quant_method_attr(host, new_method)

        assert host._modules.get("quant_method") is new_method
        assert "quant_method" not in host.__dict__


class TestLogVllmPrequantWrap:
    def test_does_not_raise_with_full_module(self):
        from quark.experimental.plugin.vllm_plugin import _log_vllm_prequant_wrap

        module = _make_fp8_linear_module("RowParallelLinear", use_scale_inv=True)
        _log_vllm_prequant_wrap(module, "QuantVLLMRowParallelLinear")  # should not raise

    def test_falls_back_through_scale_attrs(self):
        from quark.experimental.plugin.vllm_plugin import _log_vllm_prequant_wrap

        module = _make_fp8_moe_module(use_scale_inv=False)
        _log_vllm_prequant_wrap(module, "QuantVLLMFusedMoE")  # should not raise

    def test_handles_module_without_quant_method(self):
        from quark.experimental.plugin.vllm_plugin import _log_vllm_prequant_wrap

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
        from quark.experimental.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(quant_config=None)
        assert wrapper._quant_config is None
        assert wrapper._device.type in ("cuda", "cpu")  # default cuda but cpu-only CI ok
        assert wrapper._quantizer_initialized is False
        assert wrapper._float_module_cls is None
        assert wrapper._float_init_kwargs is None
        assert wrapper._weight_quantizer_inv is None
        assert wrapper._source_module is None

    def test_init_quantizers_is_noop_when_no_config(self):
        from quark.experimental.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(quant_config=None)
        wrapper._init_quantizers()  # must not raise
        assert wrapper._quantizer_initialized is False

    def test_init_quantizers_wraps_quant_method_in_fake_quant(self):
        from quark.experimental.plugin.vllm_plugin import FakeQuantLinearMethod, QuantVLLMParallelLinearBase

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
        from quark.experimental.plugin.vllm_plugin import FakeQuantLinearMethod, QuantVLLMParallelLinearBase

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
        from quark.experimental.plugin.vllm_plugin import QuantVLLMParallelLinearBase

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
        from quark.experimental.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(
            quant_config=_make_qlayer_config_empty(),
            device=torch.device("cpu"),
        )
        wrapper._init_quantizers()
        x = torch.arange(16.0).reshape(4, 4)
        # Empty config + no inverse → falls back to QuantMixin.get_quant_weight which is identity
        torch.testing.assert_close(wrapper.get_quant_weight(x), x)


class TestQuantVLLMParallelLinearBaseToFloatModule:
    def test_uses_source_module_when_present(self):
        from quark.experimental.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(quant_config=None, device=torch.device("cpu"))
        sentinel = nn.Linear(4, 4)
        wrapper._source_module = sentinel

        assert wrapper.to_float_module() is sentinel

    def test_raises_when_no_metadata(self):
        from quark.experimental.plugin.vllm_plugin import QuantVLLMParallelLinearBase

        wrapper = QuantVLLMParallelLinearBase(quant_config=None, device=torch.device("cpu"))
        with pytest.raises(ValueError, match="float-module metadata"):
            wrapper.to_float_module()

    def test_rebuilds_float_module_from_metadata(self):
        from quark.experimental.plugin.vllm_plugin import QuantVLLMParallelLinearBase

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
        from quark.experimental.plugin.vllm_plugin import QuantVLLMParallelLinearBase

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
    from quark.experimental.plugin.vllm_plugin import QuantVLLMFusedMoE

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


class TestQuantVLLMFusedMoEFromFloat:
    def test_non_prequant_module_routed_through_normal_init(self):
        from quark.experimental.plugin.vllm_plugin import QuantVLLMFusedMoE

        inner = _FakeMoEInner()  # has w13_weight/w2_weight but no quant_method
        result = QuantVLLMFusedMoE.from_float(
            float_module=inner,
            layer_quant_config=_make_qlayer_config_empty(),
            device=torch.device("cpu"),
        )
        assert isinstance(result, QuantVLLMFusedMoE)
        assert result._inner is inner

    def test_prequant_module_routed_through_from_prequantized(self, monkeypatch):
        import quark.experimental.plugin.vllm_plugin as vp

        # Stub out the runtime unquantized MoE method builder which depends on
        # vLLM internals not present in our test stubs.
        runtime_method = SimpleNamespace(name="runtime")
        monkeypatch.setattr(vp, "_build_runtime_unquantized_moe_method", lambda layer: runtime_method)

        prequant_module = _make_fp8_moe_module(use_scale_inv=True)

        result = vp.QuantVLLMFusedMoE.from_float(
            float_module=prequant_module,
            layer_quant_config=_make_qlayer_config_empty(),
            device=torch.device("cpu"),
        )

        assert isinstance(result, vp.QuantVLLMFusedMoE)
        assert result._inner is prequant_module
        # from_prequantized populated inverse quantizers
        assert result._w13_weight_quantizer_inv is not None
        assert result._w2_weight_quantizer_inv is not None
        # And replaced inner.quant_method with the runtime method
        assert prequant_module.quant_method is runtime_method


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
        from quark.experimental.plugin.vllm_plugin import (
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

    def test_replaces_moe_wrapper_with_inner(self):
        from quark.experimental.plugin.vllm_plugin import QuantVLLMFusedMoE, reset_vllm_fake_quant_model

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
        from quark.experimental.plugin.vllm_plugin import reset_vllm_fake_quant_model

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
        import quark.experimental.plugin.vllm_plugin as vp

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
        import quark.experimental.plugin.vllm_plugin as vp

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
        import quark.experimental.plugin.vllm_plugin as vp

        # Should be a no-op on a model containing no MoE wrappers
        model = nn.Sequential(nn.Linear(4, 4))
        vp.calibrate_moe_weight_params(model)  # no exception


# =============================================================================
# dequantize() — VLLMFp8LinearInverseQuantizer
# =============================================================================


class TestVLLMFp8LinearInverseQuantizerDequantize:
    def test_per_channel_scale_calls_dequantize_op(self, monkeypatch):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

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
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

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
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

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
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8LinearInverseQuantizer

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
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        inv_q = VLLMFp8MoEWeightInverseQuantizer(_make_fp8_moe_module(), "w13_weight_scale_inv")
        with pytest.raises(ValueError, match="3D"):
            inv_q.dequantize(torch.zeros(4, 4, dtype=torch.float8_e4m3fn))

    def test_1d_scale_per_expert_dequantize(self, monkeypatch):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

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
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

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
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMFp8MoEWeightInverseQuantizer

        num_experts, c, h = 2, 4, 4
        module = _make_fp8_moe_module(use_scale_inv=True)
        module.w13_weight_scale_inv = torch.ones(num_experts, 2)  # [E, 2]: unsupported
        inv_q = VLLMFp8MoEWeightInverseQuantizer(module, "w13_weight_scale_inv")

        with pytest.raises(NotImplementedError, match="scale shape"):
            inv_q.dequantize(torch.zeros(num_experts, c, h, dtype=torch.float8_e4m3fn))


# =============================================================================
# dequantize() — VLLMMxfp4MoEWeightInverseQuantizer
# =============================================================================


class TestVLLMMxfp4MoEWeightInverseQuantizerDequantize:
    def test_dequantize_calls_mx_dq_mxfp4(self, monkeypatch):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

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
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        module = _make_mxfp4_moe_module()
        module.w13_weight_scale = torch.ones(2, 1)
        inv_q = VLLMMxfp4MoEWeightInverseQuantizer(module, "w13_weight_scale")

        # Expected shape from scale [2, 1]: (2, 1*16) = (2, 16)
        packed = torch.zeros(2, 16, dtype=torch.uint8)
        result = inv_q._restore_packed_layout(packed)
        assert result.shape == (2, 16)
        assert result.is_contiguous()

    def test_restore_packed_layout_transposed_match(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        module = _make_mxfp4_moe_module()
        module.w13_weight_scale = torch.ones(2, 1)
        inv_q = VLLMMxfp4MoEWeightInverseQuantizer(module, "w13_weight_scale")

        # Transpose of (2, 16) is (16, 2), which after transpose(-2,-1) gives (2, 16)
        packed = torch.zeros(16, 2, dtype=torch.uint8)
        result = inv_q._restore_packed_layout(packed)
        assert result.shape == (2, 16)

    def test_restore_packed_layout_bad_shape_raises(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        module = _make_mxfp4_moe_module()
        module.w13_weight_scale = torch.ones(2, 1)
        inv_q = VLLMMxfp4MoEWeightInverseQuantizer(module, "w13_weight_scale")

        with pytest.raises(ValueError, match="Cannot restore"):
            inv_q._restore_packed_layout(torch.zeros(3, 7, dtype=torch.uint8))

    def test_unwrap_triton_tensor_passthrough_for_plain_tensor(self):
        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        t = torch.zeros(4, dtype=torch.uint8)
        assert VLLMMxfp4MoEWeightInverseQuantizer._unwrap_triton_tensor(t) is t

    def test_unwrap_triton_tensor_extracts_storage_data(self):
        from types import SimpleNamespace

        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        data = torch.zeros(4, dtype=torch.uint8)
        fake_tensor = SimpleNamespace(storage=SimpleNamespace(data=data))
        result = VLLMMxfp4MoEWeightInverseQuantizer._unwrap_triton_tensor(fake_tensor)
        assert result is data

    def test_unwrap_triton_tensor_bad_storage_raises(self):
        from types import SimpleNamespace

        from quark.experimental.plugin.vllm_inverse_quantizer import VLLMMxfp4MoEWeightInverseQuantizer

        bad = SimpleNamespace(storage=SimpleNamespace(data="not_a_tensor"))
        with pytest.raises(ValueError, match="Expected torch.Tensor"):
            VLLMMxfp4MoEWeightInverseQuantizer._unwrap_triton_tensor(bad)


# =============================================================================
# evaluate_ppl_offline
# =============================================================================
from quark.experimental.torch.llm.mix_precision import eval as _ppl_eval_module  # noqa: E402


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
