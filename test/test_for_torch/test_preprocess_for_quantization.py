#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Test that preprocess_for_quantization works correctly."""

import builtins
import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import quark.torch.utils.llm.model_preparation as model_preparation
from quark.common.utils.import_utils import (
    is_transformers_available,
    is_transformers_version_higher_or_equal,
)
from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.export.prequantized_config_converter import convert_prequantized_module_to_quark_config
from quark.torch.quantization.inverse_quantizer import create_inverse_quantizer, is_prequantized_linear
from quark.torch.utils.llm import preprocess_for_quantization
from quark.torch.utils.llm.module_replacement import quark_experts
from quark.torch.utils.llm.module_replacement.preprocess_registry import PREPROCESS_REGISTRY

if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    from transformers import AutoConfig, AutoModelForCausalLM, Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

    from quark.torch.utils.llm.module_replacement import QuarkExperts, QuarkQwen3MoeTopKRouter

# `FP8Experts` -- and the `ALL_FP8_EXPERTS_FUNCTIONS` / `use_experts_implementation`
# helpers the tests below use to construct it -- live under shared
# `transformers.integrations` modules and were added only in a later transformers
# release, so they are NOT present in every `transformers >= 5.0.0` (e.g. 5.2.0 lacks
# them). Gate the FP8Experts tests on actual importability rather than a version number,
# reusing the production module's already-resolved `_FP8Experts` (which is `None` when
# `FP8Experts` can't be imported) so the skip condition can never drift from the guard in
# the code under test. See issue #6042.
_FP8_EXPERTS_TEST_DEPS_AVAILABLE = False
if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    from quark.torch.utils.llm.module_replacement.quark_experts import _FP8Experts

    try:
        from transformers.integrations.finegrained_fp8 import ALL_FP8_EXPERTS_FUNCTIONS, FP8Experts
        from transformers.integrations.moe import use_experts_implementation

        from quark.torch.utils.llm.module_replacement import FP8ExpertLinear, QuarkFP8Experts

        _FP8_EXPERTS_TEST_DEPS_AVAILABLE = _FP8Experts is not None
    except ImportError:
        _FP8_EXPERTS_TEST_DEPS_AVAILABLE = False


@pytest.mark.parametrize("model_type", ["qwen3_5"])
def test_legacy_prepare_for_moe_quant_rejects_unsupported_model_type(model_type):
    class DummyConfig:
        pass

    cfg = DummyConfig()
    cfg.model_type = model_type

    model = nn.Linear(2, 2)
    model.config = cfg

    with pytest.raises(ValueError, match="not yet supported"):
        model_preparation._legacy_prepare_for_moe_quant(model)


def test_prepare_for_moe_quant_logs_deprecation_and_uses_legacy_path(monkeypatch):
    warning_messages = []
    legacy_call_args = {}

    monkeypatch.setattr(
        model_preparation,
        "is_transformers_version_higher_or_equal",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        model_preparation,
        "_legacy_prepare_for_moe_quant",
        lambda model, reload=False: legacy_call_args.update({"model": model, "reload": reload}),
    )
    monkeypatch.setattr(
        model_preparation.logger,
        "warning",
        lambda msg, *args, **kwargs: warning_messages.append(msg),
    )

    dummy_model = nn.Linear(2, 2)
    model_preparation.prepare_for_moe_quant(dummy_model, reload=True)

    assert legacy_call_args == {"model": dummy_model, "reload": True}
    assert len(warning_messages) == 1
    assert "deprecated" in warning_messages[0]
    assert "preprocess_for_quantization" in warning_messages[0]


@pytest.mark.skipif(
    not is_transformers_available() or not is_transformers_version_higher_or_equal("5.0.0"),
    reason="transformers >= 5.0.0 required for Qwen3Moe",
)
def test_quark_experts_imports_without_qwen3_5_moe(monkeypatch):
    """`qwen3_5_moe` only exists in recent transformers; its absence must not break the import.

    Re-executes `quark_experts` with that one import forced to fail (the state of an older
    transformers 5.x), which is what the `Qwen3_5MoeTopKRouter = None` fallback exists for.
    The module-level `register_quark_preprocess` calls mutate the shared
    `PREPROCESS_REGISTRY`, so it is snapshotted and restored around the re-execution.
    """

    missing_module = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing_module:
            raise ImportError(f"No module named '{missing_module}'")
        return real_import(name, *args, **kwargs)

    # A private module name: this copy must not become importable as `quark_experts`.
    spec = importlib.util.spec_from_file_location(
        "quark.torch.utils.llm.module_replacement._quark_experts_without_qwen3_5_moe",
        quark_experts.__file__,
    )
    module = importlib.util.module_from_spec(spec)

    registry_snapshot = dict(PREPROCESS_REGISTRY)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        spec.loader.exec_module(module)
    finally:
        monkeypatch.undo()
        PREPROCESS_REGISTRY.clear()
        PREPROCESS_REGISTRY.update(registry_snapshot)

    assert module.Qwen3_5MoeTopKRouter is None
    # No router class is defined, and nothing advertises one that doesn't exist.
    assert not hasattr(module, "QuarkQwen3_5MoeTopKRouter")
    assert "QuarkQwen3_5MoeTopKRouter" not in module.__all__
    # The rest of the module is unaffected.
    assert "QuarkQwen3MoeTopKRouter" in module.__all__


@pytest.mark.skipif(
    not is_transformers_available() or not is_transformers_version_higher_or_equal("5.0.0"),
    reason="transformers >= 5.0.0 required for Qwen3Moe",
)
class TestPreprocessForQuantization:
    """Test that preprocess_for_quantization works correctly."""

    def test_qwen3_moe(self):
        config_path = os.path.join(os.path.dirname(__file__), "configs", "moe_model", "qwen3")
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config=config)
        preprocess_for_quantization(model)

        found_module_replacement = False
        for _, module in model.named_modules():
            if isinstance(module, QuarkQwen3MoeTopKRouter | QuarkExperts):
                found_module_replacement = True
                break
        assert found_module_replacement, "Qwen3MoeTopKRouter or Qwen3MoeExperts not found in model"

    def test_qwen3_moe_gate_is_linear_with_no_naming_divergence(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

        config_path = os.path.join(os.path.dirname(__file__), "configs", "moe_model", "qwen3")
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config=config)
        preprocess_for_quantization(model)

        moe_blocks = [module for module in model.modules() if isinstance(module, Qwen3MoeSparseMoeBlock)]
        assert moe_blocks, "No Qwen3MoeSparseMoeBlock found in model"

        for moe_block in moe_blocks:
            assert isinstance(moe_block.gate, nn.Linear), (
                f"Expected `gate` to be an nn.Linear after preprocessing, got {type(moe_block.gate)}"
            )

        gate_linear_paths = [name for name, _ in model.named_modules() if name.endswith("gate.linear")]
        assert not gate_linear_paths, f"Found `.gate.linear` module paths (should not exist): {gate_linear_paths}"

        gate_linear_state_keys = [key for key in model.state_dict() if "gate.linear" in key]
        assert not gate_linear_state_keys, (
            f"Found `.gate.linear` state-dict keys (should not exist): {gate_linear_state_keys}"
        )

    def test_experts_accumulate_routed_outputs_in_hidden_state_dtype(self):
        """The per-architecture upstream experts loops accumulate in the hidden-state
        dtype, so the generic replacement must not silently promote to float32 (only
        `QuarkFP8Experts` does, mirroring its own upstream)."""
        config_path = os.path.join(os.path.dirname(__file__), "configs", "moe_model", "qwen3")
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config=config)
        preprocess_for_quantization(model)

        experts = [module for module in model.modules() if isinstance(module, QuarkExperts)]
        assert experts, "No QuarkExperts found in model"

        for dtype in (torch.bfloat16, torch.float32):
            hidden_states = torch.zeros(2, config.hidden_size, dtype=dtype)
            assert experts[0]._init_accumulator(hidden_states).dtype == dtype

    def test_moe_quant_skips_when_model_has_no_config(self):
        class ModelWithoutConfig(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = Qwen3MoeTopKRouter(Qwen3MoeConfig(hidden_size=8, num_experts=4, num_experts_per_tok=2))

        model = ModelWithoutConfig()
        preprocess_for_quantization(model)
        assert isinstance(model.router, Qwen3MoeTopKRouter)

    def test_moe_quant_raises_when_registered_replacement_fails(self):
        class DummyConfig:
            model_type = "qwen3_moe"

        class DummyHFModule(nn.Module):
            pass

        class DummyContainerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = DummyConfig()
                self.block = DummyHFModule()

        class BrokenReplacement(nn.Module):
            @classmethod
            def from_hf(cls, module, reload: bool = False):  # noqa: ARG003
                raise RuntimeError("intentional preprocess failure")

        model = DummyContainerModel()
        PREPROCESS_REGISTRY[DummyHFModule] = BrokenReplacement
        try:
            with pytest.raises(RuntimeError, match="intentional preprocess failure"):
                preprocess_for_quantization(model)
        finally:
            PREPROCESS_REGISTRY.pop(DummyHFModule, None)

    def test_preprocess_uses_legacy_path_when_transformers_lt5(self, monkeypatch):
        call_args = {}

        def fake_legacy(model, reload: bool = False):
            call_args["model"] = model
            call_args["reload"] = reload

        monkeypatch.setattr(model_preparation, "is_transformers_available", lambda: True)
        monkeypatch.setattr(
            model_preparation, "is_transformers_version_higher_or_equal", lambda *_args, **_kwargs: False
        )
        monkeypatch.setattr(model_preparation, "_legacy_prepare_for_moe_quant", fake_legacy)
        monkeypatch.setattr(
            model_preparation,
            "_prepare_for_moe_quant",
            lambda *_args, **_kwargs: pytest.fail("new preprocess path should not be used"),
        )

        dummy_model = nn.Linear(2, 2)
        preprocess_for_quantization(dummy_model, reload=True)
        assert call_args == {"model": dummy_model, "reload": True}


@pytest.mark.skipif(
    not _FP8_EXPERTS_TEST_DEPS_AVAILABLE,
    reason="transformers build without importable FP8Experts / ALL_FP8_EXPERTS_FUNCTIONS (see issue #6042)",
)
class TestFP8ExpertsPreprocess:
    """Regression test for issue #6042: FP8Experts (HF's fused FP8-native routed-expert
    container) must be registered for MoE preprocessing, or it silently passes through
    the standard (in-memory) quantization flow untouched."""

    def _build_fp8_experts(self, num_experts=4, hidden_size=64, intermediate_size=32, block=(32, 32), has_gate=True):
        config = SimpleNamespace(
            hidden_size=hidden_size,
            num_local_experts=num_experts,
            moe_intermediate_size=intermediate_size,
            hidden_activation="silu",
            swiglu_alpha=None,
            swiglu_limit=None,
            _experts_implementation=None,
        )
        config.dtype = torch.float32

        # Mirrors what `replace_with_fp8_linear` does at HF load time: mutate
        # `FP8Experts` in place (same class object) so `has_gate`/`is_transposed`
        # get set on instances, then construct it.
        new_class = use_experts_implementation(
            experts_class=FP8Experts, experts_interface=ALL_FP8_EXPERTS_FUNCTIONS, has_bias=False, has_gate=has_gate
        )
        assert new_class is FP8Experts
        experts = FP8Experts(
            config=config,
            block_size=block,
            activation_scheme="dynamic",
            has_bias=False,
            has_gate=has_gate,
        )
        with torch.no_grad():
            up_name = "gate_up_proj" if has_gate else "up_proj"
            up_weight = getattr(experts, up_name)
            up_weight.copy_(torch.randn_like(up_weight, dtype=torch.float32).to(torch.float8_e4m3fn))
            getattr(experts, f"{up_name}_scale_inv").uniform_(0.5, 1.5)
            experts.down_proj.copy_(torch.randn_like(experts.down_proj, dtype=torch.float32).to(torch.float8_e4m3fn))
            experts.down_proj_scale_inv.uniform_(0.5, 1.5)
        return experts

    def test_fp8_experts_is_registered(self):
        assert FP8Experts in PREPROCESS_REGISTRY, (
            "FP8Experts must be registered in PREPROCESS_REGISTRY, otherwise routed MoE "
            "experts loaded from an FP8-native checkpoint are silently never quantized "
            "(issue #6042)."
        )

    def test_fp8_experts_from_hf_produces_lazy_fp8_expert_linears(self):
        """Per-expert projections must stay FP8 (not eagerly dequantized), so an expert
        later excluded from quantization can be losslessly passed through at export
        (see FP8ExpertLinear docstring)."""
        experts = self._build_fp8_experts()
        # Reference dequantized values, computed independently of the fix, to check
        # against after replacement (per-expert block dequant).
        expected_gate_up = torch.empty(experts.gate_up_proj.shape, dtype=torch.float32)
        for i in range(experts.num_experts):
            expected_gate_up[i] = torch.ops.quark.dequantize_fp8_per_block(
                experts.gate_up_proj[i], experts.gate_up_proj_scale_inv[i], [32, 32]
            )

        source_gate_up = experts.gate_up_proj
        source_gate_up_scale_inv = experts.gate_up_proj_scale_inv
        source_down = experts.down_proj
        source_down_scale_inv = experts.down_proj_scale_inv
        replacement_class = PREPROCESS_REGISTRY[type(experts)]
        new_module = replacement_class.from_hf(experts, reload=False)

        assert isinstance(new_module, QuarkFP8Experts)
        assert new_module.num_experts == experts.num_experts

        for expert_idx in range(experts.num_experts):
            expert_module = getattr(new_module, str(expert_idx))
            assert isinstance(expert_module.gate_proj, FP8ExpertLinear)
            assert isinstance(expert_module.up_proj, FP8ExpertLinear)
            assert isinstance(expert_module.down_proj, FP8ExpertLinear)

            # Weight bytes must stay untouched in FP8 (no eager dequantization), so
            # is_prequantized_linear picks these up as passthrough-eligible, exactly
            # like a real FP8Linear.
            assert expert_module.gate_proj.weight.dtype == torch.float8_e4m3fn
            assert is_prequantized_linear(expert_module.gate_proj)
            assert is_prequantized_linear(expert_module.up_proj)
            assert is_prequantized_linear(expert_module.down_proj)

            assert (
                expert_module.gate_proj.weight.untyped_storage().data_ptr()
                == source_gate_up.untyped_storage().data_ptr()
            )
            assert (
                expert_module.gate_proj.weight_scale_inv.untyped_storage().data_ptr()
                == source_gate_up_scale_inv.untyped_storage().data_ptr()
            )
            assert (
                expert_module.down_proj.weight.untyped_storage().data_ptr() == source_down.untyped_storage().data_ptr()
            )
            assert (
                expert_module.down_proj.weight_scale_inv.untyped_storage().data_ptr()
                == source_down_scale_inv.untyped_storage().data_ptr()
            )

            gate_dequant = create_inverse_quantizer(expert_module.gate_proj).dequantize(expert_module.gate_proj.weight)
            up_dequant = create_inverse_quantizer(expert_module.up_proj).dequantize(expert_module.up_proj.weight)
            gate_weight = torch.cat([gate_dequant, up_dequant], dim=0)
            torch.testing.assert_close(gate_weight, expected_gate_up[expert_idx], rtol=1e-2, atol=1e-2)

    def test_fp8_experts_excluded_expert_survives_export_conversion(self):
        """An expert excluded from quantization must still convert cleanly to the
        export-time QParamsLinear passthrough wrapper (native FP8 passthrough)."""
        experts = self._build_fp8_experts()
        replacement_class = PREPROCESS_REGISTRY[type(experts)]
        new_module = replacement_class.from_hf(experts, reload=False)
        expert_module = getattr(new_module, "0")

        quark_config = convert_prequantized_module_to_quark_config(expert_module.gate_proj)
        assert quark_config is not None

        export_linear = QParamsLinear.from_module(
            linear=expert_module.gate_proj, custom_mode="quark", pack_method="reorder"
        )
        assert isinstance(export_linear, QParamsLinear)
        state_dict = export_linear.state_dict()
        assert "weight" in state_dict

    def test_fp8_experts_forward_runs_after_replacement(self):
        experts = self._build_fp8_experts()
        replacement_class = PREPROCESS_REGISTRY[type(experts)]
        new_module = replacement_class.from_hf(experts, reload=False)

        hidden_states = torch.randn(6, 64)
        top_k_index = torch.randint(0, experts.num_experts, (6, 2))
        top_k_weights = torch.rand(6, 2)
        out = new_module(hidden_states, top_k_index, top_k_weights)
        assert out.shape == hidden_states.shape
        assert torch.isfinite(out).all()

    def test_fp8_experts_forward_skips_sentinel_routes(self):
        experts = self._build_fp8_experts()
        replacement_class = PREPROCESS_REGISTRY[type(experts)]
        new_module = replacement_class.from_hf(experts, reload=False)

        hidden_states = torch.randn(6, 64)
        top_k_index = torch.tensor(
            [
                [experts.num_experts, 0],
                [1, experts.num_experts],
                [experts.num_experts, 2],
                [3, experts.num_experts],
                [experts.num_experts, 0],
                [1, experts.num_experts],
            ]
        )
        top_k_weights = torch.zeros(6, 2)
        top_k_weights[:, 1] = 1.0

        out = new_module(hidden_states, top_k_index, top_k_weights)
        assert out.shape == hidden_states.shape
        assert torch.isfinite(out).all()

    def test_fp8_experts_accumulate_routed_outputs_in_float32(self):
        """Upstream `FP8Experts.forward` accumulates routed expert outputs in float32
        and casts once at return, because `index_add_` accumulates in the dtype of the
        tensor written into. Inheriting the hidden-state-dtype accumulator would round
        the running sum once per routed expert on a bf16/fp16 model."""
        experts = self._build_fp8_experts()
        replacement_class = PREPROCESS_REGISTRY[type(experts)]
        new_module = replacement_class.from_hf(experts, reload=False)

        hidden_states = torch.randn(6, 64, dtype=torch.bfloat16)
        assert new_module._init_accumulator(hidden_states).dtype == torch.float32

        top_k_index = torch.randint(0, experts.num_experts, (6, 4))
        top_k_weights = torch.rand(6, 4, dtype=torch.bfloat16)
        out = new_module(hidden_states, top_k_index, top_k_weights)

        # The float32 accumulator must not leak into the MoE block's output dtype.
        assert out.dtype == hidden_states.dtype
        assert out.shape == hidden_states.shape
        assert torch.isfinite(out).all()

    def test_fp8_experts_non_gated_slices_up_proj_and_releases_fused_weights(self):
        """Non-gated FP8Experts store a single `up_proj` instead of a fused
        `gate_up_proj`, so slicing and the `from_hf` release path take their own branch."""
        experts = self._build_fp8_experts(has_gate=False)
        assert not experts.has_gate
        replacement_class = PREPROCESS_REGISTRY[type(experts)]
        new_module = replacement_class.from_hf(experts, reload=False)

        for expert_idx in range(new_module.num_experts):
            expert_module = getattr(new_module, str(expert_idx))
            assert not hasattr(expert_module, "gate_proj")
            assert isinstance(expert_module.up_proj, FP8ExpertLinear)
            assert isinstance(expert_module.down_proj, FP8ExpertLinear)
            assert expert_module.up_proj.weight.dtype == torch.float8_e4m3fn

        # `from_hf` must drop the fused source parameters, as it does for the gated layout.
        assert not hasattr(experts, "up_proj")
        assert not hasattr(experts, "up_proj_scale_inv")
        assert not hasattr(experts, "down_proj")

        hidden_states = torch.randn(6, 64)
        top_k_index = torch.randint(0, experts.num_experts, (6, 2))
        top_k_weights = torch.rand(6, 2)
        out = new_module(hidden_states, top_k_index, top_k_weights)
        assert out.shape == hidden_states.shape
        assert torch.isfinite(out).all()

    def test_fp8_expert_linear_registers_bias_when_present(self):
        """`FP8Experts` currently rejects biases upstream, but the per-expert slice must
        still register one when handed a bias so the layout matches `FP8Linear`."""
        weight = torch.randn(64, 32).to(torch.float8_e4m3fn)
        weight_scale_inv = torch.rand(2, 1) + 0.5
        bias = torch.randn(64)

        with_bias = FP8ExpertLinear(weight, weight_scale_inv, (32, 32), bias)
        assert with_bias.bias is not None
        torch.testing.assert_close(with_bias.bias, bias)
        assert with_bias(torch.randn(3, 32)).shape == (3, 64)

        without_bias = FP8ExpertLinear(weight, weight_scale_inv, (32, 32), None)
        assert without_bias.bias is None
