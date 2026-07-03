#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for shared global scale per parallel layers (Observer Sharing approach)."""

import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.common.utils.testing_utils import FROM_PRETRAINED_KWARGS, torch_device
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.template import LLMTemplate
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.model_transformation import (
    _get_per_tensor_quantizers,
    _has_glob_meta,
    setup_config_per_layer,
)
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.observer.observer import PlaceholderObserver
from quark.torch.quantization.tensor_quantize import SequentialQuantize
from quark.torch.utils.llm import preprocess_for_quantization

MODEL_NAME = "amd-quark/tiny-llama-fast-tokenizer"
torch.manual_seed(42)


def _get_dataloader(model_name=MODEL_NAME, device=torch_device):
    """Create a simple dataloader for calibration."""
    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    return DataLoader(tokenized_outputs["input_ids"].to(device))


def _get_weight_target_quantizer(quant_module):
    """Get the first per-tensor target quantizer from a QuantLinear's weight quantizer."""
    pt_qs = _get_per_tensor_quantizers(quant_module._weight_quantizer)
    return pt_qs[0] if pt_qs else None


# ──────────────────────────────────────────────────────────────────────
# Test 1: FP8 per-tensor with observer sharing
# ──────────────────────────────────────────────────────────────────────
class TestSharedScaleFP8:
    """Test observer sharing for FP8 per-tensor quantization."""

    def test_observers_shared_after_quantize(self):
        """After quantize_model, parallel layers should share the same observer object."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(torch_device)
        dataloader = _get_dataloader()
        template = LLMTemplate(model_type=model.config.model_type)

        config = template.get_config(
            scheme="fp8",
            shared_scale_groups=[["q_proj", "k_proj", "v_proj"], ["gate_proj", "up_proj"]],
        )
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model, dataloader)

        for layer_idx in range(len(quant_model.model.layers)):
            q_proj = quant_model.model.layers[layer_idx].self_attn.q_proj
            k_proj = quant_model.model.layers[layer_idx].self_attn.k_proj
            v_proj = quant_model.model.layers[layer_idx].self_attn.v_proj

            assert isinstance(q_proj, QuantLinear)
            assert isinstance(k_proj, QuantLinear)
            assert isinstance(v_proj, QuantLinear)

            # Quantizer objects are independent (NOT shared — this is Approach C)
            assert q_proj._weight_quantizer is not k_proj._weight_quantizer

            # But their observers ARE the same object
            q_obs = _get_weight_target_quantizer(q_proj).observer
            k_obs = _get_weight_target_quantizer(k_proj).observer
            v_obs = _get_weight_target_quantizer(v_proj).observer
            assert q_obs is k_obs
            assert k_obs is v_obs

            # gate_proj and up_proj share observer
            gate_proj = quant_model.model.layers[layer_idx].mlp.gate_proj
            up_proj = quant_model.model.layers[layer_idx].mlp.up_proj
            assert _get_weight_target_quantizer(gate_proj).observer is _get_weight_target_quantizer(up_proj).observer

    def test_scales_equal_after_quantize(self):
        """After quantize_model (which includes sync), all parallel layers should have identical scales."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(torch_device)
        dataloader = _get_dataloader()
        template = LLMTemplate(model_type=model.config.model_type)

        config = template.get_config(
            scheme="fp8",
            shared_scale_groups=[["q_proj", "k_proj", "v_proj"]],
        )
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model, dataloader)

        for layer_idx in range(len(quant_model.model.layers)):
            q_scale = _get_weight_target_quantizer(quant_model.model.layers[layer_idx].self_attn.q_proj).scale
            k_scale = _get_weight_target_quantizer(quant_model.model.layers[layer_idx].self_attn.k_proj).scale
            v_scale = _get_weight_target_quantizer(quant_model.model.layers[layer_idx].self_attn.v_proj).scale

            assert torch.equal(q_scale, k_scale), f"Layer {layer_idx}: q_proj.scale != k_proj.scale"
            assert torch.equal(k_scale, v_scale), f"Layer {layer_idx}: k_proj.scale != v_proj.scale"

    def test_non_shared_layers_independent(self):
        """o_proj and down_proj should NOT share observers with the shared group."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(torch_device)
        dataloader = _get_dataloader()
        template = LLMTemplate(model_type=model.config.model_type)

        config = template.get_config(
            scheme="fp8",
            shared_scale_groups=[["q_proj", "k_proj", "v_proj"], ["gate_proj", "up_proj"]],
        )
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model, dataloader)

        for layer_idx in range(len(quant_model.model.layers)):
            q_obs = _get_weight_target_quantizer(quant_model.model.layers[layer_idx].self_attn.q_proj).observer
            o_obs = _get_weight_target_quantizer(quant_model.model.layers[layer_idx].self_attn.o_proj).observer
            down_obs = _get_weight_target_quantizer(quant_model.model.layers[layer_idx].mlp.down_proj).observer

            assert q_obs is not o_obs
            assert q_obs is not down_obs

    def test_scales_equal_after_freeze(self):
        """After freeze, parallel layers should still have equal scale values with independent storage."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(torch_device)
        dataloader = _get_dataloader()
        template = LLMTemplate(model_type=model.config.model_type)

        config = template.get_config(
            scheme="fp8",
            shared_scale_groups=[["q_proj", "k_proj", "v_proj"]],
        )
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model, dataloader)
        frozen_model = ModelQuantizer.freeze(quant_model)

        for layer_idx in range(len(frozen_model.model.layers)):
            q_proj = frozen_model.model.layers[layer_idx].self_attn.q_proj
            k_proj = frozen_model.model.layers[layer_idx].self_attn.k_proj
            v_proj = frozen_model.model.layers[layer_idx].self_attn.v_proj

            # After freeze, quantizers are FrozenScaledFakeQuantize — independent objects
            assert q_proj._weight_quantizer is not k_proj._weight_quantizer

            # Scale values should be equal
            assert torch.equal(q_proj._weight_quantizer.scale, k_proj._weight_quantizer.scale)
            assert torch.equal(k_proj._weight_quantizer.scale, v_proj._weight_quantizer.scale)

            # Scale tensors should have independent storage (no tied weights)
            assert q_proj._weight_quantizer.scale.data_ptr() != k_proj._weight_quantizer.scale.data_ptr()
            assert k_proj._weight_quantizer.scale.data_ptr() != v_proj._weight_quantizer.scale.data_ptr()


# ──────────────────────────────────────────────────────────────────────
# Test 2: No sharing (default for non-NVFP4 schemes)
# ──────────────────────────────────────────────────────────────────────
class TestNoSharing:
    """Verify that without shared_scale_groups, all quantizers remain independent."""

    def test_independent_quantizers(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(torch_device)
        dataloader = _get_dataloader()
        template = LLMTemplate(model_type=model.config.model_type)

        config = template.get_config(scheme="fp8")  # No shared_scale_groups
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model, dataloader)

        for layer_idx in range(len(quant_model.model.layers)):
            q_proj = quant_model.model.layers[layer_idx].self_attn.q_proj
            k_proj = quant_model.model.layers[layer_idx].self_attn.k_proj
            v_proj = quant_model.model.layers[layer_idx].self_attn.v_proj

            # All quantizers and observers should be different
            assert q_proj._weight_quantizer is not k_proj._weight_quantizer
            q_obs = _get_weight_target_quantizer(q_proj).observer
            k_obs = _get_weight_target_quantizer(k_proj).observer
            v_obs = _get_weight_target_quantizer(v_proj).observer
            assert q_obs is not k_obs
            assert k_obs is not v_obs

        _ = ModelQuantizer.freeze(quant_model)


# ──────────────────────────────────────────────────────────────────────
# Test 3: Per-channel scheme gracefully skips sharing
# ──────────────────────────────────────────────────────────────────────
class TestPerChannelNoSharing:
    """Per-channel quantizers are not eligible for sharing; verify they remain independent."""

    def test_per_channel_not_shared(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(torch_device)
        dataloader = _get_dataloader()
        template = LLMTemplate(model_type=model.config.model_type)

        # ptpc_fp8 uses per-channel for weights — should not be shared
        config = template.get_config(
            scheme="ptpc_fp8",
            shared_scale_groups=[["q_proj", "k_proj", "v_proj"]],
        )
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model, dataloader)

        for layer_idx in range(len(quant_model.model.layers)):
            q_obs = _get_weight_target_quantizer(quant_model.model.layers[layer_idx].self_attn.q_proj)
            k_obs = _get_weight_target_quantizer(quant_model.model.layers[layer_idx].self_attn.k_proj)

            # per-channel → _get_per_tensor_quantizers returns [] → no sharing
            # The target quantizer should be None since it's per-channel
            assert q_obs is None or k_obs is None or q_obs.observer is not k_obs.observer


# ──────────────────────────────────────────────────────────────────────
# Test 4: SequentialQuantize (NVFP4) — only per-tensor observers shared
# ──────────────────────────────────────────────────────────────────────
class TestSharedScaleNVFP4:
    """Test observer sharing for NVFP4 two-stage quantization."""

    def test_sequential_quantize_observer_sharing(self):
        """For NVFP4, the per-tensor (scale) sub-quantizer's observer should be shared,
        while the per-group (FP4) sub-quantizer's observer should remain independent."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(torch_device)
        dataloader = _get_dataloader()
        template = LLMTemplate(model_type=model.config.model_type)

        config = template.get_config(
            scheme="nvfp4",
            shared_scale_groups=[["q_proj", "k_proj", "v_proj"]],
        )
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model, dataloader)

        for layer_idx in range(len(quant_model.model.layers)):
            q_proj = quant_model.model.layers[layer_idx].self_attn.q_proj
            k_proj = quant_model.model.layers[layer_idx].self_attn.k_proj
            v_proj = quant_model.model.layers[layer_idx].self_attn.v_proj

            # All should have SequentialQuantize weight quantizers
            assert isinstance(q_proj._weight_quantizer, SequentialQuantize)
            assert isinstance(k_proj._weight_quantizer, SequentialQuantize)
            assert isinstance(v_proj._weight_quantizer, SequentialQuantize)

            # SequentialQuantize objects themselves are independent
            assert q_proj._weight_quantizer is not k_proj._weight_quantizer

            # First-level (FP4 per-group) quantizers: independent objects, independent observers
            assert q_proj._weight_quantizer[0] is not k_proj._weight_quantizer[0]
            assert q_proj._weight_quantizer[0].observer is not k_proj._weight_quantizer[0].observer

            # Last-level (FP8 per-tensor scale) quantizers: independent objects, SHARED observer
            assert q_proj._weight_quantizer[-1] is not k_proj._weight_quantizer[-1]
            assert q_proj._weight_quantizer[-1].observer is k_proj._weight_quantizer[-1].observer
            assert k_proj._weight_quantizer[-1].observer is v_proj._weight_quantizer[-1].observer

    def test_sequential_quantize_scales_equal(self):
        """After quantize_model, the per-tensor scale quantizer should have identical scale values."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(torch_device)
        dataloader = _get_dataloader()
        template = LLMTemplate(model_type=model.config.model_type)

        config = template.get_config(
            scheme="nvfp4",
            shared_scale_groups=[["q_proj", "k_proj", "v_proj"]],
        )
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model, dataloader)

        for layer_idx in range(len(quant_model.model.layers)):
            q_weight_quantizer = quant_model.model.layers[layer_idx].self_attn.q_proj._weight_quantizer
            k_weight_quantizer = quant_model.model.layers[layer_idx].self_attn.k_proj._weight_quantizer
            v_weight_quantizer = quant_model.model.layers[layer_idx].self_attn.v_proj._weight_quantizer

            assert isinstance(q_weight_quantizer, SequentialQuantize)
            assert isinstance(k_weight_quantizer, SequentialQuantize)
            assert isinstance(v_weight_quantizer, SequentialQuantize)

            # After calibration, first-stage block scales are realigned and cached for export.
            assert q_weight_quantizer[0].scale.dtype == torch.float32
            assert k_weight_quantizer[0].scale.dtype == torch.float32
            assert v_weight_quantizer[0].scale.dtype == torch.float32
            assert hasattr(q_weight_quantizer[0], "_quantized_block_scale")
            assert hasattr(k_weight_quantizer[0], "_quantized_block_scale")
            assert hasattr(v_weight_quantizer[0], "_quantized_block_scale")

            q_scale = q_weight_quantizer[-1].scale
            k_scale = k_weight_quantizer[-1].scale
            v_scale = v_weight_quantizer[-1].scale

            assert torch.equal(q_scale, k_scale), f"Layer {layer_idx}: q scale != k scale"
            assert torch.equal(k_scale, v_scale), f"Layer {layer_idx}: k scale != v scale"

    def test_sequential_quantize_freeze(self):
        """After freeze, per-tensor scales should be equal with independent storage."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(torch_device)
        dataloader = _get_dataloader()
        template = LLMTemplate(model_type=model.config.model_type)

        config = template.get_config(
            scheme="nvfp4",
            shared_scale_groups=[["q_proj", "k_proj", "v_proj"]],
        )
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model, dataloader)
        frozen_model = ModelQuantizer.freeze(quant_model)

        for layer_idx in range(len(frozen_model.model.layers)):
            q_proj_quantizer = frozen_model.model.layers[layer_idx].self_attn.q_proj._weight_quantizer
            k_proj_quantizer = frozen_model.model.layers[layer_idx].self_attn.k_proj._weight_quantizer
            v_proj_quantizer = frozen_model.model.layers[layer_idx].self_attn.v_proj._weight_quantizer

            assert isinstance(q_proj_quantizer, SequentialQuantize)
            assert isinstance(k_proj_quantizer, SequentialQuantize)
            assert isinstance(v_proj_quantizer, SequentialQuantize)
            # SequentialQuantize stages are frozen during freeze().
            assert all(getattr(stage_quantizer, "frozen_params", None) is True for stage_quantizer in q_proj_quantizer)
            assert all(getattr(stage_quantizer, "frozen_params", None) is True for stage_quantizer in k_proj_quantizer)
            assert all(getattr(stage_quantizer, "frozen_params", None) is True for stage_quantizer in v_proj_quantizer)

            q_scale = q_proj_quantizer[-1].scale
            k_scale = k_proj_quantizer[-1].scale
            v_scale = v_proj_quantizer[-1].scale

            # Values equal
            assert torch.equal(q_scale, k_scale)
            assert torch.equal(k_scale, v_scale)

            # Storage independent (no tied weights for safetensors)
            assert q_scale.data_ptr() != k_scale.data_ptr()
            assert k_scale.data_ptr() != v_scale.data_ptr()


# ──────────────────────────────────────────────────────────────────────
# Test 5: MoE model — expert-internal sharing, cross-expert independence
# ──────────────────────────────────────────────────────────────────────
class TestMoEExpertSharing:
    """Test that gate_proj/up_proj share observer within each expert, but not across experts."""

    def test_moe_intra_expert_sharing(self):
        moe_model_name = "amd-quark/tiny-random-qwen3_moe"

        model = AutoModelForCausalLM.from_pretrained(moe_model_name, **FROM_PRETRAINED_KWARGS).eval().to(torch_device)

        preprocess_for_quantization(model)

        dataloader = _get_dataloader(moe_model_name)
        template = LLMTemplate(model_type=model.config.model_type)

        config = template.get_config(
            scheme="fp8",
            shared_scale_groups=[["gate_proj", "up_proj"]],
        )
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model, dataloader)

        for layer_idx in range(len(quant_model.model.layers)):
            layer_mlp = quant_model.model.layers[layer_idx].mlp

            # Within the same expert: observers should be shared
            for expert_idx in range(model.num_experts):
                expert = getattr(layer_mlp.experts, str(expert_idx))
                gate_obs = _get_weight_target_quantizer(expert.gate_proj).observer
                up_obs = _get_weight_target_quantizer(expert.up_proj).observer
                assert gate_obs is up_obs, f"Layer {layer_idx} expert {expert_idx}: observers not shared"

            # Across different experts: observers should NOT be shared
            for expert_idx in range(model.num_experts - 1):
                obs_0 = _get_weight_target_quantizer(getattr(layer_mlp.experts, str(expert_idx)).gate_proj).observer
                obs_1 = _get_weight_target_quantizer(getattr(layer_mlp.experts, str(expert_idx + 1)).gate_proj).observer
                assert obs_0 is not obs_1

    def test_moe_scales_equal_within_expert(self):
        moe_model_name = "amd-quark/tiny-random-qwen3_moe"

        model = AutoModelForCausalLM.from_pretrained(moe_model_name, **FROM_PRETRAINED_KWARGS).eval().to(torch_device)
        preprocess_for_quantization(model)

        dataloader = _get_dataloader(moe_model_name)
        template = LLMTemplate(model_type=model.config.model_type)

        config = template.get_config(
            scheme="fp8",
            shared_scale_groups=[["gate_proj", "up_proj"]],
        )
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model, dataloader)
        frozen_model = ModelQuantizer.freeze(quant_model)

        for layer_idx in range(len(frozen_model.model.layers)):
            layer_mlp = frozen_model.model.layers[layer_idx].mlp

            for expert_idx in range(model.num_experts):
                expert = getattr(layer_mlp.experts, str(expert_idx))
                gate_scale = expert.gate_proj._weight_quantizer.scale
                up_scale = expert.up_proj._weight_quantizer.scale

                # Same values
                assert torch.equal(gate_scale, up_scale), (
                    f"Layer {layer_idx} expert {expert_idx}: gate scale != up scale"
                )
                # Independent storage
                assert gate_scale.data_ptr() != up_scale.data_ptr()


# ──────────────────────────────────────────────────────────────────────
# Test 6: NVFP4 default shared_scale_groups
# ──────────────────────────────────────────────────────────────────────
class TestNVFP4Default:
    """Test that NVFP4 scheme automatically enables default shared_scale_groups."""

    def test_default_shared_groups(self):
        template = LLMTemplate(model_type="llama")
        config = template.get_config(scheme="nvfp4")
        assert config.shared_scale_groups == [["q_proj", "k_proj", "v_proj"], ["gate_proj", "up_proj"]]

    def test_non_nvfp4_no_default(self):
        template = LLMTemplate(model_type="llama")
        config = template.get_config(scheme="fp8")
        assert config.shared_scale_groups == []

    def test_explicit_override(self):
        template = LLMTemplate(model_type="llama")
        config = template.get_config(scheme="nvfp4", shared_scale_groups=[])
        assert config.shared_scale_groups == []

    def test_explicit_custom_groups(self):
        template = LLMTemplate(model_type="llama")
        config = template.get_config(scheme="fp8", shared_scale_groups=[["q_proj", "k_proj"]])
        assert config.shared_scale_groups == [["q_proj", "k_proj"]]


# ---------------------------------------------------------------------------
# setup_config_per_layer exact-name fast path
# ---------------------------------------------------------------------------


def test_setup_config_per_layer_exact_name_fast_path():
    """Exact-name excludes and layer_quant_config entries must short-circuit
    the fnmatch loop, and large exact-name exclude lists must finish quickly."""
    assert not _has_glob_meta("model.layers.0.mlp.gate_proj")
    assert _has_glob_meta("model.layers.*.mlp.gate_proj")

    tc = QTensorConfig(dtype=Dtype.float16, observer_cls=PlaceholderObserver)
    layer_cfg = QLayerConfig(input_tensors=tc, weight=tc)
    n = 2000
    named = {f"model.layers.{i}.mlp.gate_proj": torch.nn.Linear(8, 8, bias=False) for i in range(n)}

    # Exact-name exclude: must exclude every layer in well under a second.
    cfg = QConfig(global_quant_config=layer_cfg, exclude=list(named.keys()))
    out: dict = {}
    t0 = time.perf_counter()
    setup_config_per_layer(cfg, named, out)
    assert out == {} and time.perf_counter() - t0 < 1.0

    # Exact-name layer_quant_config: per-layer override wins; wildcard still works.
    override = QLayerConfig(input_tensors=tc, weight=tc)
    target = "model.layers.2.mlp.gate_proj"
    cfg = QConfig(global_quant_config=layer_cfg, layer_quant_config={target: override})
    out = {}
    setup_config_per_layer(cfg, named, out)
    assert out[target] is override and out["model.layers.0.mlp.gate_proj"] is layer_cfg

    cfg = QConfig(global_quant_config=layer_cfg, layer_quant_config={"model.layers.*.mlp.gate_proj": override})
    out = {}
    setup_config_per_layer(cfg, named, out)
    assert all(v is override for v in out.values())
