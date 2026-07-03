#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for ``quark.online_quantization.vllm`` (the vLLM online quant plugin).

Covers every file introduced by the online-quant plugin diff:

- ``utils.py``                   — ``quark_aligned_fp8_per_channel_quant``
                                   + ``materialize_meta_weight_`` (smoke)
- ``dequant.py``                 — ``dequant_layer`` + ``_dequant_fp8_block``
                                   + ``_dequant_fp8_per_channel``
                                   + ``dequant_fp8_block_per_expert``
- ``hf_quantization_configs.py`` — presets, ``_OnlineQuantHfOverride``,
                                   ``online_quant_overrides``,
                                   ``HF_QUANTIZATION_CONFIGS``
- ``quantization_config.py``     — ``QuarkVllmOnlineConfig`` proxy
- ``quant_method/linear.py``     — ``QuarkVllmOnlineFp8Method``,
                                   ``QuarkVllmOnlineMxfp4Method``,
                                   ``_quant_to_ocp_mxfp4``
- ``quant_method/moe.py``        — online MoE subclasses, helper allocators,
                                   per-expert quant ops, ``OnlineRequantMoeMethod``
- ``quant_method/requant.py``    — ``OnlineRequantMethod`` (Linear)
- ``quant_method/__init__.py``   — re-exports
- ``__init__.py``                — package re-exports

Tests that need a GPU (real fp8 quant kernels, MXFP4 packing) are guarded;
the rest run on CPU. No real model checkpoint is loaded — synthetic tensors
and ``SimpleNamespace`` "fake layers" exercise the dispatch.

Run:
    pytest test/test_for_torch/integrations/test_online_quantization_vllm.py -v
"""

import copy
import pickle
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

# vLLM is required for everything below. The directory-level conftest
# installs stub modules into ``sys.modules`` when real vLLM isn't available
# in the CI env, so this import always succeeds; tests that need real vLLM
# behavior carry the ``needs_real_vllm`` marker and auto-skip in stub mode.
vllm = pytest.importorskip("vllm")

needs_real_vllm = pytest.mark.needs_real_vllm

from quark.online_quantization.vllm import (  # noqa: E402
    HF_QUANTIZATION_CONFIGS,
    OnlineRequantMethod,
    QuarkVllmOnlineConfig,
    QuarkVllmOnlineFp8Method,
    QuarkVllmOnlineMxfp4Method,
    hf_quantization_config_linear_ptpc_fp8_moe_mxfp4,
    hf_quantization_config_mxfp4,
    hf_quantization_config_ptpc_fp8,
    online_quant_overrides,
)
from quark.online_quantization.vllm.dequant import (  # noqa: E402
    dequant_fp8_block_per_expert,
    dequant_layer,
)
from quark.online_quantization.vllm.hf_quantization_configs import (  # noqa: E402
    _OnlineQuantHfOverride,
)
from quark.online_quantization.vllm.quant_method.linear import (  # noqa: E402
    _quant_to_ocp_mxfp4,
)
from quark.online_quantization.vllm.quant_method.moe import (  # noqa: E402  # noqa: E402
    _OFFLINE_FP8_MOE_PARAM_NAMES,
    OnlineRequantMoeMethod,
    QuarkVllmOnlineFp8MoEMethod,
    QuarkVllmOnlineMxfp4MoEMethod,
    _online_moe_create_weights,
    _per_expert_fp8_quant_into_layer,
    _per_expert_mxfp4_quant_into_layer,
)
from quark.online_quantization.vllm.utils import (  # noqa: E402
    FP8_E4M3_MAX,
    quark_aligned_fp8_per_channel_quant,
)

# ---------------------------------------------------------------------------
# Skips
# ---------------------------------------------------------------------------


def _needs_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required for FP8/MXFP4 quant kernels")


# ---------------------------------------------------------------------------
# utils.py — quark_aligned_fp8_per_channel_quant
# ---------------------------------------------------------------------------


class TestQuarkAlignedFp8PerChannelQuant:
    def test_shape_and_dtype(self):
        torch.manual_seed(0)
        w = (torch.randn(8, 32) * 0.2).to(torch.bfloat16)
        qw, sc = quark_aligned_fp8_per_channel_quant(w)
        assert qw.shape == w.shape
        assert qw.dtype == torch.float8_e4m3fn
        assert sc.shape == (8,)
        assert sc.dtype == torch.float32

    def test_roundtrip_within_tolerance(self):
        torch.manual_seed(0)
        n, k = 16, 128
        w = (torch.randn(n, k) * 0.3).to(torch.bfloat16)
        qw, sc = quark_aligned_fp8_per_channel_quant(w)
        recon = qw.to(torch.float32) * sc.view(-1, 1).to(torch.float32)
        err = (recon - w.to(torch.float32)).abs().max().item()
        # ~0.1 is a generous bound for FP8 per-channel quant on random data
        assert err < 0.2

    def test_zero_row_does_not_div_by_zero(self):
        w = torch.zeros(4, 16, dtype=torch.bfloat16)
        # Add a non-zero row to make sure the zero row stays at 0
        w[2] = 1.5
        qw, sc = quark_aligned_fp8_per_channel_quant(w)
        # zero rows → scale also 0 (= amax/448)
        assert sc[0].item() == 0.0
        assert sc[1].item() == 0.0
        assert sc[3].item() == 0.0
        # zero row weights stay zero
        assert qw[0].to(torch.float32).abs().sum().item() == 0.0

    def test_scale_round_tripped_through_bf16(self):
        # The scale we return is the bf16-rounded value re-cast to f32.
        # Equivalent scales in bf16 must produce identical scale tensors.
        w = torch.full((2, 32), 0.1234, dtype=torch.bfloat16)
        _, sc = quark_aligned_fp8_per_channel_quant(w)
        # Scale value should be representable in bf16 exactly (= bf16(amax/448))
        expected = (w.abs().amax(dim=1).to(torch.float32) / FP8_E4M3_MAX).to(torch.bfloat16).to(torch.float32)
        torch.testing.assert_close(sc, expected)

    def test_fp8_max_constant(self):
        assert FP8_E4M3_MAX == 448.0


# ---------------------------------------------------------------------------
# dequant.py
# ---------------------------------------------------------------------------


def _fake_fp8_block_layer(n, k, block=(128, 128), dtype=torch.bfloat16):
    """Build a synthetic ``layer`` whose ``weight``/``weight_scale_inv``
    encode a known bf16 tensor in FP8-block format. Returns ``(layer, ref_bf16)``."""
    torch.manual_seed(0)
    ref = torch.randn(n, k, dtype=dtype) * 0.3
    b0, b1 = block
    fp8_w = torch.empty(n, k, dtype=torch.float8_e4m3fn)
    scale_inv = torch.empty((n + b0 - 1) // b0, (k + b1 - 1) // b1, dtype=torch.float32)
    f32 = ref.to(torch.float32)
    for i in range(0, n, b0):
        for j in range(0, k, b1):
            blk = f32[i : i + b0, j : j + b1]
            s = (blk.abs().amax() / FP8_E4M3_MAX).clamp_min(1e-12)
            scale_inv[i // b0, j // b1] = s
            fp8_w[i : i + b0, j : j + b1] = (blk / s).to(torch.float8_e4m3fn)
    layer = SimpleNamespace(
        weight=SimpleNamespace(data=fp8_w),
        weight_scale_inv=SimpleNamespace(data=scale_inv),
        orig_dtype=dtype,
    )
    return layer, ref


class TestDequantLayer:
    def test_dequant_layer_fp8_block(self):
        layer, ref = _fake_fp8_block_layer(64, 128, block=(128, 128))
        out = dequant_layer(layer, {"quant_method": "fp8", "weight_block_size": [128, 128]})
        assert out.shape == ref.shape
        assert out.dtype == torch.bfloat16
        # FP8 quant error bound (~0.04 typical, allow headroom)
        rel = (out.to(torch.float32) - ref.to(torch.float32)).abs().max() / (ref.to(torch.float32).abs().max() + 1e-9)
        assert rel.item() < 0.1

    def test_dequant_layer_fp8_per_channel(self):
        # Synthetic layer with per-channel scales (no weight_block_size)
        torch.manual_seed(0)
        ref = (torch.randn(8, 32) * 0.3).to(torch.bfloat16)
        amax = ref.abs().amax(dim=1, keepdim=True).to(torch.float32) / FP8_E4M3_MAX
        amax = amax.clamp_min(1e-12)
        fp8_w = (ref.to(torch.float32) / amax).to(torch.float8_e4m3fn)
        layer = SimpleNamespace(
            weight=SimpleNamespace(data=fp8_w),
            weight_scale=SimpleNamespace(data=amax.view(-1)),
            orig_dtype=torch.bfloat16,
        )
        out = dequant_layer(layer, {"quant_method": "fp8"})  # no block_size
        assert out.shape == ref.shape
        assert out.dtype == torch.bfloat16

    def test_dequant_layer_unsupported_method(self):
        layer = SimpleNamespace()
        with pytest.raises(NotImplementedError, match="offline scheme 'gptq'"):
            dequant_layer(layer, {"quant_method": "gptq"})

    def test_dequant_layer_empty_method_raises(self):
        layer = SimpleNamespace()
        with pytest.raises(NotImplementedError):
            dequant_layer(layer, {})

    def test_dequant_fp8_block_handles_nonpower_dims(self):
        # K=384 = 3*128, N=130 (last block partial along dim 0)
        layer, ref = _fake_fp8_block_layer(130, 384, block=(128, 128))
        out = dequant_layer(layer, {"quant_method": "fp8", "weight_block_size": [128, 128]})
        assert out.shape == (130, 384)

    def test_dequant_fp8_block_per_expert(self):
        # Make E independent block-quantized expert weights and dequant them
        e, n, k = 3, 64, 128
        torch.manual_seed(1)
        ref_e = (torch.randn(e, n, k) * 0.25).to(torch.bfloat16)
        fp8_e = torch.empty(e, n, k, dtype=torch.float8_e4m3fn)
        sc_inv = torch.empty(e, n // 128 + 1, k // 128, dtype=torch.float32)
        # Use a simpler per-expert block quant for the fixture
        for i in range(e):
            amax_blk = ref_e[i].abs().amax().to(torch.float32) / FP8_E4M3_MAX
            amax_blk = amax_blk.clamp_min(1e-12)
            sc_inv[i, :, :] = amax_blk
            fp8_e[i] = (ref_e[i].to(torch.float32) / amax_blk).to(torch.float8_e4m3fn)

        out = dequant_fp8_block_per_expert(fp8_e, sc_inv, block_shape=(128, 128), out_dtype=torch.bfloat16)
        assert out.shape == (e, n, k)
        assert out.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# hf_quantization_configs.py
# ---------------------------------------------------------------------------


class TestPresets:
    @pytest.mark.parametrize(
        "preset, expected_weight_dtype",
        [
            (hf_quantization_config_ptpc_fp8, "fp8_e4m3"),
            (hf_quantization_config_mxfp4, "fp4"),
        ],
    )
    def test_preset_shape_and_dtype(self, preset, expected_weight_dtype):
        for key in (
            "export",
            "exclude",
            "global_quant_config",
            "layer_quant_config",
            "layer_type_quant_config",
            "quant_method",
        ):
            assert key in preset, f"missing {key}"
        assert preset["quant_method"] == "quark_online"
        assert preset["global_quant_config"]["weight"]["dtype"] == expected_weight_dtype

    def test_mixed_preset_has_layer_override(self):
        # Current preset shape: global=MXFP4 (everything defaults to it),
        # ``*self_attn*`` overrides to FP8 per-channel. Matches the
        # checkpoint shape the preset is meant to emulate at load time.
        cfg = hf_quantization_config_linear_ptpc_fp8_moe_mxfp4
        assert cfg["global_quant_config"]["weight"]["dtype"] == "fp4"
        assert "*self_attn*" in cfg["layer_quant_config"]
        override = cfg["layer_quant_config"]["*self_attn*"]
        assert override["weight"]["dtype"] == "fp8_e4m3"
        assert override["input_tensors"]["dtype"] == "fp8_e4m3"


class TestOverridesCallable:
    def test_class_is_picklable(self):
        # vLLM's spawn engine pickles hf_overrides; closures break, classes don't.
        cb = HF_QUANTIZATION_CONFIGS["ptpc_fp8"]
        assert isinstance(cb, _OnlineQuantHfOverride)
        # Round-trip
        cb2 = pickle.loads(pickle.dumps(cb))
        assert isinstance(cb2, _OnlineQuantHfOverride)
        assert cb2._online_quant_cfg["quant_method"] == "quark_online"

    def test_scenario_a_no_offline(self):
        cb = online_quant_overrides(hf_quantization_config_ptpc_fp8)
        fake = SimpleNamespace(quantization_config=None)
        cb(fake)
        qc = fake.quantization_config
        assert qc["quant_method"] == "quark_online"
        assert "online_quant" in qc
        assert "offline_quant" not in qc

    def test_scenario_b_preserves_offline(self):
        cb = online_quant_overrides(hf_quantization_config_mxfp4)
        offline = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
        fake = SimpleNamespace(quantization_config=copy.deepcopy(offline))
        cb(fake)
        qc = fake.quantization_config
        assert qc["quant_method"] == "quark_online"
        assert qc["online_quant"]["global_quant_config"]["weight"]["dtype"] == "fp4"
        assert qc["offline_quant"] == offline

    def test_scenario_b_strips_existing_nested_keys_from_offline(self):
        # If the original config somehow already has online_quant/offline_quant
        # leftover, they shouldn't leak into the new offline_part.
        cb = online_quant_overrides(hf_quantization_config_mxfp4)
        leftover = {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "online_quant": {"junk": True},
            "offline_quant": {"junk": True},
        }
        fake = SimpleNamespace(quantization_config=copy.deepcopy(leftover))
        cb(fake)
        offline_part = fake.quantization_config["offline_quant"]
        assert "online_quant" not in offline_part
        assert "offline_quant" not in offline_part
        assert offline_part["quant_method"] == "fp8"

    def test_registry_has_expected_keys(self):
        assert set(HF_QUANTIZATION_CONFIGS.keys()) == {
            "ptpc_fp8",
            "mxfp4",
            "linear_ptpc_fp8_moe_mxfp4",
        }


# ---------------------------------------------------------------------------
# quantization_config.py — QuarkVllmOnlineConfig (proxy)
# ---------------------------------------------------------------------------


def _make_merged_cfg(online_dict, offline_dict=None):
    cfg = {"quant_method": "quark_online", "online_quant": online_dict}
    if offline_dict is not None:
        cfg["offline_quant"] = offline_dict
    return cfg


class TestQuarkVllmOnlineConfig:
    def test_from_config_scenario_a(self):
        qc = QuarkVllmOnlineConfig.from_config(_make_merged_cfg(hf_quantization_config_ptpc_fp8))
        assert qc._online_quant_config is not None
        assert qc._offline_quant_config is None

    def test_from_config_scenario_b(self):
        qc = QuarkVllmOnlineConfig.from_config(
            _make_merged_cfg(
                hf_quantization_config_mxfp4,
                offline_dict={
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "activation_scheme": "dynamic",
                    "weight_block_size": [128, 128],
                },
            )
        )
        assert qc._offline_quant_config is not None
        assert qc._offline_quant_config.get_name() == "fp8"

    def test_from_config_rejects_missing_online_quant(self):
        with pytest.raises(ValueError, match="online_quant"):
            QuarkVllmOnlineConfig.from_config({"quant_method": "quark_online"})

    def test_from_config_rejects_offline_without_quant_method(self):
        with pytest.raises(ValueError, match="quant_method"):
            QuarkVllmOnlineConfig.from_config(
                _make_merged_cfg(
                    hf_quantization_config_mxfp4,
                    offline_dict={"weight_block_size": [128, 128]},
                )
            )

    def test_get_name(self):
        assert QuarkVllmOnlineConfig.get_name() == "quark_online"

    def test_get_supported_act_dtypes(self):
        assert torch.bfloat16 in QuarkVllmOnlineConfig.get_supported_act_dtypes()
        assert torch.float16 in QuarkVllmOnlineConfig.get_supported_act_dtypes()

    def test_packed_modules_mapping_setter_propagates_to_inner(self):
        qc = QuarkVllmOnlineConfig.from_config(
            _make_merged_cfg(
                hf_quantization_config_ptpc_fp8,
                offline_dict={
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "activation_scheme": "dynamic",
                    "weight_block_size": [128, 128],
                },
            )
        )
        mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        qc.packed_modules_mapping = mapping
        assert qc.packed_modules_mapping is qc._online_quant_config.packed_modules_mapping
        assert qc._online_quant_config.packed_modules_mapping == mapping
        assert qc._offline_quant_config.packed_modules_mapping == mapping

    @needs_real_vllm
    def test_apply_vllm_mapper_forwards_to_both(self):
        # Use a fake mapper that records calls.
        class _RecordingMapper:
            def __init__(self):
                self.calls = 0

            def apply_list(self, xs):
                self.calls += 1
                return xs

            def apply_dict(self, xs):
                self.calls += 1
                return xs

        mapper = _RecordingMapper()
        qc = QuarkVllmOnlineConfig.from_config(
            _make_merged_cfg(
                hf_quantization_config_ptpc_fp8,
                offline_dict={
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "activation_scheme": "dynamic",
                    "weight_block_size": [128, 128],
                },
            )
        )
        qc.apply_vllm_mapper(mapper)
        # Both inner configs should have invoked the mapper at least once.
        assert mapper.calls > 0

    def test_get_cache_scale_prefers_offline(self):
        # Build a config with offline present; both methods return None for
        # unknown names → we should still terminate without exception.
        qc = QuarkVllmOnlineConfig.from_config(
            _make_merged_cfg(
                hf_quantization_config_ptpc_fp8,
                offline_dict={
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "activation_scheme": "dynamic",
                    "weight_block_size": [128, 128],
                },
            )
        )
        assert qc.get_cache_scale("some.random.name") is None


# ---------------------------------------------------------------------------
# quant_method/linear.py
# ---------------------------------------------------------------------------


class TestQuarkVllmOnlineFp8Method:
    def test_uses_meta_device(self):
        from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
            QuarkW8A8Fp8,
        )

        assert QuarkVllmOnlineFp8Method.uses_meta_device is True
        # Inheritance reuses QuarkScheme's per-channel finalize logic.
        assert issubclass(QuarkVllmOnlineFp8Method, QuarkW8A8Fp8)

    def test_per_channel_config_constants(self):
        # ``QuarkW8A8Fp8.__init__`` reads ``vllm_config.model_config.dtype``,
        # which means full instantiation requires a real vLLM config context
        # (too brittle to mock for a unit test). Instead, verify the scheme
        # config dicts our subclass feeds to the parent — these are what
        # drive the parent to pick ``kFp8StaticTokenSym`` (per-channel) over
        # the default ``kFp8StaticTensorSym``.
        from quark.online_quantization.vllm.quant_method.linear import (
            _PTPC_FP8_INPUT_CFG,
            _PTPC_FP8_WEIGHT_CFG,
        )

        assert _PTPC_FP8_WEIGHT_CFG["qscheme"] == "per_channel"
        assert _PTPC_FP8_WEIGHT_CFG["dtype"] == "fp8_e4m3"
        assert _PTPC_FP8_INPUT_CFG["qscheme"] == "per_channel"
        assert _PTPC_FP8_INPUT_CFG["is_dynamic"] is True


class TestQuarkVllmOnlineMxfp4Method:
    def test_uses_meta_device_and_inherits_quark_ocp_mx(self):
        from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
            QuarkOCP_MX,
        )

        assert QuarkVllmOnlineMxfp4Method.uses_meta_device is True
        # Inherits the offline OCP MX scheme for kernel-selection +
        # apply_weights; only the create_weights / quant op are the online
        # delta.
        assert issubclass(QuarkVllmOnlineMxfp4Method, QuarkOCP_MX)


@needs_real_vllm
class TestQuantToOcpMxfp4:
    def test_shapes(self):
        _needs_gpu()
        torch.manual_seed(0)
        n, k = 16, 64  # K must be divisible by OCP_MX_BLOCK_SIZE (32)
        w = (torch.randn(n, k) * 0.3).to(torch.bfloat16).cuda()
        packed, scale = _quant_to_ocp_mxfp4(w)
        assert packed.shape == (n, k // 2)
        assert packed.dtype == torch.uint8
        assert scale.shape == (n, k // 32)
        assert scale.dtype == torch.uint8

    def test_assert_on_unaligned_k(self):
        _needs_gpu()
        w = torch.empty(4, 33, dtype=torch.bfloat16).cuda()
        with pytest.raises(AssertionError):
            _quant_to_ocp_mxfp4(w)


# ---------------------------------------------------------------------------
# quant_method/moe.py
# ---------------------------------------------------------------------------


def _fake_moe_layer():
    layer = torch.nn.Module()
    layer.local_num_experts = 2
    return layer


class TestOnlineMoeCreateWeights:
    def test_meta_alloc_no_bias(self):
        layer = _fake_moe_layer()
        _online_moe_create_weights(
            layer,
            num_experts=2,
            hidden_size=32,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
            has_bias=False,
        )
        assert layer.num_experts == 2
        assert layer.orig_dtype == torch.bfloat16
        assert layer.weight_block_size is None
        assert layer.w13_weight.device.type == "meta"
        assert layer.w13_weight.shape == (2, 2 * 64, 32)
        assert layer.w13_weight.dtype == torch.bfloat16
        assert layer.w2_weight.device.type == "meta"
        assert layer.w2_weight.shape == (2, 32, 64)
        assert layer.w13_bias is None
        assert layer.w2_bias is None

    def test_meta_alloc_with_bias(self):
        layer = _fake_moe_layer()
        _online_moe_create_weights(
            layer,
            num_experts=2,
            hidden_size=32,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
            has_bias=True,
        )
        assert layer.w13_bias is not None
        assert layer.w13_bias.device.type == "meta"
        assert layer.w13_bias.shape == (2, 2 * 64)
        assert layer.w2_bias.shape == (2, 32)


@needs_real_vllm
class TestPerExpertQuantOps:
    def test_per_expert_fp8_quant_param_shapes(self):
        _needs_gpu()
        from vllm.platforms import current_platform

        e, hidden, inter = 2, 32, 64
        layer = _fake_moe_layer()
        layer.w13_weight = torch.nn.Parameter(
            (torch.randn(e, 2 * inter, hidden) * 0.2).to(torch.bfloat16).cuda(),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            (torch.randn(e, hidden, inter) * 0.2).to(torch.bfloat16).cuda(),
            requires_grad=False,
        )
        _per_expert_fp8_quant_into_layer(layer)
        # FP8 dtype depends on platform (e4m3fnuz on ROCm, e4m3fn on CUDA).
        fp8_dtype = current_platform.fp8_dtype()
        assert layer.w13_weight.dtype == fp8_dtype
        assert layer.w13_weight.shape == (e, 2 * inter, hidden)
        assert layer.w2_weight.dtype == fp8_dtype
        assert layer.w2_weight.shape == (e, hidden, inter)
        # Per-output-channel scales, 1D
        assert layer.w13_weight_scale.shape == (e, 2 * inter)
        assert layer.w2_weight_scale.shape == (e, hidden)
        assert layer.w13_input_scale is None
        assert layer.w2_input_scale is None

    def test_per_expert_mxfp4_quant_param_shapes(self):
        _needs_gpu()
        e, hidden, inter = 2, 64, 64  # both divisible by 32 for MXFP4 group
        layer = _fake_moe_layer()
        layer.w13_weight = torch.nn.Parameter(
            (torch.randn(e, 2 * inter, hidden) * 0.2).to(torch.bfloat16).cuda(),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            (torch.randn(e, hidden, inter) * 0.2).to(torch.bfloat16).cuda(),
            requires_grad=False,
        )
        _per_expert_mxfp4_quant_into_layer(layer)
        assert layer.w13_weight.dtype == torch.uint8
        assert layer.w13_weight.shape == (e, 2 * inter, hidden // 2)
        assert layer.w2_weight.dtype == torch.uint8
        assert layer.w2_weight.shape == (e, hidden, inter // 2)
        assert layer.w13_weight_scale.shape == (e, 2 * inter, hidden // 32)
        assert layer.w2_weight_scale.shape == (e, hidden, inter // 32)
        assert layer.w13_weight_scale.dtype == torch.uint8


class TestOnlineMoESubclassMRO:
    def test_online_fp8_moe_inherits_quark_w8a8_fp8_moe(self):
        from vllm.model_executor.layers.quantization.quark.quark_moe import (
            QuarkW8A8Fp8MoEMethod,
        )

        assert issubclass(QuarkVllmOnlineFp8MoEMethod, QuarkW8A8Fp8MoEMethod)
        assert QuarkVllmOnlineFp8MoEMethod.uses_meta_device is True

    def test_online_mxfp4_moe_inherits_ocp_mx_moe(self):
        from vllm.model_executor.layers.quantization.quark.quark_moe import (
            QuarkOCP_MX_MoEMethod,
        )

        assert issubclass(QuarkVllmOnlineMxfp4MoEMethod, QuarkOCP_MX_MoEMethod)
        assert QuarkVllmOnlineMxfp4MoEMethod.uses_meta_device is True

    def test_offline_fp8_moe_param_names_constant(self):
        # Used by OnlineRequantMoeMethod to drop the right set of offline params.
        for n in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
            assert n in _OFFLINE_FP8_MOE_PARAM_NAMES


class TestOnlineRequantMoeMethod:
    def test_construction_and_delegation_surface(self):
        # Build a synthetic offline + online pair, verify wrapper delegates.
        from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase

        offline = SimpleNamespace(moe=SimpleNamespace())  # only .moe needed by __init__
        online = SimpleNamespace()
        m = OnlineRequantMoeMethod(
            offline_moe=offline,
            online=online,
            offline_cfg={"quant_method": "fp8", "weight_block_size": [128, 128]},
        )
        assert m.offline_moe is offline
        assert m.online is online
        assert m.uses_meta_device is True
        assert isinstance(m, FusedMoEMethodBase)


# ---------------------------------------------------------------------------
# quant_method/requant.py — OnlineRequantMethod (Linear)
# ---------------------------------------------------------------------------


class TestOnlineRequantMethod:
    def test_construction(self):
        from vllm.model_executor.layers.linear import LinearMethodBase

        m = OnlineRequantMethod(
            offline=SimpleNamespace(),
            online=SimpleNamespace(),
            offline_cfg={"quant_method": "fp8", "weight_block_size": [128, 128]},
        )
        assert m.uses_meta_device is True
        assert isinstance(m, LinearMethodBase)

    def test_apply_delegates_to_online(self):
        recorded = {}

        class _OnlineSpy:
            def apply(self, layer, x, bias=None):
                recorded["called"] = (layer, x, bias)
                return torch.zeros(1)

        m = OnlineRequantMethod(
            offline=SimpleNamespace(),
            online=_OnlineSpy(),
            offline_cfg={"quant_method": "fp8"},
        )
        out = m.apply(layer="L", x="X", bias="B")
        assert recorded["called"] == ("L", "X", "B")
        assert out.shape == (1,)


# ---------------------------------------------------------------------------
# Package surface — quark/online_quantization/vllm/__init__.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Coverage fillers — dispatch + pipelines + small helpers
# ---------------------------------------------------------------------------


from vllm.model_executor.layers.linear import LinearBase as _LinearBase  # noqa: E402


class _FakeLinearBase(_LinearBase):
    """Real LinearBase subclass that skips the heavy distributed setup."""

    def __init__(self):
        torch.nn.Module.__init__(self)


# Context manager that gives instantiation of ``QuarkW8A8Fp8`` (and so our
# online subclasses) a vLLM config to read ``model_config.dtype`` from.
def _vllm_ctx():
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
    except ImportError:
        pytest.skip("vllm.config.set_current_vllm_config not importable")

    cfg = VllmConfig()
    # Some Quark schemes read ``vllm_config.model_config.dtype``; default the
    # field if it's missing on the synthetic config.
    if not hasattr(cfg, "model_config") or cfg.model_config is None:
        cfg.model_config = SimpleNamespace(dtype=torch.bfloat16)
    elif getattr(cfg.model_config, "dtype", None) is None:
        cfg.model_config.dtype = torch.bfloat16
    return set_current_vllm_config(cfg)


def _online_only_qc(preset):
    """Build a ``QuarkVllmOnlineConfig`` in scenario A (no offline)."""
    return QuarkVllmOnlineConfig.from_config(_make_merged_cfg(preset))


def _online_plus_fp8_block_qc(preset):
    """Build a ``QuarkVllmOnlineConfig`` in scenario B (offline = FP8 block)."""
    return QuarkVllmOnlineConfig.from_config(
        _make_merged_cfg(
            preset,
            offline_dict={
                "quant_method": "fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
            },
        )
    )


@needs_real_vllm
class TestBuildOnlineMethod:
    def test_picks_fp8_for_fp8_dtype(self):
        with _vllm_ctx():
            qc = _online_only_qc(hf_quantization_config_ptpc_fp8)
            matched = {"weight": {"dtype": "fp8_e4m3"}}
            m = qc._build_online_method(matched)
        assert isinstance(m, QuarkVllmOnlineFp8Method)

    def test_picks_mxfp4_for_fp4_dtype(self):
        with _vllm_ctx():
            qc = _online_only_qc(hf_quantization_config_mxfp4)
            matched = {"weight": {"dtype": "fp4"}}
            m = qc._build_online_method(matched)
        assert isinstance(m, QuarkVllmOnlineMxfp4Method)

    def test_defaults_to_fp8_for_unknown_dtype(self):
        with _vllm_ctx():
            qc = _online_only_qc(hf_quantization_config_ptpc_fp8)
            matched = {"weight": {"dtype": "made-up"}}
            m = qc._build_online_method(matched)
        assert isinstance(m, QuarkVllmOnlineFp8Method)


class TestOfflineDictHelper:
    def test_offline_dict_returns_inner(self):
        qc = _online_plus_fp8_block_qc(hf_quantization_config_mxfp4)
        d = qc._offline_dict()
        assert d is not None
        assert d.get("weight_block_size") == [128, 128]
        assert d.get("quant_method") == "fp8"

    def test_offline_dict_none_when_no_offline(self):
        qc = _online_only_qc(hf_quantization_config_ptpc_fp8)
        assert qc._offline_dict() is None


@needs_real_vllm
class TestGetQuantMethodDispatch:
    def test_excluded_linear_scenario_a_returns_unquantized(self):
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        with _vllm_ctx():
            qc = _online_only_qc(hf_quantization_config_ptpc_fp8)
            layer = _FakeLinearBase()
            # "lm_head" is in the preset's exclude list.
            out = qc.get_quant_method(layer, "lm_head")
        assert isinstance(out, UnquantizedLinearMethod)

    def test_linear_scenario_a_returns_online_method(self):
        with _vllm_ctx():
            qc = _online_only_qc(hf_quantization_config_ptpc_fp8)
            layer = _FakeLinearBase()
            out = qc.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
        assert isinstance(out, QuarkVllmOnlineFp8Method)

    def test_linear_scenario_b_returns_requant(self):
        with _vllm_ctx():
            qc = _online_plus_fp8_block_qc(hf_quantization_config_ptpc_fp8)
            layer = _FakeLinearBase()
            out = qc.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
        assert isinstance(out, OnlineRequantMethod)
        # Composes the offline FP8 method + our online FP8 method.
        assert isinstance(out.online, QuarkVllmOnlineFp8Method)

    def test_excluded_linear_scenario_b_returns_offline_not_requant(self):
        # When online excludes (lm_head) but offline is present, the proxy
        # delegates whatever offline.get_quant_method returns — NOT our
        # requant wrapper. Whether that's Unquantized or Fp8LinearMethod
        # depends on the offline config; the key invariant is no wrapping.
        with _vllm_ctx():
            qc = _online_plus_fp8_block_qc(hf_quantization_config_ptpc_fp8)
            layer = _FakeLinearBase()
            out = qc.get_quant_method(layer, "lm_head")
        assert not isinstance(out, OnlineRequantMethod)
        assert not isinstance(out, OnlineRequantMoeMethod)
        # And it's not our online method either (since online excluded it).
        assert not isinstance(out, QuarkVllmOnlineFp8Method)

    def test_non_linear_non_moe_returns_offline_method(self):
        # An arbitrary nn.Module (not LinearBase, not FusedMoE). Dispatch
        # should fall through to "return offline_method" without wrapping.
        with _vllm_ctx():
            qc = _online_plus_fp8_block_qc(hf_quantization_config_ptpc_fp8)

            class _Other(torch.nn.Module):
                pass

            out = qc.get_quant_method(_Other(), "model.layers.0.something")
        assert not isinstance(out, OnlineRequantMethod)
        assert not isinstance(out, OnlineRequantMoeMethod)


@needs_real_vllm
class TestOnlineRequantMethodPipeline:
    """End-to-end exercise of ``OnlineRequantMethod.process_weights_after_loading``
    on a synthetic FP8-block layer: dequant → drop offline params → stage bf16
    → call ``online.process_weights_after_loading``.
    """

    def test_pipeline_with_spy_online(self):
        # Build a synthetic FP8-block layer
        layer, _ref = _fake_fp8_block_layer(64, 128, block=(128, 128))
        # Real torch.nn.Module + Parameter for delattr to work cleanly
        real = torch.nn.Module()
        real.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        real.weight_scale_inv = torch.nn.Parameter(layer.weight_scale_inv.data, requires_grad=False)
        real.orig_dtype = torch.bfloat16
        real._offline_param_names = ["weight", "weight_scale_inv"]

        called = {}

        class _OnlineSpy:
            def process_weights_after_loading(self, layer):
                called["layer.weight.dtype"] = layer.weight.dtype
                called["layer.weight.shape"] = tuple(layer.weight.shape)

            def apply(self, layer, x, bias=None):
                return None

        offline = SimpleNamespace()
        m = OnlineRequantMethod(
            offline=offline,
            online=_OnlineSpy(),
            offline_cfg={"quant_method": "fp8", "weight_block_size": [128, 128]},
        )
        # Skip the layerwise hook side-effects; pwal is called directly.
        m.process_weights_after_loading(real)

        # By the time ``online.process_weights_after_loading`` ran, the layer
        # should have a bf16 weight (dequant result) and the offline params
        # should have been dropped.
        assert called["layer.weight.dtype"] == torch.bfloat16
        assert called["layer.weight.shape"] == (64, 128)
        assert not hasattr(real, "weight_scale_inv")

    def test_pipeline_idempotent(self):
        # Second call short-circuits via _already_called guard.
        real = torch.nn.Module()
        real._already_called_process_weights_after_loading = True
        m = OnlineRequantMethod(
            offline=SimpleNamespace(),
            online=SimpleNamespace(
                process_weights_after_loading=lambda layer: pytest.fail("online.pwal should NOT be re-invoked"),
            ),
            offline_cfg={"quant_method": "fp8"},
        )
        m.process_weights_after_loading(real)  # no-op, no fail


class TestMaterializeMetaWeight:
    def test_swaps_meta_for_real_tensor(self):
        # Both constructing a ``ModelWeightParameter`` and the internal call
        # to ``initialize_single_dummy_weight`` need a vLLM TP process group.
        # Skip cleanly — a full pipeline test would need to spin up a
        # single-rank TP group via ``init_distributed_environment``.
        pytest.skip(
            "materialize_meta_weight_ pipeline test needs vLLM TP initialization (single-rank distributed group)"
        )

    def test_no_op_when_weight_is_not_meta(self):
        from quark.online_quantization.vllm.utils import materialize_meta_weight_

        layer = torch.nn.Module()
        real = torch.nn.Parameter(torch.zeros(3, 5), requires_grad=False)
        layer.register_parameter("weight", real)
        materialize_meta_weight_(layer)
        # Same tensor identity preserved.
        assert layer.weight is real


@needs_real_vllm
class TestGetCacheScale:
    def test_returns_offline_value_when_offline_has_mapping(self):
        # Build a config + monkey-patch the offline's get_cache_scale to
        # return a known value; proxy must surface it.
        qc = _online_plus_fp8_block_qc(hf_quantization_config_ptpc_fp8)
        qc._offline_quant_config.get_cache_scale = lambda name: (
            "remapped.k_scale" if name.endswith(".kv_scale") else None
        )
        assert qc.get_cache_scale("model.layers.0.kv_scale") == "remapped.k_scale"
        assert qc.get_cache_scale("nothing.matches") is None


# ---------------------------------------------------------------------------
# hf_quantization_configs.py — ATOM-style thin schema adapter
# ---------------------------------------------------------------------------


from quark.online_quantization.vllm.hf_quantization_configs import (  # noqa: E402
    online_quant_config_to_quark,
)


class TestOnlineQuantConfigToQuark:
    def test_ptpc_fp8_global(self):
        out = online_quant_config_to_quark({"global_quant_config": "ptpc_fp8"})
        assert out["quant_method"] == "quark_online"
        assert out["global_quant_config"]["weight"]["dtype"] == "fp8_e4m3"
        assert out["global_quant_config"]["weight"]["qscheme"] == "per_channel"
        # default exclude
        assert out["exclude"] == ["lm_head"]
        # envelope present
        assert out["layer_quant_config"] == {}
        assert out["quant_mode"] == "eager_mode"

    def test_mxfp4_global(self):
        out = online_quant_config_to_quark({"global_quant_config": "mxfp4"})
        w = out["global_quant_config"]["weight"]
        assert w["dtype"] == "fp4"
        assert w["group_size"] == 32
        assert w["qscheme"] == "per_group"

    @pytest.mark.parametrize(
        ("cfg", "expected_exclude"),
        [
            (
                {"global_quant_config": "ptpc_fp8", "exclude_layer": "lm_head"},
                ["lm_head"],
            ),
            (
                {"global_quant_config": "mxfp4", "exclude_layer": ["lm_head", "embed_tokens"]},
                ["lm_head", "embed_tokens"],
            ),
            (
                {"global_quant_config": "mxfp4", "exclude_layer": ["re:.*\\.gate\\..*"]},
                ["re:.*\\.gate\\..*"],
            ),
            (
                {"global_quant_config": "mxfp4", "exclude_layer": ""},
                [],
            ),
        ],
    )
    def test_exclude_layer_normalization(self, cfg, expected_exclude):
        out = online_quant_config_to_quark(cfg)
        assert out["exclude"] == expected_exclude

    def test_exclude_layer_glob_is_converted_to_regex(self):
        out = online_quant_config_to_quark({"global_quant_config": "mxfp4", "exclude_layer": ["lm_head", "*.gate.*"]})
        assert out["exclude"][0] == "lm_head"
        assert out["exclude"][1].startswith("re:")

    def test_mixed_layer_override(self):
        out = online_quant_config_to_quark(
            {
                "global_quant_config": "mxfp4",
                "layer_quant_config": {"*self_attn*": "ptpc_fp8"},
                "exclude_layer": ["lm_head"],
            }
        )
        assert out["global_quant_config"]["weight"]["dtype"] == "fp4"
        override = out["layer_quant_config"]["*self_attn*"]
        assert override["weight"]["dtype"] == "fp8_e4m3"
        assert override["input_tensors"]["dtype"] == "fp8_e4m3"
        # only weight + input_tensors carried in a per-layer override
        assert set(override.keys()) == {"weight", "input_tensors"}

    @pytest.mark.parametrize(
        ("cfg", "exc", "match"),
        [
            ({"global_quant_config": "bogus"}, ValueError, "Unsupported online quant format"),
            (
                {"global_quant_config": "mxfp4", "layer_quant_config": {"*self_attn*": "bogus"}},
                ValueError,
                "Unsupported online quant format",
            ),
            ({}, ValueError, "global_quant_config"),
            ("ptpc_fp8", TypeError, None),
        ],
    )
    def test_invalid_configs_raise(self, cfg, exc, match):
        if match is None:
            with pytest.raises(exc):
                online_quant_config_to_quark(cfg)
        else:
            with pytest.raises(exc, match=match):
                online_quant_config_to_quark(cfg)

    def test_deepcopied_not_aliased_to_preset(self):
        # Mutating the output must not corrupt the shared preset blocks.
        out = online_quant_config_to_quark({"global_quant_config": "ptpc_fp8"})
        out["global_quant_config"]["weight"]["dtype"] = "MUTATED"
        assert hf_quantization_config_ptpc_fp8["global_quant_config"]["weight"]["dtype"] == "fp8_e4m3"


# ---------------------------------------------------------------------------
# plugin.py — hf_overrides injection from additional_config + priority
# ---------------------------------------------------------------------------


from quark.online_quantization.vllm import plugin as _plugin  # noqa: E402


class TestPluginInjectHfOverrides:
    def _engine_args(self, additional_config, hf_overrides=None):
        # EngineArgs.hf_overrides defaults to {} (empty dict = "not set").
        return SimpleNamespace(
            additional_config=additional_config,
            hf_overrides={} if hf_overrides is None else hf_overrides,
        )

    @staticmethod
    def _assert_cb_builds_quark_online(cb, expected_weight_dtype: str):
        fake = SimpleNamespace(quantization_config=None)
        cb(fake)
        assert fake.quantization_config["quant_method"] == "quark_online"
        assert (
            fake.quantization_config["online_quant"]["global_quant_config"]["weight"]["dtype"] == expected_weight_dtype
        )

    def test_injects_when_present_and_no_explicit(self):
        ea = self._engine_args({"online_quant_config": {"global_quant_config": "ptpc_fp8"}})
        _plugin._maybe_inject_hf_overrides(ea)
        assert isinstance(ea.hf_overrides, _OnlineQuantHfOverride)
        self._assert_cb_builds_quark_online(ea.hf_overrides, "fp8_e4m3")

    def test_extract_present(self):
        online_quant_cfg = {"global_quant_config": "ptpc_fp8"}
        assert (
            _plugin._online_quant_config_from_additional_config({"online_quant_config": online_quant_cfg})
            is online_quant_cfg
        )

    def test_extract_absent_returns_none(self):
        assert _plugin._online_quant_config_from_additional_config({"other": 1}) is None
        assert _plugin._online_quant_config_from_additional_config(None) is None
        assert _plugin._online_quant_config_from_additional_config({}) is None

    def test_build_callable_end_to_end(self):
        cb = _plugin._build_hf_overrides_from_online_quant_config({"global_quant_config": "ptpc_fp8"})
        assert isinstance(cb, _OnlineQuantHfOverride)
        self._assert_cb_builds_quark_online(cb, "fp8_e4m3")

    def test_built_callable_is_picklable(self):
        cb = _plugin._build_hf_overrides_from_online_quant_config({"global_quant_config": "mxfp4"})
        cb2 = pickle.loads(pickle.dumps(cb))
        assert isinstance(cb2, _OnlineQuantHfOverride)

    @pytest.mark.parametrize("additional_config", [None, {}, {"something_else": 1}])
    def test_noop_when_no_online_quant_config(self, additional_config):
        ea = self._engine_args(additional_config)
        _plugin._maybe_inject_hf_overrides(ea)
        assert ea.hf_overrides == {}

    def test_explicit_hf_overrides_wins(self):
        # A user-supplied --hf-overrides (truthy) must not be clobbered.
        sentinel = {"some": "override"}
        ea = self._engine_args(
            {"online_quant_config": {"global_quant_config": "ptpc_fp8"}},
            hf_overrides=sentinel,
        )
        _plugin._maybe_inject_hf_overrides(ea)
        assert ea.hf_overrides is sentinel

    def test_explicit_callable_hf_overrides_wins(self):
        def _user_override(cfg):
            return cfg

        ea = self._engine_args(
            {"online_quant_config": {"global_quant_config": "ptpc_fp8"}},
            hf_overrides=_user_override,
        )
        _plugin._maybe_inject_hf_overrides(ea)
        assert ea.hf_overrides is _user_override

    def test_has_explicit_hf_overrides_detection(self):
        assert _plugin._has_explicit_hf_overrides(SimpleNamespace(hf_overrides={"a": 1}))
        assert _plugin._has_explicit_hf_overrides(SimpleNamespace(hf_overrides=lambda c: c))
        assert not _plugin._has_explicit_hf_overrides(SimpleNamespace(hf_overrides={}))
        assert not _plugin._has_explicit_hf_overrides(SimpleNamespace(hf_overrides=None))

    def test_injected_override_preserves_offline_for_requant(self):
        # The injected callable must still stash a pre-existing offline cfg.
        ea = self._engine_args({"online_quant_config": {"global_quant_config": "mxfp4"}})
        _plugin._maybe_inject_hf_overrides(ea)
        offline = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        fake = SimpleNamespace(quantization_config=copy.deepcopy(offline))
        ea.hf_overrides(fake)
        assert fake.quantization_config["quant_method"] == "quark_online"
        assert fake.quantization_config["offline_quant"] == offline
        assert "online_quant" in fake.quantization_config

    def test_malformed_config_raises_in_helper(self):
        # The helper raises; the patched create_model_config wraps it in try/except.
        ea = self._engine_args({"online_quant_config": {"global_quant_config": "bogus"}})
        with pytest.raises(ValueError):
            _plugin._maybe_inject_hf_overrides(ea)

    def test_disabled_env(self, monkeypatch):
        monkeypatch.setenv("QUARK_DISABLE_VLLM_PLUGIN", "1")
        assert _plugin._disabled() is True
        monkeypatch.setenv("QUARK_DISABLE_VLLM_PLUGIN", "0")
        assert _plugin._disabled() is False
        monkeypatch.delenv("QUARK_DISABLE_VLLM_PLUGIN", raising=False)
        assert _plugin._disabled() is False

    def test_register_idempotent_and_patches_engine_args(self, monkeypatch):
        monkeypatch.delenv("QUARK_DISABLE_VLLM_PLUGIN", raising=False)

        class EngineArgs:
            def create_model_config(self):
                return "model-config"

        engine_mod = ModuleType("vllm.engine")
        arg_utils_mod = ModuleType("vllm.engine.arg_utils")
        arg_utils_mod.EngineArgs = EngineArgs
        engine_mod.__path__ = []
        engine_mod.arg_utils = arg_utils_mod
        monkeypatch.setitem(sys.modules, "vllm.engine", engine_mod)
        monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils_mod)

        orig = EngineArgs.create_model_config
        try:
            _plugin.register()
            p1 = EngineArgs.create_model_config
            assert getattr(p1, _plugin._PATCHED_FLAG, False)
            _plugin.register()
            p2 = EngineArgs.create_model_config
            assert p1 is p2  # idempotent — not double-wrapped
        finally:
            EngineArgs.create_model_config = orig


class TestPackageSurface:
    def test_public_exports(self):
        import quark.online_quantization.vllm as pkg

        for name in [
            "QuarkVllmOnlineConfig",
            "QuarkVllmOnlineFp8Method",
            "QuarkVllmOnlineMxfp4Method",
            "OnlineRequantMethod",
            "HF_QUANTIZATION_CONFIGS",
            "hf_quantization_config_ptpc_fp8",
            "hf_quantization_config_mxfp4",
            "online_quant_overrides",
            "online_quant_config_to_quark",
        ]:
            assert hasattr(pkg, name), f"missing public export {name}"

    def test_quark_online_registered_with_vllm(self):
        # Importing the package triggers @register_quantization_config("quark_online").
        from vllm.model_executor.layers.quantization import get_quantization_config

        cls = get_quantization_config("quark_online")
        assert cls is QuarkVllmOnlineConfig
