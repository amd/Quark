#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for ``quark.experimental.torch.mix_precision.perf_roofline``.

Two layers of coverage:

* Bottom-up operator formulas (GEMM / FusedMoE / SDPA / Gated Delta linear
  attention) — FLOPs and HBM bytes checked against hand-computed values, plus
  the W/A/O dtype resolution and compute-peak selection that drive the roofline.
* An end-to-end decode + prefill roofline on a Qwen3.5-35B-A3B-shaped meta
  (hybrid linear/full attention MoE), asserting the physical ordering of the
  ceilings rather than exact throughput.

Tests use CPU arithmetic or meta-device model loading: no GPU and no weight
materialization/download.
"""

from __future__ import annotations

import math
import os
from contextlib import suppress
from functools import lru_cache

import pytest
import torch
import torch.nn as nn

from quark.common.utils.import_utils import is_transformers_version_higher_or_equal
from quark.experimental.torch.mix_precision import perf_roofline as pr


# =============================================================================
# GEMM formulas
# =============================================================================
class TestGemmFormulas:
    def test_flops_is_two_mnk(self):
        assert pr._gemm_flops(4, 8, 16) == 2 * 4 * 8 * 16

    def test_bytes_separates_input_weight_output(self):
        # M*K*act + K*N*weight + M*N*out
        got = pr._gemm_bytes(4, 8, 16, weight_bpe=1.0, act_bpe=2.0, output_bpe=2.0)
        assert got == 4 * 16 * 2.0 + 16 * 8 * 1.0 + 4 * 8 * 2.0  # 320

    def test_bytes_defaults_act_and_output_to_weight(self):
        # act_bpe None -> weight_bpe; output_bpe None -> input_bytes(=act)
        got = pr._gemm_bytes(4, 8, 16, weight_bpe=1.0)
        assert got == 4 * 16 * 1.0 + 16 * 8 * 1.0 + 4 * 8 * 1.0  # 224

    def test_output_bpe_independent_of_input(self):
        # fp8 KV write: output narrower than the bf16 input read.
        got = pr._gemm_bytes(2, 2, 2, weight_bpe=1.0, act_bpe=2.0, output_bpe=1.0)
        assert got == 2 * 2 * 2.0 + 2 * 2 * 1.0 + 2 * 2 * 1.0


# =============================================================================
# FusedMoE formulas
# =============================================================================
class TestFusedMoEFormulas:
    def test_flops_gate_up_down_aggregation(self):
        M, K, N, topk = 2, 4, 8, 2
        gate_and_up = 2.0 * M * K * N * topk * 2
        down = 2.0 * M * K * N * topk
        aggregation = M * K * (2 * topk - 1)
        assert pr._fused_moe_flops(M, K, N, topk) == gate_and_up + down + aggregation

    def test_bytes_uses_coupon_active_experts(self):
        M, K, N, E, topk = 2, 4, 8, 4, 2
        active = E * (1.0 - ((E - topk) / E) ** M)  # 4 * (1 - 0.25) = 3.0
        assert active == pytest.approx(3.0)
        got = pr._fused_moe_bytes(M, K, N, E, topk, weight_bpe=1.0, act_bpe=2.0, output_bpe=2.0)
        expected = M * K * 2.0 + active * N * K * 1.0 * 2 + active * N * K * 1.0 + M * K * 2.0
        assert got == pytest.approx(expected)  # 16 + 192 + 96 + 16 = 320

    def test_active_experts_saturates_with_batch(self):
        # Large M -> almost all experts touched; small M -> ~topk experts.
        E, topk = 8, 2
        few = pr._fused_moe_bytes(1, 4, 4, E, topk, weight_bpe=1.0)
        many = pr._fused_moe_bytes(1024, 4, 4, E, topk, weight_bpe=1.0)
        # Weight term grows with active experts as batch grows.
        assert many > few


# =============================================================================
# SDPA formulas — the qk/v head-dim split (MLA) lives here.
# =============================================================================
class TestSdpaFormulas:
    def test_decode_flops_qk_plus_pv(self):
        # Non-causal decode: N_Q=1 attends to full KV.
        B, N_Q, H_Q, N_KV, H_KV, d_qk, d_v = 1, 1, 4, 10, 2, 192, 128
        qk = B * H_Q * (2.0 * N_Q * N_KV * d_qk)
        pv = B * H_Q * (2.0 * N_Q * d_v * N_KV)
        assert pr._sdpa_flops(B, N_Q, H_Q, N_KV, H_KV, d_qk, d_v, causal=False) == qk + pv

    def test_asymmetric_qk_v_increases_flops_vs_symmetric(self):
        # DeepSeek MLA: qk=192 > v=128. Modeling both as v=128 underestimates.
        args = dict(B=1, N_Q=1, H_Q=4, N_KV=10, H_KV=2, causal=False)
        mla = pr._sdpa_flops(**args, d_h_qk=192, d_h_v=128)
        collapsed = pr._sdpa_flops(**args, d_h_qk=128, d_h_v=128)
        assert mla > collapsed

    def test_causal_halves_only_when_square(self):
        square = dict(B=1, N_Q=8, H_Q=4, N_KV=8, H_KV=2, d_h_qk=64, d_h_v=64)
        full = pr._sdpa_flops(**square, causal=False)
        causal = pr._sdpa_flops(**square, causal=True)
        assert causal == pytest.approx(full / 2.0)
        # Decode (N_Q != N_KV) is never halved even if causal is requested.
        dec = dict(B=1, N_Q=1, H_Q=4, N_KV=8, H_KV=2, d_h_qk=64, d_h_v=64)
        assert pr._sdpa_flops(**dec, causal=True) == pr._sdpa_flops(**dec, causal=False)

    def test_bytes_reads_qkv_writes_output(self):
        B, N_Q, H_Q, N_KV, H_KV, d_qk, d_v = 1, 1, 4, 10, 2, 192, 128
        got = pr._sdpa_bytes(B, N_Q, H_Q, N_KV, H_KV, d_qk, d_v, causal=False, bpe=2.0, kv_bpe=1.0, output_bpe=2.0)
        expected = (
            B * N_Q * H_Q * d_qk * 2.0  # Q read
            + B * N_KV * H_KV * d_qk * 1.0  # K read (fp8 KV)
            + B * N_KV * H_KV * d_v * 1.0  # V read (fp8 KV)
            + B * N_Q * H_Q * d_v * 2.0  # output write
        )
        assert got == pytest.approx(expected)  # 1536 + 3840 + 2560 + 1024 = 8960


# =============================================================================
# Gated-Delta linear-attention formulas.
# =============================================================================
class TestLinearAttentionFormulas:
    def test_decode_flops_and_bytes(self):
        # B=2, S=1, H=2, HV=4, K=3, V=5.
        assert pr._linear_attention_flops(2, 1, 2, 4, 3, 5) == 1064.0
        assert pr._linear_attention_bytes(2, 1, 2, 4, 3, 5, 2.0, 2.0) == 1248.0

    def test_prefill_reuses_state_across_sequence(self):
        # Token tensors scale with S, but one FP32 recurrent-state read/write is
        # shared by the whole ideal prefill chunk.
        one = pr._linear_attention_bytes(1, 1, 2, 4, 3, 5, 2.0, 2.0)
        many = pr._linear_attention_bytes(1, 8, 2, 4, 3, 5, 2.0, 2.0)
        assert many < one * 8


# =============================================================================
# Compute-peak selection — B-flops fix: tag follows the input activation dtype.
# =============================================================================
class TestComputeTagForActivation:
    @pytest.mark.parametrize(
        "input_bpe, tag",
        [(0.5, "mxfp4"), (0.75, "fp8"), (1.0, "fp8"), (2.0, "bf16"), (4.0, "fp32")],
    )
    def test_tag_thresholds(self, input_bpe, tag):
        assert pr._compute_tag_for_activation(input_bpe) == tag

    def test_w4a8_uses_fp8_compute_rate_not_mxfp4(self):
        # mxfp4_fp8 activation is fp8 (1.0 bpe): compute runs at the fp8 ceiling.
        table = pr.HW_ACHIEVABLE_TFLOPS["mi355x"]
        assert pr._resolve_achievable_tflops("mi355x", 1.0) == table["fp8"]
        assert pr._resolve_achievable_tflops("mi355x", 1.0) != table["mxfp4"]

    def test_mxfp4_falls_back_to_bf16_when_target_lacks_it(self):
        # MI300X has no mxfp4 achievable entry -> bf16 degradation path.
        assert pr._resolve_achievable_tflops("mi300x", 0.5) == pr.HW_ACHIEVABLE_TFLOPS["mi300x"]["bf16"]
        # MI355X does have mxfp4.
        assert pr._resolve_achievable_tflops("mi355x", 0.5) == pr.HW_ACHIEVABLE_TFLOPS["mi355x"]["mxfp4"]


# =============================================================================
# W/A/O dtype resolution — B-bytes fix: input bytes track native activation.
# =============================================================================
class TestResolveOpDtypeInfo:
    def test_input_bytes_follow_native_activation_not_weight(self):
        # fp32 source (native_activation_bpe=4), fp8 weights. Input is READ at
        # the fp32 activation width, NOT min(weight, native)=1 floored to bf16.
        _cfg, info = pr._resolve_op_dtype_info("fp8", native_weight_bpe=4.0, native_activation_bpe=4.0)
        assert info.input_bpe == 4.0
        # Weight is still quantized (fp8) and capped so quant never inflates it.
        assert info.weight_bpe == 1.0
        # Compute width of the input stays the quant dtype (fp8) for FLOPs.
        assert info.input_compute_bpe == 1.0

    def test_bf16_source_input_is_bf16(self):
        _cfg, info = pr._resolve_op_dtype_info("fp8", native_weight_bpe=2.0, native_activation_bpe=2.0)
        assert info.input_bpe == 2.0

    def test_native_mode_is_all_native(self):
        cfg, info = pr._resolve_op_dtype_info("native", native_weight_bpe=4.0, native_activation_bpe=4.0)
        assert cfg is None
        assert info.weight_bpe == 4.0
        assert info.input_bpe == 4.0
        assert info.output_bpe == 4.0


# =============================================================================
# Geometry — head_dim split (A).
# =============================================================================
class TestHeadDimGeometry:
    def test_mla_splits_qk_and_v(self):
        cfg = {
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "num_hidden_layers": 4,
        }
        geo = pr._kv_geometry_from_config(cfg)
        assert geo["qk_head_dim"] == 192  # 128 + 64
        assert geo["v_head_dim"] == 128

    def test_standard_model_shares_one_head_dim(self):
        cfg = {"hidden_size": 4096, "num_attention_heads": 32, "head_dim": 256, "num_hidden_layers": 4}
        geo = pr._kv_geometry_from_config(cfg)
        assert geo["head_dim"] == 256
        assert geo["qk_head_dim"] == 256
        assert geo["v_head_dim"] == 256

    def test_qwen35_hybrid_layer_and_linear_head_geometry(self):
        cfg = {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        }
        geo = pr._kv_geometry_from_config(cfg)
        assert geo["full_attn_layers"] == 10
        assert geo["linear_attn_layers"] == 30
        assert geo["linear_num_key_heads"] == 16
        assert geo["linear_num_value_heads"] == 32
        assert geo["linear_key_head_dim"] == 128
        assert geo["linear_value_head_dim"] == 128


# =============================================================================
# End-to-end decode + prefill roofline on a Qwen3.5-35B-A3B-shaped meta.
# =============================================================================
def _qwen35b_a3b_meta(*, native_dtype_bytes: float = 2.0) -> pr.ModelPerfMeta:
    """Build a ModelPerfMeta with real Qwen3.5-35B-A3B geometry.

    Hybrid model: 40 layers (10 full-attention + 30 linear-attention), 256
    routed experts top-8, MoE FFN in every layer. Byte budgets are derived
    from the shapes so the meta is internally consistent at ``native_dtype_bytes``.
    """
    hidden = 2048
    num_layers = 40
    full_attn_layers = 10
    linear_attn_layers = 30
    n_heads = 16
    n_kv = 2
    head_dim = 256
    linear_n_key_heads = 16
    linear_n_value_heads = 32
    linear_key_head_dim = 128
    linear_value_head_dim = 128
    num_experts = 256
    topk = 8
    moe_inter = 512
    vocab = 248320
    wbpe = native_dtype_bytes

    attn_out = n_heads * head_dim  # 4096
    q_proj_out = 2 * attn_out  # Qwen3.5 emits query + output gate.
    kv_out = n_kv * head_dim  # 512
    attn_numel_per_layer = hidden * q_proj_out + hidden * kv_out * 2 + attn_out * hidden
    self_attn_numel = attn_numel_per_layer * full_attn_layers

    linear_key_dim = linear_n_key_heads * linear_key_head_dim  # 2048
    linear_value_dim = linear_n_value_heads * linear_value_head_dim  # 4096
    linear_qkv_out = 2 * linear_key_dim + linear_value_dim  # 8192
    linear_attn_numel_per_layer = (
        hidden * linear_qkv_out
        + hidden * linear_value_dim  # z
        + hidden * linear_n_value_heads * 2  # a + b
        + linear_value_dim * hidden  # out
    )
    linear_attn_numel = linear_attn_numel_per_layer * linear_attn_layers

    # Routed experts: gate + up + down per expert per layer.
    expert_numel_per_layer = num_experts * 3 * hidden * moe_inter
    moe_expert_numel = expert_numel_per_layer * num_layers

    linear_ops = (
        pr.LinearOpMeta("q_proj", "self_attn", hidden, q_proj_out, wbpe, repeat=full_attn_layers),
        pr.LinearOpMeta("k_proj", "self_attn", hidden, kv_out, wbpe, repeat=full_attn_layers),
        pr.LinearOpMeta("v_proj", "self_attn", hidden, kv_out, wbpe, repeat=full_attn_layers),
        pr.LinearOpMeta("o_proj", "self_attn", attn_out, hidden, wbpe, repeat=full_attn_layers),
        pr.LinearOpMeta("in_proj_qkv", "linear_attn", hidden, linear_qkv_out, wbpe, repeat=linear_attn_layers),
        pr.LinearOpMeta("in_proj_z", "linear_attn", hidden, linear_value_dim, wbpe, repeat=linear_attn_layers),
        pr.LinearOpMeta("in_proj_a", "linear_attn", hidden, linear_n_value_heads, wbpe, repeat=linear_attn_layers),
        pr.LinearOpMeta("in_proj_b", "linear_attn", hidden, linear_n_value_heads, wbpe, repeat=linear_attn_layers),
        pr.LinearOpMeta("out_proj", "linear_attn", linear_value_dim, hidden, wbpe, repeat=linear_attn_layers),
        pr.LinearOpMeta("lm_head", "other", hidden, vocab, wbpe, repeat=1),
    )

    return pr.ModelPerfMeta(
        native_dtype_bytes=native_dtype_bytes,
        num_layers=num_layers,
        hidden_size=hidden,
        num_kv_heads=n_kv,
        head_dim=head_dim,
        full_attn_layers=full_attn_layers,
        self_attn_numel=self_attn_numel,
        linear_attn_numel=linear_attn_numel,
        mlp_dense_numel=0,
        moe_expert_numel=moe_expert_numel,
        self_attn_bytes_native=int(self_attn_numel * wbpe),
        linear_attn_bytes_native=int(linear_attn_numel * wbpe),
        mlp_dense_bytes_native=0,
        moe_expert_bytes_native=int(moe_expert_numel * wbpe),
        other_bytes_native=int(vocab * hidden * wbpe * 2),  # embed + lm_head
        num_experts=num_experts,
        experts_per_tok=topk,
        num_attention_heads=n_heads,
        moe_intermediate_size=moe_inter,
        moe_layers=num_layers,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        linear_attn_layers=linear_attn_layers,
        linear_num_key_heads=linear_n_key_heads,
        linear_num_value_heads=linear_n_value_heads,
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_ops=linear_ops,
    )


class TestEndToEndRoofline35B:
    def _score(self, mlp_mode="native", **overrides):
        meta = _qwen35b_a3b_meta()
        config = {
            "self_attn_mode": overrides.get("self_attn_mode", "native"),
            "mlp_mode": mlp_mode,
            "kv_cache_mode": overrides.get("kv_cache_mode", "native"),
            "attention_mode": overrides.get("attention_mode", "native"),
        }
        return pr.compute_perf_score(
            meta,
            config,
            gpu_type="mi300x",
            num_gpus=1,
            batch=1,
            isl=8192,
            osl=1024,
            include_prefill=True,
        )

    def test_uses_op_model_and_positive_ceilings(self):
        s = self._score()
        assert s.uses_op_model is True
        assert s.selected_source == "op_roofline"
        assert s.peak_tok_per_sec > 0
        assert s.prefill_tok_per_sec > 0
        assert math.isfinite(s.peak_tok_per_sec)

    def test_prefill_throughput_exceeds_decode(self):
        # Prefill processes ISL tokens in parallel -> far higher tok/s than the
        # one-token-at-a-time decode.
        s = self._score()
        assert s.prefill_tok_per_sec > s.peak_tok_per_sec

    def test_decode_memory_bound_prefill_compute_bound(self):
        s = self._score()
        assert s.bound_kind == "memory"
        assert s.prefill_bound_kind == "compute"

    def test_op_breakdown_covers_gemm_moe_full_and_linear_attention(self):
        s = self._score()
        op_types = {op.op_type for op in s.ops}
        assert {"gemm", "fused_moe", "sdpa", "linear_attention"} <= op_types
        assert sum(op.name == "linear_attn_decode" for op in s.ops) == 1
        # Per-op time fractions sum to 1 across the decode forward.
        assert sum(op.pct_time for op in s.ops) == pytest.approx(1.0, abs=1e-6)

    def test_prefill_breakdown_present(self):
        s = self._score()
        assert s.prefill_ops
        assert sum(op.name == "linear_attn_prefill" for op in s.prefill_ops) == 1
        assert sum(op.pct_time for op in s.prefill_ops) == pytest.approx(1.0, abs=1e-6)

    def test_quantizing_mlp_speeds_up_decode(self):
        # Decode is weight-memory bound; fp8 experts halve the dominant MoE IO.
        native = self._score(mlp_mode="native")
        fp8 = self._score(mlp_mode="fp8")
        assert fp8.peak_tok_per_sec > native.peak_tok_per_sec


# =============================================================================
# Shared experts get their own partition so the op roofline honours
# ``shared_expert_mode``; otherwise FP8 and native collapse to the same score.
# =============================================================================
class TestSharedExpertPartition:
    def test_partition_mode_reads_shared_expert_mode(self):
        assert pr._partition_mode({"shared_expert_mode": "fp8"}, "shared_expert") == "fp8"
        assert pr._partition_mode({}, "shared_expert") == "native"

    def test_is_shared_expert_name_detection(self):
        assert pr._is_shared_expert("model.layers.0.mlp.shared_expert.down_proj")
        assert pr._is_shared_expert("model.layers.0.mlp.shared_experts.gate_proj")
        assert not pr._is_shared_expert("model.layers.0.mlp.experts.0.down_proj")

    def _shared_expert_meta(self, wbpe=2.0):
        hidden, inter, layers = 4096, 14336, 24
        shared_ops = (
            pr.LinearOpMeta("gate_proj", "shared_expert", hidden, inter, wbpe, repeat=layers),
            pr.LinearOpMeta("up_proj", "shared_expert", hidden, inter, wbpe, repeat=layers),
            pr.LinearOpMeta("down_proj", "shared_expert", inter, hidden, wbpe, repeat=layers),
        )
        return pr.ModelPerfMeta(
            native_dtype_bytes=wbpe,
            num_layers=layers,
            hidden_size=hidden,
            num_kv_heads=8,
            head_dim=128,
            full_attn_layers=0,
            self_attn_numel=0,
            linear_attn_numel=0,
            mlp_dense_numel=0,
            moe_expert_numel=0,
            self_attn_bytes_native=0,
            linear_attn_bytes_native=0,
            mlp_dense_bytes_native=0,
            moe_expert_bytes_native=0,
            other_bytes_native=0,
            num_experts=0,
            experts_per_tok=0,
            linear_ops=shared_ops,
        )

    def test_fp8_shared_expert_speeds_up_decode(self):
        # The regression the reviewer flagged: without a shared_expert partition,
        # fp8 and native resolve identically. They must now differ.
        meta = self._shared_expert_meta()
        native = pr.compute_op_roofline(meta, {"shared_expert_mode": "native"}, gpu_type="mi300x")
        fp8 = pr.compute_op_roofline(meta, {"shared_expert_mode": "fp8"}, gpu_type="mi300x")
        assert native is not None and fp8 is not None
        assert fp8.decode_tok_per_s > native.decode_tok_per_s
        assert all(op.partition == "shared_expert" for op in fp8.ops)
        assert all(op.mode == "fp8" for op in fp8.ops)


# =============================================================================
# Shared-expert exclusion: a shared expert excluded from QConfig must stay at
# native precision in the roofline. The meta byte-bucketing has to mirror the
# QConfig exclusion (like every other partition), otherwise the scorer shrinks
# bytes that quantization never touches and overstates performance.
# =============================================================================
class _ToyConfig:
    model_type = "default"

    def to_dict(self) -> dict[str, int]:
        return {"hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2}


class _ToyLayer(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Module()
        self.mlp.shared_expert = nn.Module()
        self.mlp.shared_expert.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.mlp.shared_expert.up_proj = nn.Linear(hidden, inter, bias=False)
        self.mlp.shared_expert.down_proj = nn.Linear(inter, hidden, bias=False)


class _ToyMoEModel(nn.Module):
    """Two-layer model with one shared expert per layer, walked by the meta builder."""

    def __init__(self, hidden: int = 8, inter: int = 16, layers: int = 2):
        super().__init__()
        self.config = _ToyConfig()
        self.layers = nn.ModuleList(_ToyLayer(hidden, inter) for _ in range(layers))


class TestSharedExpertExclusion:
    def _meta(self, exclude_patterns):
        model = _ToyMoEModel()
        meta = pr.load_model_perf_meta_from_model(model, exclude_patterns=exclude_patterns, model_type="default")
        assert meta is not None
        return meta

    def test_included_shared_expert_is_scalable(self):
        meta = self._meta(exclude_patterns=[])
        # 2 layers x (gate 8*16 + up 8*16 + down 16*8) = 768 elems in the bucket.
        assert meta.shared_expert_numel == 768
        assert meta.shared_expert_bytes_native > 0

    def test_fully_excluded_shared_expert_folds_into_native_other(self):
        full = self._meta(exclude_patterns=[])
        excl = self._meta(exclude_patterns=["*shared_expert*"])
        # No scalable shared-expert bytes remain ...
        assert excl.shared_expert_numel == 0
        assert excl.shared_expert_bytes_native == 0
        # ... they moved verbatim into the always-native "other" budget.
        assert excl.other_bytes_native == full.other_bytes_native + full.shared_expert_bytes_native
        # And an fp8 config can no longer shrink them: bytes are mode-invariant.
        native_b, _ = pr.compute_effective_weight_bytes(excl, {"shared_expert_mode": "native"})
        fp8_b, _ = pr.compute_effective_weight_bytes(excl, {"shared_expert_mode": "fp8"})
        assert fp8_b == native_b

    def test_partial_exclusion_scales_only_the_surviving_shared_expert(self):
        full = self._meta(exclude_patterns=[])
        # Exclude layer 0's shared expert only; layer 1 stays quantizable.
        meta = self._meta(exclude_patterns=["*layers.0.*shared_expert*"])
        assert meta.shared_expert_bytes_native == full.shared_expert_bytes_native // 2
        assert meta.other_bytes_native == full.other_bytes_native + full.shared_expert_bytes_native // 2
        # fp8 shrinks the surviving (layer 1) shared expert but leaves the
        # excluded layer-0 bytes (now in "other") at native precision.
        _, br_native = pr.compute_effective_weight_bytes(meta, {"shared_expert_mode": "native"})
        _, br_fp8 = pr.compute_effective_weight_bytes(meta, {"shared_expert_mode": "fp8"})
        assert br_fp8["shared_expert"] < br_native["shared_expert"]
        assert br_fp8["other"] == br_native["other"]


class TestCompressedLinearStorageAccounting:
    def test_mixed_formats_use_exact_storage_once_per_linear(self):
        class Config:
            model_type = "default"

            @staticmethod
            def to_dict():
                return {"hidden_size": 4, "num_hidden_layers": 1, "num_attention_heads": 1}

        model = nn.Module()
        model.config = Config()
        model.layers = nn.ModuleList([nn.Module()])
        model.layers[0].mlp = nn.Module()
        model.layers[0].mlp.shared_expert = nn.Module()
        gate = nn.Linear(4, 8, bias=False)
        up = nn.Linear(4, 8, bias=False)
        model.layers[0].mlp.shared_expert.gate_proj = gate
        model.layers[0].mlp.shared_expert.up_proj = up

        for linear in (gate, up):
            del linear.weight
            linear.register_parameter(
                "weight_shape",
                nn.Parameter(torch.tensor([8, 4], dtype=torch.int64), requires_grad=False),
            )

        # Logical 8x4 MXFP4 weight: 16 packed bytes + one byte of scale.
        gate.register_parameter("weight_packed", nn.Parameter(torch.empty(16, dtype=torch.uint8), requires_grad=False))
        gate.register_parameter("weight_scale", nn.Parameter(torch.empty(1, dtype=torch.uint8), requires_grad=False))

        # Logical 8x4 FP8 weight: 32 bytes + one float32 per-channel scale.
        up.register_parameter("weight_packed", nn.Parameter(torch.empty(32, dtype=torch.uint8), requires_grad=False))
        up.register_parameter("weight_scale", nn.Parameter(torch.empty(8, dtype=torch.float32), requires_grad=False))

        meta = pr.load_model_perf_meta_from_model(model, exclude_patterns=[], model_type="default")
        assert meta is not None
        assert meta.shared_expert_numel == 64
        assert meta.shared_expert_bytes_native == 16 + 1 + 32 + 8 * 4
        # Packed weights/scales/shape metadata were consumed by their Linear;
        # none may fall through and be counted again as non-Linear state.
        assert meta.other_bytes_native == 0
        assert {op.native_weight_bpe for op in meta.linear_ops if op.partition == "shared_expert"} == {
            17 / 32,
            64 / 32,
        }

    def test_weightless_linear_without_compressed_storage_fails_closed(self):
        model = _ToyMoEModel(hidden=4, inter=8, layers=1)
        del model.layers[0].self_attn.q_proj.weight

        with pytest.raises(ValueError, match="no materialized or compressed runtime weight storage"):
            pr.load_model_perf_meta_from_model(model, exclude_patterns=[], model_type="default")


# =============================================================================
# Real pipeline: meta-device load -> quark preprocess -> meta extraction ->
# roofline, mirroring MixPrecisionQuantizer.search(). Loads Qwen3.5-35B-A3B on
# the meta device (no weights materialized), so it exercises the model walker
# and expert unpacking that the hand-built meta above bypasses.
# =============================================================================
MODEL_PATH = "/group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3.5-35B-A3B"


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="CI-specific model not available")
@pytest.mark.skipif(
    not is_transformers_version_higher_or_equal("5.2.0"),
    reason="Qwen3.5 model_type=qwen3_5_moe requires transformers >= 5.2.0",
)
class TestPipelineRooflineQwen35B:
    @staticmethod
    @lru_cache(maxsize=1)
    def _load_meta():
        from quark.experimental.torch.mix_precision.run_helpers import load_transformers_model
        from quark.torch.utils.llm import preprocess_for_quantization

        # Meta device: builds the module skeleton without materializing weights.
        model = load_transformers_model(MODEL_PATH, torch_dtype="auto", device_map="meta")
        assert all(param.is_meta for param in model.parameters())
        with suppress(ValueError):
            preprocess_for_quantization(model)  # unpacks fused MoE experts -> per-expert nn.Linear
        return pr.load_model_perf_meta_from_model(model, exclude_patterns=None)

    def test_meta_geometry_matches_config(self):
        meta = self._load_meta()
        assert meta is not None
        # Config-derived geometry (Qwen3.5-35B-A3B config.json).
        assert meta.hidden_size == 2048
        assert meta.num_layers == 40
        assert meta.full_attn_layers == 10  # 10 full + 30 linear attention
        assert meta.linear_attn_layers == 30
        assert meta.num_attention_heads == 16
        assert meta.num_kv_heads == 2
        assert meta.head_dim == 256
        # Not MLA -> QK and V share head_dim.
        assert meta.qk_head_dim == 256
        assert meta.v_head_dim == 256
        assert meta.linear_num_key_heads == 16
        assert meta.linear_num_value_heads == 32
        assert meta.linear_key_head_dim == 128
        assert meta.linear_value_head_dim == 128
        assert meta.num_experts == 256
        assert meta.experts_per_tok == 8
        assert meta.moe_intermediate_size == 512
        # Walker-derived: experts must be unpacked and counted.
        assert meta.moe_layers > 0
        assert meta.moe_expert_numel > 0
        assert meta.linear_ops  # attention / router / lm_head projections

    def test_decode_and_prefill_roofline(self):
        meta = self._load_meta()
        assert meta is not None
        config = {
            "self_attn_mode": "native",
            "linear_attn_mode": "native",
            "mlp_mode": "native",
            "kv_cache_mode": "native",
            "attention_mode": "native",
        }
        score = pr.compute_perf_score(
            meta,
            config,
            gpu_type="mi300x",
            num_gpus=1,
            batch=1,
            isl=8192,
            osl=1024,
            include_prefill=True,
        )
        assert score.uses_op_model is True
        assert score.peak_tok_per_sec > 0
        assert score.prefill_tok_per_sec > score.peak_tok_per_sec
        assert score.bound_kind == "memory"
        assert score.prefill_bound_kind == "compute"
        op_types = {op.op_type for op in score.ops}
        assert {"gemm", "fused_moe", "sdpa", "linear_attention"} <= op_types
        assert any(op.name == "linear_attn_decode" for op in score.ops)
        assert any(op.name == "linear_attn_prefill" for op in score.prefill_ops)

    def test_fp8_mlp_speeds_up_decode(self):
        meta = self._load_meta()
        assert meta is not None
        base = dict(gpu_type="mi300x", num_gpus=1, batch=1, isl=8192, osl=1024)
        native = pr.compute_perf_score(meta, {"mlp_mode": "native"}, **base)
        fp8 = pr.compute_perf_score(meta, {"mlp_mode": "fp8"}, **base)
        assert fp8.peak_tok_per_sec > native.peak_tok_per_sec
