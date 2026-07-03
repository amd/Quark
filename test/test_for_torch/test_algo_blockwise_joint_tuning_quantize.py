#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Numerical tests for quantizer.py and learnable_linear.py.

All expected values are derived by hand so any code-level mistake in the
forward/backward math is caught immediately.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from quark.experimental.torch.algorithm.blockwise_joint_tuning.quantize.learnable_linear import (
    ExperimentalLearnableQuantizedLinear,
)
from quark.experimental.torch.algorithm.blockwise_joint_tuning.quantize.quantizer import (
    WeightGroupQuantizer,
    _ste_clamp,
    _ste_round,
)

# ---------------------------------------------------------------------------
# _ste_round
# ---------------------------------------------------------------------------


class TestSteRound:
    def test_forward_equals_round(self):
        x = torch.tensor([-1.7, -0.3, 0.0, 0.4, 0.6, 1.7])
        assert torch.equal(_ste_round(x), x.round())

    def test_forward_half_integers(self):
        # torch.round uses banker's rounding (round-half-to-even)
        # 0.5 → 0 (even), 1.5 → 2 (even), 2.5 → 2 (even), 3.5 → 4 (even)
        x = torch.tensor([0.5, 1.5, 2.5, 3.5])
        assert torch.equal(_ste_round(x), torch.tensor([0.0, 2.0, 2.0, 4.0]))

    def test_ste_gradient_is_one(self):
        # STE: gradient flows through as if rounding did not exist
        x = torch.tensor([-1.7, -0.3, 0.4, 0.6, 1.7], requires_grad=True)
        _ste_round(x).sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))

    def test_ste_gradient_at_half_integer(self):
        # Even at the non-differentiable rounding point, gradient = 1
        x = torch.tensor([0.5, 1.5], requires_grad=True)
        _ste_round(x).sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))

    def test_output_dtype_preserved(self):
        x = torch.tensor([1.7], dtype=torch.float64)
        assert _ste_round(x).dtype == torch.float64


# ---------------------------------------------------------------------------
# _ste_clamp
# ---------------------------------------------------------------------------


class TestSteClamp:
    def test_forward_equals_clamp(self):
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        assert torch.equal(_ste_clamp(x, -1.0, 1.0), x.clamp(-1.0, 1.0))

    def test_forward_values_below_lower(self):
        x = torch.tensor([-5.0, -2.0, 0.5])
        out = _ste_clamp(x, -1.0, 2.0)
        assert torch.equal(out, torch.tensor([-1.0, -1.0, 0.5]))

    def test_forward_values_above_upper(self):
        x = torch.tensor([0.5, 3.0, 5.0])
        out = _ste_clamp(x, -1.0, 2.0)
        assert torch.equal(out, torch.tensor([0.5, 2.0, 2.0]))

    def test_ste_gradient_is_one_for_in_range(self):
        x = torch.tensor([0.0, 0.5], requires_grad=True)
        _ste_clamp(x, -1.0, 1.0).sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))

    def test_ste_gradient_is_one_outside_range(self):
        # Key STE property: gradient = 1 even when x is outside [lo, hi]
        x = torch.tensor([-3.0, 0.5, 3.0], requires_grad=True)
        _ste_clamp(x, -1.0, 1.0).sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))

    def test_output_dtype_preserved(self):
        x = torch.tensor([0.5], dtype=torch.float64)
        assert _ste_clamp(x, 0.0, 1.0).dtype == torch.float64


# ---------------------------------------------------------------------------
# WeightGroupQuantizer – construction
# ---------------------------------------------------------------------------


class TestWeightGroupQuantizerInit:
    def _make_clean_weight(self) -> torch.Tensor:
        # [0, 1, 2, 3]: min=0, max=3 → scale=1, zp=0 with num_bits=2
        return torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    def test_quant_min_max(self):
        w = self._make_clean_weight()
        q = WeightGroupQuantizer(num_bits=2, group_size=4, weight=w)
        assert q.quant_min == 0
        assert q.quant_max == 3  # 2^2 - 1

        q8 = WeightGroupQuantizer(num_bits=8, group_size=4, weight=w)
        assert q8.quant_max == 255  # 2^8 - 1

    def test_scale_computed_correctly(self):
        # scale = (max - min) / (2^bits - 1)
        # For [0, 1, 2, 3] with bits=2: (3-0)/3 = 1.0
        w = self._make_clean_weight()
        q = WeightGroupQuantizer(num_bits=2, group_size=4, weight=w)
        assert torch.allclose(q.scale, torch.tensor([[1.0]]))

    def test_zero_point_computed_correctly(self):
        # zp = round(-min / scale) = round(0 / 1.0) = 0
        w = self._make_clean_weight()
        q = WeightGroupQuantizer(num_bits=2, group_size=4, weight=w)
        assert torch.allclose(q.zero_point, torch.tensor([[0.0]]))

    def test_scale_and_zp_for_asymmetric_range(self):
        # weight = [[-2, 0, 2, 4]] → min=-2, max=4
        # scale = (4 - (-2)) / 3 = 2.0
        # zp = round(-(-2) / 2.0) = round(1.0) = 1.0
        w = torch.tensor([[-2.0, 0.0, 2.0, 4.0]])
        q = WeightGroupQuantizer(num_bits=2, group_size=4, weight=w)
        assert torch.allclose(q.scale, torch.tensor([[2.0]]))
        assert torch.allclose(q.zero_point, torch.tensor([[1.0]]))

    def test_multiple_groups_independent_scale_zp(self):
        # Group 0: [0,1,2,3] → scale=1.0, zp=0
        # Group 1: [0,2,4,6] → scale=(6-0)/3=2.0, zp=0
        w = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.0, 2.0, 4.0, 6.0]])
        q = WeightGroupQuantizer(num_bits=2, group_size=4, weight=w)
        assert torch.allclose(q.scale[0], torch.tensor([1.0]))
        assert torch.allclose(q.scale[1], torch.tensor([2.0]))
        assert torch.allclose(q.zero_point[0], torch.tensor([0.0]))
        assert torch.allclose(q.zero_point[1], torch.tensor([0.0]))

    def test_group_size_minus_one_uses_full_last_dim(self):
        w = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
        q = WeightGroupQuantizer(num_bits=4, group_size=-1, weight=w)
        assert q.group_size == 6  # == w.shape[-1]

    def test_scale_clamped_when_weight_constant(self):
        # All-zero weight → max-min=0 → scale=0 → clamped to float32(1e-4).
        # float32 cannot represent 1e-4 exactly; use allclose instead of >=.
        w = torch.zeros(1, 4)
        q = WeightGroupQuantizer(num_bits=4, group_size=4, weight=w)
        assert torch.allclose(q.scale, torch.full_like(q.scale, 1e-4))

    def test_scale_and_zp_are_learnable_parameters(self):
        w = self._make_clean_weight()
        q = WeightGroupQuantizer(num_bits=4, group_size=4, weight=w)
        param_names = {n for n, _ in q.named_parameters()}
        assert "scale" in param_names
        assert "zero_point" in param_names
        assert q.scale.requires_grad
        assert q.zero_point.requires_grad

    def test_assert_num_bits_too_small(self):
        w = self._make_clean_weight()
        with pytest.raises(AssertionError):
            WeightGroupQuantizer(num_bits=1, group_size=4, weight=w)

    def test_assert_num_bits_too_large(self):
        w = self._make_clean_weight()
        with pytest.raises(AssertionError):
            WeightGroupQuantizer(num_bits=17, group_size=4, weight=w)

    def test_assert_weight_none(self):
        with pytest.raises(AssertionError):
            WeightGroupQuantizer(num_bits=4, group_size=4, weight=None)

    def test_assert_group_size_none(self):
        w = self._make_clean_weight()
        with pytest.raises(AssertionError):
            WeightGroupQuantizer(num_bits=4, group_size=None, weight=w)

    def test_assert_weight_not_divisible_by_group_size(self):
        w = torch.ones(1, 5)  # 5 % 3 != 0
        with pytest.raises(AssertionError):
            WeightGroupQuantizer(num_bits=4, group_size=3, weight=w)


# ---------------------------------------------------------------------------
# WeightGroupQuantizer – forward / _apply_fake_quant
# ---------------------------------------------------------------------------


class TestWeightGroupQuantizerForward:
    def _q2(self) -> tuple[WeightGroupQuantizer, torch.Tensor]:
        """2-bit quantizer initialised from [0,1,2,3]: scale=1, zp=0."""
        w = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        q = WeightGroupQuantizer(num_bits=2, group_size=4, weight=w)
        return q, w

    def test_num_bits_16_returns_input_unchanged(self):
        w = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        q = WeightGroupQuantizer(num_bits=16, group_size=4, weight=w)
        x = torch.tensor([[0.3, 0.7, 1.1, 2.8]])
        out = q(x)
        assert out is x

    def test_quant_disabled_returns_input_unchanged(self):
        q, _ = self._q2()
        q.quant_enabled = False
        x = torch.tensor([[0.3, 0.7, 1.1, 2.8]])
        out = q(x)
        assert out is x

    def test_grid_aligned_input_reconstructed_exactly(self):
        # Integer-grid inputs must survive quant/dequant without any error.
        # scale=1, zp=0 → X_int=X → X_dequant=X
        q, _ = self._q2()
        x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        out = q(x)
        assert torch.allclose(out, x)

    def test_out_of_range_values_clamped(self):
        # scale=1, zp=0, quant range [0, 3]
        # -1.0 → X_int=round(-1).clamp(0,3)=0 → dequant=0.0
        #  4.0 → X_int=round(4).clamp(0,3)=3  → dequant=3.0
        q, _ = self._q2()
        x = torch.tensor([[-1.0, 4.0, 1.0, 2.0]])
        out = q(x)
        assert torch.allclose(out, torch.tensor([[0.0, 3.0, 1.0, 2.0]]))

    def test_rounding_to_nearest_grid_point(self):
        # scale=1, zp=0; values between grid points get rounded:
        # 0.3 → 0, 0.7 → 1, 1.3 → 1, 1.7 → 2
        q, _ = self._q2()
        x = torch.tensor([[0.3, 0.7, 1.3, 1.7]])
        out = q(x)
        assert torch.allclose(out, torch.tensor([[0.0, 1.0, 1.0, 2.0]]))

    def test_multi_group_each_group_uses_own_scale(self):
        # Group 0: [0,1,2,3] → scale=1, zp=0
        # Group 1: [0,2,4,6] → scale=2, zp=0
        # Grid-aligned input for both groups → exact reconstruction
        w = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.0, 2.0, 4.0, 6.0]])
        q = WeightGroupQuantizer(num_bits=2, group_size=4, weight=w)
        x = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.0, 2.0, 4.0, 6.0]])
        out = q(x)
        assert torch.allclose(out, x)

    def test_multi_group_cross_contamination_absent(self):
        # Group 0 has scale=1; Group 1 has scale=2.
        # A value of 1.0 in group 1 dequantizes to 0.0 (nearest integer grid, X/2=0.5→0→0*2=0)
        # but the same value 1.0 in group 0 stays 1.0.  Confirms groups don't share scale.
        w = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.0, 2.0, 4.0, 6.0]])
        q = WeightGroupQuantizer(num_bits=2, group_size=4, weight=w)
        # group-0 portion of x = [[1, 1, 1, 1]], group-1 portion = [[1, 1, 1, 1]]
        x = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        out = q(x)
        # Group 0 output: round([1,1,1,1]/1+0).clamp(0,3)=[1,1,1,1]; dequant=[1,1,1,1]
        assert torch.allclose(out[:, :4], torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
        # Group 1 output: round([1,1,1,1]/2+0)=[0,0,0,0 or 1...]; actually:
        # round(0.5) = 0 (banker's, 0 is even); clamp=0; dequant=0*2=0.0
        assert torch.allclose(out[:, 4:], torch.tensor([[0.0, 0.0, 0.0, 0.0]]))

    def test_quantization_error_not_zero_for_off_grid_input(self):
        # Verify that rounding actually changes off-grid values
        q, _ = self._q2()
        x = torch.tensor([[0.3, 0.7, 1.3, 1.7]])
        out = q(x)
        assert not torch.equal(out, x)

    def test_output_shape_preserved(self):
        w = torch.randn(3, 4)  # 3 rows, group_size=4 → 3 groups
        q = WeightGroupQuantizer(num_bits=4, group_size=4, weight=w)
        x = torch.randn(3, 4)
        assert q(x).shape == x.shape

    def test_fake_quant_is_differentiable_via_ste(self):
        # Gradients must flow back through fake-quant (via STE)
        w = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        q = WeightGroupQuantizer(num_bits=2, group_size=4, weight=w)
        x = torch.tensor([[0.3, 0.7, 1.3, 1.7]], requires_grad=True)
        out = q(x)
        out.sum().backward()
        # STE: gradient w.r.t. x is 1 everywhere (not zeroed by rounding)
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))


# ---------------------------------------------------------------------------
# ExperimentalLearnableQuantizedLinear – construction
# ---------------------------------------------------------------------------


class TestLearnableQuantizedLinearInit:
    def _source(self, bias: bool = False) -> nn.Linear:
        lin = nn.Linear(4, 2, bias=bias)
        torch.manual_seed(0)
        nn.init.normal_(lin.weight)
        if bias:
            nn.init.zeros_(lin.bias)
        return lin

    def test_in_out_features(self):
        src = self._source()
        q = ExperimentalLearnableQuantizedLinear(src, num_bits=4, group_size=4)
        assert q.in_features == 4
        assert q.out_features == 2

    def test_weight_is_same_parameter_object(self):
        src = self._source()
        q = ExperimentalLearnableQuantizedLinear(src, num_bits=4, group_size=4)
        assert q.weight is src.weight

    def test_weight_is_parameter_not_buffer(self):
        src = self._source()
        q = ExperimentalLearnableQuantizedLinear(src, num_bits=4, group_size=4)
        param_names = {n for n, _ in q.named_parameters()}
        buffer_names = {n for n, _ in q.named_buffers()}
        assert "weight" in param_names
        assert "weight" not in buffer_names

    def test_no_bias_when_source_has_none(self):
        src = self._source(bias=False)
        q = ExperimentalLearnableQuantizedLinear(src, num_bits=4, group_size=4)
        assert q.bias is None

    def test_bias_values_match_source(self):
        src = self._source(bias=True)
        q = ExperimentalLearnableQuantizedLinear(src, num_bits=4, group_size=4)
        assert q.bias is not None
        assert torch.equal(q.bias, src.bias.data)

    def test_bias_is_buffer_not_parameter(self):
        src = self._source(bias=True)
        q = ExperimentalLearnableQuantizedLinear(src, num_bits=4, group_size=4)
        buffer_names = {n for n, _ in q.named_buffers()}
        param_names = {n for n, _ in q.named_parameters()}
        assert "bias" in buffer_names
        assert "bias" not in param_names

    def test_weight_quant_disabled_by_default(self):
        src = self._source()
        q = ExperimentalLearnableQuantizedLinear(src, num_bits=4, group_size=4)
        assert q.weight_quant_enabled is False

    def test_scale_and_zero_point_are_learnable(self):
        src = self._source()
        q = ExperimentalLearnableQuantizedLinear(src, num_bits=4, group_size=4)
        param_names = dict(q.named_parameters())
        assert "weight_quantizer.scale" in param_names
        assert "weight_quantizer.zero_point" in param_names
        assert param_names["weight_quantizer.scale"].requires_grad
        assert param_names["weight_quantizer.zero_point"].requires_grad


# ---------------------------------------------------------------------------
# ExperimentalLearnableQuantizedLinear – forward
# ---------------------------------------------------------------------------


class TestLearnableQuantizedLinearForward:
    def _build(self, bias: bool = False) -> tuple[nn.Linear, ExperimentalLearnableQuantizedLinear]:
        lin = nn.Linear(4, 2, bias=bias)
        torch.manual_seed(42)
        nn.init.normal_(lin.weight)
        if bias:
            nn.init.normal_(lin.bias)
        q = ExperimentalLearnableQuantizedLinear(lin, num_bits=4, group_size=4)
        return lin, q

    def test_quant_disabled_matches_raw_linear(self):
        lin, q = self._build()
        x = torch.randn(3, 4)
        with torch.no_grad():
            assert torch.allclose(q(x), lin(x))

    def test_quant_disabled_matches_raw_linear_with_bias(self):
        lin, q = self._build(bias=True)
        x = torch.randn(3, 4)
        with torch.no_grad():
            assert torch.allclose(q(x), lin(x))

    def test_quant_enabled_differs_from_raw_for_lossy_weight(self):
        # 2-bit quantization is very coarse; any non-trivial weight will be lossy
        lin = nn.Linear(4, 2, bias=False)
        torch.manual_seed(7)
        nn.init.normal_(lin.weight)  # random weight, almost certainly not on 2-bit grid
        q = ExperimentalLearnableQuantizedLinear(lin, num_bits=2, group_size=4)
        q.enable_weight_quant()
        x = torch.randn(3, 4)
        with torch.no_grad():
            raw = lin(x)
            quant = q(x)
        assert not torch.allclose(raw, quant)

    def test_quant_enabled_with_grid_aligned_weight_matches_raw(self):
        # A weight that IS on the 2-bit grid should be reconstructed exactly.
        # scale=1, zp=0 for rows [[0,1,2,3], [1,2,0,3]]
        lin = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 0.0, 3.0]]))
        q = ExperimentalLearnableQuantizedLinear(lin, num_bits=2, group_size=4)
        q.enable_weight_quant()
        x = torch.randn(3, 4)
        with torch.no_grad():
            assert torch.allclose(q(x), lin(x), atol=1e-5)

    def test_quant_known_output_values(self):
        # Use a 1-output layer so we can compute by hand.
        # Weight = [[0.0, 1.0, 2.0, 3.0]], num_bits=2, group_size=4
        # → scale=1, zp=0 → Q(W)=W (exact)
        # Input = [[1.0, 0.0, 0.0, 0.0]]
        # Expected output = 0*1 + 1*0 + 2*0 + 3*0 = 0.0
        lin = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor([[0.0, 1.0, 2.0, 3.0]]))
        q = ExperimentalLearnableQuantizedLinear(lin, num_bits=2, group_size=4)
        q.enable_weight_quant()
        x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        with torch.no_grad():
            out = q(x)
        assert torch.allclose(out, torch.tensor([[0.0]]))

    def test_quant_known_output_values_full_sum(self):
        # Same setup, input = [[1,1,1,1]] → output = 0+1+2+3 = 6.0
        lin = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor([[0.0, 1.0, 2.0, 3.0]]))
        q = ExperimentalLearnableQuantizedLinear(lin, num_bits=2, group_size=4)
        q.enable_weight_quant()
        x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        with torch.no_grad():
            out = q(x)
        assert torch.allclose(out, torch.tensor([[6.0]]))

    def test_enable_disable_toggle(self):
        _, q = self._build()
        q.enable_weight_quant(True)
        assert q.weight_quant_enabled is True
        q.disable_weight_quant()
        assert q.weight_quant_enabled is False
        q.enable_weight_quant()
        assert q.weight_quant_enabled is True
        q.enable_weight_quant(False)
        assert q.weight_quant_enabled is False

    def test_output_shape(self):
        lin = nn.Linear(4, 2, bias=False)
        q = ExperimentalLearnableQuantizedLinear(lin, num_bits=4, group_size=4)
        x = torch.randn(5, 4)
        with torch.no_grad():
            out = q(x)
        assert out.shape == (5, 2)

    def test_gradients_flow_through_quantized_weight(self):
        # With quant enabled, weight.grad must be non-None after backward
        lin = nn.Linear(4, 2, bias=False)
        q = ExperimentalLearnableQuantizedLinear(lin, num_bits=4, group_size=4)
        q.enable_weight_quant()
        x = torch.randn(3, 4)
        out = q(x)
        out.sum().backward()
        assert q.weight.grad is not None
        assert not torch.all(q.weight.grad == 0)

    def test_gradients_reach_scale_and_zero_point(self):
        # Joint-tuning relies on qparam grads existing when quant is enabled.
        lin = nn.Linear(4, 2, bias=False)
        q = ExperimentalLearnableQuantizedLinear(lin, num_bits=4, group_size=4)
        q.enable_weight_quant()
        x = torch.randn(3, 4)
        out = q(x)
        out.sum().backward()

        scale_grad = q.weight_quantizer.scale.grad
        zp_grad = q.weight_quantizer.zero_point.grad
        assert scale_grad is not None
        assert zp_grad is not None
        assert torch.isfinite(scale_grad).all()
        assert torch.isfinite(zp_grad).all()
