#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest

import torch
import torch.nn as nn

from quark.experimental.torch.autoround.wrapper import (
    WrapperLinear,
    WrapperLinearMXFP4,
    attach_int4_fakequant,
    make_wrapper,
)


class _TinyLinear(nn.Module):
    def __init__(self, in_features=64, out_features=32) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_int4_quant_linear(in_features=64, out_features=32, group_size=32, bits=4):
    lin = nn.Linear(in_features, out_features, bias=False)
    return attach_int4_fakequant(lin, group_size=group_size, bits=bits)


def _make_int4_sym_quant_linear(in_features=64, out_features=32, group_size=32):
    """Real, calibrated symmetric INT4 (`int4_wo_sym`-equivalent) quantized linear, built via
    ModelQuantizer (not the asymmetric-only `attach_int4_fakequant` test fixture)."""
    from quark.torch import ModelQuantizer
    from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
    from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
    from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver

    weight_spec = QTensorConfig(
        dtype=Dtype.int4,
        observer_cls=PerGroupMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        qscheme=QSchemeType.per_group,
        ch_axis=1,
        group_size=group_size,
        is_dynamic=False,
    )
    quant_config = QConfig(global_quant_config=QLayerConfig(weight=weight_spec))
    qmodel = ModelQuantizer(quant_config).quantize_model(_TinyLinear(in_features, out_features).eval(), None)
    return qmodel.fc


def _make_mxfp4_quant_linear(in_features=64, out_features=32, scale_calculation_mode=None, with_activation=False):
    from quark.torch import ModelQuantizer
    from quark.torch.quantization.config.config import OCP_MXFP4Spec, QConfig, QLayerConfig

    kwargs = {} if scale_calculation_mode is None else {"scale_calculation_mode": scale_calculation_mode}
    weight_spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False, **kwargs).to_quantization_spec()
    input_spec = (
        OCP_MXFP4Spec(ch_axis=-1, is_dynamic=True, **kwargs).to_quantization_spec() if with_activation else None
    )
    quant_config = QConfig(global_quant_config=QLayerConfig(weight=weight_spec, input_tensors=input_spec))
    qmodel = ModelQuantizer(quant_config).quantize_model(_TinyLinear(in_features, out_features).eval(), None)
    return qmodel.fc


class TestWrapperValueBound(unittest.TestCase):
    """V's zero-init and [-0.5, 0.5] clamp are shared _WrapperLinearBase behavior -- checked once
    across both format subclasses rather than duplicated per class."""

    def test_V_initialized_zero_and_bounded(self):
        for wrapper_cls, linear in (
            (WrapperLinear, _make_int4_quant_linear()),
            (WrapperLinearMXFP4, _make_mxfp4_quant_linear()),
        ):
            with self.subTest(wrapper_cls=wrapper_cls.__name__):
                w = wrapper_cls(linear)
                self.assertTrue(torch.equal(w.value, torch.zeros_like(w.value)))
                w.value.data.fill_(10.0)
                _ = w(torch.randn(4, 64))
                self.assertLessEqual(w.value_clamped().max().item(), 0.5)
                self.assertGreaterEqual(w.value_clamped().min().item(), -0.5)


class TestWrapperLinear(unittest.TestCase):
    def _make_quant_linear(self):
        return _make_int4_quant_linear()

    def test_zero_V_equals_plain_fakequant(self):
        qlin = self._make_quant_linear()
        w = WrapperLinear(qlin)
        x = torch.randn(4, 64)
        self.assertTrue(torch.allclose(w(x), qlin(x), atol=1e-5))

    def test_minmax_params_present_only_when_enabled(self):
        # Off (default): no clip params.
        w_off = WrapperLinear(self._make_quant_linear())
        names_off = {n for n, _ in w_off.named_parameters()}
        self.assertNotIn("min_scale", names_off)
        self.assertNotIn("max_scale", names_off)

        # On: min_scale/max_scale are trainable params, init 1.0, shape [out, n_groups, 1].
        w_on = WrapperLinear(self._make_quant_linear(), enable_minmax_tuning=True)
        names_on = {n for n, _ in w_on.named_parameters()}
        self.assertIn("min_scale", names_on)
        self.assertIn("max_scale", names_on)
        self.assertTrue(w_on.min_scale.requires_grad)
        self.assertTrue(w_on.max_scale.requires_grad)
        # out=32, in=64, group_size=32 -> n_groups=2
        self.assertEqual(tuple(w_on.min_scale.shape), (32, 2, 1))
        self.assertEqual(tuple(w_on.max_scale.shape), (32, 2, 1))
        self.assertTrue(torch.equal(w_on.min_scale, torch.ones_like(w_on.min_scale)))
        self.assertTrue(torch.equal(w_on.max_scale, torch.ones_like(w_on.max_scale)))

    def test_minmax_at_unit_scale_matches_observer_reference(self):
        # With min_scale=max_scale=1.0 and V=0, the recomputed scale/zp must equal a
        # hand-computed reference using the Quark observer formula (observer.py:288-292):
        #   min_neg = clamp(min, max=0); max_pos = clamp(max, min=0)
        #   scale = max((max_pos - min_neg)/(qmax-qmin), eps)
        #   zp = clamp(qmin - round(min_neg/scale), qmin, qmax)
        qlin = self._make_quant_linear()
        w = WrapperLinear(qlin, enable_minmax_tuning=True)

        weight = qlin.weight.detach()
        out_f, in_f = weight.shape
        gs = 32
        ng = in_f // gs
        wg = weight.reshape(out_f, ng, gs)
        qmin, qmax = 0, 15
        min_neg = wg.min(dim=-1, keepdim=True).values.clamp(max=0.0)
        max_pos = wg.max(dim=-1, keepdim=True).values.clamp(min=0.0)
        eps = torch.finfo(torch.float32).eps
        ref_scale = torch.clamp((max_pos - min_neg) / float(qmax - qmin), min=eps)
        ref_zp = torch.clamp(qmin - torch.round(min_neg / ref_scale), qmin, qmax)

        scale, zp = w.get_scale_zero_point()  # 2D [out, ng]
        self.assertTrue(torch.allclose(scale, ref_scale.squeeze(-1), atol=1e-7))
        self.assertTrue(torch.equal(zp, ref_zp.squeeze(-1)))

        # And the full quantized weight matches a reference built with the same qparams. Round
        # w/scale FIRST, THEN add zp -- NOT round(w/scale + zp) as a single combined rounding
        # (see test_zero_point_order_matches_official_and_production for why this distinction
        # is load-bearing, not stylistic).
        w_int = torch.clamp(torch.round(wg / ref_scale) + ref_zp, qmin, qmax)
        w_deq = ((w_int - ref_zp) * ref_scale).reshape(out_f, in_f)
        self.assertTrue(torch.allclose(w._quantize_weight(), w_deq, atol=1e-6))

    def test_minmax_scale_projected_into_bound_bug_fix(self):
        # Regression test: min_scale/max_scale must be projected into MINMAX_SCALE_BOUND
        # (matches the official auto-round repo's WrapperLinear.minmax_scale_bound = (0.0, 1.0)).
        # Before the fix, these params had no bound and unconstrained sign-SGD could drift them
        # to extreme/negative values, overfitting calibration MSE at the cost of downstream
        # accuracy (observed as a wikitext2-ppl improvement alongside an MMLU regression).
        w = WrapperLinear(self._make_quant_linear(), enable_minmax_tuning=True)
        self.assertEqual(w.MINMAX_SCALE_BOUND, (0.0, 1.0))

        w.min_scale.data.fill_(-3.0)
        w.max_scale.data.fill_(5.0)
        _ = w._quantize_weight()  # any call that recomputes scale/zp must project in place

        self.assertTrue(torch.all(w.min_scale >= 0.0))
        self.assertTrue(torch.all(w.min_scale <= 1.0))
        self.assertTrue(torch.all(w.max_scale >= 0.0))
        self.assertTrue(torch.all(w.max_scale <= 1.0))
        self.assertTrue(torch.equal(w.min_scale, torch.zeros_like(w.min_scale)))
        self.assertTrue(torch.equal(w.max_scale, torch.ones_like(w.max_scale)))

    def test_zero_point_ste_gradient_reaches_minmax_scale(self):
        # Regression test: zero_point must use round_ste (not a plain, non-differentiable round)
        # so gradient reaches min_scale/max_scale through the zero-point term, matching the
        # official auto-round repo's data_type/int.py (zp = round_ste(-wmin/scale)). Before the
        # fix, this path was dead (torch.round has zero local gradient almost everywhere), so
        # min_scale/max_scale could only learn through the scale-division term.
        w = WrapperLinear(self._make_quant_linear(), enable_minmax_tuning=True)
        _, zero_point = w._recompute_scale_zp()
        (min_scale_grad,) = torch.autograd.grad(zero_point.sum(), w.min_scale, retain_graph=True)
        self.assertTrue(torch.any(min_scale_grad != 0.0))

    def test_zero_point_order_matches_official_and_production(self):
        # Regression test: round(w/scale + V) THEN add zero_point, NOT round(w/scale + zero_point
        # + V) as a single combined rounding. round(x) + zp == round(x + zp) only when zp is
        # even -- when zp is odd and w/scale lands on an exact tie, adding zp before vs after
        # rounding can flip which neighboring integer round-half-to-even picks. An earlier
        # version combined zp into the round(), diverging from both the official auto-round
        # repo's quant_tensor_asym (`int_w = round_ste(tensor/scale + v); q = int_w + zp`) and
        # Quark's own real production kernel (hw_emulation_interface.py::
        # fake_quantize_per_channel_affine: `torch.round(input*inv_scale) + zero_point`) --
        # found via a real ~0.1%-of-elements mismatch on a real Llama-3.1-8B weight tensor.
        w = WrapperLinear(self._make_quant_linear(), enable_minmax_tuning=True)
        scale, zp = w.get_scale_zero_point()
        # Force an odd zero_point and a weight that lands exactly on a rounding tie once
        # divided by scale, so the old (buggy) and fixed orderings disagree.
        with torch.no_grad():
            w.wmin.fill_(-1.0)
            w.wmax.fill_(1.0)
        # qmin=0, qmax=15 (int4 unsigned): scale = (1-(-1))/15, zp = 0 - round(-1/scale) = round(15/2) = round(7.5).
        # round-half-to-even(7.5) = 8 (even) -> zp = -8 (odd magnitude neighbor check below).
        scale2, zp2 = w.get_scale_zero_point()
        self.assertNotEqual(int(zp2.flatten()[0].item()) % 2, 0, "test setup must produce an odd zero_point")

        with torch.no_grad():
            w.linear.weight.zero_()
            # Set one element so that weight/scale lands exactly on a x.5 tie.
            s = scale2.flatten()[0].item()
            w.linear.weight[0, 0] = 1.5 * s

        out = w._quantize_weight()
        # Reproduce the CORRECT (fixed) computation by hand: round(w/scale) first, then add zp.
        expected_int = torch.round(torch.tensor(1.5)) + zp2.flatten()[0]
        expected_int = torch.clamp(expected_int, w.quant_min, w.quant_max)
        expected = (expected_int - zp2.flatten()[0]) * s
        self.assertAlmostEqual(out[0, 0].item(), expected.item(), places=5)

    def test_zero_point_unclamped_stays_in_bound_matches_official(self):
        # Regression test: _recompute_scale_zp deliberately does NOT clamp zero_point on its own
        # (the official repo only clamps the final combined index, clamp(int_w + zp, 0, maxq), not
        # zp alone). Given min_scale/max_scale are bounded to [0,1] (MINMAX_SCALE_BOUND), zero
        # always lies within [wmin_s, wmax_s] by construction, so the unclamped zero_point is
        # mathematically guaranteed to already land in [quant_min, quant_max] -- verify this
        # invariant holds across the corners of the min_scale/max_scale range, so the removed
        # clamp is provably a no-op here (matching official) rather than an unverified assumption.
        w = WrapperLinear(self._make_quant_linear(), enable_minmax_tuning=True)
        for min_s, max_s in [(0.0, 1.0), (1.0, 0.0), (0.3, 0.3), (1.0, 1.0), (0.0, 0.0)]:
            w.min_scale.data.fill_(min_s)
            w.max_scale.data.fill_(max_s)
            _, zero_point = w._recompute_scale_zp()
            self.assertTrue(torch.all(zero_point >= w.quant_min))
            self.assertTrue(torch.all(zero_point <= w.quant_max))


class TestWrapperLinearIntSymmetric(unittest.TestCase):
    """Regression coverage for symmetric INT4 (e.g. `int4_wo_sym`) reaching WrapperLinearInt.

    Before the fix, minmax tuning always used the asymmetric wmin/wmax formula regardless of
    ``weight_quantizer.symmetric``, so a symmetric scheme's zero_point could drift away from 0.
    """

    def _make_quant_linear(self):
        return _make_int4_sym_quant_linear()

    def test_zero_V_equals_plain_fakequant(self):
        qlin = self._make_quant_linear()
        w = WrapperLinear(qlin)
        x = torch.randn(4, 64)
        self.assertTrue(torch.allclose(w(x), qlin(x), atol=1e-5))

    def test_minmax_params_present_only_when_enabled_no_min_scale(self):
        # Off (default): no clip params.
        w_off = WrapperLinear(self._make_quant_linear())
        names_off = {n for n, _ in w_off.named_parameters()}
        self.assertNotIn("min_scale", names_off)
        self.assertNotIn("max_scale", names_off)

        # On: ONLY max_scale (no min_scale -- symmetric has no min side to tune, matching
        # WrapperLinearMXFP4's single-clip-param shape).
        w_on = WrapperLinear(self._make_quant_linear(), enable_minmax_tuning=True)
        names_on = {n for n, _ in w_on.named_parameters()}
        self.assertIn("max_scale", names_on)
        self.assertNotIn("min_scale", names_on)
        self.assertEqual(w_on.clip_params(), [w_on.max_scale])

    def test_zero_point_stays_zero_under_minmax_tuning(self):
        w = WrapperLinear(self._make_quant_linear(), enable_minmax_tuning=True)
        _, zero_point = w.get_scale_zero_point()
        self.assertTrue(torch.equal(zero_point, torch.zeros_like(zero_point)))

        # Still zero after the clip param has actually moved away from its 1.0 init -- not just
        # at the untuned starting point.
        w.max_scale.data.fill_(0.5)
        _, zero_point2 = w.get_scale_zero_point()
        self.assertTrue(torch.equal(zero_point2, torch.zeros_like(zero_point2)))

    def test_zero_V_max_scale_1_matches_calibrated_quantizer(self):
        qlin = self._make_quant_linear()
        w = WrapperLinear(qlin, enable_minmax_tuning=True)
        x = torch.randn(4, 64)
        self.assertTrue(torch.allclose(w(x), qlin(x), atol=1e-5))

    def test_gradient_reaches_max_scale_through_ste(self):
        w = WrapperLinear(self._make_quant_linear(), enable_minmax_tuning=True)
        out = w(torch.randn(4, 64))
        (grad,) = torch.autograd.grad(out.sum(), w.max_scale, retain_graph=True)
        self.assertTrue(torch.any(grad != 0.0))


class TestWrapperLinearMXFP4(unittest.TestCase):
    """Uses a REAL Quark OCP_MXFP4Spec-built weight_quantizer (calibrated via ModelQuantizer),
    not a hand-rolled test-only fixture -- the shared-exponent scale computation is intricate
    enough that an independent test-only reimplementation risks sharing a bug with the real
    implementation."""

    def _make_mxfp4_quant_linear(self, in_features=64, out_features=32):
        return _make_mxfp4_quant_linear(in_features, out_features)

    def _make_mxfp4_quant_linear_with_mode(self, scale_calculation_mode, in_features=64, out_features=32):
        return _make_mxfp4_quant_linear(in_features, out_features, scale_calculation_mode=scale_calculation_mode)

    def test_scale_calculation_mode_survives_calibration_freeze(self):
        # Regression test: weight_quantizer is a StaticScaledFakeQuantize post-calibration, which
        # has no scale_calculation_mode attribute of its own (only weight_quantizer.observer
        # does) -- a wrapper that read the attribute directly off weight_quantizer would always
        # silently fall back to "even", making --mx_scale_mode floor a no-op under the driver's
        # default enable_minmax_tuning=True. Must read via weight_quantizer.observer instead.
        torch.manual_seed(0)
        even_linear = self._make_mxfp4_quant_linear_with_mode("even")
        torch.manual_seed(0)
        floor_linear = self._make_mxfp4_quant_linear_with_mode("floor")
        floor_linear.weight.data.copy_(even_linear.weight.data)

        w_even = WrapperLinearMXFP4(even_linear, enable_minmax_tuning=True)
        w_floor = WrapperLinearMXFP4(floor_linear, enable_minmax_tuning=True)
        self.assertEqual(w_even.scale_calculation_mode, "even")
        self.assertEqual(w_floor.scale_calculation_mode, "floor")

        x = torch.randn(4, 64)
        self.assertFalse(
            torch.allclose(w_even(x), w_floor(x)),
            "even and floor scale_calculation_mode produced identical output — the mode isn't "
            "actually reaching mxfp4_scale (regression of the StaticScaledFakeQuantize bug)",
        )

    def test_zero_V_max_scale_1_matches_calibrated_quantizer(self):
        # The single most important point-value test: with V=0 and (if minmax tuning is on)
        # max_scale=1.0, the wrapper's quantized output must match the plain calibrated
        # weight_quantizer's own forward exactly -- this would catch a systematic
        # scale/exponent-formula mismatch immediately.
        qlin = self._make_mxfp4_quant_linear()
        w = WrapperLinearMXFP4(qlin)
        x = torch.randn(4, 64)
        self.assertTrue(torch.allclose(w(x), qlin(x), atol=1e-5))

        w_on = WrapperLinearMXFP4(self._make_mxfp4_quant_linear(), enable_minmax_tuning=True)
        qlin_on = w_on.linear
        self.assertTrue(torch.allclose(w_on(x), qlin_on(x), atol=1e-5))

    def test_minmax_params_present_only_when_enabled_no_min_scale(self):
        # Off (default): no clip params.
        w_off = WrapperLinearMXFP4(self._make_mxfp4_quant_linear())
        names_off = {n for n, _ in w_off.named_parameters()}
        self.assertNotIn("max_scale", names_off)
        self.assertNotIn("min_scale", names_off)

        # On: ONLY max_scale (no min_scale -- MXFP4 is symmetric, unlike INT4's asymmetric
        # min/max pair; regression test for §1's deviation table).
        w_on = WrapperLinearMXFP4(self._make_mxfp4_quant_linear(), enable_minmax_tuning=True)
        names_on = {n for n, _ in w_on.named_parameters()}
        self.assertIn("max_scale", names_on)
        self.assertNotIn("min_scale", names_on)
        self.assertTrue(w_on.max_scale.requires_grad)
        self.assertTrue(torch.equal(w_on.max_scale, torch.ones_like(w_on.max_scale)))
        self.assertEqual(w_on.clip_params(), [w_on.max_scale])
        self.assertEqual(w_off.clip_params(), [])

    def test_max_scale_bound_matches_official_repo(self):
        # Confirmed by reading auto_round/wrapper.py (plan3-autoround-mxfp4.md §7 item 2): the
        # official repo's minmax_scale_bound=(0,1) is applied unconditionally regardless of
        # format, so MXFP4's max_scale shares INT4's bound. Regression test: an earlier version
        # of WrapperLinearMXFP4 never clamped max_scale at all (unlike WrapperLinearInt), letting
        # unconstrained sign-SGD drift it outside [0, 1] -- a real divergence from the official
        # repo's tuning trajectory. Must assert the clamp actually happened, not just "no NaN"
        # (an unclamped max_scale=5.0 doesn't NaN, it just silently searches the wrong range).
        w = WrapperLinearMXFP4(self._make_mxfp4_quant_linear(), enable_minmax_tuning=True)
        w.max_scale.data.fill_(5.0)
        out = w(torch.randn(4, 64))
        self.assertFalse(torch.isnan(out).any())
        self.assertLessEqual(w.max_scale.max().item(), 1.0)
        self.assertGreaterEqual(w.max_scale.min().item(), 0.0)

        w2 = WrapperLinearMXFP4(self._make_mxfp4_quant_linear(), enable_minmax_tuning=True)
        w2.max_scale.data.fill_(-3.0)
        w2(torch.randn(4, 64))
        self.assertGreaterEqual(w2.max_scale.min().item(), 0.0)

    def test_gradient_reaches_max_scale_through_ste(self):
        w = WrapperLinearMXFP4(self._make_mxfp4_quant_linear(), enable_minmax_tuning=True)
        out = w(torch.randn(4, 64))
        (grad,) = torch.autograd.grad(out.sum(), w.max_scale, retain_graph=True)
        self.assertTrue(torch.any(grad != 0.0))

    def test_get_scale_zero_point_returns_zero_zp(self):
        # MXFP4 has no real zero-point; get_scale_zero_point() must still return a
        # shape-compatible zero tensor for AutoRoundProcessor._freeze_block's write-back
        # (plan3-autoround-mxfp4.md §7 item 3).
        w = WrapperLinearMXFP4(self._make_mxfp4_quant_linear(), enable_minmax_tuning=True)
        scale, zero_point = w.get_scale_zero_point()
        self.assertEqual(scale.shape, zero_point.shape)
        self.assertTrue(torch.equal(zero_point, torch.zeros_like(zero_point)))

    def test_make_wrapper_dispatches_by_dtype(self):
        int_wrapper = make_wrapper(_make_int4_quant_linear())
        self.assertIsInstance(int_wrapper, WrapperLinear)
        mx_wrapper = make_wrapper(self._make_mxfp4_quant_linear())
        self.assertIsInstance(mx_wrapper, WrapperLinearMXFP4)


class TestWrapperLinearMXFP4WeightActivation(unittest.TestCase):
    """Weight+activation MXFP4: weight keeps AutoRound's learnable V/max_scale (same as
    weight-only); activation quantization is fixed (no learnable parameters, dynamic per-forward
    scale) -- reuses Quark's real DynamicScaledFakeQuantize directly, no reimplementation."""

    def _make_mxfp4_wa_quant_linear(self, in_features=64, out_features=32):
        return _make_mxfp4_quant_linear(in_features, out_features, with_activation=True)

    def _make_mxfp4_quant_linear(self, in_features=64, out_features=32):
        return _make_mxfp4_quant_linear(in_features, out_features)

    def test_activation_is_quantized_when_input_quantizer_present(self):
        w = WrapperLinearMXFP4(self._make_mxfp4_wa_quant_linear())
        self.assertTrue(w.quantize_activation)
        x = torch.randn(4, 64)
        self.assertFalse(torch.equal(w._quantize_activation(x), x))

    def test_activation_passthrough_for_weight_only_scheme(self):
        # Regression test: the existing weight-only mxfp4 scheme (no input_quantizer) must be
        # completely unaffected by adding weight+activation support.
        w = WrapperLinearMXFP4(self._make_mxfp4_quant_linear())
        self.assertFalse(w.quantize_activation)
        x = torch.randn(4, 64)
        self.assertTrue(torch.equal(w._quantize_activation(x), x))

    def test_no_learnable_parameters_added_for_activation(self):
        # The instruction this feature was built to satisfy: activation quantization must add
        # ZERO trainable parameters -- only weight-side V/max_scale are ever learnable.
        # named_parameters(recurse=False) excludes `self.linear`'s own (frozen) weight submodule.
        w_off = WrapperLinearMXFP4(self._make_mxfp4_wa_quant_linear())
        names_off = {n for n, _ in w_off.named_parameters(recurse=False)}
        self.assertEqual(names_off, {"value"})

        w_on = WrapperLinearMXFP4(self._make_mxfp4_wa_quant_linear(), enable_minmax_tuning=True)
        names_on = {n for n, _ in w_on.named_parameters(recurse=False)}
        self.assertEqual(names_on, {"value", "max_scale"})

    def test_forward_runs_and_differs_from_weight_only(self):
        torch.manual_seed(0)
        wa_linear = self._make_mxfp4_wa_quant_linear()
        torch.manual_seed(0)
        w_only_linear = self._make_mxfp4_quant_linear()
        w_only_linear.weight.data.copy_(wa_linear.weight.data)

        w_wa = WrapperLinearMXFP4(wa_linear)
        w_only = WrapperLinearMXFP4(w_only_linear)
        x = torch.randn(4, 64)
        out_wa = w_wa(x)
        out_only = w_only(x)
        self.assertEqual(out_wa.shape, out_only.shape)
        self.assertFalse(
            torch.allclose(out_wa, out_only),
            "weight+activation output identical to weight-only — activation quantization isn't actually being applied",
        )

    def test_gradient_still_reaches_max_scale_with_activation_quant_on(self):
        # Activation quantization must not break the weight-side gradient path.
        w = WrapperLinearMXFP4(self._make_mxfp4_wa_quant_linear(), enable_minmax_tuning=True)
        out = w(torch.randn(4, 64))
        (grad,) = torch.autograd.grad(out.sum(), w.max_scale, retain_graph=True)
        self.assertTrue(torch.any(grad != 0.0))


if __name__ == "__main__":
    unittest.main()
