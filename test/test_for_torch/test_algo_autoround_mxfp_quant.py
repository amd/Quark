import unittest

import torch

import quark.torch  # noqa: F401 -- import the top package first to avoid a circular-import

# ordering issue when quark.experimental.* is imported standalone (pre-existing, unrelated to
# this module).
from quark.experimental.torch.autoround.mxfp_quant import (
    FP4_MAX_NORM,
    mxfp4_fake_quantize,
    mxfp4_quantize_dequantize,
    mxfp4_scale,
    mxfp4_scale_floor,
)
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.utils import even_round


class TestMXFP4Scale(unittest.TestCase):
    def test_matches_quark_even_round_reference(self):
        # Per plan3-autoround-mxfp4.md §7 item 1: the fidelity target is Quark's own kernel
        # (even_round), NOT the official auto-round repo's plain floor(log2)-emax formula.
        amax = torch.tensor([0.1, 1.0, 3.9999, 4.0, 4.0001, 6.0, 100.0])
        mine = mxfp4_scale(amax)
        theirs = even_round(amax, Dtype.fp4)
        self.assertTrue(torch.allclose(mine, theirs))

    def test_zero_amax_does_not_produce_nan_or_inf(self):
        amax = torch.tensor([0.0])
        scale = mxfp4_scale(amax)
        self.assertFalse(torch.isnan(scale).any())
        self.assertFalse(torch.isinf(scale).any())

    def test_gradient_reaches_max_abs(self):
        # Adversarial autograd test (mirrors the INT4 wrapper's
        # test_zero_point_ste_gradient_reaches_minmax_scale): the discrete bit-trick scale must
        # still carry a nonzero gradient back to whatever produced it (the AutoRound-tunable
        # max_scale, once wired into the wrapper).
        max_abs = torch.tensor([2.5], requires_grad=True)
        scale = mxfp4_scale(max_abs)
        (grad,) = torch.autograd.grad(scale.sum(), max_abs)
        self.assertNotEqual(grad.item(), 0.0)


class TestMXFP4ScaleFloorMode(unittest.TestCase):
    """'floor' scale_calculation_mode: matches the official auto-round repo's formula exactly
    (unlike the default 'even' mode, which matches Quark's HIP kernel but not the official
    repo) -- see plan3-autoround-mxfp4.md's follow-up on aligning with the official repo."""

    def test_matches_quark_observer_floor_mode_reference(self):
        from quark.torch.quantization.config.config import OCP_MXFP4Spec
        from quark.torch.quantization.observer.observer import PerBlockMXObserver

        spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False, scale_calculation_mode="floor").to_quantization_spec()
        obs = PerBlockMXObserver(spec)
        amax = torch.tensor([0.1, 1.0, 3.9999, 4.0, 4.0001, 6.0, 100.0])
        theirs = obs.get_scale(amax, torch.finfo(torch.float32).eps, "e8m0")
        mine = mxfp4_scale_floor(amax)
        self.assertTrue(torch.allclose(mine, theirs))

    def test_dispatch_by_name(self):
        amax = torch.tensor([2.5])
        self.assertTrue(torch.equal(mxfp4_scale(amax, "floor"), mxfp4_scale_floor(amax)))
        with self.assertRaises(ValueError):
            mxfp4_scale(amax, "bogus")

        # floor and even are genuinely different formulas -- must disagree somewhere, or "floor"
        # isn't wired to anything different.
        boundary = torch.tensor([3.9999])
        self.assertFalse(torch.allclose(mxfp4_scale_floor(boundary), mxfp4_scale(boundary, "even")))

    def test_gradient_reaches_max_abs(self):
        max_abs = torch.tensor([2.5], requires_grad=True)
        scale = mxfp4_scale_floor(max_abs)
        (grad,) = torch.autograd.grad(scale.sum(), max_abs)
        self.assertNotEqual(grad.item(), 0.0)


class TestMXFP4QuantizeDequantize(unittest.TestCase):
    def test_matches_quark_kernel_reference(self):
        # The real reference is the compiled op Quark's actual static-quantization dispatch
        # (fake_quantize_fp4_fp6_per_group_with_scale) and real/packed quantization
        # (real_quantize_fp4_fp6_per_group) both call -- NOT
        # quark/torch/kernel/mx/triton.py::downcast_to_mxfp_torch, which has no real dispatch
        # caller anywhere in the codebase (see plan3-autoround-mxfp4.md §7 item 1's correction).
        from quark.torch.quantization.config.type import Dtype
        from quark.torch.quantization.utils import calculate_qmin_qmax, get_dtype_params

        x = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                5.0,
                5.1,
                6.0,
                5.9,
                -0.75,
                -1.25,
                -2.9,
                0.1,
                0.2,
                0.3,
                0.4,
                0.6,
                0.7,
                0.8,
                0.9,
                1.1,
                1.2,
                1.3,
                1.4,
                1.6,
                1.7,
                1.8,
                1.9,
            ]
        )
        ebits, mbits, _ = get_dtype_params(Dtype.fp4)
        _, max_norm = calculate_qmin_qmax(Dtype.fp4)
        theirs = torch.ops.quark.fake_quantize_to_low_precision_fp(x, ebits, mbits, max_norm, 0)

        mine = mxfp4_quantize_dequantize(x)
        self.assertTrue(torch.equal(mine, theirs))

    def test_beyond_max_norm_clamps_without_nan(self):
        # Non-finite input is out of scope: Quark's own reference kernel also produces NaN/inf
        # garbage for inf inputs (the shared-exponent scale itself blows up), and real weight
        # tensors are always finite, so this isn't a fidelity gap -- only finite large magnitudes
        # need to clamp cleanly.
        big = torch.tensor([100.0, -100.0])
        out = mxfp4_quantize_dequantize(big)
        self.assertTrue(torch.equal(out.abs(), torch.full((2,), FP4_MAX_NORM)))
        self.assertFalse(torch.isnan(out).any())

    def test_gradient_is_straight_through(self):
        x_scaled = torch.tensor([0.83], requires_grad=True)
        out = mxfp4_quantize_dequantize(x_scaled)
        (grad,) = torch.autograd.grad(out.sum(), x_scaled)
        self.assertEqual(grad.item(), 1.0)

    def test_gradient_is_zero_beyond_max_norm(self):
        # Regression test: matches the official auto-round repo's quant_mx, which clamps
        # x/scale+v with plain (non-STE) torch.clamp before the STE-differentiable rounding step
        # -- real clamp semantics give zero gradient once saturated. An earlier version wrapped
        # the whole op (including saturation) in one top-level STE, which always returned
        # gradient 1 even for already-saturated inputs, unlike the official repo.
        for x_val in (8.0, -9.0, FP4_MAX_NORM + 1e-3):
            x_scaled = torch.tensor([x_val], requires_grad=True)
            out = mxfp4_quantize_dequantize(x_scaled)
            (grad,) = torch.autograd.grad(out.sum(), x_scaled)
            self.assertEqual(out.abs().item(), FP4_MAX_NORM, f"forward should saturate at x={x_val}")
            self.assertEqual(grad.item(), 0.0, f"gradient should be zero once saturated, x={x_val}")

        # Just inside the boundary: still gets a real (nonzero, finite) gradient -- not
        # necessarily exactly 1.0, since the in-range gradient is value-dependent (see
        # test_gradient_matches_official_repo_grid_derivative).
        x_scaled = torch.tensor([FP4_MAX_NORM - 1e-3], requires_grad=True)
        out = mxfp4_quantize_dequantize(x_scaled)
        (grad,) = torch.autograd.grad(out.sum(), x_scaled)
        self.assertGreater(grad.item(), 0.0)
        self.assertTrue(torch.isfinite(grad).item())

    def test_gradient_matches_official_repo_grid_derivative(self):
        # Regression test: an earlier version used a flat top-level STE (gradient always exactly
        # 1.0, independent of input value) for the grid-rounding step. The official auto-round
        # repo's quant_element differentiates through each element's own private-exponent
        # normalization (extract exponent -> rescale -> round -> rescale back), giving a gradient
        # that depends on where in its exponent octave the value falls -- NOT a constant 1. This
        # test calls the official repo's own quant_element (via paper-reprise's checked-out
        # commit) with its real mx_fp4 parameters (ebits=2, mbits=3 -- note: mbits=3 in the
        # official repo's OWN bit-counting convention, which differs from FP4_MBITS=1 used
        # elsewhere in this module for the same E2M1 format) and asserts Quark's gradient matches
        # it exactly (forward output too, as a sanity check) across values spanning subnormal,
        # normal, and near-saturation regions.
        import importlib.util
        import os

        repo_path = os.environ.get(
            "AUTO_ROUND_REPO_PATH",
            "/proj/rdi/staff/zhaolin/code/paper-reprise/runs/"
            "optimize-weight-rounding-via-signed-grad-2309.05516-20260704-023345/repo",
        )
        mxfp_path = os.path.join(repo_path, "auto_round", "data_type", "mxfp.py")
        if not os.path.exists(mxfp_path):
            self.skipTest(f"official auto-round repo checkout not found at {repo_path}")

        spec = importlib.util.spec_from_file_location("_official_mxfp", mxfp_path)
        official_mxfp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official_mxfp)
        ebits, mbits, _emax, max_norm, _min_norm = official_mxfp.MXFP_FORMAT_CACHE["mx_fp4"]
        self.assertEqual(max_norm, FP4_MAX_NORM)

        for x_val in (0.05, 0.3, 0.7, 1.0, 1.3, 1.75, 2.2, 2.5, 3.5, 4.9, 5.9, 8.0, -0.83, -2.7, -5.5, -9.0):
            x_quark = torch.tensor([x_val], requires_grad=True)
            out_quark = mxfp4_quantize_dequantize(x_quark)
            (g_quark,) = torch.autograd.grad(out_quark.sum(), x_quark)

            x_official = torch.tensor([x_val], requires_grad=True)
            clamped = torch.clamp(x_official, min=-max_norm, max=max_norm)
            out_official = official_mxfp.quant_element(
                clamped, ebits=ebits, mbits=mbits, max_norm=max_norm, mantissa_rounding="even"
            )
            (g_official,) = torch.autograd.grad(out_official.sum(), x_official)

            self.assertEqual(out_quark.item(), out_official.item(), f"forward mismatch at x={x_val}")
            self.assertAlmostEqual(g_quark.item(), g_official.item(), places=5, msg=f"gradient mismatch at x={x_val}")


class TestMXFP4FakeQuantize(unittest.TestCase):
    def test_v_zero_max_scale_default_matches_plain_quantize(self):
        x = torch.tensor([2.5, -1.25, 0.3])
        scale = mxfp4_scale(x.abs().max().unsqueeze(0))
        via_fake_quantize = mxfp4_fake_quantize(x, scale, v=0.0)
        via_direct = mxfp4_quantize_dequantize(x / scale) * scale
        self.assertTrue(torch.equal(via_fake_quantize, via_direct))

    def test_v_shifts_the_quantized_output(self):
        x = torch.tensor([1.0])
        scale = torch.tensor([1.0])
        base = mxfp4_fake_quantize(x, scale, v=0.0)
        shifted = mxfp4_fake_quantize(x, scale, v=0.4)
        self.assertNotEqual(base.item(), shifted.item())


if __name__ == "__main__":
    unittest.main()
