#
# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Unit tests for native FP8 inference support using AMD Aiter kernels.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from quark.common.utils.import_utils import is_aiter_available
from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.nn.modules.aiter_fp8_inference_linear import (
    AITER_NATIVE_CLASS_FOR_MODE,
    AiterFP8PerTensorNativeInferenceLinear,
)
from quark.torch.quantization.nn.modules.native_inference_linear_common import (
    NativeInferenceLinear,
    NativeInferenceMode,
    _preshuffle_weight,
    _unshuffle_weight,
)
from quark.torch.quantization.utils import (
    disable_native_inference,
    enable_native_inference,
)


class TestNativeInferenceMode:
    """Tests for NativeInferenceMode enum."""

    def test_fp8_modes_exist(self):
        """Test that the FP8 inference modes currently supported are defined."""
        assert hasattr(NativeInferenceMode, "FP8_PER_TENSOR")


class TestAiterAvailability:
    """Tests for Aiter availability checking."""

    def test_is_aiter_available_returns_bool(self):
        """Test that is_aiter_available returns a boolean."""
        result = is_aiter_available()
        assert isinstance(result, bool)

    def test_aiter_kernel_wrapper_is_aiter_available(self):
        """Test that kernel wrapper has is_aiter_available function."""
        from quark.torch.kernel.aiter import is_aiter_available as kernel_is_aiter_available

        result = kernel_is_aiter_available()
        assert isinstance(result, bool)


class TestNativeInferenceLinearMixin:
    """Tests for NativeInferenceLinear marker mixin."""

    def test_mixin_has_enabled_flag(self):
        """Test that the mixin defines _native_inference_enabled = True."""
        assert hasattr(NativeInferenceLinear, "_native_inference_enabled")
        assert NativeInferenceLinear._native_inference_enabled is True

    def test_aiter_classes_are_subclasses(self):
        """Test that all Aiter native classes inherit NativeInferenceLinear."""
        assert issubclass(AiterFP8PerTensorNativeInferenceLinear, NativeInferenceLinear)

    def test_aiter_classes_are_not_qparamslinear_subclasses(self):
        """Test that native classes are decoupled from QParamsLinear inheritance."""
        from quark.torch.export.nn.modules.qparamslinear import QParamsLinear

        assert not issubclass(AiterFP8PerTensorNativeInferenceLinear, QParamsLinear)


class TestAiterNativeClassForMode:
    """Tests for AITER_NATIVE_CLASS_FOR_MODE dispatch dict."""

    def test_dispatch_dict_has_all_fp8_modes(self):
        """Test that dispatch dict covers all currently-supported FP8 modes."""
        assert NativeInferenceMode.FP8_PER_TENSOR in AITER_NATIVE_CLASS_FOR_MODE

    def test_dispatch_dict_maps_to_correct_classes(self):
        """Test that dispatch dict maps modes to the correct classes."""
        assert AITER_NATIVE_CLASS_FOR_MODE[NativeInferenceMode.FP8_PER_TENSOR] is AiterFP8PerTensorNativeInferenceLinear

    def test_dispatch_dict_values_are_subclasses(self):
        """Test that dispatch dict values are NativeInferenceLinear subclasses."""
        for mode, cls in AITER_NATIVE_CLASS_FOR_MODE.items():
            assert issubclass(cls, NativeInferenceLinear), (
                f"{cls} for mode {mode} is not a NativeInferenceLinear subclass"
            )


class TestEnableNativeInferenceAPI:
    """Tests for the enable_native_inference API."""

    def test_enable_native_inference_without_aiter_raises(self):
        """Test that enabling native inference without Aiter raises ImportError."""
        model = nn.Sequential(nn.Linear(64, 32))

        with patch("quark.torch.kernel.aiter.is_aiter_available", return_value=False):
            with pytest.raises(ImportError) as exc_info:
                enable_native_inference(model)

            assert "AMD Aiter" in str(exc_info.value)

    def test_disable_native_inference_no_error(self):
        """Test that disable_native_inference works on a plain model."""
        model = nn.Sequential(nn.Linear(64, 32))
        count = disable_native_inference(model)
        assert count == 0

    def test_enable_on_model_without_qparamslinear(self):
        """Test enable_native_inference on model with no QParamsLinear layers."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
        )

        with patch("quark.torch.kernel.aiter.is_aiter_available", return_value=True):
            count = enable_native_inference(model)
            assert count == 0


class TestAiterGemmWrappers:
    """Tests for Aiter GEMM wrapper argument normalization."""

    def test_gemm_fp8_bpreshuffle_normalizes_scales(self, monkeypatch):
        """Ensure bpreshuffle wrapper normalizes and casts scale tensors."""
        import quark.torch.kernel.aiter.gemm as aiter_gemm

        try:
            from aiter.ops.shuffle import shuffle_weight
        except ImportError:
            pytest.skip("aiter.ops.shuffle is not available in this environment")

        assert callable(shuffle_weight)

        captured: dict[str, torch.Tensor | torch.dtype | None] = {}

        def _fake_check_available() -> None:
            return None

        def _fake_bpreshuffle_kernel(
            XQ: torch.Tensor,
            WQ: torch.Tensor,
            x_scale: torch.Tensor,
            w_scale: torch.Tensor,
            bias: torch.Tensor | None = None,
            dtype: torch.dtype | None = None,
        ) -> torch.Tensor:
            captured["WQ"] = WQ
            captured["x_scale"] = x_scale
            captured["w_scale"] = w_scale
            captured["dtype"] = dtype
            return torch.zeros((XQ.shape[0], WQ.shape[0]), dtype=dtype or torch.bfloat16)

        monkeypatch.setattr(aiter_gemm, "_check_aiter_available", _fake_check_available)
        monkeypatch.setattr(aiter_gemm, "_aiter_gemm_a8w8_bpreshuffle", _fake_bpreshuffle_kernel, raising=False)

        if not torch.cuda.is_available():
            pytest.skip("shuffle_weight requires CUDA/ROCm device")

        xq = torch.zeros((4, 8), dtype=torch.float16)
        # Build FP8 weight and run real Aiter shuffle op to produce preshuffled layout.
        wq = torch.arange(128 * 128, device="cuda", dtype=torch.float32).view(128, 128).to(torch.float8_e4m3fn)
        try:
            wq_reshuffled = shuffle_weight(wq)
        except Exception as exc:
            pytest.skip(f"shuffle_weight is unavailable for this runtime: {exc}")
        x_scale = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)  # 1D [M] -> expected [M, 1]
        w_scale = torch.tensor(0.25, dtype=torch.float32)  # scalar -> expected [1, N]

        out = aiter_gemm.gemm_fp8_bpreshuffle(
            XQ=xq,
            WQ=wq_reshuffled,
            x_scale=x_scale,
            w_scale=w_scale,
            bias=None,
            output_dtype=torch.bfloat16,
        )

        M, N = 4, 128
        assert out.shape == (M, N)
        assert out.dtype == torch.bfloat16
        assert isinstance(captured["x_scale"], torch.Tensor)
        assert isinstance(captured["w_scale"], torch.Tensor)
        assert isinstance(captured["WQ"], torch.Tensor)
        assert torch.equal(captured["WQ"], wq_reshuffled)
        assert captured["WQ"].shape == wq.shape
        assert captured["WQ"].dtype == wq.dtype
        assert captured["x_scale"].shape == (M, 1), f"expected [M,1], got {captured['x_scale'].shape}"
        assert captured["w_scale"].shape == (1, N), f"expected [1,N], got {captured['w_scale'].shape}"
        assert captured["x_scale"].dtype == torch.float32
        assert captured["w_scale"].dtype == torch.float32
        assert captured["dtype"] == torch.bfloat16

    def test_gemm_fp8_bpreshuffle_matches_torch_gemm(self):
        """Compare real bpreshuffle GEMM output against Torch GEMM reference."""
        if not torch.cuda.is_available():
            pytest.skip("Aiter kernels require CUDA/ROCm device")

        import quark.torch.kernel.aiter.gemm as aiter_gemm

        if not aiter_gemm.is_aiter_available():
            pytest.skip("Aiter GEMM kernels are not available")

        try:
            from aiter.ops.shuffle import shuffle_weight
        except ImportError:
            pytest.skip("aiter.ops.shuffle is not available in this environment")

        from quark.torch.kernel.aiter.quant import dynamic_per_tensor_quant_fp8

        torch.manual_seed(0)
        # Aiter heuristic dispatch generally requires K > 192 and K % 64 == 0.
        m, k, n = 16, 256, 256

        # Build original quantized FP8 weight and preshuffled runtime layout.
        w_float = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.1
        w_scale = torch.tensor(0.25, device="cuda", dtype=torch.float32)
        wq = (w_float / w_scale).to(torch.float8_e4m3fn)
        wq_preshuffled = shuffle_weight(wq)
        assert wq_preshuffled.dtype == torch.float8_e4m3fn
        assert w_scale.dtype == torch.float32

        # Quantize input with Aiter dynamic FP8 path to match runtime flow.
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        xq, x_scale = dynamic_per_tensor_quant_fp8(x)
        assert xq.dtype == torch.float8_e4m3fn
        assert x_scale.dtype == torch.float32

        # Real Aiter preshuffle GEMM (skip unsupported shapes/configs).
        # Pass raw scalar scales — the wrapper handles reshape and bf16 cast.
        try:
            out_aiter = aiter_gemm.gemm_fp8_bpreshuffle(
                XQ=xq,
                WQ=wq_preshuffled,
                x_scale=x_scale,
                w_scale=w_scale,
                bias=None,
                output_dtype=torch.bfloat16,
            )
        except RuntimeError as exc:
            if "not supported" in str(exc).lower() or "unsupported" in str(exc).lower():
                pytest.skip(f"Aiter bpreshuffle GEMM unsupported for this config: {exc}")
            raise

        # Torch reference in dequantized domain.
        x_scale_scalar = x_scale.float().reshape(-1)[0]
        w_scale_scalar = w_scale.float().reshape(-1)[0]
        x_deq = xq.float() * x_scale_scalar
        w_deq = wq.float() * w_scale_scalar
        out_ref = torch.matmul(x_deq, w_deq.t()).to(torch.bfloat16)

        assert torch.all(torch.isfinite(out_aiter))
        assert torch.all(torch.isfinite(out_ref))
        torch.testing.assert_close(out_aiter, out_ref, atol=2e-3, rtol=2e-3)

    def test_fake_fp8_gemm_simulation_error_budget(self):
        """Simulate fake-FP8 GEMM vs bf16 GEMM and validate expected max diff."""
        torch.manual_seed(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m, k, n = 16, 256, 256

        # Match bpreshuffle test setup.
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
        w_float = torch.randn(n, k, device=device, dtype=torch.float32) * 0.1
        w_scale = torch.tensor(0.25, device=device, dtype=torch.float32)

        # Simulate dynamic_per_tensor_quant_fp8 input quantization.
        x_max = x.float().abs().max()
        x_scale = (x_max / 448.0).clamp(min=1e-8)
        xq = (x.float() / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        # Simulate pre-quantized FP8 weight.
        wq = (w_float / w_scale).to(torch.float8_e4m3fn)

        x_deq = xq.float() * x_scale
        w_deq = wq.float() * w_scale

        # "Reference" path: float accumulation then cast.
        out_ref = (x_deq @ w_deq.t()).to(torch.bfloat16)
        # "Kernel-like" path: bf16 operands/accumulation behavior.
        out_bf16 = (x_deq.to(torch.bfloat16) @ w_deq.t().to(torch.bfloat16)).to(torch.bfloat16)

        abs_diff = (out_ref.float() - out_bf16.float()).abs()
        rel_diff = abs_diff / (out_ref.float().abs() + 1e-12)
        max_abs = float(abs_diff.max())

        # 1/512 == 0.001953125 is a common bf16 quantization unit.
        assert max_abs <= (1.0 / 512.0 + 1e-7)

        # Relative error near zero is unstable; check on stable magnitudes.
        stable_mask = out_ref.float().abs() > 0.1
        if bool(stable_mask.any()):
            max_rel_stable = float(rel_diff[stable_mask].max())
            assert max_rel_stable <= 0.02


class TestPreshuffleRoundTrip:
    """Tests for preshuffle round-trip correctness."""

    def test_preshuffle_unshuffle_restores_weight(self):
        """Verify preshuffled weight can be reversed back exactly."""
        if not torch.cuda.is_available():
            pytest.skip("preshuffle requires CUDA/ROCm device")

        torch.manual_seed(0)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)

        try:
            preshuffled = _preshuffle_weight(weight)
            restored = _unshuffle_weight(preshuffled)
        except (ImportError, RuntimeError) as exc:
            pytest.skip(f"preshuffle op is unavailable for this runtime: {exc}")
        assert restored.shape == weight.shape
        assert restored.dtype == weight.dtype
        assert torch.equal(restored, weight)


class TestAiterPerTensorForward:
    """Forward-path tests for AiterFP8PerTensorNativeInferenceLinear."""

    @staticmethod
    def _build_module(
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        use_preshuffle: bool,
    ) -> AiterFP8PerTensorNativeInferenceLinear:
        mod = AiterFP8PerTensorNativeInferenceLinear.__new__(AiterFP8PerTensorNativeInferenceLinear)
        nn.Module.__init__(mod)
        mod.in_features = weight_fp8.shape[1]
        mod.out_features = weight_fp8.shape[0]
        mod.bias = nn.Parameter(
            torch.randn(mod.out_features, device=weight_fp8.device, dtype=torch.bfloat16), requires_grad=False
        )
        mod._output_dtype = torch.bfloat16
        mod.use_preshuffle = use_preshuffle
        mod._weight_postprocessed = False
        mod.register_buffer("_kernel_scale", weight_scale.view(1), persistent=False)

        if use_preshuffle:
            try:
                from aiter.ops.shuffle import shuffle_weight
            except ImportError as exc:
                pytest.skip(f"aiter.ops.shuffle is unavailable: {exc}")
            mod.weight = nn.Parameter(shuffle_weight(weight_fp8.clone()), requires_grad=False)
        else:
            mod.weight = nn.Parameter(weight_fp8, requires_grad=False)
        return mod

    @pytest.mark.parametrize("use_preshuffle", [False, True])
    @pytest.mark.parametrize(
        "x_shape,seed",
        [
            ((16, 256), 0),
            ((2, 8, 256), 1),
            ((4, 8, 256), 2),
        ],
    )
    def test_forward_matches_kernel_reference(self, use_preshuffle: bool, x_shape: tuple[int, ...], seed: int):
        if not torch.cuda.is_available():
            pytest.skip("Aiter forward test requires CUDA/ROCm device")

        import quark.torch.kernel.aiter.gemm as aiter_gemm

        if not aiter_gemm.is_aiter_available():
            pytest.skip("Aiter GEMM kernels are not available")

        from quark.torch.kernel.aiter import dynamic_per_tensor_quant_fp8, gemm_fp8
        from quark.torch.kernel.aiter.gemm import gemm_fp8_bpreshuffle

        torch.manual_seed(seed)
        n, k = 256, 256
        w_float = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.1
        w_scale = torch.tensor(0.25, device="cuda", dtype=torch.float32)
        w_fp8 = (w_float / w_scale).to(torch.float8_e4m3fn)

        mod = self._build_module(w_fp8, w_scale, use_preshuffle=use_preshuffle)
        x = torch.randn(*x_shape, device="cuda", dtype=torch.bfloat16) * 0.1

        out_mod = mod.forward(x)

        x_2d = x.view(-1, mod.in_features)
        x_quant, x_scale = dynamic_per_tensor_quant_fp8(x_2d)
        kernel_weight = mod._get_kernel_weight()

        try:
            if use_preshuffle:
                out_ref = gemm_fp8_bpreshuffle(
                    x_quant,
                    kernel_weight,
                    x_scale,
                    mod._kernel_scale,
                    bias=None,
                    output_dtype=mod._output_dtype,
                )
            else:
                out_ref = gemm_fp8(
                    x_quant,
                    kernel_weight,
                    x_scale.repeat(x_quant.shape[0]),
                    mod._kernel_scale.repeat(kernel_weight.shape[0]),
                    bias=None,
                    output_dtype=mod._output_dtype,
                )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "not supported" in msg or "unsupported" in msg:
                pytest.skip(f"Aiter GEMM unsupported for this shape/config: {exc}")
            raise

        if mod.bias is not None:
            out_ref = out_ref + mod.bias
        out_ref = out_ref.view(*x_shape[:-1], mod.out_features)

        assert torch.all(torch.isfinite(out_mod))
        assert torch.all(torch.isfinite(out_ref))
        torch.testing.assert_close(out_mod, out_ref, atol=5e-3, rtol=5e-3)


class TestAiterPerTensorE2EFromModule:
    """End-to-end create->forward test through from_module API."""

    @staticmethod
    def _make_qparams_linear_source(
        *,
        n: int,
        k: int,
        use_fp8_weight: bool = True,
        seed: int = 123,
    ) -> QParamsLinear:
        torch.manual_seed(seed)
        qpl = QParamsLinear.__new__(QParamsLinear)
        nn.Module.__init__(qpl)

        qpl.in_features = k
        qpl.out_features = n
        qpl.weight = nn.Parameter(
            (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.1).to(torch.float8_e4m3fn)
            if use_fp8_weight
            else (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.1),
            requires_grad=False,
        )
        qpl.bias = nn.Parameter(torch.randn(n, device="cuda", dtype=torch.bfloat16), requires_grad=False)

        # Minimal weight quantizer contract required by native conversion.
        qspec = SimpleNamespace(
            dtype=Dtype.fp8_e4m3,
            qscheme=QSchemeType.per_tensor,
            group_size=None,
            ch_axis=0,
            round_method=SimpleNamespace(value="half_even"),
        )
        weight_quantizer = SimpleNamespace(
            qspec=qspec,
            scale=torch.tensor(0.25, device="cuda", dtype=torch.float32),
            quant_min=-448,
            quant_max=448,
        )
        qpl.weight_quantizer = weight_quantizer
        qpl.input_quantizer = None
        qpl.output_quantizer = None
        qpl.bias_quantizer = None
        qpl._custom_mode = "quark"
        qpl._quant_config = None
        qpl._quant_dict = None
        qpl.algo_config = None
        return qpl

    @pytest.mark.parametrize("use_preshuffle", [False, True])
    def test_from_module_then_forward_matches_kernel_reference(self, use_preshuffle: bool):
        if not torch.cuda.is_available():
            pytest.skip("Aiter e2e test requires CUDA/ROCm device")

        import quark.torch.kernel.aiter.gemm as aiter_gemm

        if not aiter_gemm.is_aiter_available():
            pytest.skip("Aiter GEMM kernels are not available")

        from quark.torch.kernel.aiter import dynamic_per_tensor_quant_fp8, gemm_fp8
        from quark.torch.kernel.aiter.gemm import gemm_fp8_bpreshuffle

        source = self._make_qparams_linear_source(n=256, k=256, use_fp8_weight=True)
        x = torch.randn(2, 8, 256, device="cuda", dtype=torch.bfloat16) * 0.1

        try:
            mod = AiterFP8PerTensorNativeInferenceLinear.from_module(
                source,
                use_preshuffle=use_preshuffle,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "not supported" in msg or "unsupported" in msg:
                pytest.skip(f"Aiter conversion unsupported for this config: {exc}")
            raise

        out_mod = mod.forward(x)

        x_2d = x.view(-1, mod.in_features)
        x_quant, x_scale = dynamic_per_tensor_quant_fp8(x_2d)
        kernel_weight = mod._get_kernel_weight()

        try:
            if use_preshuffle:
                out_ref = gemm_fp8_bpreshuffle(
                    x_quant,
                    kernel_weight,
                    x_scale,
                    mod._kernel_scale,
                    bias=None,
                    output_dtype=mod._output_dtype,
                )
            else:
                out_ref = gemm_fp8(
                    x_quant,
                    kernel_weight,
                    x_scale.repeat(x_quant.shape[0]),
                    mod._kernel_scale.repeat(kernel_weight.shape[0]),
                    bias=None,
                    output_dtype=mod._output_dtype,
                )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "not supported" in msg or "unsupported" in msg:
                pytest.skip(f"Aiter GEMM unsupported for this shape/config: {exc}")
            raise

        if mod.bias is not None:
            out_ref = out_ref + mod.bias
        out_ref = out_ref.view(*x.shape[:-1], mod.out_features)

        assert torch.all(torch.isfinite(out_mod))
        assert torch.all(torch.isfinite(out_ref))
        torch.testing.assert_close(out_mod, out_ref, atol=1e-5, rtol=1e-5)
