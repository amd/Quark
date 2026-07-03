#
# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Tests targeting uncovered lines in native inference modules:
- quark/torch/quantization/api.py (lines 428-429)
- quark/torch/quantization/utils.py (lines 33,35,355,381,385-386,392-394,400,428-442,445)
- quark/torch/quantization/nn/modules/native_inference_linear_common.py
- quark/torch/quantization/nn/modules/aiter_fp8_inference_linear.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
import torch.nn as nn

from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.export.nn.modules.realquantizer import SequentialRealQuantizer
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.nn.modules.aiter_fp8_inference_linear import (
    AiterFP8PerTensorNativeInferenceLinear,
    aiter_native_linear_from_module,
)
from quark.torch.quantization.nn.modules.native_inference_linear_common import (
    NativeInferenceLinear,
    NativeInferenceMode,
    _determine_output_dtype,
    _ensure_weight_real_quantized,
    _extract_input_scale,
    _extract_kernel_state,
    _get_weight_scale,
    _KernelState,
    _preshuffle_weight,
    _require_aiter,
    _resolve_dtype_qscheme_from_source,
    _unshuffle_weight,
    determine_inference_mode,
)
from quark.torch.quantization.utils import (
    RuntimeOptions,
    disable_native_inference,
    enable_native_inference,
    is_aiter_available,
)

# ---------------------------------------------------------------------------
# Helper: minimal QParamsLinear stub for testing
# ---------------------------------------------------------------------------


def _make_stub_qparams_linear(
    *,
    n: int = 32,
    k: int = 64,
    dtype: Dtype = Dtype.fp8_e4m3,
    qscheme: QSchemeType = QSchemeType.per_tensor,
    group_size: int | None = None,
    weight_dtype=torch.float8_e4m3fn,
    scale_val: float = 0.25,
    use_sequential_quantizer: bool = False,
    device: str = "cpu",
) -> QParamsLinear:
    """Build a minimal QParamsLinear stub with controllable quantizer metadata."""
    qpl = QParamsLinear.__new__(QParamsLinear)
    nn.Module.__init__(qpl)

    qpl.in_features = k
    qpl.out_features = n
    w = torch.randn(n, k, dtype=torch.float32) * 0.1
    if weight_dtype is not None:
        w = w.to(weight_dtype)
    qpl.weight = nn.Parameter(w.to(device), requires_grad=False)
    qpl.bias = nn.Parameter(torch.randn(n, dtype=torch.bfloat16).to(device), requires_grad=False)

    qspec = SimpleNamespace(
        dtype=dtype,
        qscheme=qscheme,
        group_size=group_size,
        ch_axis=0,
        round_method=SimpleNamespace(value="half_even"),
    )
    single_quantizer = SimpleNamespace(
        qspec=qspec,
        scale=torch.tensor(scale_val, dtype=torch.float32).to(device),
        quant_min=-448,
        quant_max=448,
    )

    if use_sequential_quantizer:
        seq = MagicMock(spec=SequentialRealQuantizer)
        seq.__getitem__ = MagicMock(return_value=single_quantizer)
        seq.__len__ = MagicMock(return_value=1)
        seq.scale = single_quantizer.scale
        qpl.weight_quantizer = seq
    else:
        qpl.weight_quantizer = single_quantizer

    qpl.input_quantizer = None
    qpl.output_quantizer = None
    qpl.bias_quantizer = None
    qpl._custom_mode = "quark"
    qpl._quant_config = None
    qpl._quant_dict = None
    qpl.algo_config = None
    return qpl


# ===========================================================================
# Tests for quark/torch/quantization/utils.py
# ===========================================================================


class TestIsAiterAvailable:
    """Cover utils.is_aiter_available (lines 33, 35)."""

    def test_returns_bool(self):
        result = is_aiter_available()
        assert isinstance(result, bool)

    def test_returns_false_when_not_available(self):
        with patch(
            "quark.torch.kernel.aiter.is_aiter_available",
            return_value=False,
        ):
            assert is_aiter_available() is False

    def test_returns_true_when_available(self):
        with patch(
            "quark.torch.kernel.aiter.is_aiter_available",
            return_value=True,
        ):
            assert is_aiter_available() is True


class TestEnableNativeInferenceConversion:
    """Cover enable_native_inference with actual QParamsLinear conversion
    (lines 355, 381, 385-386, 392-394, 400).
    """

    def test_enable_with_qparams_linear_converts_layer(self):
        """Successful conversion of QParamsLinear -> NativeInferenceLinear."""
        qpl = _make_stub_qparams_linear()
        model = nn.Sequential()
        model.add_module("linear", qpl)

        fake_native = MagicMock(spec=NativeInferenceLinear)
        fake_native._native_inference_enabled = True

        with (
            patch("quark.torch.kernel.aiter.is_aiter_available", return_value=True),
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.aiter_native_linear_from_module",
                return_value=fake_native,
            ) as mock_factory,
        ):
            count = enable_native_inference(model)

        assert count == 1
        mock_factory.assert_called_once()

    def test_enable_skips_native_inference_linear(self):
        """NativeInferenceLinear modules are skipped (not double-converted)."""
        native_mock = MagicMock(spec=NativeInferenceLinear)
        native_mock._native_inference_enabled = True
        native_mock.named_modules = MagicMock(return_value=iter([("", native_mock)]))

        model = nn.Sequential()
        model.add_module("layer", native_mock)

        with patch("quark.torch.kernel.aiter.is_aiter_available", return_value=True):
            count = enable_native_inference(model)
        assert count == 0

    def test_enable_catches_conversion_error(self):
        """ValueError during conversion is caught and logged as warning."""
        qpl = _make_stub_qparams_linear()
        model = nn.Sequential()
        model.add_module("linear", qpl)

        with (
            patch("quark.torch.kernel.aiter.is_aiter_available", return_value=True),
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.aiter_native_linear_from_module",
                side_effect=ValueError("unsupported mode"),
            ),
        ):
            count = enable_native_inference(model)
        assert count == 0

    def test_enable_with_invalid_mode_raises(self):
        model = nn.Sequential(nn.Linear(8, 4))
        options = RuntimeOptions(native_linear_mode="invalid_mode")
        with (
            patch("quark.torch.kernel.aiter.is_aiter_available", return_value=True),
            pytest.raises(ValueError, match="Unsupported RuntimeOptions"),
        ):
            enable_native_inference(model, runtime_options=options)

    def test_enable_with_runtime_options_passes_forced_mode(self):
        qpl = _make_stub_qparams_linear()
        model = nn.Sequential()
        model.add_module("linear", qpl)

        fake_native = MagicMock(spec=NativeInferenceLinear)
        options = RuntimeOptions(
            native_linear_mode="fp8_per_tensor",
            use_preshuffle=True,
        )

        with (
            patch("quark.torch.kernel.aiter.is_aiter_available", return_value=True),
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.aiter_native_linear_from_module",
                return_value=fake_native,
            ) as mock_factory,
        ):
            count = enable_native_inference(model, runtime_options=options)

        assert count == 1
        call_kwargs = mock_factory.call_args
        assert call_kwargs.kwargs["use_preshuffle"] is True
        assert call_kwargs.kwargs["forced_mode"] == NativeInferenceMode.FP8_PER_TENSOR


class TestDisableNativeInference:
    """Cover disable_native_inference with actual NativeInferenceLinear
    (lines 428-442, 445, 447-448).
    """

    def test_disable_converts_native_to_qparams(self):
        """NativeInferenceLinear is reverted to QParamsLinear."""
        native = AiterFP8PerTensorNativeInferenceLinear.__new__(AiterFP8PerTensorNativeInferenceLinear)
        nn.Module.__init__(native)
        native.in_features = 64
        native.out_features = 32
        native.weight = nn.Parameter(torch.randn(32, 64))
        native.bias = nn.Parameter(torch.randn(32))
        native.weight_quantizer = SimpleNamespace(scale=torch.tensor(0.1))
        native.input_quantizer = None
        native.output_quantizer = None
        native.bias_quantizer = None
        native._custom_mode = "quark"
        native._quant_config = None
        native._quant_dict = None
        native.algo_config = None

        model = nn.Sequential()
        model.add_module("layer", native)

        count = disable_native_inference(model)
        assert count == 1
        replaced = model.layer
        assert isinstance(replaced, QParamsLinear)
        assert replaced.in_features == 64
        assert replaced.out_features == 32


# ===========================================================================
# Tests for native_inference_linear_common.py
# ===========================================================================


class TestRequireAiter:
    """Cover _require_aiter (lines 123-127)."""

    def test_raises_when_aiter_unavailable(self):
        with (
            patch(
                "quark.torch.quantization.nn.modules.native_inference_linear_common.is_aiter_available",
                return_value=False,
            ),
            pytest.raises(ImportError, match="AMD Aiter"),
        ):
            _require_aiter()

    def test_passes_when_aiter_available(self):
        with patch(
            "quark.torch.quantization.nn.modules.native_inference_linear_common.is_aiter_available", return_value=True
        ):
            _require_aiter()


class TestResolveDtypeQscheme:
    """Cover _resolve_dtype_qscheme_from_source (lines 131-147)."""

    def test_from_qparams_linear_simple(self):
        qpl = _make_stub_qparams_linear(dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor)
        dtype, qscheme = _resolve_dtype_qscheme_from_source(qpl)
        assert dtype == Dtype.fp8_e4m3
        assert qscheme == QSchemeType.per_tensor

    def test_from_qparams_linear_sequential_quantizer(self):
        qpl = _make_stub_qparams_linear(
            dtype=Dtype.fp8_e5m2,
            qscheme=QSchemeType.per_channel,
            use_sequential_quantizer=True,
        )
        dtype, qscheme = _resolve_dtype_qscheme_from_source(qpl)
        assert dtype == Dtype.fp8_e5m2
        assert qscheme == QSchemeType.per_channel

    def test_from_qparams_linear_none_quantizer_raises(self):
        qpl = _make_stub_qparams_linear()
        qpl.weight_quantizer = None
        with pytest.raises(ValueError, match="weight_quantizer is None"):
            _resolve_dtype_qscheme_from_source(qpl)

    def test_from_quant_linear_list_qspec(self):
        """Cover the isinstance(wqspec, list) branch."""
        from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear

        ql = MagicMock(spec=QuantLinear)
        type(ql).weight_qspec = PropertyMock(
            return_value=[
                SimpleNamespace(dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor),
                SimpleNamespace(dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor),
            ]
        )
        # Make isinstance check work for QuantLinear
        ql.__class__ = QuantLinear

        dtype, qscheme = _resolve_dtype_qscheme_from_source(ql)
        assert dtype == Dtype.fp8_e4m3
        assert qscheme == QSchemeType.per_tensor

    def test_from_quant_linear_none_qspec_raises(self):
        from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear

        ql = MagicMock(spec=QuantLinear)
        type(ql).weight_qspec = PropertyMock(return_value=None)
        ql.__class__ = QuantLinear

        with pytest.raises(ValueError, match="weight_qspec is None"):
            _resolve_dtype_qscheme_from_source(ql)

    def test_from_quant_linear_single_qspec(self):
        from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear

        ql = MagicMock(spec=QuantLinear)
        type(ql).weight_qspec = PropertyMock(
            return_value=SimpleNamespace(dtype=Dtype.fp8_e5m2, qscheme=QSchemeType.per_channel),
        )
        ql.__class__ = QuantLinear

        dtype, qscheme = _resolve_dtype_qscheme_from_source(ql)
        assert dtype == Dtype.fp8_e5m2
        assert qscheme == QSchemeType.per_channel

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported source type"):
            _resolve_dtype_qscheme_from_source(nn.Linear(4, 4))


class TestDetermineInferenceMode:
    """Cover determine_inference_mode (lines 150-165)."""

    def test_fp8_per_tensor(self):
        qpl = _make_stub_qparams_linear(dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor)
        assert determine_inference_mode(qpl) == NativeInferenceMode.FP8_PER_TENSOR

    def test_fp8_e5m2_per_tensor(self):
        qpl = _make_stub_qparams_linear(dtype=Dtype.fp8_e5m2, qscheme=QSchemeType.per_tensor)
        assert determine_inference_mode(qpl) == NativeInferenceMode.FP8_PER_TENSOR

    def test_mxfp4_per_group(self):
        qpl = _make_stub_qparams_linear(
            dtype=Dtype.fp4,
            qscheme=QSchemeType.per_group,
            group_size=32,
            weight_dtype=torch.uint8,
        )
        assert determine_inference_mode(qpl) == NativeInferenceMode.MXFP4

    def test_unsupported_dtype_raises(self):
        qpl = _make_stub_qparams_linear(dtype=Dtype.int8, qscheme=QSchemeType.per_tensor)
        with pytest.raises(ValueError, match="Unsupported quantization configuration"):
            determine_inference_mode(qpl)

    def test_unsupported_qscheme_raises(self):
        qpl = _make_stub_qparams_linear(dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_channel)
        with pytest.raises(ValueError, match="Unsupported quantization configuration"):
            determine_inference_mode(qpl)


class TestGetWeightScale:
    """Cover _get_weight_scale (lines 237-249)."""

    def test_simple_quantizer(self):
        qpl = _make_stub_qparams_linear(scale_val=0.5)
        scale = _get_weight_scale(qpl)
        assert float(scale) == pytest.approx(0.5)

    def test_sequential_quantizer(self):
        qpl = _make_stub_qparams_linear(scale_val=0.3, use_sequential_quantizer=True)
        scale = _get_weight_scale(qpl)
        assert float(scale) == pytest.approx(0.3)


class TestDetermineOutputDtype:
    """Cover _determine_output_dtype (lines 260-264)."""

    def test_bfloat16_weight(self):
        qpl = _make_stub_qparams_linear(weight_dtype=torch.bfloat16)
        assert _determine_output_dtype(qpl) == torch.bfloat16

    def test_float16_weight(self):
        qpl = _make_stub_qparams_linear(weight_dtype=torch.float16)
        assert _determine_output_dtype(qpl) == torch.float16

    def test_float32_weight(self):
        qpl = _make_stub_qparams_linear(weight_dtype=torch.float32)
        assert _determine_output_dtype(qpl) == torch.float32

    def test_fp8_weight_defaults_to_bfloat16(self):
        qpl = _make_stub_qparams_linear(weight_dtype=torch.float8_e4m3fn)
        assert _determine_output_dtype(qpl) == torch.bfloat16


class TestExtractInputScale:
    """Cover _extract_input_scale helper across input-quantizer shapes."""

    def test_returns_none_when_no_input_quantizer_attr(self):
        src = SimpleNamespace()  # no input_quantizer
        assert _extract_input_scale(src) is None

    def test_returns_none_when_input_quantizer_is_none(self):
        src = SimpleNamespace(input_quantizer=None)
        assert _extract_input_scale(src) is None

    def test_returns_none_when_no_qspec(self):
        iq = SimpleNamespace(scale=torch.tensor(0.5))  # no qspec / quant_spec
        src = SimpleNamespace(input_quantizer=iq)
        assert _extract_input_scale(src) is None

    def test_returns_none_when_dynamic_quant(self):
        qspec = SimpleNamespace(is_dynamic=True)
        iq = SimpleNamespace(qspec=qspec, scale=torch.tensor(0.5))
        src = SimpleNamespace(input_quantizer=iq)
        assert _extract_input_scale(src) is None

    def test_returns_none_when_static_but_no_scale(self):
        qspec = SimpleNamespace(is_dynamic=False)
        iq = SimpleNamespace(qspec=qspec, scale=None)
        src = SimpleNamespace(input_quantizer=iq)
        assert _extract_input_scale(src) is None

    def test_returns_scale_for_static_quantizer(self):
        qspec = SimpleNamespace(is_dynamic=False)
        scale = torch.tensor(0.123, dtype=torch.float32)
        iq = SimpleNamespace(qspec=qspec, scale=scale)
        src = SimpleNamespace(input_quantizer=iq)

        out = _extract_input_scale(src)
        assert out is not None
        assert out.dtype == torch.float32
        assert out.shape == (1,)
        assert torch.allclose(out, scale.reshape(-1))

    def test_accepts_legacy_quant_spec_attr(self):
        """Older quantizers exposed ``quant_spec`` instead of ``qspec``."""
        quant_spec = SimpleNamespace(is_dynamic=False)
        scale = torch.tensor(0.42, dtype=torch.float32)
        iq = SimpleNamespace(qspec=None, quant_spec=quant_spec, scale=scale)
        src = SimpleNamespace(input_quantizer=iq)

        out = _extract_input_scale(src)
        assert out is not None
        assert torch.allclose(out, scale.reshape(-1))


class TestExtractKernelState:
    """Cover _extract_kernel_state on a typical FP8 per-tensor source."""

    def test_returns_kernel_state_with_expected_fields(self):
        qpl = _make_stub_qparams_linear(scale_val=0.25)
        state = _extract_kernel_state(qpl)

        assert isinstance(state, _KernelState)
        # ``Parameter.data`` returns a fresh view each access, so identity
        # (`is`) doesn't hold; compare underlying storage instead. (FP8
        # tensors don't support ``torch.equal`` directly — view-as-uint8.)
        assert state.weight.data_ptr() == qpl.weight.data_ptr()
        assert state.weight.shape == qpl.weight.shape
        assert torch.equal(state.weight.view(torch.uint8), qpl.weight.data.view(torch.uint8))
        assert state.weight_scale.dtype == torch.float32
        assert state.weight_scale.shape == (1,)
        assert torch.allclose(state.weight_scale, torch.tensor([0.25]))
        assert state.bias.data_ptr() == qpl.bias.data_ptr()
        assert torch.equal(state.bias, qpl.bias.data)
        assert state.in_features == qpl.in_features
        assert state.out_features == qpl.out_features
        assert state.output_dtype == torch.bfloat16  # fp8 weight defaults
        assert state.input_scale is None  # stub has no input quantizer

    def test_propagates_static_input_scale(self):
        qpl = _make_stub_qparams_linear()
        qpl.input_quantizer = SimpleNamespace(
            qspec=SimpleNamespace(is_dynamic=False),
            scale=torch.tensor(0.7, dtype=torch.float32),
        )
        state = _extract_kernel_state(qpl)

        assert state.input_scale is not None
        assert torch.allclose(state.input_scale, torch.tensor([0.7]))

    def test_kernel_state_is_frozen(self):
        from dataclasses import FrozenInstanceError

        qpl = _make_stub_qparams_linear()
        state = _extract_kernel_state(qpl)
        with pytest.raises(FrozenInstanceError):
            state.weight = torch.zeros(1)  # type: ignore[misc]


class TestApplyKernelState:
    """Cover NativeInferenceLinear._apply_kernel_state default impl."""

    @staticmethod
    def _empty_mod() -> AiterFP8PerTensorNativeInferenceLinear:
        mod = AiterFP8PerTensorNativeInferenceLinear.__new__(
            AiterFP8PerTensorNativeInferenceLinear,
        )
        nn.Module.__init__(mod)
        # Provide a weight so the override's ``mode = determine_inference_mode(self)``
        # check has something to inspect — though we'll bypass the override and call
        # the base impl directly via ``NativeInferenceLinear._apply_kernel_state``.
        mod.in_features = 64
        mod.out_features = 32
        return mod

    def test_default_registers_kernel_scale_and_output_dtype(self):
        mod = self._empty_mod()
        state = _KernelState(
            weight=torch.empty(0),
            weight_scale=torch.tensor([0.5], dtype=torch.float32),
            bias=None,
            in_features=64,
            out_features=32,
            output_dtype=torch.float16,
            input_scale=None,
        )
        # Bypass the Aiter override to exercise the *base* default impl.
        NativeInferenceLinear._apply_kernel_state(mod, state)

        assert torch.equal(mod._kernel_scale, torch.tensor([0.5]))
        assert mod._input_scale is None
        assert mod._output_dtype == torch.float16

    def test_default_registers_input_scale_when_present(self):
        mod = self._empty_mod()
        state = _KernelState(
            weight=torch.empty(0),
            weight_scale=torch.tensor([0.25], dtype=torch.float32),
            bias=None,
            in_features=64,
            out_features=32,
            output_dtype=torch.bfloat16,
            input_scale=torch.tensor([0.9], dtype=torch.float32),
        )
        NativeInferenceLinear._apply_kernel_state(mod, state)

        assert mod._input_scale is not None
        assert torch.allclose(mod._input_scale, torch.tensor([0.9]))


class TestToQParamsLinear:
    """Cover NativeInferenceLinear.to_qparams_linear round-trip."""

    def test_rebuilds_qparams_linear_from_state(self):
        qpl = _make_stub_qparams_linear()
        with patch(
            "quark.torch.quantization.nn.modules.native_inference_linear_common.is_aiter_available",
            return_value=True,
        ):
            native = AiterFP8PerTensorNativeInferenceLinear.from_qparams_linear(qpl)

        rebuilt = native.to_qparams_linear()
        assert isinstance(rebuilt, QParamsLinear)
        assert rebuilt.in_features == qpl.in_features
        assert rebuilt.out_features == qpl.out_features
        assert rebuilt.weight is qpl.weight
        assert rebuilt.bias is qpl.bias
        assert rebuilt.weight_quantizer is qpl.weight_quantizer
        assert rebuilt._custom_mode == qpl._custom_mode

    def test_unshuffles_weight_when_preshuffled(self):
        """If the weight was preshuffled, ``to_qparams_linear`` un-shuffles
        first via ``postprocess_weight`` so the resulting QPL has the
        canonical on-disk layout."""
        mod = AiterFP8PerTensorNativeInferenceLinear.__new__(
            AiterFP8PerTensorNativeInferenceLinear,
        )
        nn.Module.__init__(mod)
        mod.in_features = 32
        mod.out_features = 16
        mod.weight = nn.Parameter(torch.randn(16, 32), requires_grad=False)
        mod.bias = None
        mod.weight_quantizer = None
        mod.input_quantizer = None
        mod.output_quantizer = None
        mod.bias_quantizer = None
        mod._custom_mode = "quark"
        mod._quant_config = None
        mod._quant_dict = None
        mod.algo_config = None
        mod.use_preshuffle = True
        mod._weight_postprocessed = False

        sentinel = torch.zeros(16, 32)
        with patch(
            "quark.torch.quantization.nn.modules.native_inference_linear_common._unshuffle_weight",
            return_value=sentinel,
        ) as mock_unshuffle:
            qpl = mod.to_qparams_linear()

        mock_unshuffle.assert_called_once()
        assert torch.equal(qpl.weight.data, sentinel)


class TestQParamsLinearBridge:
    """Cover the dedicated _QParamsLinearBridge adapter directly.

    The bridge owns the only QPL field-list used for native-inference
    state transfer; these tests pin its public surface (FIELDS,
    build_from_source, adopt, materialize) so QPL evolution shows up as
    a localized failure here rather than a mysterious downstream bug.
    """

    def test_fields_match_qparams_linear_attributes(self):
        """The declared FIELDS tuple must match real QPL attribute names.

        Guards against typos / drift: every name in ``_FIELDS`` must be
        a readable attribute on a fresh QParamsLinear stub.
        """
        from quark.torch.quantization.nn.modules.qparamslinear_bridge import _QParamsLinearBridge

        qpl = _make_stub_qparams_linear()
        for field in _QParamsLinearBridge._FIELDS:
            assert hasattr(qpl, field), f"_FIELDS references missing QPL attr: {field}"

    def test_fields_includes_expected_core_attrs(self):
        """Sanity check on the field set so accidental deletions are caught."""
        from quark.torch.quantization.nn.modules.qparamslinear_bridge import _QParamsLinearBridge

        expected = {
            "in_features",
            "out_features",
            "weight",
            "bias",
            "weight_quantizer",
            "input_quantizer",
            "output_quantizer",
            "bias_quantizer",
            "_custom_mode",
            "_quant_config",
            "_quant_dict",
            "algo_config",
        }
        assert expected.issubset(set(_QParamsLinearBridge._FIELDS))

    def test_build_from_source_returns_qpl_unchanged(self):
        """If the source is already a QParamsLinear, no rebuild happens."""
        from quark.torch.quantization.nn.modules.qparamslinear_bridge import _QParamsLinearBridge

        qpl = _make_stub_qparams_linear()
        result = _QParamsLinearBridge.build_from_source(qpl)
        assert result is qpl

    def test_build_from_source_invokes_qparams_from_module_for_plain_linear(self):
        """For a plain ``nn.Linear``, the bridge calls ``QParamsLinear.from_module``."""
        from quark.torch.quantization.nn.modules import qparamslinear_bridge

        plain = nn.Linear(8, 16)
        sentinel = object()
        with patch.object(
            qparamslinear_bridge.QParamsLinear,
            "from_module",
            return_value=sentinel,
        ) as mock_from:
            result = qparamslinear_bridge._QParamsLinearBridge.build_from_source(
                plain,
                custom_mode="awq",
                pack_method="reorder",
                quant_config="cfg",
                algo_config="algo",
            )

        mock_from.assert_called_once_with(
            linear=plain,
            custom_mode="awq",
            pack_method="reorder",
            quant_config="cfg",
            algo_config="algo",
        )
        assert result is sentinel

    def test_adopt_copies_every_field_by_reference(self):
        """``adopt`` must transfer every ``_FIELDS`` entry to the target by reference."""
        from quark.torch.quantization.nn.modules.qparamslinear_bridge import _QParamsLinearBridge

        qpl = _make_stub_qparams_linear()
        target = AiterFP8PerTensorNativeInferenceLinear.__new__(
            AiterFP8PerTensorNativeInferenceLinear,
        )
        nn.Module.__init__(target)

        _QParamsLinearBridge.adopt(target, qpl)

        for field in _QParamsLinearBridge._FIELDS:
            tgt_val = getattr(target, field)
            src_val = getattr(qpl, field)
            # Tensors / Parameters: identity holds because nn.Module
            # __setattr__ stores Parameters directly (no copy). Plain
            # Python attrs: simple equality.
            if isinstance(src_val, torch.nn.Parameter):
                assert tgt_val is src_val, f"Parameter field {field} not shared by reference"
            else:
                assert tgt_val is src_val or tgt_val == src_val, (
                    f"Field {field} mismatch: target={tgt_val!r}, source={src_val!r}"
                )

    def test_materialize_round_trips_a_freshly_adopted_module(self):
        """adopt → materialize round-trip must yield a QPL with identical fields."""
        from quark.torch.quantization.nn.modules.qparamslinear_bridge import _QParamsLinearBridge

        qpl = _make_stub_qparams_linear()
        target = AiterFP8PerTensorNativeInferenceLinear.__new__(
            AiterFP8PerTensorNativeInferenceLinear,
        )
        nn.Module.__init__(target)
        _QParamsLinearBridge.adopt(target, qpl)

        rebuilt = _QParamsLinearBridge.materialize(target)
        assert isinstance(rebuilt, QParamsLinear)
        # Geometry, references, metadata all preserved.
        for field in _QParamsLinearBridge._FIELDS:
            assert getattr(rebuilt, field) is getattr(qpl, field) or (getattr(rebuilt, field) == getattr(qpl, field)), (
                f"Field {field} not round-tripped"
            )

    def test_materialize_bypasses_qparams_linear_init(self):
        """``materialize`` uses ``__new__`` + ``nn.Module.__init__`` so QPL.__init__
        (which expects a non-quantized source) is not invoked."""
        from quark.torch.quantization.nn.modules import qparamslinear_bridge

        qpl = _make_stub_qparams_linear()
        target = AiterFP8PerTensorNativeInferenceLinear.__new__(
            AiterFP8PerTensorNativeInferenceLinear,
        )
        nn.Module.__init__(target)
        qparamslinear_bridge._QParamsLinearBridge.adopt(target, qpl)

        with patch.object(
            qparamslinear_bridge.QParamsLinear,
            "__init__",
            side_effect=AssertionError("QParamsLinear.__init__ should not be called"),
        ):
            # Should succeed without calling QPL.__init__.
            rebuilt = qparamslinear_bridge._QParamsLinearBridge.materialize(target)
        assert isinstance(rebuilt, QParamsLinear)


class TestNativeInferenceLinearPrePostProcess:
    """Cover preprocess_weight / postprocess_weight."""

    def test_postprocess_unshuffles_when_use_preshuffle(self):
        mod = AiterFP8PerTensorNativeInferenceLinear.__new__(AiterFP8PerTensorNativeInferenceLinear)
        nn.Module.__init__(mod)
        original_data = torch.randn(16, 32)
        mod.weight = nn.Parameter(original_data.clone(), requires_grad=False)
        mod.use_preshuffle = True
        mod._weight_postprocessed = False

        sentinel = torch.zeros(16, 32)
        with patch(
            "quark.torch.quantization.nn.modules.native_inference_linear_common._unshuffle_weight",
            return_value=sentinel,
        ) as mock_unshuffle:
            mod.postprocess_weight()

        mock_unshuffle.assert_called_once()
        assert mod._weight_postprocessed is True
        assert torch.equal(mod.weight.data, sentinel)

    def test_preprocess_reshuffles_when_postprocessed(self):
        mod = AiterFP8PerTensorNativeInferenceLinear.__new__(AiterFP8PerTensorNativeInferenceLinear)
        nn.Module.__init__(mod)
        mod.weight = nn.Parameter(torch.randn(16, 32), requires_grad=False)
        mod.use_preshuffle = True
        mod._weight_postprocessed = True

        sentinel = torch.ones(16, 32)
        with patch(
            "quark.torch.quantization.nn.modules.native_inference_linear_common._preshuffle_weight",
            return_value=sentinel,
        ) as mock_preshuffle:
            mod.preprocess_weight()

        mock_preshuffle.assert_called_once()
        assert mod._weight_postprocessed is False
        assert torch.equal(mod.weight.data, sentinel)

    def test_no_op_when_not_use_preshuffle(self):
        mod = AiterFP8PerTensorNativeInferenceLinear.__new__(AiterFP8PerTensorNativeInferenceLinear)
        nn.Module.__init__(mod)
        mod.weight = nn.Parameter(torch.randn(16, 32), requires_grad=False)
        mod.use_preshuffle = False
        mod._weight_postprocessed = False

        original = mod.weight.data.clone()
        mod.postprocess_weight()
        assert torch.equal(mod.weight.data, original)

        mod.preprocess_weight()
        assert torch.equal(mod.weight.data, original)

    def test_postprocess_noop_when_already_postprocessed(self):
        mod = AiterFP8PerTensorNativeInferenceLinear.__new__(AiterFP8PerTensorNativeInferenceLinear)
        nn.Module.__init__(mod)
        mod.weight = nn.Parameter(torch.randn(16, 32), requires_grad=False)
        mod.use_preshuffle = True
        mod._weight_postprocessed = True

        original = mod.weight.data.clone()
        mod.postprocess_weight()
        assert torch.equal(mod.weight.data, original)

    def test_preprocess_noop_when_not_postprocessed(self):
        mod = AiterFP8PerTensorNativeInferenceLinear.__new__(AiterFP8PerTensorNativeInferenceLinear)
        nn.Module.__init__(mod)
        mod.weight = nn.Parameter(torch.randn(16, 32), requires_grad=False)
        mod.use_preshuffle = True
        mod._weight_postprocessed = False

        original = mod.weight.data.clone()
        mod.preprocess_weight()
        assert torch.equal(mod.weight.data, original)


class TestNativeInferenceLinearFromModule:
    """Cover NativeInferenceLinear.from_module registry-based dispatch."""

    def test_from_module_dispatches_via_registry(self):
        """``from_module`` looks up the registered backend and calls
        ``from_qparams_linear`` with preshuffle gated by ``supports_preshuffle``."""
        qpl = _make_stub_qparams_linear()
        fake_result = MagicMock(spec=NativeInferenceLinear)

        with (
            patch(
                "quark.torch.quantization.nn.modules.native_inference_linear_common.is_aiter_available",
                return_value=True,
            ),
            patch.object(
                AiterFP8PerTensorNativeInferenceLinear,
                "from_qparams_linear",
                return_value=fake_result,
            ) as mock_factory,
        ):
            result = NativeInferenceLinear.from_module(
                qpl,
                use_preshuffle=True,
            )

        assert result is fake_result
        mock_factory.assert_called_once()
        call_kw = mock_factory.call_args.kwargs
        # supports_preshuffle is True for the FP8 per-tensor backend.
        assert call_kw["use_preshuffle"] is True

    def test_from_module_gates_preshuffle_when_unsupported(self):
        """When ``supports_preshuffle`` is False the backend never sees True."""
        qpl = _make_stub_qparams_linear()
        fake_result = MagicMock(spec=NativeInferenceLinear)

        with (
            patch(
                "quark.torch.quantization.nn.modules.native_inference_linear_common.is_aiter_available",
                return_value=True,
            ),
            patch.object(
                AiterFP8PerTensorNativeInferenceLinear,
                "supports_preshuffle",
                False,
            ),
            patch.object(
                AiterFP8PerTensorNativeInferenceLinear,
                "from_qparams_linear",
                return_value=fake_result,
            ) as mock_factory,
        ):
            NativeInferenceLinear.from_module(qpl, use_preshuffle=True)

        assert mock_factory.call_args.kwargs["use_preshuffle"] is False

    def test_from_module_unknown_mode_raises(self):
        """Unknown forced_mode (no backend registered) raises a clear error."""
        qpl = _make_stub_qparams_linear()
        sentinel_mode = MagicMock(spec=NativeInferenceMode)
        sentinel_mode.name = "FAKE_MODE"

        with (
            patch(
                "quark.torch.quantization.nn.modules.native_inference_linear_common.is_aiter_available",
                return_value=True,
            ),
            pytest.raises(ValueError, match="No native inference backend registered"),
        ):
            NativeInferenceLinear.from_module(qpl, forced_mode=sentinel_mode)


class TestEnsureWeightRealQuantized:
    """Cover _ensure_weight_real_quantized (lines 187-234)."""

    def test_noop_when_quantizer_is_none(self):
        qpl = _make_stub_qparams_linear()
        qpl.weight_quantizer = None
        original_w = qpl.weight.data.clone()
        _ensure_weight_real_quantized(qpl)
        assert torch.equal(qpl.weight.data, original_w)

    def test_noop_when_scale_already_set(self):
        qpl = _make_stub_qparams_linear(scale_val=0.5)
        original_w = qpl.weight.data.clone()
        _ensure_weight_real_quantized(qpl)
        assert torch.equal(qpl.weight.data, original_w)

    def test_dynamic_quantization_computes_scale(self):
        """Cover the dynamic quantization path that calls update_dynamic_params."""
        qpl = _make_stub_qparams_linear(weight_dtype=torch.float32)
        # Simulate dynamic quantizer: scale=None, has update_dynamic_params
        qpl.weight_quantizer.scale = None

        fake_scale = torch.tensor(0.1, dtype=torch.float32)
        fake_real_weight = torch.randn_like(qpl.weight.data).to(torch.float8_e4m3fn)

        def mock_update_dynamic_params(w):
            qpl.weight_quantizer.scale = fake_scale

        qpl.weight_quantizer.update_dynamic_params = mock_update_dynamic_params
        qpl.weight_quantizer.zero_point = None

        with patch("quark.torch.kernel.scaled_real_quantize", return_value=fake_real_weight):
            _ensure_weight_real_quantized(qpl)

        assert qpl.weight_quantizer.scale is not None

    def test_sequential_quantizer_dynamic(self):
        """Cover SequentialRealQuantizer branch for dynamic quantization."""
        qpl = _make_stub_qparams_linear(use_sequential_quantizer=True)
        inner = qpl.weight_quantizer[0]
        inner.scale = None

        fake_scale = torch.tensor(0.2, dtype=torch.float32)
        fake_real_weight = torch.randn(qpl.out_features, qpl.in_features).to(torch.float8_e4m3fn)

        def mock_update(w):
            inner.scale = fake_scale

        inner.update_dynamic_params = mock_update
        inner.zero_point = None

        with patch("quark.torch.kernel.scaled_real_quantize", return_value=fake_real_weight):
            _ensure_weight_real_quantized(qpl)


class TestPreshuffleWeight:
    """Cover _preshuffle_weight / _unshuffle_weight (lines 267-297)."""

    def test_preshuffle_raises_when_no_aiter_shuffle(self):
        with (
            patch(
                "quark.torch.quantization.nn.modules.native_inference_linear_common._aiter_shuffle_weight",
                None,
            ),
            pytest.raises(ImportError, match="Pre-shuffle requires"),
        ):
            _preshuffle_weight(torch.randn(16, 16))

    def test_unshuffle_basic(self):
        """_unshuffle_weight reshapes correctly for a compatible tensor."""
        w = torch.randn(32, 32, dtype=torch.float32)
        restored = _unshuffle_weight(w)
        assert restored.shape == w.shape


# ===========================================================================
# Tests for aiter_fp8_inference_linear.py
# ===========================================================================


class TestAiterFP8PerTensorFromModule:
    """Cover AiterFP8PerTensorNativeInferenceLinear.from_module (lines 142-208)."""

    def _make_source(self, *, n=32, k=64, fp8=True):
        return _make_stub_qparams_linear(
            n=n,
            k=k,
            weight_dtype=torch.float8_e4m3fn if fp8 else torch.float32,
        )

    def test_from_module_no_preshuffle(self):
        source = self._make_source()
        with patch(
            "quark.torch.quantization.nn.modules.native_inference_linear_common.is_aiter_available",
            return_value=True,
        ):
            mod = AiterFP8PerTensorNativeInferenceLinear.from_module(source)

        assert isinstance(mod, AiterFP8PerTensorNativeInferenceLinear)
        assert mod.in_features == 64
        assert mod.out_features == 32
        assert mod.use_preshuffle is False
        assert hasattr(mod, "_kernel_scale")
        assert mod._output_dtype == torch.bfloat16

    def test_from_module_wrong_mode_raises(self):
        source = _make_stub_qparams_linear(
            dtype=Dtype.fp8_e4m3,
            qscheme=QSchemeType.per_group,
        )
        with (
            patch(
                "quark.torch.quantization.nn.modules.native_inference_linear_common.is_aiter_available",
                return_value=True,
            ),
            pytest.raises(ValueError, match="Unsupported quantization configuration"),
        ):
            AiterFP8PerTensorNativeInferenceLinear.from_module(source)

    def test_from_module_preshuffle_when_K_supported(self):
        """K is bpreshuffle-compatible -> weight is shuffled in-place."""
        source = self._make_source(k=256, fp8=True)
        fake_shuffled = torch.randn(32, 256).to(torch.float8_e4m3fn)

        with (
            patch(
                "quark.torch.quantization.nn.modules.native_inference_linear_common.is_aiter_available",
                return_value=True,
            ),
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear._preshuffle_weight",
                return_value=fake_shuffled,
            ),
        ):
            mod = AiterFP8PerTensorNativeInferenceLinear.from_module(
                source,
                use_preshuffle=True,
            )

        assert mod.use_preshuffle is True
        assert mod._weight_postprocessed is False
        assert mod.weight.data.data_ptr() == fake_shuffled.data_ptr()
        assert not hasattr(mod, "_preshuffled_weight")

    def test_from_module_preshuffle_silently_disabled_when_K_too_small(self):
        """K is not bpreshuffle-compatible -> use_preshuffle is silently False
        and the weight is left unshuffled."""
        source = self._make_source(k=64, fp8=True)
        original_ptr = source.weight.data.data_ptr()

        with (
            patch(
                "quark.torch.quantization.nn.modules.native_inference_linear_common.is_aiter_available",
                return_value=True,
            ),
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear._preshuffle_weight",
            ) as mock_preshuffle,
        ):
            mod = AiterFP8PerTensorNativeInferenceLinear.from_module(
                source,
                use_preshuffle=True,
            )

        assert mod.use_preshuffle is False
        mock_preshuffle.assert_not_called()
        assert mod.weight.data.data_ptr() == original_ptr


class TestAiterFP8PerTensorGetKernelWeight:
    """Cover _get_kernel_weight."""

    def test_returns_weight_data(self):
        """``_get_kernel_weight`` must return a no-copy view onto ``self.weight``.

        The contract is "same storage as ``self.weight`` so the GEMM kernel
        sees zero-copy weight access". ``Parameter.data`` is a property that
        may return a fresh ``Tensor`` wrapper on each access, so Python ``is``
        identity is not a reliable check — assert storage identity via
        ``data_ptr()`` plus shape / dtype instead.
        """
        mod = AiterFP8PerTensorNativeInferenceLinear.__new__(AiterFP8PerTensorNativeInferenceLinear)
        nn.Module.__init__(mod)
        w = torch.randn(16, 32)
        mod.weight = nn.Parameter(w, requires_grad=False)

        kernel_weight = mod._get_kernel_weight()
        assert kernel_weight.data_ptr() == mod.weight.data_ptr()
        assert kernel_weight.shape == mod.weight.shape
        assert kernel_weight.dtype == mod.weight.dtype


class TestAiterFP8PerTensorForwardMocked:
    """Cover forward() paths via mocked GEMM kernels."""

    @staticmethod
    def _build_mod(
        *,
        use_preshuffle=False,
        bias=True,
        in_features=64,
        input_scale: torch.Tensor | None = None,
    ):
        mod = AiterFP8PerTensorNativeInferenceLinear.__new__(AiterFP8PerTensorNativeInferenceLinear)
        nn.Module.__init__(mod)
        mod.in_features = in_features
        mod.out_features = 32
        w = torch.randn(32, in_features).to(torch.float8_e4m3fn)
        mod.weight = nn.Parameter(w, requires_grad=False)
        mod.bias = nn.Parameter(torch.randn(32, dtype=torch.bfloat16), requires_grad=False) if bias else None
        mod._output_dtype = torch.bfloat16
        mod.use_preshuffle = use_preshuffle
        mod._weight_postprocessed = False
        mod.register_buffer("_kernel_scale", torch.tensor([0.25], dtype=torch.float32), persistent=False)
        # Mirror the runtime invariant established by ``_apply_kernel_state``:
        # ``_input_scale`` is either a registered buffer (static input quant)
        # or ``None`` (dynamic input quant). Forward reads it unconditionally.
        if input_scale is not None:
            mod.register_buffer("_input_scale", input_scale, persistent=False)
        else:
            mod._input_scale = None
        return mod

    def test_forward_no_preshuffle(self):
        mod = self._build_mod(use_preshuffle=False)
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        fake_out = torch.randn(4, 32, dtype=torch.bfloat16)

        with (
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.dynamic_per_tensor_quant_fp8",
                return_value=(torch.randn(4, 64).to(torch.float8_e4m3fn), torch.tensor(1.0)),
            ),
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.gemm_fp8",
                return_value=fake_out,
            ),
        ):
            out = mod.forward(x)

        assert out.shape == (4, 32)

    def test_forward_preshuffle_uses_bpreshuffle_kernel(self):
        """When use_preshuffle is True, forward calls gemm_fp8_bpreshuffle."""
        mod = self._build_mod(use_preshuffle=True, in_features=256)
        x = torch.randn(4, 256, dtype=torch.bfloat16)
        fake_out = torch.randn(4, 32, dtype=torch.bfloat16)

        with (
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.dynamic_per_tensor_quant_fp8",
                return_value=(torch.randn(4, 256).to(torch.float8_e4m3fn), torch.tensor(1.0)),
            ),
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.gemm_fp8_bpreshuffle",
                return_value=fake_out,
            ) as mock_bp,
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.gemm_fp8",
            ) as mock_regular,
        ):
            out = mod.forward(x)

        assert out.shape == (4, 32)
        mock_bp.assert_called_once()
        mock_regular.assert_not_called()

    def test_forward_bias_addition(self):
        mod = self._build_mod(use_preshuffle=False, bias=True)
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        fake_out = torch.zeros(4, 32, dtype=torch.bfloat16)

        with (
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.dynamic_per_tensor_quant_fp8",
                return_value=(torch.randn(4, 64).to(torch.float8_e4m3fn), torch.tensor(1.0)),
            ),
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.gemm_fp8",
                return_value=fake_out,
            ),
        ):
            out = mod.forward(x)

        assert torch.allclose(out, mod.bias.unsqueeze(0).expand_as(out).to(out.dtype), atol=1e-5)

    def test_forward_no_bias(self):
        mod = self._build_mod(use_preshuffle=False, bias=False)
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        fake_out = torch.randn(4, 32, dtype=torch.bfloat16)

        with (
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.dynamic_per_tensor_quant_fp8",
                return_value=(torch.randn(4, 64).to(torch.float8_e4m3fn), torch.tensor(1.0)),
            ),
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.gemm_fp8",
                return_value=fake_out,
            ),
        ):
            out = mod.forward(x)

        assert torch.equal(out, fake_out)

    def test_forward_threads_static_input_scale_to_kernel(self):
        """Static input quantization → forward passes the calibrated scale
        to ``dynamic_per_tensor_quant_fp8`` so the kernel skips per-batch
        amax."""
        static_scale = torch.tensor([0.7], dtype=torch.float32)
        mod = self._build_mod(use_preshuffle=False, bias=False, input_scale=static_scale)
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        fake_out = torch.randn(4, 32, dtype=torch.bfloat16)

        with (
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.dynamic_per_tensor_quant_fp8",
                return_value=(torch.randn(4, 64).to(torch.float8_e4m3fn), torch.tensor(1.0)),
            ) as mock_quant,
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.gemm_fp8",
                return_value=fake_out,
            ),
        ):
            mod.forward(x)

        passed_scale = mock_quant.call_args.kwargs["scale"]
        assert passed_scale is mod._input_scale

    def test_forward_passes_none_scale_when_dynamic(self):
        """No ``_input_scale`` buffer → forward passes ``scale=None`` so the
        kernel falls back to per-batch dynamic quantization."""
        mod = self._build_mod(use_preshuffle=False, bias=False, input_scale=None)
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        fake_out = torch.randn(4, 32, dtype=torch.bfloat16)

        with (
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.dynamic_per_tensor_quant_fp8",
                return_value=(torch.randn(4, 64).to(torch.float8_e4m3fn), torch.tensor(1.0)),
            ) as mock_quant,
            patch(
                "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.gemm_fp8",
                return_value=fake_out,
            ),
        ):
            mod.forward(x)

        assert mock_quant.call_args.kwargs["scale"] is None


# ===========================================================================
# Tests for aiter_fp4_inference_linear.py
# ===========================================================================


def _make_mxfp4_qpl(
    *,
    weight: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    group_size: int = 32,
    use_sequential_quantizer: bool = False,
) -> QParamsLinear:
    qpl = QParamsLinear.__new__(QParamsLinear)
    nn.Module.__init__(qpl)
    weight = weight if weight is not None else torch.randn(2, 32, dtype=torch.bfloat16)
    qpl.weight = nn.Parameter(weight, requires_grad=False)
    qpl.bias = None
    qpl.in_features = weight.shape[-1] * 2 if weight.dtype == torch.uint8 else weight.shape[-1]
    qpl.out_features = weight.shape[0]

    quantizer = SimpleNamespace(
        qspec=SimpleNamespace(dtype=Dtype.fp4, qscheme=QSchemeType.per_group, group_size=group_size),
        scale=scale,
    )
    if use_sequential_quantizer:
        seq = MagicMock(spec=SequentialRealQuantizer)
        seq.__getitem__ = MagicMock(return_value=quantizer)
        seq.__len__ = MagicMock(return_value=1)
        seq.scale = scale
        qpl.weight_quantizer = seq
    else:
        qpl.weight_quantizer = quantizer

    qpl.input_quantizer = None
    qpl.output_quantizer = None
    qpl.bias_quantizer = None
    qpl._custom_mode = "quark"
    qpl._quant_config = None
    qpl._quant_dict = None
    qpl.algo_config = None
    return qpl


class TestAiterMXFP4WeightRecovery:
    def test_returns_weight_when_no_quantizer(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        qpl = _make_mxfp4_qpl()
        qpl.weight_quantizer = None
        out = fp4._get_mxfp4_float_weight(qpl)
        assert out.data_ptr() == qpl.weight.data_ptr()

    def test_returns_weight_when_dynamic_scale_missing(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        qpl = _make_mxfp4_qpl(scale=None)
        out = fp4._get_mxfp4_float_weight(qpl)
        assert out.data_ptr() == qpl.weight.data_ptr()

    def test_uint8_weight_and_uint8_scale_use_mxfp4_dequant(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        weight = torch.arange(32, dtype=torch.uint8).view(2, 16)
        scale = torch.full((2, 1), 127, dtype=torch.uint8)
        qpl = _make_mxfp4_qpl(weight=weight, scale=scale)
        expected = torch.randn(2, 32, dtype=torch.bfloat16)

        with patch("quark.torch.kernel.mx.hip.dq_mxfp4_hip", return_value=expected) as mock_dq:
            out = fp4._get_mxfp4_float_weight(qpl)

        mock_dq.assert_called_once()
        called_weight, called_scale, called_dtype = mock_dq.call_args.args
        assert torch.equal(called_weight, weight)
        assert torch.equal(called_scale, scale)
        assert called_dtype is torch.bfloat16
        assert out is expected

    def test_float_fallback_applies_uint8_e8m0_scale(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        weight = torch.ones(2, 32, dtype=torch.bfloat16)
        scale = torch.tensor([[127], [128]], dtype=torch.uint8)
        qpl = _make_mxfp4_qpl(weight=weight, scale=scale)

        out = fp4._get_mxfp4_float_weight(qpl)

        assert out.dtype == torch.bfloat16
        assert torch.allclose(out[0].float(), torch.ones(32))
        assert torch.allclose(out[1].float(), torch.full((32,), 2.0))

    def test_float_fallback_with_non_uint8_scale_uses_float_branch(self):
        """Cover the ``return s.float()`` branch of ``_scale_as_float``.

        The uint8-scale case is the e8m0 dequant path (``2**(s-127)``); a
        float-dtype scale must take the trivial cast branch instead so the
        per-group multiplication uses the literal scale value.
        """
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        weight = torch.ones(2, 32, dtype=torch.bfloat16)
        scale = torch.tensor([[2.0], [4.0]], dtype=torch.float32)
        qpl = _make_mxfp4_qpl(weight=weight, scale=scale)

        out = fp4._get_mxfp4_float_weight(qpl)

        assert out.dtype == torch.bfloat16
        assert torch.allclose(out[0].float(), torch.full((32,), 2.0))
        assert torch.allclose(out[1].float(), torch.full((32,), 4.0))

    def test_float4_e2m1fn_x2_branch_dequants_and_applies_scale(self):
        """Cover the sub-byte ``float4_e2m1fn_x2`` weight branch.

        ``weight.to(bfloat16)`` is the real FP4->bf16 dequant that already
        doubles the trailing dim; the per-group e8m0 scale is then applied
        on top. ``.to(bfloat16)`` is not implemented for ``float4_e2m1fn_x2``
        on CPU, so the weight is faked with a Mock that returns a known
        bf16 tensor.
        """
        if not hasattr(torch, "float4_e2m1fn_x2"):
            pytest.skip("torch.float4_e2m1fn_x2 not available")

        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        fake_bf16 = torch.full((2, 32), 3.0, dtype=torch.bfloat16)

        weight_data = MagicMock()
        weight_data.dtype = torch.float4_e2m1fn_x2
        weight_data.to = MagicMock(return_value=fake_bf16)

        weight_param = MagicMock()
        weight_param.data = weight_data

        scale = torch.full((2, 1), 127, dtype=torch.uint8)  # 2**0 = 1.0

        qpl = SimpleNamespace(
            weight=weight_param,
            weight_quantizer=SimpleNamespace(
                qspec=SimpleNamespace(group_size=32),
                scale=scale,
            ),
        )

        out = fp4._get_mxfp4_float_weight(qpl)

        weight_data.to.assert_called_once_with(torch.bfloat16)
        assert out.dtype == torch.bfloat16
        assert out.shape == (2, 32)
        assert torch.allclose(out.float(), torch.full((2, 32), 3.0))


class TestAiterMXFP4Packers:
    def test_check_triton_fp4_available_raises_with_import_error(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        with (
            patch.object(fp4, "_fp4_kernels_available", False),
            patch.object(fp4, "_fp4_import_error", "missing kernels"),
            pytest.raises(ImportError, match="missing kernels"),
        ):
            fp4._check_triton_fp4_available()

    def test_pack_weight_triton_converts_float32_and_calls_downcast(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        packed = torch.ones(2, 16, dtype=torch.uint8)
        scale = torch.ones(2, 1, dtype=torch.uint8)

        def fake_downcast(weight, out_dtype, axis):
            assert weight.dtype == torch.bfloat16
            assert out_dtype is torch.uint8
            assert axis == -1
            return packed, scale, None

        with (
            patch.object(fp4, "_fp4_kernels_available", True),
            patch.object(fp4, "_downcast_to_mxfp", fake_downcast),
        ):
            out_w, out_s = fp4._pack_weight_triton(torch.randn(2, 32, dtype=torch.float32))

        assert out_w is packed
        assert out_s is scale

    def test_pack_weight_asm_requires_quant_kernel(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        with (
            patch.object(fp4, "_per_1x32_f4_quant_hip", None),
            pytest.raises(ImportError, match="per_1x32_f4_quant_hip"),
        ):
            fp4._pack_weight_asm(torch.randn(2, 32, dtype=torch.bfloat16))

    def test_pack_weight_asm_small_k_uses_unshuffled_quant(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        packed = torch.ones(2, 16, dtype=torch.uint8)
        scale = torch.ones(2, 1, dtype=torch.uint8)

        def fake_quant(weight, *, shuffle):
            assert weight.dtype == torch.bfloat16
            assert shuffle is False
            return packed, scale

        with patch.object(fp4, "_per_1x32_f4_quant_hip", fake_quant):
            out_w, out_s = fp4._pack_weight_asm(torch.randn(2, 64, dtype=torch.float32))

        assert out_w is packed
        assert out_s is scale

    def test_pack_weight_asm_large_k_requires_shuffle_helpers(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        with (
            patch.object(fp4, "_per_1x32_f4_quant_hip", lambda weight, shuffle: (weight, weight)),
            patch.object(fp4, "_get_triton_quant", None),
            pytest.raises(ImportError, match="get_triton_quant"),
        ):
            fp4._pack_weight_asm(torch.randn(2, 256, dtype=torch.bfloat16))

    def test_pack_weight_asm_large_k_quantizes_and_shuffles_weight(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        quantized = torch.ones(2, 128, dtype=torch.uint8)
        shuffled = torch.full((2, 128), 2, dtype=torch.uint8)
        scale = torch.ones(2, 8, dtype=torch.uint8)
        quant_type = SimpleNamespace(per_1x32="per_1x32")

        def fake_get_triton_quant(kind):
            assert kind == "per_1x32"

            def fake_quant(weight, *, shuffle):
                assert shuffle is True
                return quantized, scale

            return fake_quant

        with (
            patch.object(fp4, "_per_1x32_f4_quant_hip", lambda weight, shuffle: (weight, weight)),
            patch.object(fp4, "_get_triton_quant", fake_get_triton_quant),
            patch.object(fp4, "_QuantType", quant_type),
            patch.object(fp4, "_shuffle_weight", MagicMock(return_value=shuffled)) as mock_shuffle,
        ):
            out_w, out_s = fp4._pack_weight_asm(torch.randn(2, 256, dtype=torch.bfloat16))

        mock_shuffle.assert_called_once_with(quantized, layout=(16, 16))
        assert out_w is shuffled
        assert out_s is scale


class TestAiterMXFP4GemmWithDynamicQuant:
    def test_triton_path_preallocates_output_and_calls_gemm(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        x = torch.randn(3, 32, dtype=torch.bfloat16)
        weight = torch.ones(4, 16, dtype=torch.uint8)
        weight_scale = torch.ones(4, 1, dtype=torch.uint8)
        x_q = torch.ones(3, 16, dtype=torch.uint8)
        x_s = torch.ones(3, 1, dtype=torch.uint8)

        def fake_gemm(xq, w, xs, ws, dtype, y):
            assert xq is x_q
            assert w is weight
            assert xs is x_s
            assert ws is weight_scale
            assert dtype is torch.bfloat16
            y.copy_(torch.arange(12, dtype=torch.bfloat16).view(3, 4))

        with (
            patch.object(fp4, "_fp4_kernels_available", True),
            patch.object(fp4, "_dynamic_mxfp4_quant", MagicMock(return_value=(x_q, x_s))),
            patch.object(fp4, "_gemm_afp4wfp4", fake_gemm),
        ):
            out = fp4._gemm_with_dynamic_quant(x, weight, weight_scale, use_asm_gemm=False, out_dtype=torch.bfloat16)

        assert out.shape == (3, 4)
        assert torch.equal(out, torch.arange(12, dtype=torch.bfloat16).view(3, 4))

    def test_asm_path_requires_gemm_kernel(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        with (
            patch.object(fp4, "_fp4_kernels_available", True),
            patch.object(fp4, "_gemm_a4w4", None),
            pytest.raises(ImportError, match="gemm_a4w4"),
        ):
            fp4._gemm_with_dynamic_quant(
                torch.randn(2, 32),
                torch.ones(4, 16, dtype=torch.uint8),
                torch.ones(4, 1, dtype=torch.uint8),
                use_asm_gemm=True,
                out_dtype=torch.bfloat16,
            )

    def test_asm_small_k_fallback_dequantizes_and_matmuls(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        x = torch.randn(2, 64, dtype=torch.bfloat16)
        weight = torch.ones(3, 32, dtype=torch.uint8)
        weight_scale = torch.ones(3, 2, dtype=torch.uint8)
        x_q = torch.ones(2, 32, dtype=torch.uint8)
        x_s = torch.ones(2, 2, dtype=torch.uint8)
        x_dq = torch.arange(128, dtype=torch.bfloat16).view(2, 64)
        w_dq = torch.arange(192, dtype=torch.bfloat16).view(3, 64)

        def fake_quant(inp, *, shuffle):
            assert inp is x
            assert shuffle is False
            return x_q, x_s

        def fake_dq(tensor, scale, dtype):
            if torch.equal(tensor, x_q):
                return x_dq
            if torch.equal(tensor, weight):
                return w_dq
            raise AssertionError("unexpected tensor")

        with (
            patch.object(fp4, "_fp4_kernels_available", True),
            patch.object(fp4, "_gemm_a4w4", object()),
            patch.object(fp4, "_per_1x32_f4_quant_hip", fake_quant),
            patch("quark.torch.kernel.mx.hip.dq_mxfp4_hip", side_effect=fake_dq),
        ):
            out = fp4._gemm_with_dynamic_quant(x, weight, weight_scale, use_asm_gemm=True, out_dtype=torch.bfloat16)

        assert torch.equal(out, x_dq @ w_dq.T)

    def test_asm_large_k_captures_returned_output_and_slices_m(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        x = torch.randn(3, 256, dtype=torch.bfloat16)
        weight = torch.ones(4, 128, dtype=torch.uint8)
        weight_scale = torch.ones(4, 8, dtype=torch.uint8)
        x_q = torch.ones(32, 128, dtype=torch.uint8)
        x_s = torch.ones(32, 8, dtype=torch.uint8)
        returned = torch.arange(32 * 4, dtype=torch.bfloat16).view(32, 4)
        quant_type = SimpleNamespace(per_1x32="per_1x32")

        def fake_get_triton_quant(kind):
            assert kind == "per_1x32"

            def fake_quant(inp, *, shuffle):
                assert inp is x
                assert shuffle is True
                return x_q, x_s

            return fake_quant

        with (
            patch.object(fp4, "_fp4_kernels_available", True),
            patch.object(fp4, "_gemm_a4w4", MagicMock(return_value=returned)) as mock_gemm,
            patch.object(fp4, "_per_1x32_f4_quant_hip", object()),
            patch.object(fp4, "_get_triton_quant", fake_get_triton_quant),
            patch.object(fp4, "_QuantType", quant_type),
        ):
            out = fp4._gemm_with_dynamic_quant(x, weight, weight_scale, use_asm_gemm=True, out_dtype=torch.bfloat16)

        assert torch.equal(out, returned[:3])
        _, kwargs = mock_gemm.call_args
        assert kwargs["dtype"] is torch.bfloat16
        assert kwargs["bpreshuffle"] is True


class TestEnsureCompiledAsmOpsRegistered:
    """Coverage for the torch.compile bridge entry point."""

    def test_returns_true_when_already_registered(self):
        """Hits the early-return short-circuit (line 139)."""
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        with patch.object(fp4, "_compiled_asm_ops_registered", True):
            assert fp4._ensure_compiled_asm_ops_registered() is True

    def test_returns_false_when_aiter_kernel_missing(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        with (
            patch.object(fp4, "_compiled_asm_ops_registered", False),
            patch.object(fp4, "_gemm_a4w4", None),
            patch.object(fp4, "_per_1x32_f4_quant_hip", object()),
            patch.object(fp4, "_get_triton_quant", object()),
            patch.object(fp4, "_QuantType", object()),
        ):
            assert fp4._ensure_compiled_asm_ops_registered() is False


def _force_register_compiled_asm_ops():
    """Register the opaque ops once for the test process if not already.

    Registration requires all four Aiter symbols to be non-None at predicate
    evaluation time; the bodies use module-attribute lookups so each test can
    plug in its own mocks afterwards. After this returns the
    ``torch.ops.quark.mxfp4_asm_linear_*`` ops are available regardless of
    later patching.
    """
    from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

    if fp4._compiled_asm_ops_registered:
        return
    with (
        patch.object(fp4, "_gemm_a4w4", object()),
        patch.object(fp4, "_per_1x32_f4_quant_hip", object()),
        patch.object(fp4, "_get_triton_quant", object()),
        patch.object(fp4, "_QuantType", object()),
    ):
        fp4._ensure_compiled_asm_ops_registered()


class TestMxfp4AsmOpaqueOpBodies:
    """Cover the two ``torch.library.custom_op`` bodies and their fakes.

    These exercise the kernels through the registered opaque ops so the
    code under test is both the op body (eager dispatch) and the registered
    fake (FakeTensor dispatch).
    """

    def test_normal_k_op_body_dispatches_quant_and_gemm(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        _force_register_compiled_asm_ops()

        x = torch.randn(3, 256, dtype=torch.bfloat16)
        weight = torch.ones(4, 128, dtype=torch.uint8)
        weight_scale = torch.ones(4, 8, dtype=torch.uint8)
        x_q = torch.ones(32, 128, dtype=torch.uint8)
        x_s = torch.ones(32, 8, dtype=torch.uint8)
        returned = torch.arange(32 * 4, dtype=torch.bfloat16).view(32, 4)
        quant_type = SimpleNamespace(per_1x32="per_1x32")

        def fake_get_triton_quant(kind):
            assert kind == "per_1x32"

            def fq(inp, *, shuffle):
                assert shuffle is True
                return x_q, x_s

            return fq

        with (
            patch.object(fp4, "_gemm_a4w4", MagicMock(return_value=returned)) as mg,
            patch.object(fp4, "_get_triton_quant", fake_get_triton_quant),
            patch.object(fp4, "_QuantType", quant_type),
        ):
            out = torch.ops.quark.mxfp4_asm_linear_normal_k(x, weight, weight_scale)

        assert torch.equal(out, returned[:3])
        _, kwargs = mg.call_args
        assert kwargs["dtype"] is torch.bfloat16
        assert kwargs["bpreshuffle"] is True

    def test_normal_k_fake_returns_bf16_empty_with_correct_shape(self):
        """Hits the register_fake body (line 168)."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        _force_register_compiled_asm_ops()

        with FakeTensorMode():
            x = torch.empty(3, 256, dtype=torch.bfloat16)
            w = torch.empty(4, 128, dtype=torch.uint8)
            ws = torch.empty(4, 8, dtype=torch.uint8)
            out = torch.ops.quark.mxfp4_asm_linear_normal_k(x, w, ws)

        assert out.shape == (3, 4)
        assert out.dtype is torch.bfloat16

    def test_small_k_op_body_quants_dequants_and_matmuls(self):
        """Hits the small-K op body (lines 176, 178-179, 182, 185)."""
        from quark.torch.kernel.mx import hip as hip_mod
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        _force_register_compiled_asm_ops()

        x = torch.randn(2, 64, dtype=torch.bfloat16)
        weight = torch.ones(3, 32, dtype=torch.uint8)
        weight_scale = torch.ones(3, 2, dtype=torch.uint8)
        x_q = torch.ones(2, 32, dtype=torch.uint8)
        x_s = torch.ones(2, 2, dtype=torch.uint8)
        x_dq = torch.arange(128, dtype=torch.bfloat16).view(2, 64)
        w_dq = torch.arange(192, dtype=torch.bfloat16).view(3, 64)

        def fake_quant(inp, *, shuffle):
            assert shuffle is False
            return x_q, x_s

        def fake_dq(tensor, scale, dtype):
            assert dtype is torch.bfloat16
            if torch.equal(tensor, x_q):
                return x_dq
            if torch.equal(tensor, weight):
                return w_dq
            raise AssertionError("unexpected tensor passed to dq_mxfp4_hip")

        with (
            patch.object(fp4, "_per_1x32_f4_quant_hip", fake_quant),
            patch.object(hip_mod, "dq_mxfp4_hip", fake_dq),
        ):
            out = torch.ops.quark.mxfp4_asm_linear_small_k(x, weight, weight_scale)

        assert torch.equal(out, x_dq @ w_dq.T)

    def test_small_k_fake_returns_bf16_empty_with_correct_shape(self):
        """Hits the register_fake body (line 189)."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        _force_register_compiled_asm_ops()

        with FakeTensorMode():
            x = torch.empty(2, 64, dtype=torch.bfloat16)
            w = torch.empty(3, 32, dtype=torch.uint8)
            ws = torch.empty(3, 2, dtype=torch.uint8)
            out = torch.ops.quark.mxfp4_asm_linear_small_k(x, w, ws)

        assert out.shape == (2, 3)
        assert out.dtype is torch.bfloat16

    def test_gemm_with_dynamic_quant_small_k_dispatches_to_opaque_op(self):
        """Hits the small-K opaque-op route in _gemm_with_dynamic_quant (line 352)."""
        from quark.torch.kernel.mx import hip as hip_mod
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        _force_register_compiled_asm_ops()

        x = torch.randn(2, 64, dtype=torch.bfloat16)
        weight = torch.ones(3, 32, dtype=torch.uint8)
        weight_scale = torch.ones(3, 2, dtype=torch.uint8)
        x_q = torch.ones(2, 32, dtype=torch.uint8)
        x_s = torch.ones(2, 2, dtype=torch.uint8)
        x_dq = torch.arange(128, dtype=torch.bfloat16).view(2, 64)
        w_dq = torch.arange(192, dtype=torch.bfloat16).view(3, 64)

        def fake_quant(inp, *, shuffle):
            assert shuffle is False
            return x_q, x_s

        def fake_dq(tensor, scale, dtype):
            if torch.equal(tensor, x_q):
                return x_dq
            if torch.equal(tensor, weight):
                return w_dq
            raise AssertionError("unexpected tensor passed to dq_mxfp4_hip")

        with (
            patch.object(fp4, "_fp4_kernels_available", True),
            patch.object(fp4, "_gemm_a4w4", object()),
            patch.object(fp4, "_per_1x32_f4_quant_hip", fake_quant),
            patch.object(fp4, "_get_triton_quant", object()),
            patch.object(fp4, "_QuantType", object()),
            patch.object(fp4, "_compiled_asm_ops_registered", True),
            patch.object(hip_mod, "dq_mxfp4_hip", fake_dq),
        ):
            out = fp4._gemm_with_dynamic_quant(
                x,
                weight,
                weight_scale,
                use_asm_gemm=True,
                out_dtype=torch.bfloat16,
            )

        assert torch.equal(out, x_dq @ w_dq.T)


class TestAiterMXFP4NativeInferenceLinear:
    def _make_state(self) -> _KernelState:
        return _KernelState(
            weight=torch.randn(2, 32, dtype=torch.bfloat16),
            weight_scale=torch.tensor([[1.0]], dtype=torch.float32),
            bias=None,
            in_features=32,
            out_features=2,
            output_dtype=torch.bfloat16,
            input_scale=None,
        )

    def _make_mod(self):
        from quark.torch.quantization.nn.modules.aiter_fp4_inference_linear import AiterMXFP4NativeInferenceLinear

        mod = AiterMXFP4NativeInferenceLinear.__new__(AiterMXFP4NativeInferenceLinear)
        nn.Module.__init__(mod)
        mod.in_features = 32
        mod.out_features = 2
        mod.weight = nn.Parameter(torch.randn(2, 32, dtype=torch.bfloat16), requires_grad=False)
        mod.bias = None
        mod.weight_quantizer = SimpleNamespace(
            qspec=SimpleNamespace(dtype=Dtype.fp4, qscheme=QSchemeType.per_group, group_size=32),
            scale=torch.ones(2, 1),
        )
        return mod

    def test_reset_parameters_is_noop(self):
        """``reset_parameters`` is intentionally a no-op for the MXFP4
        backend: kernel buffers are materialized via ``_apply_kernel_state``
        rather than being randomly initialized by ``Linear`` semantics.
        """
        from quark.torch.quantization.nn.modules.aiter_fp4_inference_linear import AiterMXFP4NativeInferenceLinear

        mod = AiterMXFP4NativeInferenceLinear.__new__(AiterMXFP4NativeInferenceLinear)
        nn.Module.__init__(mod)
        mod.reset_parameters()

    def test_apply_kernel_state_uses_asm_by_default(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        mod = self._make_mod()
        kernel_weight = torch.ones(2, 16, dtype=torch.uint8)
        kernel_scale = torch.ones(2, 1, dtype=torch.uint8)

        with (
            patch.object(fp4, "_require_aiter"),
            patch.object(fp4, "_get_mxfp4_float_weight", return_value=torch.randn(2, 32, dtype=torch.bfloat16)),
            patch.object(fp4, "_pack_weight_asm", return_value=(kernel_weight, kernel_scale)) as mock_pack,
            patch.object(fp4, "_pack_weight_triton") as mock_triton_pack,
        ):
            mod._apply_kernel_state(self._make_state())

        mock_pack.assert_called_once()
        mock_triton_pack.assert_not_called()
        assert mod._use_asm_gemm is True
        assert mod._kernel_weight is kernel_weight
        assert mod._kernel_scale is kernel_scale
        assert "_kernel_weight" not in mod.state_dict()
        assert "_kernel_scale" not in mod.state_dict()

    def test_apply_kernel_state_asm_uses_asm_packer(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        mod = self._make_mod()
        kernel_weight = torch.ones(2, 16, dtype=torch.uint8)
        kernel_scale = torch.ones(2, 1, dtype=torch.uint8)

        with (
            patch.object(fp4, "_require_aiter"),
            patch.object(fp4, "_get_mxfp4_float_weight", return_value=torch.randn(2, 32, dtype=torch.bfloat16)),
            patch.object(fp4, "_pack_weight_asm", return_value=(kernel_weight, kernel_scale)) as mock_pack,
        ):
            mod._apply_kernel_state(self._make_state())

        mock_pack.assert_called_once()
        assert mod._use_asm_gemm is True
        assert mod._get_kernel_weight() is kernel_weight

    def test_forward_calls_helper_and_restores_original_shape_with_bias(self):
        from quark.torch.quantization.nn.modules import aiter_fp4_inference_linear as fp4

        mod = self._make_mod()
        mod.in_features = 4
        mod.out_features = 3
        mod.bias = nn.Parameter(torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16), requires_grad=False)
        mod._kernel_weight = torch.ones(3, 2, dtype=torch.uint8)
        mod._kernel_scale = torch.ones(3, 1, dtype=torch.uint8)
        mod._use_asm_gemm = True
        mod._output_dtype = torch.bfloat16
        x = torch.randn(2, 5, 4, dtype=torch.bfloat16)
        helper_out = torch.zeros(10, 3, dtype=torch.bfloat16)

        with patch.object(fp4, "_gemm_with_dynamic_quant", return_value=helper_out) as mock_gemm:
            out = mod.forward(x)

        assert out.shape == (2, 5, 3)
        assert torch.equal(out[0, 0], mod.bias)
        assert mock_gemm.call_args.kwargs["use_asm_gemm"] is True


class TestAiterNativeLinearFromModule:
    """Cover aiter_native_linear_from_module dispatch (the legacy wrapper)."""

    def test_dispatch_per_tensor(self):
        source = _make_stub_qparams_linear(dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor)

        with patch(
            "quark.torch.quantization.nn.modules.native_inference_linear_common.is_aiter_available",
            return_value=True,
        ):
            mod = aiter_native_linear_from_module(source)

        assert isinstance(mod, AiterFP8PerTensorNativeInferenceLinear)

    def test_dispatch_with_forced_mode(self):
        source = _make_stub_qparams_linear(dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor)

        with patch(
            "quark.torch.quantization.nn.modules.native_inference_linear_common.is_aiter_available",
            return_value=True,
        ):
            mod = aiter_native_linear_from_module(
                source,
                forced_mode=NativeInferenceMode.FP8_PER_TENSOR,
            )

        assert isinstance(mod, AiterFP8PerTensorNativeInferenceLinear)


# ===========================================================================
# Tests for quark/torch/quantization/api.py lines 428-429
# ===========================================================================


class TestModelQuantizerFreezeWithRuntimeOptions:
    """Cover freeze with runtime_options (api.py lines 427-429)."""

    def test_freeze_calls_enable_native_inference(self):
        from quark.torch.quantization.api import ModelQuantizer

        model = nn.Sequential(nn.Linear(8, 4))
        options = RuntimeOptions()

        with patch("quark.torch.quantization.api.enable_native_inference") as mock_enable:
            ModelQuantizer.freeze(model, runtime_options=options)

        mock_enable.assert_called_once_with(model, runtime_options=options)

    def test_freeze_skips_native_inference_without_options(self):
        from quark.torch.quantization.api import ModelQuantizer

        model = nn.Sequential(nn.Linear(8, 4))

        with patch("quark.torch.quantization.api.enable_native_inference") as mock_enable:
            ModelQuantizer.freeze(model)

        mock_enable.assert_not_called()
