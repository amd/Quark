#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""CPU-only coverage tests for SVDQuant native inference (no GPU / no Aiter).

These exercise the orchestration logic -- construction, forward composition,
round-trip, and the enable/disable wiring -- with the Aiter-dependent kernel
calls mocked, so the module is covered on a CPU-only CI runner. The real
GPU+Aiter correctness/parity tests live in ``test_svdquant_native_inference.py``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from quark.torch.algorithm.svdquant.svdquant import ErrorCorrectedModule, LowRankCorrectionModule
from quark.torch.quantization import utils as quant_utils
from quark.torch.quantization.nn.modules.aiter_svdquant_inference_linear import (
    AiterSVDQuantMXFP4NativeInferenceLinear,
    svdquant_native_linear_from_error_corrected_module,
)
from quark.torch.quantization.nn.modules.native_inference_linear_common import NativeInferenceMode
from quark.torch.quantization.utils import RuntimeOptions, disable_native_inference, enable_native_inference

MODULE = "quark.torch.quantization.nn.modules.aiter_svdquant_inference_linear"
_CLS = AiterSVDQuantMXFP4NativeInferenceLinear


def _bare_instance() -> AiterSVDQuantMXFP4NativeInferenceLinear:
    """A minimally-constructed instance (bypasses the Aiter build path)."""
    obj = _CLS.__new__(_CLS)
    nn.Module.__init__(obj)
    return obj


def _fake_ecm(smooth: torch.Tensor | None = None) -> nn.Module:
    ecm = nn.Module()
    ecm.layer = nn.Linear(8, 8, bias=False)
    ecm.correction = LowRankCorrectionModule(8, 8, rank=2)
    ecm.smooth_factor = smooth
    return ecm


# ---------------------------------------------------------------------------
# _combine (pure tensor math)
# ---------------------------------------------------------------------------


def test_combine_bf16():
    a = torch.ones(2, 3, dtype=torch.bfloat16)
    b = torch.ones(2, 3, dtype=torch.bfloat16)
    out = _CLS._combine(a, b, torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), torch.full((2, 3), 2.0))


def test_combine_fp16_upcasts():
    a = torch.ones(2, 3, dtype=torch.bfloat16)
    b = torch.ones(2, 3, dtype=torch.float16)
    out = _CLS._combine(a, b, torch.float16)
    assert out.dtype == torch.float16
    assert torch.allclose(out.float(), torch.full((2, 3), 2.0))


# ---------------------------------------------------------------------------
# from_error_corrected_module: validation + (mocked) build path
# ---------------------------------------------------------------------------


def test_from_ecm_rejects_non_ecm():
    with pytest.raises(ValueError, match="ErrorCorrectedModule"):
        _CLS.from_error_corrected_module(object())


def test_from_ecm_rejects_non_mxfp4_residual():
    with (
        patch(f"{MODULE}.determine_inference_mode", return_value=NativeInferenceMode.FP8_PER_TENSOR),
        pytest.raises(ValueError, match="MXFP4 residual"),
    ):
        _CLS.from_error_corrected_module(_fake_ecm())


@pytest.mark.parametrize("smooth", [None, torch.ones(8)])
def test_from_ecm_build_path_mocked(smooth):
    ecm = _fake_ecm(smooth=smooth)
    bare = _bare_instance()
    with (
        patch(f"{MODULE}.determine_inference_mode", return_value=NativeInferenceMode.MXFP4),
        patch(f"{MODULE}._QParamsLinearBridge.build_from_source", return_value=object()),
        patch.object(_CLS, "from_qparams_linear", return_value=bare),
    ):
        out = _CLS.from_error_corrected_module(ecm, overlap_streams=True)
    assert out is bare
    assert out.correction is ecm.correction
    assert out.overlap_streams is True
    assert out._side_stream is None
    if smooth is None:
        assert out.smooth_factor is None
    else:
        assert out.smooth_factor is not None


def test_wrapper_delegates():
    with pytest.raises(ValueError):
        svdquant_native_linear_from_error_corrected_module(object())


# ---------------------------------------------------------------------------
# forward composition (residual mocked; correction + combine real, on CPU)
# ---------------------------------------------------------------------------


def _forward_instance(smooth: torch.Tensor | None) -> AiterSVDQuantMXFP4NativeInferenceLinear:
    inst = _bare_instance()
    inst.correction = LowRankCorrectionModule(8, 4, rank=2).to(torch.bfloat16)
    inst.overlap_streams = False
    inst._side_stream = None
    if smooth is not None:
        inst.register_buffer("smooth_factor", smooth)
    else:
        inst.smooth_factor = None
    # Stand in for the Aiter MXFP4 residual GEMM (which needs a GPU kernel).
    inst._residual_forward = lambda x: torch.zeros(x.shape[0], 4, dtype=torch.bfloat16)  # type: ignore[method-assign]
    return inst


def test_forward_sequential_no_smooth():
    inst = _forward_instance(smooth=None)
    out = inst.forward(torch.randn(3, 8, dtype=torch.bfloat16))
    assert out.shape == (3, 4)
    assert out.dtype == torch.bfloat16


def test_forward_applies_smooth():
    inst = _forward_instance(smooth=torch.ones(8, dtype=torch.bfloat16))
    out = inst.forward(torch.randn(3, 8, dtype=torch.bfloat16))
    assert out.shape == (3, 4)


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def test_to_qparams_linear_not_supported():
    with pytest.raises(NotImplementedError):
        _bare_instance().to_qparams_linear()


def test_to_error_corrected_module_mocked():
    inst = _bare_instance()
    inst.correction = LowRankCorrectionModule(8, 4, rank=2)
    inst.smooth_factor = None
    inst.use_preshuffle = False  # postprocess_weight becomes a no-op
    with patch(f"{MODULE}._QParamsLinearBridge.materialize", return_value=object()):
        ecm = inst.to_error_corrected_module()
    assert isinstance(ecm, ErrorCorrectedModule)
    assert ecm.correction is inst.correction


# ---------------------------------------------------------------------------
# utils: _hf_hook transfer + enable/disable wiring
# ---------------------------------------------------------------------------


def test_transfer_hf_hook_no_hook_is_noop():
    new = nn.Linear(2, 2)
    quant_utils._transfer_hf_hook(nn.Linear(2, 2), new)
    assert not hasattr(new, "_hf_hook")


def test_transfer_hf_hook_clones_hook():
    pytest.importorskip("accelerate")
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    old = nn.Linear(2, 2)
    add_hook_to_module(old, AlignDevicesHook(execution_device="cpu", offload=False, io_same_device=True))
    new = nn.Linear(2, 2)
    quant_utils._transfer_hf_hook(old, new)
    assert hasattr(new, "_hf_hook")
    assert new._hf_hook.execution_device == "cpu"


def _model_with_ecm() -> nn.Module:
    model = nn.Module()
    model.blk = ErrorCorrectedModule(LowRankCorrectionModule(8, 8, rank=2), nn.Linear(8, 8, bias=False), None)
    return model


def test_enable_native_inference_skips_unconvertible_ecm():
    # The residual is a plain nn.Linear (no quantizer) -> conversion raises and
    # is caught; the ErrorCorrectedModule is left in place (eager).
    model = _model_with_ecm()
    with patch("quark.torch.kernel.aiter.is_aiter_available", return_value=True):
        converted = enable_native_inference(model, runtime_options=RuntimeOptions(native_linear_mode="mxfp4"))
    assert converted == 0
    assert isinstance(model.blk, ErrorCorrectedModule)


def test_enable_native_inference_fp8_mode_leaves_ecm():
    # fp8_per_tensor forces convert_ecm=False, so the MXFP4 ErrorCorrectedModule
    # is left untouched (covers the False branch of `if convert_ecm`).
    model = _model_with_ecm()
    with patch("quark.torch.kernel.aiter.is_aiter_available", return_value=True):
        converted = enable_native_inference(model, runtime_options=RuntimeOptions(native_linear_mode="fp8_per_tensor"))
    assert converted == 0
    assert isinstance(model.blk, ErrorCorrectedModule)


def test_enable_native_inference_converts_ecm_mocked():
    model = _model_with_ecm()
    stub = nn.Identity()
    with (
        patch("quark.torch.kernel.aiter.is_aiter_available", return_value=True),
        patch(f"{MODULE}.svdquant_native_linear_from_error_corrected_module", return_value=stub),
    ):
        converted = enable_native_inference(model, runtime_options=RuntimeOptions(native_linear_mode="mxfp4"))
    assert converted == 1
    assert model.blk is stub


def test_enable_native_inference_converts_plain_qparamslinear_mocked():
    from quark.torch.export.nn.modules.qparamslinear import QParamsLinear

    qpl = QParamsLinear.__new__(QParamsLinear)
    nn.Module.__init__(qpl)
    qpl.weight_quantizer = object()  # marks it convertible
    model = nn.Module()
    model.lin = qpl
    stub = nn.Identity()
    with (
        patch("quark.torch.kernel.aiter.is_aiter_available", return_value=True),
        patch(
            "quark.torch.quantization.nn.modules.aiter_fp8_inference_linear.aiter_native_linear_from_module",
            return_value=stub,
        ),
    ):
        converted = enable_native_inference(model, runtime_options=RuntimeOptions(native_linear_mode="mxfp4"))
    assert converted == 1
    assert model.lin is stub


def test_disable_native_inference_reverts_plain_native_mocked():
    # A plain (non-SVDQuant) NativeInferenceLinear reverts via to_qparams_linear().
    from quark.torch.quantization.nn.modules.aiter_fp8_inference_linear import (
        AiterFP8PerTensorNativeInferenceLinear,
    )

    inst = AiterFP8PerTensorNativeInferenceLinear.__new__(AiterFP8PerTensorNativeInferenceLinear)
    nn.Module.__init__(inst)
    stub_qpl = nn.Identity()
    inst.to_qparams_linear = lambda: stub_qpl  # type: ignore[method-assign]
    model = nn.Module()
    model.lin = inst
    reverted = disable_native_inference(model)
    assert reverted == 1
    assert model.lin is stub_qpl


def test_disable_native_inference_reverts_svdquant_mocked():
    inst = _bare_instance()
    inst.child = nn.Linear(2, 2)  # child triggers the prefix-skip branch
    stub_ecm = nn.Identity()
    inst.to_error_corrected_module = lambda: stub_ecm  # type: ignore[method-assign]
    model = nn.Module()
    model.blk = inst
    reverted = disable_native_inference(model)
    assert reverted == 1
    assert model.blk is stub_ecm
