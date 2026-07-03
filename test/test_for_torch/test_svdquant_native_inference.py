#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""End-to-end tests for SVDQuant MXFP4 native inference.

These exercise the full path on real hardware: a tiny model is SVDQuant-ed and
quantized to MXFP4 (so each layer becomes an ``ErrorCorrectedModule`` whose
residual is an MXFP4 ``QuantLinear`` / ``QParamsLinear``), then
:func:`enable_native_inference` replaces the wrappers with
:class:`AiterSVDQuantMXFP4NativeInferenceLinear` (Aiter MXFP4 residual GEMM +
low-rank correction). We check that:

* the wrappers are converted (and the inner residual is not double-converted),
* the native output closely matches the eager ``ErrorCorrectedModule`` output,
* 3-D (B, S, C) inputs work,
* the optional second-stream overlap produces the same result,
* :func:`disable_native_inference` restores an ``ErrorCorrectedModule`` that
  reproduces the eager output (round-trip).

Requires a GPU and AMD Aiter; skipped otherwise.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quark.common.utils.import_utils import is_aiter_available
from quark.torch import ModelQuantizer
from quark.torch.algorithm.svdquant.svdquant import (
    ErrorCorrectedModule,
    SVDQuantProcessor,
    build_quant_layer_config,
)
from quark.torch.quantization.config.config import QConfig, SVDQuantConfig
from quark.torch.quantization.nn.modules.aiter_svdquant_inference_linear import (
    AiterSVDQuantMXFP4NativeInferenceLinear,
)
from quark.torch.quantization.utils import (
    RuntimeOptions,
    disable_native_inference,
    enable_native_inference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_aiter_available(),
    reason="SVDQuant native inference requires a GPU and AMD Aiter.",
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
D = 512  # in/out features; >=256 and multiple of 32 for the MXFP4 ASM GEMM path.


class _TinyMLP(nn.Module):
    def __init__(self, d_in: int = D, d_hidden: int = D, d_out: int = D) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=True)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def _snr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    ref_f = ref.flatten().float()
    noise = (ref_f - test.flatten().float()).pow(2).mean()
    return (10 * torch.log10(ref_f.pow(2).mean() / noise.clamp_min(1e-12))).item()


def _build_quantized_svdquant_model() -> nn.Module:
    """SVDQuant + MXFP4 quantize + freeze a tiny model.

    The result has ``ErrorCorrectedModule`` layers whose residual is MXFP4.
    """
    torch.manual_seed(0)
    model = _TinyMLP().to(DEVICE, DTYPE)
    calib = torch.randn(16, D, device=DEVICE, dtype=DTYPE)
    dl = DataLoader(TensorDataset(calib), batch_size=4)

    svd_config = SVDQuantConfig(
        name="svdquant",
        svd_rank=16,
        smooth_alpha=0.5,
        search_alpha=False,
        alpha_candidates=None,
        alpha_search_max_samples=4,
        exclude_patterns=[],
        min_layer_size=1,
        use_gptq=False,
        gptq_n_bits=4,
        gptq_symmetric=True,
        gptq_group_size=-1,
        gptq_blocksize=128,
        gptq_percdamp=0.01,
        gptq_actorder=False,
    )
    SVDQuantProcessor(model=model, quant_algo_config=svd_config, calib_data=dl).apply()

    quant_config = QConfig(global_quant_config=build_quant_layer_config("mxfp4"), exclude=["*correction*"])
    model = ModelQuantizer(quant_config).quantize_model(model, dl)
    model = ModelQuantizer.freeze(model)
    model.eval()
    return model


@pytest.fixture(scope="module")
def bundle() -> dict:
    """Build the quantized model once; record the eager reference output."""
    model = _build_quantized_svdquant_model()
    n_ecm = sum(isinstance(m, ErrorCorrectedModule) for m in model.modules())
    assert n_ecm > 0, "expected SVDQuant to produce ErrorCorrectedModule layers"

    x2d = torch.randn(8, D, device=DEVICE, dtype=DTYPE)
    x3d = torch.randn(2, 4, D, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        y2d = model(x2d).clone()
        y3d = model(x3d).clone()
    return {"model": model, "n_ecm": n_ecm, "x2d": x2d, "x3d": x3d, "y2d": y2d, "y3d": y3d}


def _enabled(model: nn.Module, **opts):
    """Context manager: enable native inference, then restore on exit."""

    class _Ctx:
        def __enter__(self_inner):
            self_inner.n = enable_native_inference(model, runtime_options=RuntimeOptions(**opts))
            return self_inner.n

        def __exit__(self_inner, *exc):
            disable_native_inference(model)
            return False

    return _Ctx()


def test_enable_converts_error_corrected_modules(bundle):
    model = bundle["model"]
    with _enabled(model, native_linear_mode="mxfp4") as n_converted:
        assert n_converted == bundle["n_ecm"]
        natives = [m for m in model.modules() if isinstance(m, AiterSVDQuantMXFP4NativeInferenceLinear)]
        assert len(natives) == bundle["n_ecm"]
        # The inner residual must not survive as a standalone native linear at
        # the ECM's old `.layer` path — it lives inside the composite now.
        assert not any(name.endswith(".layer") for name, _ in model.named_modules())


def test_native_matches_eager_2d(bundle):
    model, x, y_eager = bundle["model"], bundle["x2d"], bundle["y2d"]
    with _enabled(model, native_linear_mode="mxfp4"):
        with torch.no_grad():
            y_native = model(x)
        assert y_native.shape == y_eager.shape
        assert y_native.dtype == y_eager.dtype
        assert torch.isfinite(y_native).all()
        assert _cosine(y_eager, y_native) > 0.99
        assert _snr_db(y_eager, y_native) > 20.0


def test_native_matches_eager_3d(bundle):
    model, x, y_eager = bundle["model"], bundle["x3d"], bundle["y3d"]
    with _enabled(model, native_linear_mode="mxfp4"):
        with torch.no_grad():
            y_native = model(x)
        assert y_native.shape == y_eager.shape
        assert torch.isfinite(y_native).all()
        assert _cosine(y_eager, y_native) > 0.99


def test_overlap_streams_matches_sequential(bundle):
    model, x = bundle["model"], bundle["x2d"]
    with _enabled(model, native_linear_mode="mxfp4", svdquant_overlap_streams=False), torch.no_grad():
        y_seq = model(x).clone()
    with _enabled(model, native_linear_mode="mxfp4", svdquant_overlap_streams=True), torch.no_grad():
        y_overlap = model(x).clone()
    assert torch.isfinite(y_overlap).all()
    # Same math, just a different stream schedule -> should be identical.
    assert _cosine(y_seq, y_overlap) > 0.9999
    assert (y_seq - y_overlap).abs().max().item() < 1e-2


def test_disable_round_trip(bundle):
    model, x, y_eager = bundle["model"], bundle["x2d"], bundle["y2d"]
    enable_native_inference(model, runtime_options=RuntimeOptions(native_linear_mode="mxfp4"))
    assert any(isinstance(m, AiterSVDQuantMXFP4NativeInferenceLinear) for m in model.modules())

    n_reverted = disable_native_inference(model)
    assert n_reverted == bundle["n_ecm"]
    assert not any(isinstance(m, AiterSVDQuantMXFP4NativeInferenceLinear) for m in model.modules())
    assert sum(isinstance(m, ErrorCorrectedModule) for m in model.modules()) == bundle["n_ecm"]

    with torch.no_grad():
        y_back = model(x)
    assert _cosine(y_eager, y_back) > 0.999


def test_to_qparams_linear_not_supported(bundle):
    model = bundle["model"]
    with _enabled(model, native_linear_mode="mxfp4"):
        native = next(m for m in model.modules() if isinstance(m, AiterSVDQuantMXFP4NativeInferenceLinear))
        with pytest.raises(NotImplementedError):
            native.to_qparams_linear()
        # The supported round-trip path rebuilds an ErrorCorrectedModule.
        assert isinstance(native.to_error_corrected_module(), ErrorCorrectedModule)


def test_auto_mode_also_converts_svdquant(bundle):
    """``native_linear_mode='auto'`` should still pick up SVDQuant layers."""
    model = bundle["model"]
    with _enabled(model, native_linear_mode="auto") as n_converted:
        assert n_converted == bundle["n_ecm"]
        assert any(isinstance(m, AiterSVDQuantMXFP4NativeInferenceLinear) for m in model.modules())
