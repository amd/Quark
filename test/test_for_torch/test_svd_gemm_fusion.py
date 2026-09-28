#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for the fused SVDQuant a8w4 GEMM kernel (gfx950).

Two levels of coverage, both against a **bf16 baseline** (per the project's
low-precision test policy) and each repeated 3x for repeatability:

1. Kernel level (`_gemm_flydsl_svdquant`): the fused
   ``y = dequant(quant(x)) @ R^T + d @ L2^T`` must match the SAME computation
   done as a separate a8w4 residual GEMM plus a torch low-rank add (isolates the
   epilogue fusion), and must be close to a full-bf16 reference matmul.
2. Module level (`FlyDSLSVDQuantNativeInferenceLinear` via
   ``enable_native_inference(native_linear_mode="flydsl_svdquant")``): the fused
   native SVDQuant path must be at least as accurate as the aiter MXFP4 SVDQuant
   path when both are measured against the unquantized bf16 model.

Metrics reported: cosine similarity, max error, min error, min abs error,
mean abs error, and percentage error. Run:

    HIP_VISIBLE_DEVICES=<idle> pytest -s tools/flydsl/tests/test_svd_gemm_fusion.py
"""

from __future__ import annotations

import copy

import pytest
import torch

try:
    from flydsl.runtime.device import get_rocm_arch

    _IS_GFX950 = str(get_rocm_arch()) == "gfx950"
except Exception:  # noqa: BLE001
    _IS_GFX950 = False

pytestmark = [
    pytest.mark.flydsl_gfx950,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not _IS_GFX950,
        reason="FlyDSL fused SVDQuant a8w4 kernel requires a gfx950 GPU.",
    ),
]

DEV = "cuda"
REPS = 3


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _metrics(test: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    t = test.flatten().float()
    r = ref.flatten().float()
    diff = (t - r).abs()
    denom = r.abs().clamp_min(1e-6)
    return dict(
        cos=torch.nn.functional.cosine_similarity(t, r, dim=0).item(),
        # Relative L2 (normalized RMSE). Cosine similarity is blind to scale -- a
        # uniform 5% gain error still reports cos=1.000000 -- so assert on this too.
        rel_l2=(torch.linalg.vector_norm(t - r) / torch.linalg.vector_norm(r).clamp_min(1e-6)).item(),
        max_err=diff.max().item(),
        min_err=diff.min().item(),  # min signed-magnitude error == min abs here
        min_abs=diff.min().item(),
        mean_abs=diff.mean().item(),
        pct_mean=(diff / denom).mean().item() * 100.0,
    )


def _report(tag: str, m: dict[str, float]) -> None:
    print(
        f"  [{tag}] cos={m['cos']:.6f} rel_l2={m['rel_l2']:.5f} max_err={m['max_err']:.4g} "
        f"min_err={m['min_err']:.4g} min_abs={m['min_abs']:.4g} "
        f"mean_abs={m['mean_abs']:.4g} pct_mean={m['pct_mean']:.3f}%"
    )


# --------------------------------------------------------------------------- #
# 1. Kernel-level fusion correctness
# --------------------------------------------------------------------------- #
# (M, N, K, rank) — production Wan-like shapes. N must be >= 128 and a multiple
# of 128, K >= 256 and a multiple of 256 (see _validate_a8w4_inputs); shapes outside
# that domain are covered by test_validate_rejects_unsupported_shapes below.
_KERNEL_SHAPES = [
    (256, 128, 256, 32),  # smallest supported N (tile_n=128)
    (1024, 384, 3072, 32),  # N % 256 != 0, so prefill M falls back to tile_n=128
    (256, 512, 256, 32),
    (1024, 3072, 3072, 32),
    (512, 3072, 3072, 16),
    (1800, 3072, 3072, 32),  # ragged M (not a multiple of tile_m)
]


@pytest.mark.parametrize("shape", _KERNEL_SHAPES, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}_r{s[3]}")
def test_fused_kernel_matches_separate_and_bf16(shape):
    from quark.torch.quantization.nn.modules.aiter_fp4_inference_linear import _pack_weight_asm
    from quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear import (
        _gemm_flydsl_a8w4,
        _gemm_flydsl_svdquant,
    )

    M, N, K, rank = shape
    for rep in range(REPS):
        torch.manual_seed(rep)
        x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.5
        Wf = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.1
        L2 = torch.randn(N, rank, device=DEV, dtype=torch.bfloat16) * 0.3
        d = torch.randn(M, rank, device=DEV, dtype=torch.bfloat16) * 0.3

        kw, ks = _pack_weight_asm(Wf)

        # Reference A: separate residual GEMM + torch low-rank add (isolates fusion).
        y_res = _gemm_flydsl_a8w4(x, kw.view(torch.uint8), ks.view(torch.uint8), torch.bfloat16)
        y_sep = y_res.float() + (d.float() @ L2.float().T)

        # Fused path.
        y_fused = _gemm_flydsl_svdquant(x, kw.view(torch.uint8), ks.view(torch.uint8), d, L2, torch.bfloat16)

        m_sep = _metrics(y_fused, y_sep)
        _report(f"rep{rep} fused-vs-separate", m_sep)
        assert m_sep["cos"] > 0.999, f"fusion diverged from separate path: {m_sep}"
        assert m_sep["rel_l2"] < 0.01, f"fusion diverged in magnitude from separate: {m_sep}"

        # Reference B: the unquantized fp32 matmul. The separate path shares the same
        # quantized residual as the fused one, so it cannot catch an error in that
        # residual -- only an independent reference can.
        y_ref = x.float() @ Wf.float().T + d.float() @ L2.float().T
        m_ref = _metrics(y_fused, y_ref)
        _report(f"rep{rep} fused-vs-fp32", m_ref)
        assert torch.isfinite(y_fused).all()
        assert m_ref["cos"] > 0.98, f"fused diverged from fp32 reference: {m_ref}"
        assert m_ref["rel_l2"] < 0.2, f"fused diverged in magnitude from fp32: {m_ref}"


# (M, N, K) outside the kernel's supported domain, plus why.
_REJECTED_SHAPES = [
    (256, 192, 256, "N not a multiple of 128"),
    (256, 64, 256, "N below 128"),
    (256, 512, 192, "K not a multiple of 256"),
    (256, 512, 128, "K below 256"),
]


@pytest.mark.parametrize("shape", _REJECTED_SHAPES, ids=lambda s: s[3].replace(" ", "_"))
def test_validate_rejects_unsupported_shapes(shape):
    """Shapes outside the documented domain must raise rather than compute a wrong
    answer: an N that is not a multiple of 128 selects a tile_n the MX-scale MFMA path
    cannot issue, which used to store a zero result instead of failing."""
    from quark.torch.quantization.nn.modules.aiter_fp4_inference_linear import _pack_weight_asm
    from quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear import (
        _gemm_flydsl_a8w4,
        _gemm_flydsl_svdquant,
    )

    M, N, K, _why = shape
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    Wf = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
    L2 = torch.randn(N, 32, device=DEV, dtype=torch.bfloat16)
    d = torch.randn(M, 32, device=DEV, dtype=torch.bfloat16)
    kw, ks = _pack_weight_asm(Wf)

    with pytest.raises(ValueError):
        _gemm_flydsl_a8w4(x, kw.view(torch.uint8), ks.view(torch.uint8), torch.bfloat16)
    with pytest.raises(ValueError):
        _gemm_flydsl_svdquant(x, kw.view(torch.uint8), ks.view(torch.uint8), d, L2, torch.bfloat16)


def test_validate_rejects_unsupported_dtype_and_device():
    from quark.torch.quantization.nn.modules.aiter_fp4_inference_linear import _pack_weight_asm
    from quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear import _gemm_flydsl_a8w4

    x = torch.randn(256, 256, device=DEV, dtype=torch.bfloat16)
    Wf = torch.randn(512, 256, device=DEV, dtype=torch.bfloat16)
    kw, ks = _pack_weight_asm(Wf)

    with pytest.raises(ValueError):
        _gemm_flydsl_a8w4(x, kw.view(torch.uint8), ks.view(torch.uint8), torch.float32)
    with pytest.raises(ValueError):
        _gemm_flydsl_a8w4(x.cpu(), kw.cpu().view(torch.uint8), ks.cpu().view(torch.uint8), torch.bfloat16)


def test_safe_base_cfg_never_selects_an_unsupported_tile_n():
    """tile_n < 128 cannot be issued by the MX-scale MFMA path, and used to return a
    zero result rather than failing, so the baseline config must never pick one."""
    from quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear import _safe_base_cfg

    for k in (256, 512, 3072, 13824):
        assert _safe_base_cfg(k)["tile_n"] == 128


def test_ineligible_layer_stays_on_the_eager_path():
    """A layer whose features violate the tile rule must be skipped by
    enable_native_inference, not converted into a module that computes zeros."""
    from quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear import (
        FlyDSLA8W4NativeInferenceLinear,
    )

    mod = FlyDSLA8W4NativeInferenceLinear.__new__(FlyDSLA8W4NativeInferenceLinear)
    mod.in_features = 256
    mod.out_features = 192
    with pytest.raises(ValueError, match="multiple of 128"):
        FlyDSLA8W4NativeInferenceLinear._apply_kernel_state(mod, None)


# --------------------------------------------------------------------------- #
# 2. Module-level: fused native SVDQuant vs bf16 baseline
# --------------------------------------------------------------------------- #
_D = 512


class _TinyMLP(torch.nn.Module):
    def __init__(self, d: int = _D) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(d, d, bias=True)
        self.fc2 = torch.nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _build_svdquant_model(bf16_model):
    from torch.utils.data import DataLoader, TensorDataset

    from quark.torch import ModelQuantizer
    from quark.torch.algorithm.svdquant.svdquant import (
        SVDQuantProcessor,
        build_quant_layer_config,
    )
    from quark.torch.quantization.config.config import QConfig, SVDQuantConfig

    model = copy.deepcopy(bf16_model)
    calib = torch.randn(16, _D, device=DEV, dtype=torch.bfloat16)
    dl = DataLoader(TensorDataset(calib), batch_size=4)
    svd = SVDQuantConfig(
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
    SVDQuantProcessor(model=model, quant_algo_config=svd, calib_data=dl).apply()
    qc = QConfig(global_quant_config=build_quant_layer_config("mxfp4"), exclude=["*correction*"])
    model = ModelQuantizer(qc).quantize_model(model, dl)
    return ModelQuantizer.freeze(model).eval()


def test_native_svdquant_vs_bf16_baseline():
    from quark.torch.quantization.nn.modules.flydsl_svdquant_inference_linear import (
        FlyDSLSVDQuantNativeInferenceLinear,
    )
    from quark.torch.quantization.utils import (
        RuntimeOptions,
        disable_native_inference,
        enable_native_inference,
    )

    for rep in range(REPS):
        torch.manual_seed(rep)
        bf16_model = _TinyMLP().to(DEV, torch.bfloat16).eval()
        x = torch.randn(8, _D, device=DEV, dtype=torch.bfloat16)
        with torch.no_grad():
            y_bf16 = bf16_model(x).clone()  # true bf16 baseline

        model = _build_svdquant_model(bf16_model)

        # aiter MXFP4 SVDQuant (separate correction) for comparison.
        enable_native_inference(model, runtime_options=RuntimeOptions(native_linear_mode="mxfp4"))
        with torch.no_grad():
            y_aiter = model(x).clone()
        disable_native_inference(model)

        # FlyDSL fused a8w4 SVDQuant.
        n = enable_native_inference(model, runtime_options=RuntimeOptions(native_linear_mode="flydsl_svdquant"))
        natives = [m for m in model.modules() if isinstance(m, FlyDSLSVDQuantNativeInferenceLinear)]
        with torch.no_grad():
            y_flydsl = model(x).clone()
        disable_native_inference(model)

        assert n > 0 and len(natives) == n
        m_fl = _metrics(y_flydsl, y_bf16)
        m_ai = _metrics(y_aiter, y_bf16)
        _report(f"rep{rep} flydsl_svd vs bf16", m_fl)
        _report(f"rep{rep} aiter_mxfp4 vs bf16", m_ai)

        assert torch.isfinite(y_flydsl).all()
        # Fused fp8-activation SVDQuant should be at least as accurate as the
        # a4w4 MXFP4 path against bf16 (fp8 activation is more precise), within
        # a small tolerance for run-to-run quant noise.
        assert m_fl["cos"] >= m_ai["cos"] - 0.01, (m_fl, m_ai)
        assert m_fl["cos"] > 0.95
        # Cosine cannot see a scale error, so hold the relative L2 as well; measured
        # 0.143-0.153 for the fused path against 0.193-0.209 for the aiter path.
        assert m_fl["rel_l2"] <= m_ai["rel_l2"] + 0.01, (m_fl, m_ai)
        assert m_fl["rel_l2"] < 0.25, m_fl


def test_bias_epilogue_matches_a_separate_add() -> None:
    """The fused bias epilogue must equal a post-hoc add.

    ``FlyDSLA8W4NativeInferenceLinear.forward`` takes ``epilogue="bias"`` for every layer
    that has a bias, so this is the shipped path, not an option.
    """
    from quark.torch.quantization.nn.modules.aiter_fp4_inference_linear import _pack_weight_asm
    from quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear import _gemm_flydsl_a8w4

    M, N, K = 256, 512, 256
    torch.manual_seed(0)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.5
    Wf = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(N, device=DEV, dtype=torch.bfloat16)
    kw, ks = _pack_weight_asm(Wf)

    y = _gemm_flydsl_a8w4(x, kw.view(torch.uint8), ks.view(torch.uint8), torch.bfloat16)
    y_fused = _gemm_flydsl_a8w4(
        x, kw.view(torch.uint8), ks.view(torch.uint8), torch.bfloat16, bias=bias, epilogue="bias"
    )

    m = _metrics(y_fused, y.float() + bias.float())
    _report("bias epilogue vs separate add", m)
    assert m["cos"] > 0.9999, f"fused bias diverged from a separate add: {m}"
    assert m["rel_l2"] < 0.01, f"fused bias diverged in magnitude: {m}"
