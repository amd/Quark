#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quark.common.utils.testing_utils import skip_if_no_gpu, torch_device
from quark.torch.algorithm.svdquant.svdquant import (
    QUANT_MODE_TO_SCHEME,
    ActivationSmoother,
    ErrorCorrectedModule,
    HessianCollector,
    LowRankCorrectionModule,
    SVDQuantProcessor,
    _replace_module,
    _simulate_quant,
    _svd_decompose,
    apply_smoothing,
    build_quant_layer_config,
    gptq_quantize_residual,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _SVDQuantConfig:
    """Minimal config mirroring SVDQuantConfig for testing."""

    name: str = "svdquant"
    svd_rank: int = 4
    smooth_alpha: float = 0.5
    search_alpha: bool = False
    alpha_candidates: list[float] | None = None
    alpha_search_max_samples: int = 4
    exclude_patterns: list[str] = field(default_factory=list)
    min_layer_size: int = 1
    use_gptq: bool = False
    gptq_n_bits: int = 4
    gptq_symmetric: bool = True
    gptq_group_size: int = -1
    gptq_blocksize: int = 128
    gptq_percdamp: float = 0.01
    gptq_actorder: bool = False


class TinyModel(nn.Module):
    def __init__(self, d_in: int = 16, d_hidden: int = 32, d_out: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _make_dataloader(
    n_samples: int = 4, d_in: int = 16, batch_size: int = 2, device: torch.device | str = "cpu"
) -> DataLoader[torch.Tensor]:
    data = torch.randn(n_samples, d_in, device=device)
    return DataLoader(TensorDataset(data), batch_size=batch_size)


def _make_psd_matrix(n: int, device: torch.device | str = "cpu") -> torch.Tensor:
    A = torch.randn(n, n, device=device)
    return A.t() @ A + 0.1 * torch.eye(n, device=device)


# ---------------------------------------------------------------------------
# Unit tests for individual components (CPU, no GPU needed)
# ---------------------------------------------------------------------------


def test_hessian_collector():
    model = TinyModel()
    hc = HessianCollector()
    hc.register_hooks(model, {"fc1"})

    with torch.no_grad():
        model(torch.randn(2, 16))
        model(torch.randn(3, 16))

    assert "fc1" in hc.hessians
    assert hc.hessians["fc1"].shape == (16, 16)
    assert hc.nsamples["fc1"] == 5

    s = torch.ones(16) * 2.0
    H_before = hc.hessians["fc1"].clone()
    hc.adjust_for_smoothing({"fc1": s})
    assert torch.allclose(hc.hessians["fc1"], H_before * (0.5**2), atol=1e-5)

    hc.adjust_for_smoothing({"nonexistent": torch.ones(4)})

    hc.remove_hooks()
    assert hc._hooks == []


def test_build_quant_layer_config():
    for mode in QUANT_MODE_TO_SCHEME:
        assert build_quant_layer_config(mode) is not None
    assert build_quant_layer_config("W4A16") is not None
    with pytest.raises(ValueError, match="Unknown quant mode"):
        build_quant_layer_config("bogus_mode")


def test_low_rank_correction_module():
    m = LowRankCorrectionModule(16, 8, rank=4)
    assert m(torch.randn(2, 16)).shape == (2, 8)
    assert m.l1.bias is None and m.l2.bias is None


def test_error_corrected_module():
    layer = nn.Linear(16, 8, bias=True)
    correction = LowRankCorrectionModule(16, 8, rank=4)

    ecm_no_smooth = ErrorCorrectedModule(correction, layer, smooth_factor=None)
    assert ecm_no_smooth(torch.randn(2, 16)).shape == (2, 8)
    assert ecm_no_smooth.smooth_factor is None
    assert ecm_no_smooth.weight is layer.weight
    assert ecm_no_smooth.bias is layer.bias
    assert ecm_no_smooth.in_features == 16
    assert ecm_no_smooth.out_features == 8

    ecm_smooth = ErrorCorrectedModule(correction, layer, smooth_factor=torch.ones(16) * 2.0)
    assert ecm_smooth(torch.randn(2, 16)).shape == (2, 8)
    assert ecm_smooth.smooth_factor is not None


def test_simulate_quant():
    W = torch.randn(8, 16)
    assert _simulate_quant(W, n_bits=4, symmetric=True).shape == W.shape
    assert _simulate_quant(W, n_bits=4, symmetric=False).shape == W.shape
    assert _simulate_quant(W, n_bits=4, symmetric=True, group_size=8).shape == W.shape


def test_activation_smoother():
    model = TinyModel()
    smoother = ActivationSmoother(alpha=0.5)
    smoother.register_hooks(model)

    with torch.no_grad():
        model(torch.randn(4, 16))

    smoother.remove_hooks()
    assert "fc1" in smoother.activation_max

    factors = smoother.compute_smooth_factors(model, exclude_patterns=[])
    assert "fc1" in factors
    factors_excl = smoother.compute_smooth_factors(model, exclude_patterns=["fc1"])
    assert "fc1" not in factors_excl
    factors_alpha = smoother.compute_smooth_factors(model, exclude_patterns=[], per_layer_alpha={"fc1": 0.8})
    assert "fc1" in factors_alpha


def test_apply_smoothing():
    model = TinyModel()
    model.fc1.weight.data = model.fc1.weight.data.half()
    w_before = model.fc1.weight.data.clone()
    s = torch.ones(16) * 2.0

    apply_smoothing(model, {"fc1": s}, keep_float32=True)
    assert model.fc1.weight.data.dtype == torch.float32

    model.fc1.weight.data = w_before
    applied = apply_smoothing(model, {"fc1": s}, keep_float32=False)
    assert "fc1" in applied
    assert model.fc1.weight.data.dtype == torch.float16


def test_replace_module():
    model = TinyModel()
    _replace_module(model, "fc1", nn.Identity())
    assert isinstance(model.fc1, nn.Identity)


def test_should_quantize():
    proc = SVDQuantProcessor(TinyModel(), _SVDQuantConfig(exclude_patterns=["time_*"], min_layer_size=256), None)
    assert proc._should_quantize("layer", nn.Linear(256, 256)) is True
    assert proc._should_quantize("conv", nn.Conv2d(3, 3, 3)) is False
    assert proc._should_quantize("time_embedding", nn.Linear(256, 256)) is False
    assert proc._should_quantize("small", nn.Linear(4, 8)) is False
    layer_1d = nn.Linear(16, 32)
    layer_1d.weight = nn.Parameter(torch.randn(32))
    assert proc._should_quantize("bad", layer_1d) is False


# ---------------------------------------------------------------------------
# GPU-required integration tests
# ---------------------------------------------------------------------------


@skip_if_no_gpu
def test_gptq_quantize_residual():
    W = torch.randn(8, 16, device=torch_device)
    H = _make_psd_matrix(16, device=torch_device)
    Q = gptq_quantize_residual(W, H, n_bits=4, symmetric=True, group_size=8, actorder=True, blocksize=4)
    assert Q.shape == W.shape
    assert Q.dtype == W.dtype


@skip_if_no_gpu
def test_gptq_quantize_residual_asymmetric():
    W = torch.randn(8, 16, device=torch_device)
    H = _make_psd_matrix(16, device=torch_device)
    Q = gptq_quantize_residual(W, H, n_bits=4, symmetric=False, group_size=8)
    assert Q.shape == W.shape


@skip_if_no_gpu
def test_gptq_cholesky_fallback():
    W = torch.randn(4, 4, device=torch_device)
    H = torch.zeros(4, 4, device=torch_device)
    Q = gptq_quantize_residual(W, H, n_bits=4)
    assert Q.shape == W.shape


@skip_if_no_gpu
def test_svd_decompose():
    W = torch.randn(8, 16, device=torch_device)
    L1, L2, R = _svd_decompose(W, rank=4)
    assert torch.allclose(W, L2 @ L1 + R, atol=1e-4)

    W_small = torch.randn(4, 4, device=torch_device)
    L1s, L2s, Rs = _svd_decompose(W_small, rank=100)
    assert torch.allclose(W_small, L2s @ L1s + Rs, atol=1e-4)


@skip_if_no_gpu
def test_apply_no_calib_data():
    model = TinyModel().to(torch_device)
    SVDQuantProcessor(model, _SVDQuantConfig(), calib_data=None).apply()
    assert isinstance(model.fc1, ErrorCorrectedModule)
    assert isinstance(model.fc2, ErrorCorrectedModule)


@skip_if_no_gpu
def test_apply_with_calib_data():
    model = TinyModel().to(torch_device)
    dl = _make_dataloader(device=torch_device)
    SVDQuantProcessor(model, _SVDQuantConfig(), calib_data=dl).apply()
    assert isinstance(model.fc1, ErrorCorrectedModule)


@skip_if_no_gpu
def test_apply_with_gptq():
    model = TinyModel().to(torch_device)
    dl = _make_dataloader(device=torch_device)
    SVDQuantProcessor(model, _SVDQuantConfig(use_gptq=True), calib_data=dl).apply()
    assert isinstance(model.fc1, ErrorCorrectedModule)


@skip_if_no_gpu
def test_apply_with_alpha_search():
    model = TinyModel().to(torch_device)
    config = _SVDQuantConfig(search_alpha=True, alpha_candidates=[0.3, 0.5, 0.7], alpha_search_max_samples=4)
    dl = _make_dataloader(device=torch_device)
    SVDQuantProcessor(model, config, calib_data=dl).apply()
    assert isinstance(model.fc1, ErrorCorrectedModule)


@skip_if_no_gpu
def test_apply_with_alpha_search_default_candidates():
    model = TinyModel().to(torch_device)
    config = _SVDQuantConfig(search_alpha=True, alpha_candidates=None, alpha_search_max_samples=4)
    dl = _make_dataloader(device=torch_device)
    SVDQuantProcessor(model, config, calib_data=dl).apply()
    assert isinstance(model.fc1, ErrorCorrectedModule)


@skip_if_no_gpu
def test_apply_dict_data():
    class DictModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8, bias=False)

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            return self.fc(x)

    model = DictModel().to(torch_device)
    dl = DataLoader([{"x": torch.randn(2, 8, device=torch_device)} for _ in range(3)], batch_size=None)
    SVDQuantProcessor(model, _SVDQuantConfig(svd_rank=2), calib_data=dl).apply()
    assert isinstance(model.fc, ErrorCorrectedModule)


@skip_if_no_gpu
def test_apply_diffusion_data():
    class DiffusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8, bias=False)

        def forward(self, data):
            args, kwargs = data[0]
            return self.fc(args)

    model = DiffusionModel().to(torch_device)
    sample = ((torch.randn(2, 8, device=torch_device), {"timestep": 1}),)
    dl = DataLoader([sample], batch_size=None)
    SVDQuantProcessor(model, _SVDQuantConfig(svd_rank=2), calib_data=dl).apply()
    assert isinstance(model.fc, ErrorCorrectedModule)


# ---------------------------------------------------------------------------
# E2E regression tests for the fp16 SVDQuant black-image bug
# ---------------------------------------------------------------------------


class _DeepTransformerBlock(nn.Module):
    """Single transformer-like block: two linear layers with residual."""

    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(torch.relu(self.fc1(x)))


class _DeepModel(nn.Module):
    """A deep sequential model that mimics SD3's 24-block transformer."""

    def __init__(self, d: int = 256, n_blocks: int = 24):
        super().__init__()
        self.blocks = nn.ModuleList([_DeepTransformerBlock(d) for _ in range(n_blocks)])
        self.head = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _make_deep_dataloader(
    n_samples: int = 8,
    d: int = 256,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> DataLoader[torch.Tensor]:
    data = torch.randn(n_samples, d, device=device, dtype=dtype)
    return DataLoader(TensorDataset(data), batch_size=batch_size)


@skip_if_no_gpu
def test_smooth_factor_absorbed_after_apply():
    """After SVDQuant with smooth_alpha > 0, smooth_factor must be None
    (absorbed into weights), not a runtime buffer."""
    model = TinyModel().to(torch_device).half()
    data = torch.randn(4, 16, device=torch_device, dtype=torch.float16)
    dl = DataLoader(TensorDataset(data), batch_size=2)
    cfg = _SVDQuantConfig(smooth_alpha=0.5, search_alpha=False)
    SVDQuantProcessor(model, cfg, calib_data=dl).apply()

    for name, module in model.named_modules():
        if isinstance(module, ErrorCorrectedModule):
            assert module.smooth_factor is None, (
                f"ErrorCorrectedModule '{name}' should have smooth_factor=None (absorbed into weights), but it is set."
            )


@skip_if_no_gpu
def test_fp16_svdquant_no_nan_or_zero():
    """Regression test: SVDQuant with smooth_alpha > 0 on a deep fp16 model
    must not produce NaN or all-zero outputs (the black-image bug)."""
    d = 256
    model = _DeepModel(d=d, n_blocks=24).to(torch_device).half()

    dl = _make_deep_dataloader(d=d, device=torch_device, dtype=torch.float16)
    cfg = _SVDQuantConfig(smooth_alpha=0.5, search_alpha=False, svd_rank=4, min_layer_size=1)
    SVDQuantProcessor(model, cfg, calib_data=dl).apply()

    x = torch.randn(4, d, device=torch_device, dtype=torch.float16)
    with torch.no_grad():
        out = model(x)

    assert not torch.isnan(out).any(), "SVDQuant output contains NaN"
    assert not torch.isinf(out).any(), "SVDQuant output contains Inf"
    assert out.abs().mean() > 1e-6, "SVDQuant output is all-zero (black-image bug)"


@skip_if_no_gpu
def test_fp16_ecm_forward_no_overflow():
    """Directly test ErrorCorrectedModule with large fan-in and large
    activations that would overflow in fp16 without the fp32 compute path."""
    fan_in = 6144
    fan_out = 1024
    layer = nn.Linear(fan_in, fan_out, bias=False).to(torch_device).half()
    correction = LowRankCorrectionModule(fan_in, fan_out, rank=4).to(torch_device).half()
    ecm = ErrorCorrectedModule(correction, layer, smooth_factor=None)

    x = torch.full((2, fan_in), 350.0, device=torch_device, dtype=torch.float16)
    with torch.no_grad():
        out = ecm(x)

    assert not torch.isnan(out).any(), "ECM forward produced NaN with large fp16 activations"
    assert not torch.isinf(out).any(), "ECM forward produced Inf with large fp16 activations"


# ---------------------------------------------------------------------------
# _hf_hook preservation in _replace_module (accelerate device_map support)
# ---------------------------------------------------------------------------


def test_replace_module_preserves_hf_hook():
    """When the old module has an accelerate _hf_hook, _replace_module must
    transfer it to the new module so device-placement survives the swap."""
    pytest.importorskip("accelerate")
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    class _Container(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4, bias=False)

    model = _Container()
    original_hook = AlignDevicesHook(execution_device="cpu", offload=False, io_same_device=True)
    add_hook_to_module(model.fc, original_hook)
    assert hasattr(model.fc, "_hf_hook")

    new_linear = nn.Linear(4, 4, bias=False)
    _replace_module(model, "fc", new_linear)

    assert model.fc is new_linear
    assert hasattr(model.fc, "_hf_hook"), "_hf_hook was dropped during _replace_module"
    assert model.fc._hf_hook.execution_device == original_hook.execution_device
    assert model.fc._hf_hook.io_same_device == original_hook.io_same_device
