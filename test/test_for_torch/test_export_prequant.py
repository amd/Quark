#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Tests for export flow support of pre-quantized models."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from quark.torch.export.config.config import JsonExporterConfig
from quark.torch.export.main_export.model_post_process import ModelPostProcessor
from quark.torch.export.prequantized_layer_handler import _resolve_model_default_dtype
from quark.torch.export.utils import _quant_tensor_configs_match, _route_prequantized_layers
from quark.torch.quantization.config.config import QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver


class FakeCompressedLinear(nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.in_features = 64
        self.out_features = 32
        self.weight = nn.Parameter(torch.randn(32, 64).to(torch.float8_e4m3fn))
        self.bias = None
        self.weight_scale = torch.ones(32, 1)
        self.weight_zero_point = None
        self.weight_shape = None
        self.weight_g_idx = None


class FakeModelConfig:
    def __init__(self, dtype: object | None = None, torch_dtype: object | None = None) -> None:
        self.dtype = dtype
        self.torch_dtype = torch_dtype


class ModelWithPrequant(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        model_dtype: torch.dtype | None = None,
        config_dtype: object | None = None,
        config_torch_dtype: object | None = None,
    ) -> None:
        super().__init__()
        self.prequant = FakeCompressedLinear()
        self._model_dtype = model_dtype
        self.config = FakeModelConfig(dtype=config_dtype, torch_dtype=config_torch_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @property
    def dtype(self) -> torch.dtype | None:
        return self._model_dtype


def test_prequant_linear_converted_to_nn_linear() -> None:
    """ModelPostProcessor prioritizes model.dtype for pre-quantized export fallback."""
    model = ModelWithPrequant(
        model_dtype=torch.bfloat16,
        config_dtype="float32",
        config_torch_dtype=torch.float32,
    )
    config = JsonExporterConfig(weight_format="fake_quantized")
    received_dtype: torch.dtype | None = None

    def fake_dequantize(module: nn.Module, dtype: torch.dtype | None = None) -> nn.Linear:
        nonlocal received_dtype
        received_dtype = dtype
        w = module.weight.detach().float() * module.weight_scale.float()
        if dtype is not None:
            w = w.to(dtype)
        lin = nn.Linear(module.in_features, module.out_features, bias=False, dtype=w.dtype)
        lin.weight = nn.Parameter(w, requires_grad=False)
        lin.bias = None
        return lin

    with (
        patch(
            "quark.torch.export.prequantized_layer_handler.find_prequantized_linears",
            side_effect=lambda m: [(n, mod) for n, mod in m.named_modules() if isinstance(mod, FakeCompressedLinear)],
        ),
        patch(
            "quark.torch.export.prequantized_layer_handler.dequantize_prequantized_to_linear",
            side_effect=fake_dequantize,
        ),
    ):
        processed = ModelPostProcessor(model, config, custom_mode="default", output_quant=False).get_processed_model()

    assert isinstance(processed.prequant, nn.Linear)
    assert received_dtype == torch.bfloat16
    assert processed.prequant.weight.dtype == torch.bfloat16
    assert processed.prequant.weight.shape == (32, 64)


@pytest.mark.parametrize(
    ("model_dtype", "config_dtype", "config_torch_dtype", "expected"),
    [
        (None, None, "BFLOAT16", torch.bfloat16),  # string resolved case-insensitively
        (None, None, "UNKNOWN_DTYPE", None),  # unknown string → None
        (torch.float64, "float16", None, torch.float16),  # unsupported torch.dtype ignored
    ],
)
def test_resolve_model_default_dtype(model_dtype, config_dtype, config_torch_dtype, expected) -> None:
    model = ModelWithPrequant(model_dtype=model_dtype, config_dtype=config_dtype, config_torch_dtype=config_torch_dtype)
    assert _resolve_model_default_dtype(model) == expected


# ============================================================================
# _quant_tensor_configs_match
# ============================================================================

_FP8_PER_TENSOR = QTensorConfig(
    dtype=Dtype.fp8_e4m3,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    is_dynamic=False,
)


def test_configs_match_none_handling() -> None:
    """None matches None but not a real config."""
    assert _quant_tensor_configs_match(None, None) is True
    assert _quant_tensor_configs_match(_FP8_PER_TENSOR, None) is False
    assert _quant_tensor_configs_match(None, _FP8_PER_TENSOR) is False


def test_configs_match_equivalent_specs() -> None:
    other = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
    )
    assert _quant_tensor_configs_match(_FP8_PER_TENSOR, other) is True


def test_configs_match_differs_on_dtype() -> None:
    other = QTensorConfig(
        dtype=Dtype.fp8_e5m2,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
    )
    assert _quant_tensor_configs_match(_FP8_PER_TENSOR, other) is False


def test_model_post_processor_routes_to_preserve_when_keep_flag_set() -> None:
    """keep_prequantized_layers=True routes through preserve_prequantized_layers (not dequantize)."""
    model = ModelWithPrequant(model_dtype=torch.bfloat16)
    fake_quant_config = type("QC", (), {"keep_prequantized_layers": True})()

    config = JsonExporterConfig(weight_format="fake_quantized")

    with (
        patch("quark.torch.export.main_export.model_post_process.preserve_prequantized_layers") as preserve_mock,
        patch("quark.torch.export.main_export.model_post_process.dequantize_prequantized_linears") as dequant_mock,
    ):
        post = ModelPostProcessor(model, config, custom_mode="default", output_quant=False)
        post._quantization_config = fake_quant_config
        # Only the routing branch matters here; the rest of the pipeline may fail without a real config.
        with contextlib.suppress(Exception):
            post.get_processed_model()

    preserve_mock.assert_called_once()
    dequant_mock.assert_not_called()


def test_configs_match_block_size_normalised() -> None:
    """block_size as list vs tuple should still compare equal."""
    a = type("X", (), {})()
    a.dtype = Dtype.fp8_e4m3
    a.qscheme = QSchemeType.per_block
    a.group_size = None
    a.block_size = [128, 128]

    b = type("X", (), {})()
    b.dtype = Dtype.fp8_e4m3
    b.qscheme = QSchemeType.per_block
    b.group_size = None
    b.block_size = (128, 128)

    assert _quant_tensor_configs_match(a, b) is True


# ---------------------------------------------------------------------------
# _route_prequantized_layers — 4 decision branches
# ---------------------------------------------------------------------------


class _StubModule(nn.Module):  # type: ignore[misc]
    pass


@dataclass
class _WeightCfg:
    """Stand-in for a layer / native config exposing a ``.weight`` spec."""

    weight: object


@dataclass
class _Spec:
    """Stand-in for a QTensorConfig weight spec compared by _quant_tensor_configs_match."""

    dtype: object
    qscheme: object
    group_size: object = None
    block_size: object = None


def _run_route(layer_config: object, native_config: object) -> tuple[int, int, object]:
    """Drive _route_prequantized_layers with mocked boundaries; return counts + dequant call."""
    model = _StubModule()
    fake_module = _StubModule()
    replacement = nn.Linear(2, 2)

    with (
        patch(
            "quark.torch.export.utils.find_prequantized_linears",
            return_value=[("layer0", fake_module)],
        ),
        patch("quark.torch.export.utils.get_layer_quant_config", return_value=layer_config),
        patch(
            "quark.torch.export.utils.convert_prequantized_module_to_quark_config",
            return_value=native_config,
        ),
        patch("quark.torch.export.utils.dequantize_prequantized_to_linear", return_value=replacement) as dequant_mock,
        patch("quark.torch.export.utils.setattr_recursive") as setattr_mock,
    ):
        converted, preserved = _route_prequantized_layers(model, quantization_config=object(), model_dtype=None)
    return converted, preserved, (dequant_mock, setattr_mock, replacement)


_FP8 = _Spec(Dtype.fp8_e4m3, QSchemeType.per_tensor)
_INT8 = _Spec(Dtype.int8, QSchemeType.per_tensor)

# _route_prequantized_layers returns (converted_count, preserved_count).
_DEQUANTIZED = (1, 0)
_PRESERVED = (0, 1)


@pytest.mark.parametrize(
    ("layer_config", "native_config", "expected"),
    [
        pytest.param(None, _WeightCfg(object()), _DEQUANTIZED, id="no_layer_config"),
        pytest.param(_WeightCfg(object()), None, _DEQUANTIZED, id="no_native_config"),
        pytest.param(_WeightCfg(_FP8), _WeightCfg(_INT8), _DEQUANTIZED, id="config_mismatch"),
        pytest.param(_WeightCfg(_FP8), _WeightCfg(_FP8), _PRESERVED, id="config_match"),
    ],
)
def test_route_prequantized_layers_branches(layer_config, native_config, expected) -> None:
    converted, preserved, (dequant_mock, setattr_mock, replacement) = _run_route(layer_config, native_config)
    assert (converted, preserved) == expected
    if expected == _PRESERVED:
        dequant_mock.assert_not_called()
        setattr_mock.assert_not_called()
    else:
        dequant_mock.assert_called_once()
        setattr_mock.assert_called_once()
        assert setattr_mock.call_args.args[1] == "layer0"
        assert setattr_mock.call_args.args[2] is replacement
