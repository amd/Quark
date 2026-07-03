#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Tests for :mod:`quark.torch.export.prequantized_config_converter`.

Focused unit tests for both concrete converters and the public dispatch entry point.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from quark.torch.export.prequantized_config_converter import (
    CompressedTensorsConfigConverter,
    FP8LinearConfigConverter,
    convert_prequantized_module_to_quark_config,
)
from quark.torch.quantization.config.type import Dtype, QSchemeType

# ============================================================================
#  CompressedTensorsConfigConverter
# ============================================================================


class _QArgs:
    """Stand-in for compressed_tensors.QuantizationArgs."""

    def __init__(
        self,
        num_bits=8,
        qtype="float",
        strategy="tensor",
        symmetric=True,
        group_size=None,
        block_structure=None,
        dynamic=False,
    ):
        self.num_bits = num_bits
        self.type = qtype
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = group_size
        self.block_structure = block_structure
        self.dynamic = dynamic


class _Scheme:
    def __init__(self, weights=None, input_activations=None):
        self.weights = weights
        self.input_activations = input_activations


def _compressed_module(scheme):
    m = nn.Module()
    m.quantization_scheme = scheme
    return m


@pytest.mark.parametrize(
    ("module_factory", "case_id"),
    [
        (lambda: nn.Module(), "no_scheme_attr"),
        (lambda: _compressed_module(_Scheme(None, None)), "empty_scheme"),
        (
            lambda: _compressed_module(_Scheme(weights=_QArgs(num_bits=5, qtype="int", strategy="tensor"))),
            "unsupported_dtype",
        ),
        (
            lambda: _compressed_module(_Scheme(weights=_QArgs(num_bits=8, qtype="float", strategy="frob"))),
            "unsupported_strategy",
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_compressed_converter_returns_none(module_factory, case_id):
    """Every unsupported / empty input shape collapses to None."""
    assert CompressedTensorsConfigConverter().convert(module_factory()) is None


def test_compressed_converter_fp8_per_tensor():
    """FP8 per_tensor weight → maps to fp8_e4m3 + per_tensor."""
    sch = _Scheme(weights=_QArgs(num_bits=8, qtype="float", strategy="tensor"))
    out = CompressedTensorsConfigConverter().convert(_compressed_module(sch))
    assert out is not None
    assert out.weight.dtype == Dtype.fp8_e4m3
    assert out.weight.qscheme == QSchemeType.per_tensor


def test_compressed_converter_int8_per_channel_weight_axis_zero():
    """per_channel weight → ch_axis=0; non-weight tensor → ch_axis=-1."""
    sch = _Scheme(
        weights=_QArgs(num_bits=8, qtype="int", strategy="channel"),
        input_activations=_QArgs(num_bits=8, qtype="int", strategy="channel", dynamic=True),
    )
    out = CompressedTensorsConfigConverter().convert(_compressed_module(sch))
    assert out.weight.ch_axis == 0
    assert out.input_tensors.ch_axis == -1
    assert out.input_tensors.is_dynamic is True


def test_compressed_converter_per_group_axis_minus_one():
    """per_group always uses ch_axis=-1."""
    sch = _Scheme(weights=_QArgs(num_bits=4, qtype="int", strategy="group", group_size=32))
    out = CompressedTensorsConfigConverter().convert(_compressed_module(sch))
    assert out.weight.qscheme == QSchemeType.per_group
    assert out.weight.ch_axis == -1
    assert out.weight.group_size == 32


def test_compressed_converter_per_block_carries_block_size():
    """per_block strategy with block_structure → block_size populated on output."""
    sch = _Scheme(
        weights=_QArgs(num_bits=8, qtype="float", strategy="block", block_structure=[128, 128]),
    )
    out = CompressedTensorsConfigConverter().convert(_compressed_module(sch))
    assert out.weight.qscheme == QSchemeType.per_block
    assert out.weight.block_size == [128, 128]


def test_compressed_converter_handles_enum_like_strategy_and_type():
    """type/strategy values may be enum-like with `.value`; both branches are exercised."""

    class EnumLike:
        def __init__(self, v):
            self.value = v

    qargs = _QArgs(num_bits=8, qtype=EnumLike("float"), strategy=EnumLike("tensor"))
    sch = _Scheme(weights=qargs)
    out = CompressedTensorsConfigConverter().convert(_compressed_module(sch))
    assert out is not None
    assert out.weight.dtype == Dtype.fp8_e4m3


def test_compressed_converter_handles_non_bool_dynamic_truthy():
    """dynamic carried as a non-bool (e.g. DynamicType enum) is coerced via != False."""

    class DynLike:
        def __eq__(self, other):
            # Non-False match for the "!= False" coercion.
            return False

    qargs = _QArgs(num_bits=8, qtype="float", strategy="tensor", dynamic=DynLike())
    sch = _Scheme(input_activations=qargs)
    out = CompressedTensorsConfigConverter().convert(_compressed_module(sch))
    assert out is not None
    assert out.input_tensors.is_dynamic is True


# ============================================================================
#  FP8LinearConfigConverter
# ============================================================================


class _FP8LinearLike(nn.Module):
    """Stand-in for transformers.FP8Linear."""

    def __init__(self, weight_dtype=torch.float8_e4m3fn, activation_scheme="dynamic", block_size=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4).to(weight_dtype), requires_grad=False)
        self.activation_scheme = activation_scheme
        self.block_size = block_size


def _fp8_module_without_weight():
    return nn.Module()


def _fp8_module_with_unsupported_weight_dtype():
    m = nn.Module()
    m.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.float32), requires_grad=False)
    m.activation_scheme = "dynamic"
    m.block_size = None
    return m


@pytest.mark.parametrize(
    "module_factory",
    [_fp8_module_without_weight, _fp8_module_with_unsupported_weight_dtype],
    ids=["weight_missing", "unsupported_weight_dtype"],
)
def test_fp8_converter_returns_none(module_factory):
    assert FP8LinearConfigConverter().convert(module_factory()) is None


def test_fp8_converter_per_tensor_dynamic_default():
    """FP8 with no block_size and dynamic activations → per_tensor weight + dynamic input."""
    m = _FP8LinearLike(activation_scheme="dynamic", block_size=None)
    cfg = FP8LinearConfigConverter().convert(m)
    assert cfg is not None
    assert cfg.weight.qscheme == QSchemeType.per_tensor
    assert cfg.input_tensors.qscheme == QSchemeType.per_tensor
    assert cfg.input_tensors.is_dynamic is True


def test_fp8_converter_per_tensor_static():
    """activation_scheme='static' → input is non-dynamic per_tensor."""
    m = _FP8LinearLike(activation_scheme="static", block_size=None)
    cfg = FP8LinearConfigConverter().convert(m)
    assert cfg.input_tensors.is_dynamic is False


def test_fp8_converter_per_block_dynamic_routes_input_to_per_group():
    """per_block weight + dynamic activations → input becomes per_group with matching group_size."""
    m = _FP8LinearLike(activation_scheme="dynamic", block_size=[128, 128])
    cfg = FP8LinearConfigConverter().convert(m)
    assert cfg.weight.qscheme == QSchemeType.per_block
    assert cfg.weight.block_size == [128, 128]
    assert cfg.input_tensors.qscheme == QSchemeType.per_group
    assert cfg.input_tensors.group_size == 128
    assert cfg.input_tensors.is_dynamic is True


def test_fp8_converter_per_block_static_keeps_input_per_tensor():
    """per_block weight + static activations → input remains per_tensor (static)."""
    m = _FP8LinearLike(activation_scheme="static", block_size=[128, 128])
    cfg = FP8LinearConfigConverter().convert(m)
    assert cfg.input_tensors.qscheme == QSchemeType.per_tensor
    assert cfg.input_tensors.is_dynamic is False


# ============================================================================
#  convert_prequantized_module_to_quark_config dispatch
# ============================================================================


def test_dispatch_unknown_module_returns_none():
    """Unregistered module type with no quantization_scheme → returns None."""
    m = nn.Linear(8, 8)
    assert convert_prequantized_module_to_quark_config(m) is None


def test_dispatch_routes_quantization_scheme_attr_to_compressed_converter():
    """nn.Linear with quantization_scheme attribute → routed to CompressedLinear converter."""
    m = nn.Linear(8, 8)
    m.quantization_scheme = _Scheme(weights=_QArgs(num_bits=8, qtype="float", strategy="tensor"))
    out = convert_prequantized_module_to_quark_config(m)
    assert out is not None
    assert out.weight.dtype == Dtype.fp8_e4m3
