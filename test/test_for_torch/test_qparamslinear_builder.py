#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Tests for :mod:`quark.torch.export.nn.modules.qparamslinear_builder`."""

from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from quark.torch.export.nn.modules import qparamslinear_builder as _builder_module
from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.export.nn.modules.qparamslinear_builder import (
    ExportBuilder,
    ImportBuilder,
    PreserveBuilder,
    create_builder,
)
from quark.torch.export.nn.modules.realquantizer import SequentialRealQuantizer
from quark.torch.quantization.config.config import QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver, PerTensorMinMaxObserver

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FP8_PER_TENSOR_SPEC = QTensorConfig(
    dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
)


def test_export_builder_build_quantizers():
    """Cover ExportBuilder.build_quantizers all four branches (weight, bias, input, output)
    and build_bias returning None when source has no bias.
    """
    mock_source = Mock(
        spec=[
            "weight_qspec",
            "weight_quantizer",
            "bias_qspec",
            "bias_quantizer",
            "input_qspec",
            "input_quantizer",
            "output_qspec",
            "output_quantizer",
            "is_prequantized",
            "weight",
            "bias",
        ],
    )
    mock_source.weight_qspec = FP8_PER_TENSOR_SPEC
    mock_source.weight_quantizer = Mock()
    mock_source.bias_qspec = FP8_PER_TENSOR_SPEC
    mock_source.bias_quantizer = Mock()
    mock_source.input_qspec = FP8_PER_TENSOR_SPEC
    mock_source.input_quantizer = Mock()
    mock_source.output_qspec = FP8_PER_TENSOR_SPEC
    mock_source.output_quantizer = Mock()
    mock_source.bias = None

    builder = ExportBuilder(mock_source, reorder=False, custom_mode="fp8")

    target = Mock(spec=QParamsLinear)
    target.weight_quantizer = None
    target.bias_quantizer = None
    target.input_quantizer = None
    target.output_quantizer = None

    mock_real_quantizer = Mock()
    with patch(
        "quark.torch.export.nn.modules.qparamslinear_builder.get_real_quantizer",
        return_value=mock_real_quantizer,
    ) as mock_get_real_quantizer:
        builder.build_quantizers(target, device=torch.device(DEVICE))
        assert mock_get_real_quantizer.call_count == 4
        assert target.weight_quantizer == mock_real_quantizer
        assert target.bias_quantizer == mock_real_quantizer
        assert target.input_quantizer == mock_real_quantizer
        assert target.output_quantizer == mock_real_quantizer

    assert builder.build_bias(device=torch.device(DEVICE)) is None


def test_export_builder_real_quantize_branches():
    """Cover ExportBuilder._real_quantize: float_weight=None path, float_weight
    without weight quantizer path, and bias quantization path.
    """
    # Case 1: float_weight=None with weight_quantizer
    target_case1 = Mock(spec=QParamsLinear)
    target_case1.weight = nn.Parameter(torch.randn(64, 64, device=DEVICE), requires_grad=False)
    target_case1.bias = None
    mock_weight_quantizer = Mock()
    mock_weight_quantizer.is_dynamic = False
    mock_weight_quantizer.to_real_quantize_params = Mock(return_value=torch.randn(64, 64, device=DEVICE))
    mock_weight_quantizer.maybe_convert_and_transpose_scale = Mock()
    mock_weight_quantizer.pack_zero_point = Mock()
    target_case1.weight_quantizer = mock_weight_quantizer
    target_case1.bias_quantizer = None
    target_case1.input_quantizer = None
    target_case1.output_quantizer = None

    ExportBuilder._real_quantize(target_case1, float_weight=None)
    mock_weight_quantizer.to_real_quantize_params.assert_called_once()

    # Case 2: float_weight provided but no weight_quantizer
    target_case2 = Mock(spec=QParamsLinear)
    target_case2.weight = nn.Parameter(torch.randn(64, 64, device=DEVICE), requires_grad=False)
    target_case2.bias = None
    target_case2.weight_quantizer = None
    target_case2.bias_quantizer = None
    target_case2.input_quantizer = None
    target_case2.output_quantizer = None

    float_weight = torch.randn(64, 64, device=DEVICE)
    ExportBuilder._real_quantize(target_case2, float_weight=float_weight)
    assert isinstance(target_case2.weight, nn.Parameter)

    # Case 3: bias quantization
    target_case3 = Mock(spec=QParamsLinear)
    target_case3.weight = nn.Parameter(torch.randn(64, 64, device=DEVICE), requires_grad=False)
    target_case3.bias = nn.Parameter(torch.randn(64, device=DEVICE), requires_grad=False)
    target_case3.weight_quantizer = None
    mock_bias_quantizer = Mock()
    mock_bias_quantizer.is_dynamic = False
    mock_bias_quantizer.to_real_quantize_params = Mock(return_value=torch.randn(64, device=DEVICE))
    mock_bias_quantizer.maybe_convert_and_transpose_scale = Mock()
    mock_bias_quantizer.pack_zero_point = Mock()
    target_case3.bias_quantizer = mock_bias_quantizer
    target_case3.input_quantizer = None
    target_case3.output_quantizer = None

    ExportBuilder._real_quantize(target_case3, float_weight=None)
    mock_bias_quantizer.to_real_quantize_params.assert_called_once()


class FP8Linear(nn.Linear):
    """Simulates transformers.FP8Linear for prequantized path testing."""

    pass


def test_preserve_builder_from_prequantized_module():
    """Cover PreserveBuilder: config derivation, tensor extraction, resolve_device, build_bias."""
    source = FP8Linear(in_features=64, out_features=64, bias=True).to(DEVICE)
    source.weight_scale_inv = torch.tensor([1.0], device=DEVICE)

    fake_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    with patch.object(
        _builder_module,
        "convert_prequantized_module_to_quark_config",
        return_value=fake_config,
    ):
        builder = PreserveBuilder(source, reorder=False)

    resolved_device = builder.resolve_device()
    assert resolved_device.type == torch.device(DEVICE).type

    original_bias = source.bias.detach().clone()
    bias_parameter = builder.build_bias(device=torch.device(DEVICE))
    assert bias_parameter is not None
    assert torch.allclose(bias_parameter.data, original_bias)
    assert not bias_parameter.requires_grad


def test_import_builder_prequantized_module():
    """Cover ImportBuilder: resolve_device via weight, and empty bias placeholder."""
    source_with_weight = FP8Linear(in_features=64, out_features=64, bias=True).to(DEVICE)
    quantization_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    builder = ImportBuilder(
        source_with_weight, reorder=False, custom_mode="fp8", quantization_config=quantization_config
    )
    assert builder.resolve_device().type == torch.device(DEVICE).type

    bias_parameter = builder.build_bias(device=torch.device(DEVICE))
    assert bias_parameter is not None
    assert bias_parameter.shape == (64,)
    assert not bias_parameter.requires_grad


def test_import_builder_quantizer_config_branches():
    """Cover ImportBuilder.build_quantizers with weight=None, bias config,
    and output config via QParamsLinear.from_module integration.
    """
    # weight=None → falls through to empty weight allocation
    quantization_config_no_weight = QLayerConfig(weight=None, input_tensors=FP8_PER_TENSOR_SPEC)
    float_module = nn.Linear(in_features=64, out_features=64, bias=True, dtype=torch.float32).to(DEVICE)
    qparams_linear_no_weight = QParamsLinear.from_module(
        float_module, custom_mode="fp8", pack_method=None, quant_config=quantization_config_no_weight
    )
    assert qparams_linear_no_weight.weight_quantizer is None
    assert qparams_linear_no_weight.input_quantizer is not None

    # bias config
    quantization_config_with_bias = QLayerConfig(weight=FP8_PER_TENSOR_SPEC, bias=FP8_PER_TENSOR_SPEC)
    float_module_bias = nn.Linear(in_features=64, out_features=64, bias=True, dtype=torch.float32).to(DEVICE)
    qparams_linear_bias = QParamsLinear.from_module(
        float_module_bias, custom_mode="fp8", pack_method=None, quant_config=quantization_config_with_bias
    )
    assert qparams_linear_bias.bias_quantizer is not None

    # output config
    quantization_config_with_output = QLayerConfig(weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC)
    float_module_output = nn.Linear(in_features=64, out_features=64, bias=False, dtype=torch.float32).to(DEVICE)
    qparams_linear_output = QParamsLinear.from_module(
        float_module_output, custom_mode="fp8", pack_method=None, quant_config=quantization_config_with_output
    )
    assert qparams_linear_output.output_quantizer is not None

    # _create_and_set_quantizer "weight"
    target = Mock(spec=QParamsLinear)
    target.weight_quantizer = None
    target.bias_quantizer = None
    target.input_quantizer = None
    target.output_quantizer = None
    ImportBuilder._create_and_set_quantizer(
        target,
        "weight",
        FP8_PER_TENSOR_SPEC,
        reorder=False,
        device=torch.device(DEVICE),
        float_dtype=torch.float32,
        real_quantized=True,
    )
    assert target.weight_quantizer is not None


# ============================================================================
#  PreserveBuilder extra coverage
# ============================================================================


def test_preserve_builder_raises_on_unsupported_format():
    """PreserveBuilder raises when no converter handles the source module."""
    plain_module = nn.Linear(in_features=8, out_features=8, bias=False).to(DEVICE)
    with (
        patch.object(_builder_module, "convert_prequantized_module_to_quark_config", return_value=None),
        pytest.raises(ValueError, match="Unsupported prequantized format"),
    ):
        PreserveBuilder(plain_module, reorder=False)


def test_preserve_builder_extract_uses_weight_scale_when_inv_missing():
    """_extract_weight_tensors falls back to weight_scale when weight_scale_inv is absent."""
    source = FP8Linear(in_features=8, out_features=8, bias=False).to(DEVICE)
    source.weight_scale = torch.tensor([0.5], device=DEVICE)
    source.weight_zero_point = None
    fake_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    weight, scale, zp = PreserveBuilder._extract_weight_tensors(source, fake_config, reorder=False)
    assert torch.equal(scale, source.weight_scale)
    assert zp is None
    assert weight.shape == source.weight.shape


def test_preserve_builder_extract_raises_when_no_weight():
    """_extract_weight_tensors raises when the source module has no usable weight attribute."""
    bare = nn.Module()
    fake_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    with pytest.raises(ValueError, match="Cannot extract quantized weight"):
        PreserveBuilder._extract_weight_tensors(bare, fake_config, reorder=False)


def test_preserve_builder_extract_raises_when_no_scale():
    """_extract_weight_tensors raises when neither weight_scale nor weight_scale_inv exist."""
    bare = nn.Module()
    bare.weight = nn.Parameter(torch.randn(4, 4, device=DEVICE), requires_grad=False)
    fake_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    with pytest.raises(ValueError, match="Cannot extract weight scale"):
        PreserveBuilder._extract_weight_tensors(bare, fake_config, reorder=False)


def test_preserve_builder_build_bias_returns_none_when_source_has_no_bias():
    """build_bias returns None when source.bias is None."""
    source = FP8Linear(in_features=8, out_features=8, bias=False).to(DEVICE)
    source.weight_scale_inv = torch.tensor([1.0], device=DEVICE)
    fake_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    with patch.object(_builder_module, "convert_prequantized_module_to_quark_config", return_value=fake_config):
        builder = PreserveBuilder(source, reorder=False)
    assert builder.build_bias(device=torch.device(DEVICE)) is None


def test_preserve_builder_finalize_sets_quant_config():
    """finalize copies the derived quantization config onto the target."""
    source = FP8Linear(in_features=8, out_features=8, bias=False).to(DEVICE)
    source.weight_scale_inv = torch.tensor([1.0], device=DEVICE)
    fake_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    with patch.object(_builder_module, "convert_prequantized_module_to_quark_config", return_value=fake_config):
        builder = PreserveBuilder(source, reorder=False)
    target = Mock(spec=QParamsLinear)
    target._quant_config = None
    builder.finalize(target)
    assert target._quant_config is fake_config


def test_preserve_builder_copy_scale_and_zero_point_noop_without_scale():
    """_copy_scale_and_zero_point silently returns when quantizer has no scale attr."""
    quantizer_without_scale = Mock(spec=[])  # no attributes
    # Just shouldn't raise:
    PreserveBuilder._copy_scale_and_zero_point(
        quantizer_without_scale, weight_scale=torch.tensor([1.0]), weight_zero_point=None
    )


def test_preserve_builder_copy_scale_skips_zero_point_when_quantizer_lacks_attr():
    """_copy_scale_and_zero_point copies scale but skips zero_point when missing."""

    class Q:
        pass

    q = Q()
    q.scale = type("S", (), {"data": torch.zeros(1)})()
    PreserveBuilder._copy_scale_and_zero_point(q, weight_scale=torch.tensor([3.5]), weight_zero_point=torch.tensor([1]))
    assert torch.allclose(q.scale.data, torch.tensor([3.5]))


def test_preserve_builder_build_quantizers_raises_on_sequential():
    """build_quantizers raises if get_real_quantizer returns a SequentialRealQuantizer."""
    source = FP8Linear(in_features=8, out_features=8, bias=False).to(DEVICE)
    source.weight_scale_inv = torch.tensor([1.0], device=DEVICE)
    fake_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    with patch.object(_builder_module, "convert_prequantized_module_to_quark_config", return_value=fake_config):
        builder = PreserveBuilder(source, reorder=False)

    target = Mock(spec=QParamsLinear)
    target.weight_quantizer = None
    target.bias_quantizer = None
    target.input_quantizer = None
    target.output_quantizer = None

    mock_sequential = Mock(spec=SequentialRealQuantizer)
    with (
        patch.object(_builder_module, "get_real_quantizer", return_value=mock_sequential),
        pytest.raises(ValueError, match="SequentialRealQuantizer"),
    ):
        builder.build_quantizers(target, device=torch.device(DEVICE))


def test_preserve_builder_build_quantizers_returns_early_when_weight_config_missing():
    """build_quantizers returns after setting weight when quantization_config.weight is None."""
    source = FP8Linear(in_features=8, out_features=8, bias=False).to(DEVICE)
    source.weight_scale_inv = torch.tensor([1.0], device=DEVICE)
    fake_config_with_weight = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    with patch.object(
        _builder_module, "convert_prequantized_module_to_quark_config", return_value=fake_config_with_weight
    ):
        builder = PreserveBuilder(source, reorder=False)

    # Mutate the cached config to drop the weight spec, simulating the no-weight path.
    builder._quantization_config = QLayerConfig(weight=None)

    target = Mock(spec=QParamsLinear)
    target.weight_quantizer = None
    builder.build_quantizers(target, device=torch.device(DEVICE))
    assert target.weight_quantizer is None


def test_preserve_builder_copy_scale_writes_zero_point_when_present():
    """_copy_scale_and_zero_point copies zero_point when both quantizer attr and value exist."""

    class Q:
        pass

    q = Q()
    q.scale = type("S", (), {"data": torch.zeros(1)})()
    q.zero_point = type("Z", (), {"data": torch.zeros(1, dtype=torch.int32)})()
    PreserveBuilder._copy_scale_and_zero_point(
        q, weight_scale=torch.tensor([2.0]), weight_zero_point=torch.tensor([3], dtype=torch.int32)
    )
    assert torch.allclose(q.scale.data, torch.tensor([2.0]))
    assert torch.equal(q.zero_point.data, torch.tensor([3], dtype=torch.int32))


def test_preserve_builder_build_quantizers_builds_other_quantizers_from_config():
    """build_quantizers happy path proceeds to _build_other_quantizers_from_config (cover 597-603)."""
    source = FP8Linear(in_features=8, out_features=8, bias=False).to(DEVICE)
    source.weight_scale_inv = torch.tensor([1.0], device=DEVICE)
    fake_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC, input_tensors=FP8_PER_TENSOR_SPEC)
    with patch.object(_builder_module, "convert_prequantized_module_to_quark_config", return_value=fake_config):
        builder = PreserveBuilder(source, reorder=False)

    target = Mock(spec=QParamsLinear)
    target.weight_quantizer = None
    target.bias_quantizer = None
    target.input_quantizer = None
    target.output_quantizer = None

    real_q = Mock()
    with patch.object(_builder_module, "get_real_quantizer", return_value=real_q):
        builder.build_quantizers(target, device=torch.device(DEVICE))

    assert target.weight_quantizer is real_q
    assert target.input_quantizer is real_q


# ============================================================================
#  create_builder dispatch
# ============================================================================


def test_create_builder_dispatches_preserve_for_prequantized_module():
    """create_builder returns PreserveBuilder for prequantized modules."""
    source = FP8Linear(in_features=8, out_features=8, bias=False).to(DEVICE)
    source.weight_scale_inv = torch.tensor([1.0], device=DEVICE)
    fake_config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    with patch.object(_builder_module, "convert_prequantized_module_to_quark_config", return_value=fake_config):
        builder = create_builder(source, reorder=False, custom_mode="fp8", quantization_config=None)
    assert isinstance(builder, PreserveBuilder)


def test_create_builder_dispatches_import_for_nn_linear_with_config():
    """create_builder returns ImportBuilder for nn.Linear with quantization_config."""
    source = nn.Linear(in_features=8, out_features=8, bias=False).to(DEVICE)
    config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    builder = create_builder(source, reorder=False, custom_mode="fp8", quantization_config=config)
    assert isinstance(builder, ImportBuilder)


def test_create_builder_raises_for_unsupported_combination():
    """create_builder raises ValueError when source / config combination isn't handled."""
    source = nn.Linear(in_features=8, out_features=8, bias=False).to(DEVICE)
    with pytest.raises(ValueError, match="No builder available"):
        create_builder(source, reorder=False, custom_mode="fp8", quantization_config=None)


# ============================================================================
#  Packed weight extraction (compressed-tensors W4A16)
# ============================================================================

INT4_PER_GROUP_SPEC = QTensorConfig(
    dtype=Dtype.int4,
    qscheme=QSchemeType.per_group,
    observer_cls=PerGroupMinMaxObserver,
    is_dynamic=False,
    group_size=32,
    ch_axis=-1,
    symmetric=True,
    round_method=RoundType.half_even,
    scale_type=ScaleType.float,
)


def test_extract_packed_raises_when_weight_config_missing():
    source = nn.Module()
    config = QLayerConfig(weight=None)
    with pytest.raises(ValueError, match="requires a weight QTensorConfig"):
        PreserveBuilder._extract_packed_weight_tensors(source, config, reorder=False)


def test_extract_packed_raises_on_list_weight_config():
    source = nn.Module()
    config = QLayerConfig(weight=[INT4_PER_GROUP_SPEC, INT4_PER_GROUP_SPEC])
    with pytest.raises(ValueError, match="list-typed weight configs"):
        PreserveBuilder._extract_packed_weight_tensors(source, config, reorder=False)


def test_extract_packed_raises_on_non_int4_per_group():
    source = nn.Module()
    config = QLayerConfig(weight=FP8_PER_TENSOR_SPEC)
    with pytest.raises(ValueError, match="only implemented for int4 per_group"):
        PreserveBuilder._extract_packed_weight_tensors(source, config, reorder=False)


def test_extract_packed_raises_when_weight_scale_missing():
    """Packed extraction needs weight_scale on the source."""
    out_features, in_features = 8, 32
    source = nn.Module()
    source.weight_packed = torch.zeros((out_features, in_features // 8), dtype=torch.int32)
    source.weight_shape = torch.tensor([out_features, in_features])
    config = QLayerConfig(weight=INT4_PER_GROUP_SPEC)
    with pytest.raises(ValueError, match="missing weight_scale"):
        PreserveBuilder._extract_packed_weight_tensors(source, config, reorder=False)


def test_extract_packed_raises_when_weight_shape_missing():
    out_features, in_features = 8, 32
    group_size = 32
    source = nn.Module()
    source.weight_packed = torch.zeros((out_features, in_features // 8), dtype=torch.int32)
    source.weight_scale = torch.ones((out_features, in_features // group_size))
    config = QLayerConfig(weight=INT4_PER_GROUP_SPEC)
    with pytest.raises(ValueError, match="missing weight_shape"):
        PreserveBuilder._extract_packed_weight_tensors(source, config, reorder=False)


def test_extract_packed_int4_round_trip_shapes_and_zero_point_default():
    """Successful packed W4A16 extraction returns correctly-shaped repacked tensors."""
    out_features, in_features = 8, 32
    group_size = 32
    source = nn.Module()
    # int32 storage with packed_dim=1, 8 nibbles per int32.
    source.weight_packed = torch.zeros((out_features, in_features // 8), dtype=torch.int32)
    source.weight_scale = torch.ones((out_features, in_features // group_size))
    source.weight_shape = torch.tensor([out_features, in_features])
    # No weight_zero_point → exercise the default zero_point allocation branch.
    config = QLayerConfig(weight=INT4_PER_GROUP_SPEC)

    repacked_weight, repacked_scale, repacked_zp = PreserveBuilder._extract_packed_weight_tensors(
        source, config, reorder=False
    )
    # Quark stores [in, out/8] int32.
    assert repacked_weight.shape == (in_features, out_features // 8)
    assert repacked_weight.dtype == torch.int32
    # Scale transposed to [in/group, out].
    assert repacked_scale.shape == (in_features // group_size, out_features)
    # Zero point allocated as [in/group, out/8] int32 zeros.
    assert repacked_zp is not None
    assert repacked_zp.shape == (in_features // group_size, out_features // 8)
    assert torch.equal(repacked_zp, torch.zeros_like(repacked_zp))


def test_extract_packed_with_zero_point_provided():
    """Source weight_zero_point present → exercises the packed-zp branch."""
    out_features, in_features = 8, 32
    group_size = 32
    source = nn.Module()
    source.weight_packed = torch.zeros((out_features, in_features // 8), dtype=torch.int32)
    source.weight_scale = torch.ones((out_features, in_features // group_size))
    source.weight_shape = torch.tensor([out_features, in_features])
    # Packed zero_point: [ceil(out/8), in/group].
    source.weight_zero_point = torch.zeros((out_features // 8, in_features // group_size), dtype=torch.int32)
    config = QLayerConfig(weight=INT4_PER_GROUP_SPEC)
    _, _, repacked_zp = PreserveBuilder._extract_packed_weight_tensors(source, config, reorder=False)
    assert repacked_zp is not None
    assert repacked_zp.shape == (in_features // group_size, out_features // 8)


def test_extract_weight_tensors_routes_packed_branch():
    """is_compressed_tensors_module + weight_packed → packed extraction path."""

    class CompressedStatus:
        value = "compressed"

    out_features, in_features = 8, 32
    group_size = 32
    source = nn.Module()
    source.quantization_status = CompressedStatus()
    source.weight_packed = torch.zeros((out_features, in_features // 8), dtype=torch.int32)
    source.weight_scale = torch.ones((out_features, in_features // group_size))
    source.weight_shape = torch.tensor([out_features, in_features])
    config = QLayerConfig(weight=INT4_PER_GROUP_SPEC)
    weight, scale, _ = PreserveBuilder._extract_weight_tensors(source, config, reorder=False)
    assert weight.shape == (in_features, out_features // 8)
    assert scale.shape == (in_features // group_size, out_features)
