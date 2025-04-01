#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
import torch
from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
from quark.torch.quantization import QuantizationSpec, Int8PerTensorSpec, Uint4PerTensorSpec
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver, PerChannelMinMaxObserver, PerTensorHistogramObserver, PerTensorPercentileObserver


def test_calculate_int_quant_params():
    # Test Symmetric
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = Int8PerTensorSpec(observer_method="min_max",
                                                         symmetric=True,
                                                         scale_type="float",
                                                         round_method="half_even",
                                                         is_dynamic=False).to_quantization_spec()
    observer = PerTensorMinMaxObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    min_val = torch.Tensor([0.0, 0.0])
    max_val = torch.Tensor([1.0, 1.0])
    scale, zero_point = observer.calculate_int_quant_params(min_val, max_val)
    assert torch.allclose(scale, torch.Tensor([0.00784314, 0.00784314]), atol=1e-6)
    assert torch.equal(zero_point, torch.Tensor([0, 0]))

    # Test Asymmetric
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = Uint4PerTensorSpec(observer_method="min_max",
                                                          symmetric=False,
                                                          scale_type="float",
                                                          round_method="half_even",
                                                          is_dynamic=False).to_quantization_spec()
    observer = PerTensorMinMaxObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    min_val = torch.Tensor([-1.0, -1.0])
    max_val = torch.Tensor([10.0, 10.0])
    scale, zero_point = observer.calculate_int_quant_params(min_val, max_val)
    assert torch.allclose(scale, torch.Tensor([0.733333333333, 0.733333333333]), atol=1e-6)
    assert torch.equal(zero_point, torch.Tensor([1, 1]))


@pytest.mark.parametrize("dtype,max_norm,observer_cls", [
    (Dtype.fp8_e4m3, 448, PerTensorMinMaxObserver),
    (Dtype.fp8_e5m2, 57344, PerTensorMinMaxObserver),
])
def test_calculate_fp8_quant_parameters(dtype, max_norm, observer_cls):
    FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=dtype,
                                           qscheme=QSchemeType.per_tensor,
                                           observer_cls=observer_cls,
                                           is_dynamic=False)
    observer = observer_cls(FP8_PER_TENSOR_SPEC)
    min_val = torch.Tensor([0.0])
    max_val = torch.Tensor([1.0])
    scale, zero_point = observer.calculate_fp8_quant_parameters(min_val, max_val)
    assert torch.allclose(scale, torch.Tensor([1 / max_norm]), atol=1e-6)
    assert torch.equal(zero_point, torch.Tensor([0]))


def test_PerTensorMinMaxObserver():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                                        qscheme=QSchemeType.per_tensor,
                                                        observer_cls=PerTensorMinMaxObserver,
                                                        symmetric=True,
                                                        scale_type=ScaleType.float,
                                                        round_method=RoundType.half_even,
                                                        is_dynamic=False)
    observer = PerTensorMinMaxObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    x_orig = torch.Tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    x_after_observer = observer(x_orig)
    assert torch.equal(x_orig, x_after_observer)
    assert torch.allclose(observer.max_val, torch.Tensor([3.0]), atol=1e-6)
    assert torch.allclose(observer.min_val, torch.Tensor([-1.0]), atol=1e-6)


def test_PerChannelMinMaxObserver():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                                        qscheme=QSchemeType.per_tensor,
                                                        observer_cls=PerChannelMinMaxObserver,
                                                        symmetric=True,
                                                        scale_type=ScaleType.float,
                                                        round_method=RoundType.half_even,
                                                        is_dynamic=False,
                                                        ch_axis=1)
    observer = PerChannelMinMaxObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    x_orig = torch.Tensor([[-1.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    x_after_observer = observer(x_orig)
    assert torch.equal(x_orig, x_after_observer)
    assert torch.allclose(observer.max_val, torch.Tensor([3.0, 4.0]), atol=1e-6)
    assert torch.allclose(observer.min_val, torch.Tensor([-1.0, 0.0]), atol=1e-6)


def test_PerTensorHistogramObserver():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                                        qscheme=QSchemeType.per_tensor,
                                                        observer_cls=PerTensorHistogramObserver,
                                                        symmetric=True,
                                                        scale_type=ScaleType.float,
                                                        round_method=RoundType.half_even,
                                                        is_dynamic=False)
    observer = PerTensorHistogramObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    x_orig = torch.Tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    x_after_observer = observer(x_orig)
    assert torch.equal(x_orig, x_after_observer)
    assert torch.allclose(observer.calib_bin_edges.sum(), torch.Tensor([3073.5000]), atol=1e-6)
    assert torch.allclose(observer.calib_hist.sum(), torch.Tensor([5]), atol=1e-6)

def test_PerTensorPercentileObserver():
    DEFAULT_INT8_PER_TENSOR_ASYM_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                                         qscheme=QSchemeType.per_tensor,
                                                         observer_cls=PerTensorPercentileObserver,
                                                         symmetric=False,
                                                         scale_type=ScaleType.float,
                                                         round_method=RoundType.half_even,
                                                         is_dynamic=False)
    observer = PerTensorPercentileObserver(DEFAULT_INT8_PER_TENSOR_ASYM_SPEC)
    x_orig = torch.Tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    x_after_observer = observer(x_orig)
    assert torch.equal(x_orig, x_after_observer)
    x_orig = torch.Tensor([-2.0, 0.0, 1.0, 2.0, 4.0])
    x_after_observer = observer(x_orig)
    assert torch.equal(x_orig, x_after_observer)
