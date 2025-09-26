#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
import torch

from quark.torch.quantization import Int8PerTensorSpec, OCP_MXFP8E4M3Spec, QuantizationSpec, Uint4PerTensorSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import (
    PerBlockBFPObserver,
    PerBlockMXObserver,
    PerChannelMinMaxObserver,
    PerChannelPowOf2MinMaxObserver,
    PerChannelPowOf2MinMSEObserver,
    PerTensorHistogramObserver,
    PerTensorMinMaxObserver,
    PerTensorPercentileObserver,
    PerTensorPowOf2MinMaxObserver,
    PerTensorPowOf2MinMSEObserver,
)


def test_calculate_int_quant_params():
    # Test Symmetric
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = Int8PerTensorSpec(
        observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
    ).to_quantization_spec()
    observer = PerTensorMinMaxObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    min_val = torch.Tensor([0.0, 0.0])
    max_val = torch.Tensor([1.0, 1.0])
    scale, zero_point = observer.calculate_int_quant_params(min_val, max_val)
    assert torch.allclose(scale, torch.Tensor([0.00784314, 0.00784314]), atol=1e-6)
    assert torch.equal(zero_point, torch.Tensor([0, 0]))

    # Test Asymmetric
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = Uint4PerTensorSpec(
        observer_method="min_max", symmetric=False, scale_type="float", round_method="half_even", is_dynamic=False
    ).to_quantization_spec()
    observer = PerTensorMinMaxObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    min_val = torch.Tensor([-1.0, -1.0])
    max_val = torch.Tensor([10.0, 10.0])
    scale, zero_point = observer.calculate_int_quant_params(min_val, max_val)
    assert torch.allclose(scale, torch.Tensor([0.733333333333, 0.733333333333]), atol=1e-6)
    assert torch.equal(zero_point, torch.Tensor([1, 1]))


@pytest.mark.parametrize(
    "dtype,max_norm,observer_cls",
    [
        (Dtype.fp8_e4m3, 448, PerTensorMinMaxObserver),
        (Dtype.fp8_e5m2, 57344, PerTensorMinMaxObserver),
    ],
)
def test_calculate_fp8_quant_parameters(dtype, max_norm, observer_cls):
    FP8_PER_TENSOR_SPEC = QuantizationSpec(
        dtype=dtype, qscheme=QSchemeType.per_tensor, observer_cls=observer_cls, is_dynamic=False
    )
    observer = observer_cls(FP8_PER_TENSOR_SPEC)
    min_val = torch.Tensor([0.0])
    max_val = torch.Tensor([1.0])
    scale, zero_point = observer.calculate_fp8_quant_parameters(min_val, max_val)
    assert torch.allclose(scale, torch.Tensor([1 / max_norm]), atol=1e-6)
    assert torch.equal(zero_point, torch.Tensor([0]))


def test_PerTensorMinMaxObserver():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    observer = PerTensorMinMaxObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    x_orig = torch.Tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    x_after_observer = observer(x_orig)
    assert torch.equal(x_orig, x_after_observer)
    assert torch.allclose(observer.max_val, torch.Tensor([3.0]), atol=1e-6)
    assert torch.allclose(observer.min_val, torch.Tensor([-1.0]), atol=1e-6)


def test_PerChannelMinMaxObserver():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_channel,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
        ch_axis=1,
    )
    observer = PerChannelMinMaxObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    x_orig = torch.Tensor([[-1.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    x_after_observer = observer(x_orig)
    assert torch.equal(x_orig, x_after_observer)
    assert torch.allclose(observer.max_val, torch.Tensor([3.0, 4.0]), atol=1e-6)
    assert torch.allclose(observer.min_val, torch.Tensor([-1.0, 0.0]), atol=1e-6)


def test_PerTensorHistogramObserver():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorHistogramObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    observer = PerTensorHistogramObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    x_orig = torch.Tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    x_after_observer = observer(x_orig)
    assert torch.equal(x_orig, x_after_observer)
    assert torch.allclose(observer.calib_bin_edges.sum(), torch.Tensor([3073.5000]), atol=1e-6)
    assert torch.allclose(observer.calib_hist.sum(), torch.Tensor([5]), atol=1e-6)


def test_PerTensorPercentileObserver():
    DEFAULT_INT8_PER_TENSOR_ASYM_SPEC = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorPercentileObserver,
        symmetric=False,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    observer = PerTensorPercentileObserver(DEFAULT_INT8_PER_TENSOR_ASYM_SPEC)
    x_orig = torch.Tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    x_after_observer = observer(x_orig)
    assert torch.equal(x_orig, x_after_observer)
    x_orig = torch.Tensor([-2.0, 0.0, 1.0, 2.0, 4.0])
    x_after_observer = observer(x_orig)
    assert torch.equal(x_orig, x_after_observer)


@pytest.mark.parametrize(
    "observer_cls",
    [
        (PerBlockMXObserver),
        (PerBlockBFPObserver),
        (PerChannelMinMaxObserver),
    ],
)
def test_reset_state(observer_cls):
    if observer_cls is PerBlockMXObserver:
        spec = OCP_MXFP8E4M3Spec().to_quantization_spec()
    elif observer_cls is PerBlockBFPObserver:
        spec = QuantizationSpec(
            dtype=Dtype.bfp16,
            observer_cls=PerBlockBFPObserver,
            qscheme=QSchemeType.per_group,
            ch_axis=-1,
            group_size=8,
            is_dynamic=True,
            round_method=RoundType.half_even,
        )
    elif observer_cls is PerChannelMinMaxObserver:
        spec = QuantizationSpec(
            dtype=Dtype.fp4,
            observer_cls=PerChannelMinMaxObserver,
            qscheme=QSchemeType.per_channel,
            ch_axis=-1,
            is_dynamic=False,
            round_method=RoundType.half_even,
        )

    observer = observer_cls(qspec=spec)
    observer.reset_state()
    if observer_cls is PerBlockMXObserver:
        assert torch.equal(observer.amax, torch.tensor(0.0))
        del observer.amax
        with pytest.raises(AttributeError):
            observer.reset_state()

    if observer_cls is PerBlockBFPObserver:
        assert torch.equal(observer.min_val, torch.tensor(float("inf")))
        assert torch.equal(observer.max_val, torch.tensor(float("-inf")))

    if observer_cls is PerChannelMinMaxObserver:
        assert torch.equal(observer.min_val, torch.tensor(float("inf")))
        assert torch.equal(observer.max_val, torch.tensor(float("-inf")))
        del observer.min_val
        del observer.max_val
        with pytest.raises(AttributeError):
            observer.reset_state()


def test_PerTensorPowOf2MinMaxObserver():
    dtype = [Dtype.int8, Dtype.uint8]
    symmetric = [True, False]
    count = 0
    scale_list = [
        torch.tensor(1 / (2**5)),
        torch.tensor(1 / (2**5)),
        torch.tensor(1 / (2**6)),
        torch.tensor(1 / (2**6)),
    ]
    zp_list = [torch.tensor(0), torch.tensor(128), torch.tensor(-64), torch.tensor(63)]
    for each_symmetric in symmetric:
        for each_dtype in dtype:
            DEFAULT_POF2_INT8_PER_TENSOR_SPEC = QuantizationSpec(
                dtype=each_dtype,
                qscheme=QSchemeType.per_tensor,
                observer_cls=PerTensorPowOf2MinMaxObserver,
                symmetric=each_symmetric,
                scale_type=ScaleType.float,
                round_method=RoundType.half_even,
                is_dynamic=False,
            )

            observer = PerTensorPowOf2MinMaxObserver(DEFAULT_POF2_INT8_PER_TENSOR_SPEC)
            x_orig = torch.Tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
            x_after_observer = observer(x_orig)
            assert torch.equal(x_orig, x_after_observer)
            assert torch.allclose(observer.max_val, torch.Tensor([3.0]), atol=1e-6)
            assert torch.allclose(observer.min_val, torch.Tensor([-1.0]), atol=1e-6)
            powof2_scale, zp = observer.calculate_int_quant_params(observer.min_val, observer.max_val)
            assert torch.equal(powof2_scale, scale_list[count])
            assert torch.equal(zp, zp_list[count])
            count += 1


def test_PerTensorPowOf2MinMSEObserver():
    dtype = [Dtype.int8, Dtype.uint8]
    symmetric = [True, False]
    count = 0
    scale_ = torch.tensor(1 / (2**3))

    zp_list = [torch.tensor(0), torch.tensor(128), torch.tensor(-42), torch.tensor(85)]
    for each_symmetric in symmetric:
        for each_dtype in dtype:
            POF2_INT8_PER_TENSOR_MSE_SPEC = QuantizationSpec(
                dtype=each_dtype,
                qscheme=QSchemeType.per_tensor,
                observer_cls=PerTensorPowOf2MinMSEObserver,
                symmetric=each_symmetric,
                scale_type=ScaleType.float,
                round_method=RoundType.half_even,
                is_dynamic=False,
            )

            observer = PerTensorPowOf2MinMSEObserver(POF2_INT8_PER_TENSOR_MSE_SPEC)
            x_orig_1 = torch.Tensor([-4, -2.1, 0.1, 2.0, 3.9, 7.9, 8.1])
            x_orig_2 = torch.Tensor([-4.1, -1.9, -0.1, 2.1, 4.1, 7.99, 7.85])
            x_after_observer_1 = observer(x_orig_1)
            x_after_observer_2 = observer(x_orig_2)
            assert torch.equal(x_orig_1, x_after_observer_1) and torch.equal(x_orig_2, x_after_observer_2)
            assert torch.allclose(observer.max_val, torch.Tensor([8.1]), atol=1e-6)
            assert torch.allclose(observer.min_val, torch.Tensor([-4.1]), atol=1e-6)
            powof2_scale, zp = observer.calculate_int_quant_params(observer.min_val, observer.max_val)
            assert torch.equal(powof2_scale, scale_)
            assert torch.equal(zp, zp_list[count])
            count += 1


def test_PerChannelPowOf2MinMaxObserver():
    dtype = [Dtype.int8, Dtype.uint8]
    symmetric = [True, False]
    count = 0
    scale = [
        torch.tensor([1 / (2**5), 1 / (2**6), 1 / (2**10), 1 / (2**6), 1 / (2**5), 1 / (2**4), 1 / (2**4)]),
        torch.tensor([1 / (2**5), 1 / (2**6), 1 / (2**10), 1 / (2**6), 1 / (2**5), 1 / (2**4), 1 / (2**4)]),
        torch.tensor([1 / (2**6), 1 / (2**7), 1 / (2**10), 1 / (2**7), 1 / (2**6), 1 / (2**5), 1 / (2**5)]),
        torch.tensor([1 / (2**6), 1 / (2**7), 1 / (2**10), 1 / (2**7), 1 / (2**6), 1 / (2**5), 1 / (2**5)]),
    ]
    zp_list = [
        torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        torch.tensor([128, 128, 128, 128, 128, 128, 128], dtype=torch.int32),
        torch.tensor([127, 127, 0, -128, -128, -128, -128], dtype=torch.int32),
        torch.tensor([255, 255, 127, 0, 0, 0, 0], dtype=torch.int32),
    ]
    for each_symmetric in symmetric:
        for each_dtype in dtype:
            POF2_INT8_PER_CHANNEL_MSE_SPEC = QuantizationSpec(
                dtype=each_dtype,
                qscheme=QSchemeType.per_channel,
                observer_cls=PerChannelPowOf2MinMaxObserver,
                symmetric=each_symmetric,
                ch_axis=0,
                scale_type=ScaleType.float,
                round_method=RoundType.half_even,
                is_dynamic=False,
            )

            observer = PerChannelPowOf2MinMaxObserver(POF2_INT8_PER_CHANNEL_MSE_SPEC)
            x_orig_1 = torch.Tensor([-4, -2.1, 0.1, 2.0, 3.9, 7.9, 8.1])
            x_orig_2 = torch.Tensor([-4.1, -1.9, -0.1, 2.1, 4.1, 7.99, 7.85])
            x_after_observer_1 = observer(x_orig_1)
            x_after_observer_2 = observer(x_orig_2)
            assert torch.equal(x_orig_1, x_after_observer_1) and torch.equal(x_orig_2, x_after_observer_2)
            assert torch.allclose(observer.max_val, torch.Tensor([-4, -1.9, 0.1, 2.1, 4.1, 7.99, 8.1]))
            assert torch.allclose(observer.min_val, torch.Tensor([-4.1, -2.1, -0.1, 2.0, 3.9, 7.9, 7.85]))
            powof2_scale, zp = observer.calculate_int_quant_params(observer.min_val, observer.max_val)
            assert torch.allclose(powof2_scale, scale[count])
            assert torch.allclose(zp, zp_list[count])
            count += 1


def test_PerChannelPowOf2MinMSEObserver():
    dtype = [Dtype.int8, Dtype.uint8]
    symmetric = [True, False]
    count = 0
    scale = [
        torch.tensor([1 / (2**1), 1 / (2**2), 1 / (2**2), 1 / (2**1)]),
        torch.tensor([1 / (2**1), 1 / (2**2), 1 / (2**2), 1 / (2**1)]),
        torch.tensor([1 / (2**2), 1 / (2**3), 1 / (2**3), 1 / (2**2)]),
        torch.tensor([1 / (2**2), 1 / (2**3), 1 / (2**3), 1 / (2**2)]),
    ]
    zp_list = [
        torch.tensor([0, 0, 0, 0], dtype=torch.int32),
        torch.tensor([128, 128, 128, 128], dtype=torch.int32),
        torch.tensor([126, 126, -128, -128], dtype=torch.int32),
        torch.tensor([254, 254, 0, 0], dtype=torch.int32),
    ]

    # ch_anix = 0
    for each_symmetric in symmetric:
        for each_dtype in dtype:
            POF2_INT8_PER_CHANNEL_MSE_SPEC = QuantizationSpec(
                dtype=each_dtype,
                qscheme=QSchemeType.per_channel,
                observer_cls=PerChannelPowOf2MinMSEObserver,
                symmetric=each_symmetric,
                ch_axis=0,
                scale_type=ScaleType.float,
                round_method=RoundType.half_even,
                is_dynamic=False,
            )

            observer = PerChannelPowOf2MinMSEObserver(POF2_INT8_PER_CHANNEL_MSE_SPEC)
            x_orig_1 = torch.arange(1, 4 * 3 * 3 * 1 + 1).view(4, 1, 3, 3).to(dtype=torch.float) - 18
            x_orig_2 = torch.arange(1, 4 * 3 * 3 * 1 + 1).view(4, 1, 3, 3).to(dtype=torch.float) - 18
            x_after_observer_1 = observer(x_orig_1)
            x_after_observer_2 = observer(x_orig_2)
            assert torch.equal(x_orig_1, x_after_observer_1) and torch.equal(x_orig_2, x_after_observer_2)
            assert torch.allclose(observer.max_val, torch.Tensor([-9.0, 0.0, 9.0, 18.0]))
            assert torch.allclose(observer.min_val, torch.Tensor([-17.0, -8.0, 1.0, 10.0]))
            powof2_scale, zp = observer.calculate_int_quant_params(observer.min_val, observer.max_val)
            assert torch.allclose(powof2_scale, scale[count])
            assert torch.allclose(zp, zp_list[count])
            count += 1

    # ch_anix != 1
    scale = [
        torch.tensor([1 / (2**3), 1 / (2**2), 1 / (2**2)]),
        torch.tensor([1 / (2**3), 1 / (2**2), 1 / (2**2)]),
        torch.tensor([1 / (2**3), 1 / (2**3), 1 / (2**3)]),
        torch.tensor([1 / (2**3), 1 / (2**3), 1 / (2**3)]),
    ]
    zp_list = [
        torch.tensor([0, 0, 0], dtype=torch.int32),
        torch.tensor([128, 128, 128], dtype=torch.int32),
        torch.tensor([-14, -43, -71], dtype=torch.int32),
        torch.tensor([113, 84, 56], dtype=torch.int32),
    ]
    count = 0
    for each_symmetric in symmetric:
        for each_dtype in dtype:
            POF2_INT8_PER_CHANNEL_MSE_SPEC = QuantizationSpec(
                dtype=each_dtype,
                qscheme=QSchemeType.per_channel,
                observer_cls=PerChannelPowOf2MinMSEObserver,
                symmetric=each_symmetric,
                ch_axis=1,
                scale_type=ScaleType.float,
                round_method=RoundType.half_even,
                is_dynamic=False,
            )

            observer = PerChannelPowOf2MinMSEObserver(POF2_INT8_PER_CHANNEL_MSE_SPEC)
            x_orig_1 = torch.arange(1, 4 * 3 + 1).view(4, 3).to(dtype=torch.float) - 5
            x_orig_2 = torch.arange(1, 4 * 3 + 1).view(4, 3).to(dtype=torch.float) - 6
            x_after_observer_1 = observer(x_orig_1)
            powof2_scale_0, zp_0 = observer.calculate_int_quant_params(observer.min_val, observer.max_val)
            x_after_observer_2 = observer(x_orig_2)
            powof2_scale, zp = observer.calculate_int_quant_params(observer.min_val, observer.max_val)
            assert torch.equal(x_orig_1, x_after_observer_1) and torch.equal(x_orig_2, x_after_observer_2)
            assert torch.allclose(observer.max_val, torch.Tensor([5.0, 6.0, 7.0]))
            assert torch.allclose(observer.min_val, torch.Tensor([-5, -4.0, -3.0]))
            assert torch.allclose(powof2_scale, scale[count])
            assert torch.allclose(zp, zp_list[count])
            count += 1


if __name__ == "__main__":
    torch.cuda.empty_cache()
    test_calculate_int_quant_params()
    test_PerTensorMinMaxObserver()
    test_PerChannelMinMaxObserver()
    test_PerTensorHistogramObserver()
    test_PerTensorPercentileObserver()
    test_reset_state()
    test_PerTensorPowOf2MinMaxObserver()
    test_PerTensorPowOf2MinMSEObserver()
    test_PerChannelPowOf2MinMaxObserver()
    test_PerChannelPowOf2MinMSEObserver()
    torch.cuda.empty_cache()
