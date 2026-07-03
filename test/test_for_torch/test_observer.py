#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


import pytest
import torch

from quark.common.data_type import BaseFP4, BaseFP8_E4M3, BaseFP8_E5M3
from quark.torch.export.nn.modules.realquantizer import DynamicScaledQuantizer
from quark.torch.quantization import Int8PerTensorSpec, OCP_MXFP8E4M3Spec, QTensorConfig, Uint4PerTensorSpec
from quark.torch.quantization.config.config import (
    AmdFP4Spec,
    FP4PerGroupSpec,
)
from quark.torch.quantization.config.template import AmdFP4GlobalScaleScheme, FP4Block16ScaleE4M3Scheme
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import (
    PerBlockBFPObserver,
    PerBlockMXBufferReuseObserver,
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
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, ScaledFakeQuantize


def _create_mx_observer(spec: QTensorConfig) -> PerBlockMXObserver:
    if getattr(spec, "enable_buffer_reuse", False):
        return PerBlockMXBufferReuseObserver(qspec=spec)
    return PerBlockMXObserver(qspec=spec)


def test_calculate_int_quant_params():
    # Test Symmetric
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = Int8PerTensorSpec(is_dynamic=False).to_quantization_spec()
    observer = PerTensorMinMaxObserver(DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
    min_val = torch.Tensor([0.0, 0.0])
    max_val = torch.Tensor([1.0, 1.0])
    scale, zero_point = observer.calculate_int_quant_params(min_val, max_val)
    assert torch.allclose(scale, torch.Tensor([0.00784314, 0.00784314]), atol=1e-6)
    assert torch.equal(zero_point, torch.Tensor([0, 0]))

    # Test Asymmetric
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = Uint4PerTensorSpec(is_dynamic=False).to_quantization_spec()
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
    FP8_PER_TENSOR_SPEC = QTensorConfig(
        dtype=dtype, qscheme=QSchemeType.per_tensor, observer_cls=observer_cls, is_dynamic=False
    )
    observer = observer_cls(FP8_PER_TENSOR_SPEC)
    min_val = torch.Tensor([0.0])
    max_val = torch.Tensor([1.0])
    scale, zero_point = observer.calculate_fp8_quant_parameters(min_val, max_val)
    assert torch.allclose(scale, torch.Tensor([1 / max_norm]), atol=1e-6)
    assert torch.equal(zero_point, torch.Tensor([0]))


def test_PerTensorMinMaxObserver():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QTensorConfig(
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
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QTensorConfig(
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
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QTensorConfig(
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
    DEFAULT_INT8_PER_TENSOR_ASYM_SPEC = QTensorConfig(
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
        spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    elif observer_cls is PerBlockBFPObserver:
        spec = QTensorConfig(
            dtype=Dtype.bfp16,
            observer_cls=PerBlockBFPObserver,
            qscheme=QSchemeType.per_group,
            ch_axis=-1,
            group_size=8,
            is_dynamic=True,
            round_method=RoundType.half_even,
        )
    elif observer_cls is PerChannelMinMaxObserver:
        spec = QTensorConfig(
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
        with pytest.raises(RuntimeError):
            observer.reset_state()

    if observer_cls is PerBlockBFPObserver:
        assert torch.equal(observer.min_val, torch.tensor(float("inf")))
        assert torch.equal(observer.max_val, torch.tensor(float("-inf")))

    if observer_cls is PerChannelMinMaxObserver:
        assert torch.equal(observer.min_val, torch.tensor(float("inf")))
        assert torch.equal(observer.max_val, torch.tensor(float("-inf")))
        del observer.min_val
        del observer.max_val
        with pytest.raises(RuntimeError):
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
            DEFAULT_POF2_INT8_PER_TENSOR_SPEC = QTensorConfig(
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
            POF2_INT8_PER_TENSOR_MSE_SPEC = QTensorConfig(
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
            POF2_INT8_PER_CHANNEL_MSE_SPEC = QTensorConfig(
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
            POF2_INT8_PER_CHANNEL_MSE_SPEC = QTensorConfig(
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
            POF2_INT8_PER_CHANNEL_MSE_SPEC = QTensorConfig(
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


def test_amdfp4_observer():
    # Seed the RNG so this test is deterministic. Without a seed the result
    # depends on whatever ran before in the same pytest invocation, which
    # makes the |ratio - quant_max| tolerance below flake roughly 1 run in 10.
    torch.manual_seed(0)

    amdfp4_spec = AmdFP4Spec(ch_axis=-1, group_size=16, is_dynamic=True).to_quantization_spec()

    observer = _create_mx_observer(amdfp4_spec)

    x_orig = torch.rand(10, 32) * 10.0

    x_after_observer = observer(x_orig)

    assert torch.equal(x_orig, x_after_observer)

    scale, zero_point = observer.calculate_qparams()

    quant_max = BaseFP4.max_value

    ratio = observer.amax / scale

    assert scale.shape == observer.amax.shape
    # AmdFP4Spec uses scale_format="e5m3" + round_method=half_even (see
    # quark/torch/quantization/config/config.py::AmdFP4Spec). fp8_e5m3 has
    # 3 mantissa bits + implicit 1 -> 4 significant bits -> ULP ~= 12.5%
    # between adjacent representable scales. Round-to-nearest-even can
    # therefore shift the stored scale by up to +/- 6.25% relative to the
    # ideal amax/quant_max, which lets the observed `amax/scale` deviate
    # from quant_max=6.0 by as much as 6.0 * (1/16) = 0.375 in the
    # worst case (scale rounded *down*). The previous tolerance of 0.25
    # was tighter than this natural E5M3 rounding spread and produced
    # ~12% flake rate (verified by sweeping 50 RNG seeds). 0.4 covers
    # the worst case with a small margin without masking real bugs.
    assert torch.all((ratio - quant_max).abs() < 0.4)
    assert torch.all(zero_point == 0)


def test_amdfp4_global_nvfp4():
    """Test observer for amdfp4 global scale scheme with FP8 E5M3 scale quantization.

    This test verifies the two-stage quantization for amdfp4 global scale schemes:
    - 1st stage: FP4 per-group quantization producing float32 scales
    - 2nd stage: FP8 E5M3 per-tensor quantization of those float32 scales
    """

    group_size = 16

    # Create two-stage quantization spec (FP4 + FP8 E5M3 scale quantization)
    quant_spec_list_amdfp4_global = AmdFP4GlobalScaleScheme(group_size=group_size).config.input_tensors
    quant_spec_list_nvfp4 = FP4Block16ScaleE4M3Scheme().config.input_tensors

    quantizer_amdfp4_global = FakeQuantizeBase.get_fake_quantize(quant_spec_list_amdfp4_global)
    quantizer_nvfp4 = FakeQuantizeBase.get_fake_quantize(quant_spec_list_nvfp4)

    shape = (1024, 256)

    x_orig = torch.empty(group_size * shape[0] * shape[1]).normal_(mean=0, std=20)
    x_orig = x_orig.reshape(shape[0], shape[1] * group_size)

    # Enable observer to collect statistics
    quantizer_amdfp4_global.enable_observer()
    quantizer_nvfp4.enable_observer()

    # Run the quantizer on the input
    x_quantized_amdfp4_global = quantizer_amdfp4_global(x_orig)
    x_quantized_nvfp4 = quantizer_nvfp4(x_orig)

    quantizer_amdfp4_global[0].observer(x_orig)
    quantizer_nvfp4[0].observer(x_orig)

    scale_f32, _ = quantizer_amdfp4_global[0].observer.calculate_qparams()
    scale_f32_nvfp4, _ = quantizer_nvfp4[0].observer.calculate_qparams()

    assert torch.equal(scale_f32, scale_f32_nvfp4)

    amdfp4_global_scale = quantizer_amdfp4_global[1].scale
    nvfp4_global_scale = quantizer_nvfp4[1].scale

    assert nvfp4_global_scale == x_orig.abs().max() / (BaseFP4.max_value * BaseFP8_E4M3.max_value)
    assert amdfp4_global_scale == x_orig.abs().max() / (BaseFP4.max_value * BaseFP8_E5M3.max_value)

    # Reference per-block FP4 quantization, with FP32 scales with group_size=16.
    quant_spec_fp4_per_group_16 = QTensorConfig(
        dtype=Dtype.fp4,
        observer_cls=PerBlockMXObserver,
        symmetric=None,
        scale_type=ScaleType.float,
        scale_format="float32",
        scale_calculation_mode=None,
        round_method=RoundType.half_even,
        qscheme=QSchemeType.per_group,
        ch_axis=-1,
        is_dynamic=True,
        group_size=group_size,
    )

    quantizer_fp4_ref = FakeQuantizeBase.get_fake_quantize(quant_spec_fp4_per_group_16)
    quantizer_fp4_ref.enable_observer()

    x_quantized_fp4_ref = quantizer_fp4_ref(x_orig)

    loss_fn = torch.nn.L1Loss()

    l1_ref = loss_fn(x_orig, x_quantized_fp4_ref).item()
    l1_amdfp4_global = loss_fn(x_orig, x_quantized_amdfp4_global).item()
    l1_nvfp4 = loss_fn(x_orig, x_quantized_nvfp4).item()

    assert torch.equal(quantizer_amdfp4_global[0].scale, quantizer_nvfp4[0].scale)

    # Feed the raw amax (not scale_f32) to the second-stage quantizer, matching
    # what SequentialQuantize.forward does internally. The second-stage observer
    # has quant_max_first_level = FP4_max set, so it expects amax and computes
    # global_scale = amax / (FP4_max * FP8_max) in a single division.
    amax = quantizer_amdfp4_global[0].observer.amax
    fp4_scale_qdq_e4m3 = quantizer_nvfp4[1](amax.clone())
    fp4_scale_qdq_e5m3 = quantizer_amdfp4_global[1](amax.clone())

    scale_f32_scaled_e5m3 = scale_f32 / quantizer_amdfp4_global[1].scale
    scale_f32_scaled_e4m3 = scale_f32 / quantizer_nvfp4[1].scale

    assert torch.allclose(scale_f32_scaled_e5m3.max(), torch.tensor(BaseFP8_E5M3.max_value))
    assert torch.allclose(scale_f32_scaled_e4m3.max(), torch.tensor(BaseFP8_E4M3.max_value))

    # E5M3 and E4M3 are BOTH wasting their range.
    assert scale_f32_scaled_e5m3.min() >= 500
    assert scale_f32_scaled_e4m3.min() >= 2

    assert scale_f32_scaled_e5m3.min() / scale_f32_scaled_e4m3.min() > 100

    assert not torch.equal(amax, fp4_scale_qdq_e4m3)
    assert not torch.equal(amax, fp4_scale_qdq_e5m3)
    assert torch.equal(fp4_scale_qdq_e4m3, fp4_scale_qdq_e5m3)

    assert l1_amdfp4_global == l1_nvfp4
    assert l1_ref < l1_amdfp4_global
    assert l1_ref < l1_nvfp4

    fp4_scale = quantizer_amdfp4_global[0].scale
    assert fp4_scale is not None
    assert fp4_scale.dtype == torch.float32

    # Verify the FP4 scale
    assert fp4_scale.shape == (shape[0], shape[1])

    # Verify second level scale
    fp8_scale = quantizer_amdfp4_global[1].scale
    assert fp8_scale is not None
    assert fp8_scale.dtype == torch.float32
    assert fp8_scale.numel() == 1


def test_FP4PerGroupSpec_scale_type_default_backward_compatible():
    """Verify that FP4PerGroupSpec default scale_type='float' produces ScaleType.float,
    matching the original hardcoded behavior before the scale_type field was added."""
    spec = FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=True).to_quantization_spec()
    assert spec.scale_type == ScaleType.float, (
        f"Expected default scale_type to be ScaleType.float, but got {spec.scale_type}"
    )


def test_FP4PerGroupSpec_scale_type_float32():
    """Verify that FP4PerGroupSpec(scale_type='float32') produces ScaleType.float32."""
    spec = FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=True, scale_type="float32").to_quantization_spec()
    assert spec.scale_type == ScaleType.float32, (
        f"Expected scale_type to be ScaleType.float32, but got {spec.scale_type}"
    )


def test_PerBlockMXObserver_scale_float32_with_bfloat16_input():
    """Verify that when scale_type='float32', the scale is computed in float32 precision
    even when the model input is BF16.

    This is the core fix for the NVFP4 scale dtype issue: with scale_type='float32',
    the amax is cast to float32 before the division, preventing precision loss.
    """
    spec_float32 = FP4PerGroupSpec(
        ch_axis=-1, group_size=16, is_dynamic=True, scale_type="float32"
    ).to_quantization_spec()
    observer = PerBlockMXObserver(qspec=spec_float32)

    input_bf16 = (torch.rand(10, 32) * 10.0).to(torch.bfloat16)
    observer(input_bf16)
    scale, _ = observer.calculate_qparams()

    assert scale.dtype == torch.float32, (
        f"Expected scale dtype to be float32 when scale_type='float32', but got {scale.dtype}"
    )


def test_PerBlockMXObserver_scale_precision_bf16_vs_fp32():
    """Verify that BF16 and FP32 inputs produce close scale values with scale_type='float32'.

    With scale_type='float32', the scale computation is done in float32 regardless of
    input dtype. The results should be close because the only difference comes from
    BF16 weight truncation, not from the scale computation itself.
    """
    spec = FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=True, scale_type="float32").to_quantization_spec()

    torch.manual_seed(42)
    input_fp32 = torch.rand(10, 32) * 10.0
    input_bf16 = input_fp32.to(torch.bfloat16)

    observer_fp32 = PerBlockMXObserver(qspec=spec)
    observer_bf16 = PerBlockMXObserver(qspec=spec)

    observer_fp32(input_fp32)
    observer_bf16(input_bf16)

    scale_fp32, _ = observer_fp32.calculate_qparams()
    scale_bf16, _ = observer_bf16.calculate_qparams()

    assert scale_fp32.dtype == torch.float32
    assert scale_bf16.dtype == torch.float32
    assert torch.allclose(scale_fp32, scale_bf16, rtol=1e-2, atol=1e-6), (
        f"Scale mismatch between FP32 and BF16 inputs exceeds tolerance. "
        f"Max diff: {(scale_fp32 - scale_bf16).abs().max().item()}"
    )


def test_combined_division_via_quant_max_first_level():
    """Verify that setting quant_max_first_level enables single combined division.

    When SequentialQuantize sets the second stage observer's quant_max_first_level
    to element_format_max (e.g. 6.0), the scale is computed as:
    amax / (6.0 * 448.0) in a single division, matching the reference implementation.
    """
    fp8_spec = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
    )
    observer = PerTensorMinMaxObserver(fp8_spec)

    raw_amax = torch.tensor(3.14159265, dtype=torch.float32)
    element_format_max = 6.0
    scale_format_max = 448.0

    # Set combined divisor (normally done by SequentialQuantize.__init__)
    observer.quant_max_first_level = element_format_max

    # Feed raw amax (normally passed by SequentialQuantize.forward via first_stage.observer.amax)
    observer(raw_amax.unsqueeze(0))

    scale, _ = observer._calculate_qparams()

    # Expected: single combined division
    expected_scale = raw_amax / (element_format_max * scale_format_max)

    assert torch.allclose(scale, expected_scale), (
        f"Expected single-division scale {expected_scale.item()}, but got {scale.item()}"
    )


def test_combined_division_accumulates_across_shared_layers():
    """Verify that combined division works correctly with shared observer (shared scale).

    In shared scale groups, the same observer receives raw amax from multiple layers.
    The observer's min/max accumulation naturally tracks the global max across layers.
    """
    fp8_spec = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
    )
    shared_observer = PerTensorMinMaxObserver(fp8_spec)
    shared_observer.quant_max_first_level = 6.0

    # Layer 1: raw amax = 5.0
    shared_observer(torch.tensor([5.0]))
    # Layer 2: raw amax = 8.0 (largest)
    shared_observer(torch.tensor([8.0]))
    # Layer 3: raw amax = 3.0
    shared_observer(torch.tensor([3.0]))

    scale, _ = shared_observer._calculate_qparams()

    # Expected: max(5, 8, 3) / (6.0 * 448.0) = 8.0 / 2688.0
    expected_scale = torch.tensor(8.0 / (6.0 * 448.0))

    assert torch.allclose(scale, expected_scale, rtol=1e-6), (
        f"Expected accumulated scale {expected_scale.item()}, but got {scale.item()}"
    )


# New tests for memory buffer reuse
def test_PerBlockMXObserver_buffer_reuse():
    """Test buffer reuse with multiple forward passes of same shape."""
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec.enable_buffer_reuse = True
    observer = _create_mx_observer(spec)

    # Reset global stats
    PerBlockMXBufferReuseObserver._gpu_stats.clear()
    PerBlockMXBufferReuseObserver._device_buffers.clear()

    device = torch.device("cpu")
    shape = (2, 8, 16)

    # First forward pass - should allocate
    x1 = torch.randn(*shape, device=device)
    observer(x1)

    # Second forward pass - should reuse buffer
    x2 = torch.randn(*shape, device=device)
    observer(x2)

    # Third forward pass - should reuse buffer
    x3 = torch.randn(*shape, device=device)
    observer(x3)

    # Check stats
    device_str = str(device)
    assert device_str in PerBlockMXBufferReuseObserver._gpu_stats
    assert PerBlockMXBufferReuseObserver._gpu_stats[device_str]["call_count"] >= 3
    # Should have successful reuses
    assert PerBlockMXBufferReuseObserver._gpu_stats[device_str]["reuse_count"] >= 1


def test_PerBlockMXObserver_different_shapes():
    """Test observer handles different input shapes correctly."""
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec.enable_buffer_reuse = True
    observer = _create_mx_observer(spec)

    # Reset global buffers
    PerBlockMXBufferReuseObserver._device_buffers.clear()

    # Test various shapes
    shapes = [
        (2, 8, 16),
        (4, 16, 32),
        (1, 4, 8),
        (8, 32, 64),
    ]

    for shape in shapes:
        x = torch.randn(*shape)
        x_output = observer(x)
        assert torch.equal(x, x_output)
        assert observer.amax is not None


def test_PerBlockMXObserver_buffer_reuse_with_growing_shapes():
    """Test buffer allocation grows when needed."""
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec.enable_buffer_reuse = True
    observer = _create_mx_observer(spec)

    # Reset global buffers
    PerBlockMXBufferReuseObserver._device_buffers.clear()
    PerBlockMXBufferReuseObserver._gpu_stats.clear()

    device = torch.device("cpu")
    device_str = str(device)

    # Small shape first
    x1 = torch.randn(2, 8, 16, device=device)
    observer(x1)

    # Larger shape - should reallocate
    x2 = torch.randn(4, 16, 32, device=device)
    observer(x2)

    # Same large shape - should reuse
    x3 = torch.randn(4, 16, 32, device=device)
    observer(x3)

    # Smaller shape again - should reuse existing large buffer
    x4 = torch.randn(2, 8, 16, device=device)
    observer(x4)

    # Should have some buffer reuses
    assert device_str in PerBlockMXBufferReuseObserver._gpu_stats
    assert PerBlockMXBufferReuseObserver._gpu_stats[device_str]["reuse_count"] >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_PerBlockMXObserver_cuda_buffer_reuse():
    """Test buffer reuse on CUDA device."""
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec.enable_buffer_reuse = True
    observer = _create_mx_observer(spec)

    # Reset global buffers
    PerBlockMXBufferReuseObserver._device_buffers.clear()
    PerBlockMXBufferReuseObserver._gpu_stats.clear()

    device = torch.device("cuda:0")
    device_str = str(device)

    # Multiple forward passes on CUDA
    for _ in range(10):
        x = torch.randn(4, 16, 32, device=device)
        x_output = observer(x)
        assert x_output.device == device
        assert torch.equal(x, x_output)

    # Should have buffer reuses
    assert device_str in PerBlockMXBufferReuseObserver._gpu_stats
    assert PerBlockMXBufferReuseObserver._gpu_stats[device_str]["call_count"] == 10
    assert PerBlockMXBufferReuseObserver._gpu_stats[device_str]["reuse_count"] >= 5


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="Multiple CUDA devices not available"
)
def test_PerBlockMXObserver_multi_gpu_buffers():
    """Test separate buffer pools for different GPUs."""
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec.enable_buffer_reuse = True
    observer = _create_mx_observer(spec)

    # Reset global buffers
    PerBlockMXBufferReuseObserver._device_buffers.clear()
    PerBlockMXBufferReuseObserver._gpu_stats.clear()

    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    # Run on GPU 0
    for _ in range(5):
        x = torch.randn(4, 16, 32, device=device0)
        observer(x)

    # Run on GPU 1
    for _ in range(5):
        x = torch.randn(4, 16, 32, device=device1)
        observer(x)

    # Back to GPU 0 - should reuse existing buffer
    for _ in range(5):
        x = torch.randn(4, 16, 32, device=device0)
        observer(x)

    # Both devices should have their own stats
    assert str(device0) in PerBlockMXBufferReuseObserver._gpu_stats
    assert str(device1) in PerBlockMXBufferReuseObserver._gpu_stats

    # Both should have buffer reuses
    assert PerBlockMXBufferReuseObserver._gpu_stats[str(device0)]["reuse_count"] >= 3
    assert PerBlockMXBufferReuseObserver._gpu_stats[str(device1)]["reuse_count"] >= 3


def test_PerBlockMXObserver_large_tensor_fallback():
    """Test fallback to regular path for very large tensors."""
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec.enable_buffer_reuse = True
    spec.max_input_numel = 1000  # Set very small limit
    observer = _create_mx_observer(spec)

    # Large tensor exceeding limit - should use fallback path
    x_large = torch.randn(100, 200, 300)  # Much larger than limit
    x_output = observer(x_large)

    assert torch.equal(x_large, x_output)
    assert observer.amax is not None


def test_PerBlockMXObserver_buffer_disabled():
    """Test with buffer reuse disabled."""
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec.enable_buffer_reuse = False
    observer = _create_mx_observer(spec)

    # Multiple forward passes
    for _ in range(5):
        x = torch.randn(4, 16, 32)
        x_output = observer(x)
        assert torch.equal(x, x_output)

    # Should work but not track buffer reuse stats


def test_PerBlockMXObserver_different_dtypes():
    """Test observer with different input dtypes."""
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    observer = _create_mx_observer(spec)

    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    for dtype in dtypes:
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            continue  # bfloat16 may not be supported on CPU

        x = torch.randn(4, 16, 32, dtype=dtype)
        x_output = observer(x)
        assert x_output.dtype == dtype
        assert torch.equal(x, x_output)


def test_PerBlockMXObserver_empty_input():
    """Test observer handles empty tensors gracefully."""
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    observer = _create_mx_observer(spec)

    # Empty tensor
    x = torch.randn(0, 16, 32)
    x_output = observer(x)
    assert torch.equal(x, x_output)


def test_PerBlockMXObserver_calculate_qparams():
    """Test quantization parameter calculation."""
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    observer = _create_mx_observer(spec)

    # Forward pass
    x = torch.randn(4, 16, 32)
    observer(x)

    # Calculate qparams
    scale, zero_point = observer.calculate_qparams()

    assert scale is not None
    assert zero_point is not None
    assert scale.numel() > 0


def test_PerBlockMXObserver_buffer_reuse_vs_original():
    """Compare outputs with buffer reuse enabled vs disabled to ensure correctness."""
    # Create two observers - one with buffer reuse, one without
    spec_with_reuse = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec_with_reuse.enable_buffer_reuse = True
    observer_with_reuse = _create_mx_observer(spec_with_reuse)

    spec_without_reuse = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec_without_reuse.enable_buffer_reuse = False
    observer_without_reuse = _create_mx_observer(spec_without_reuse)

    # Reset global buffers for clean test
    PerBlockMXBufferReuseObserver._device_buffers.clear()
    PerBlockMXBufferReuseObserver._gpu_stats.clear()

    # Test with various random inputs
    test_shapes = [
        (2, 8, 16),
        (4, 16, 32),
        (1, 4, 8),
        (8, 32, 64),
        (2, 8, 16),  # Repeat to test buffer reuse path
    ]

    for i, shape in enumerate(test_shapes):
        # Use same random input for both observers
        torch.manual_seed(42 + i)
        x_copy1 = torch.randn(*shape)

        torch.manual_seed(42 + i)
        x_copy2 = torch.randn(*shape)

        # Forward passes
        output_with_reuse = observer_with_reuse(x_copy1)
        amax_with_reuse = observer_with_reuse.amax.clone()

        output_without_reuse = observer_without_reuse(x_copy2)
        amax_without_reuse = observer_without_reuse.amax.clone()

        # Verify inputs are identical
        assert torch.equal(x_copy1, x_copy2)

        # Verify outputs are identical
        assert torch.equal(output_with_reuse, output_without_reuse)

        # Verify amax values are very close
        amax_diff = torch.abs(amax_with_reuse - amax_without_reuse).max().item()
        assert amax_diff < 1e-6, f"amax difference {amax_diff} too large"


def test_PerBlockMXObserver_numerical_stability():
    """Test that buffer reuse maintains numerical stability across multiple passes."""
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec.enable_buffer_reuse = True
    observer = _create_mx_observer(spec)

    # Reset buffers
    PerBlockMXBufferReuseObserver._device_buffers.clear()

    shape = (4, 16, 32)
    num_iterations = 20

    # Collect amax values over many iterations
    amax_values = []

    for i in range(num_iterations):
        # Use consistent input to check for numerical drift
        torch.manual_seed(100)  # Same seed for all iterations
        x = torch.randn(*shape)
        observer(x)
        amax_values.append(observer.amax.clone())

    # All amax values should be identical (same input, same computation)
    for i in range(1, num_iterations):
        amax_diff = torch.abs(amax_values[i] - amax_values[0]).max().item()
        assert amax_diff < 1e-10, f"Iteration {i}: amax drift detected: {amax_diff:.2e}"


def test_PerBlockMXObserver_qparams_equivalence():
    """Test that quantization parameters are identical with/without buffer reuse."""
    # Create two observers
    spec_with_reuse = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec_with_reuse.enable_buffer_reuse = True
    observer_with_reuse = _create_mx_observer(spec_with_reuse)

    spec_without_reuse = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec_without_reuse.enable_buffer_reuse = False
    observer_without_reuse = _create_mx_observer(spec_without_reuse)

    # Same input for both
    torch.manual_seed(123)
    x = torch.randn(4, 16, 32)

    # Forward passes
    observer_with_reuse(x)
    observer_without_reuse(x)

    # Calculate qparams
    scale_with, zp_with = observer_with_reuse.calculate_qparams()
    scale_without, zp_without = observer_without_reuse.calculate_qparams()

    # Compare qparams
    scale_diff = torch.abs(scale_with - scale_without).max().item()
    zp_diff = torch.abs(zp_with - zp_without).max().item()

    assert scale_diff < 1e-6, f"Scale difference too large: {scale_diff}"
    assert zp_diff < 1e-6, f"Zero-point difference too large: {zp_diff}"


def test_PerBlockMXBufferReuseObserver_buffer_layout_and_bytes():
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec.enable_buffer_reuse = True
    observer = _create_mx_observer(spec)
    assert isinstance(observer, PerBlockMXBufferReuseObserver)

    PerBlockMXBufferReuseObserver._device_buffers.clear()
    PerBlockMXBufferReuseObserver._gpu_stats.clear()

    x = torch.randn(2, 8, 16)
    observer(x)

    device_str = str(x.device)
    buffers = PerBlockMXBufferReuseObserver._device_buffers[device_str]
    stats = PerBlockMXBufferReuseObserver._gpu_stats[device_str]
    shape0, shape1, shape2 = buffers["shape"]

    assert "block_x" in buffers and "amax" in buffers
    assert "abs_block_x" not in buffers
    assert "amax_indices" not in buffers

    expected_total_bytes = shape0 * shape1 * shape2 * 4 + shape0 * shape1 * 4
    assert stats["total_bytes"] == expected_total_bytes


def test_PerBlockMXBufferReuseObserver_debug_env_flag(monkeypatch: pytest.MonkeyPatch):
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec.enable_buffer_reuse = True

    monkeypatch.setenv("QUARK_BUFFER_REUSE_DEBUG", "1")
    observer_debug = _create_mx_observer(spec)
    assert isinstance(observer_debug, PerBlockMXBufferReuseObserver)
    assert observer_debug.buffer_reuse_debug

    monkeypatch.setenv("QUARK_BUFFER_REUSE_DEBUG", "0")
    observer_no_debug = _create_mx_observer(spec)
    assert isinstance(observer_no_debug, PerBlockMXBufferReuseObserver)
    assert not observer_no_debug.buffer_reuse_debug


def test_PerBlockMXBufferReuseObserver_allocate_buffers_initializes_stats():
    PerBlockMXBufferReuseObserver._device_buffers.clear()
    PerBlockMXBufferReuseObserver._gpu_stats.clear()

    device = torch.device("cpu")
    block_shape = (2, 4, 16)
    PerBlockMXBufferReuseObserver._allocate_buffers(block_shape, device)

    device_str = str(device)
    assert device_str in PerBlockMXBufferReuseObserver._gpu_stats
    assert PerBlockMXBufferReuseObserver._gpu_stats[device_str]["reuse_count"] == 0
    assert PerBlockMXBufferReuseObserver._gpu_stats[device_str]["call_count"] == 0


def test_PerBlockMXBufferReuseObserver_static_paths_and_dtype_cast():
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
    spec.enable_buffer_reuse = True
    observer = _create_mx_observer(spec)
    assert isinstance(observer, PerBlockMXBufferReuseObserver)

    PerBlockMXBufferReuseObserver._device_buffers.clear()
    PerBlockMXBufferReuseObserver._gpu_stats.clear()

    # First call exercises static else-branch assignment (line with torch.max(...)).
    x0 = torch.randn(2, 8, 16, dtype=torch.float16)
    observer(x0)
    assert observer.amax.dtype == torch.float16

    # Same shape exercises static in-place maximum path.
    x1 = torch.randn(2, 8, 16, dtype=torch.float16)
    observer(x1)

    # Shape change exercises static shape-mismatch reset path.
    x2 = torch.randn(1, 4, 16, dtype=torch.float16)
    observer(x2)


def test_PerBlockMXBufferReuseObserver_debug_forward_calls_print(monkeypatch: pytest.MonkeyPatch):
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec.enable_buffer_reuse = True
    monkeypatch.setenv("QUARK_BUFFER_REUSE_DEBUG", "1")
    observer = _create_mx_observer(spec)
    assert isinstance(observer, PerBlockMXBufferReuseObserver)

    PerBlockMXBufferReuseObserver._device_buffers.clear()
    PerBlockMXBufferReuseObserver._gpu_stats.clear()
    called: list[str] = []

    def _fake_print_stats(cls, device: torch.device) -> None:
        called.append(str(device))

    monkeypatch.setattr(PerBlockMXBufferReuseObserver, "_print_stats", classmethod(_fake_print_stats))

    observer(torch.randn(2, 8, 16))
    assert called == [str(torch.device("cpu"))]


def test_PerBlockMXBufferReuseObserver_print_stats_with_cuda_summary(monkeypatch: pytest.MonkeyPatch):
    device = torch.device("cpu")
    device_str = str(device)
    PerBlockMXBufferReuseObserver._gpu_stats[device_str] = {"reuse_count": 3, "total_bytes": 1024, "call_count": 4}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_summary", lambda device=None, abbreviated=False: "mock-memory-summary")
    PerBlockMXBufferReuseObserver._print_stats(device)


def test_create_observer_uses_buffer_reuse_for_scaled_quantize_and_dynamic_real_quantize():
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1).to_quantization_spec()
    spec.enable_buffer_reuse = True

    scaled_observer = ScaledFakeQuantize.create_observer(spec)
    dynamic_observer = DynamicScaledQuantizer.create_observer(spec)

    assert isinstance(scaled_observer, PerBlockMXBufferReuseObserver)
    assert isinstance(dynamic_observer, PerBlockMXBufferReuseObserver)


def test_dynamic_scale_observer_uses_latest_batch_range():
    """Verify that ``quant_max_first_level`` does not make dynamic scale observers accumulate ranges."""
    fp8_spec = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=True,
        is_scale_quant=True,
    )
    observer = PerTensorMinMaxObserver(fp8_spec)
    observer.quant_max_first_level = 6.0

    observer(torch.tensor([8.0], dtype=torch.float32))
    observer(torch.tensor([2.0], dtype=torch.float32))

    scale, _ = observer._calculate_qparams()
    expected_scale = torch.tensor(2.0 / (6.0 * 448.0))

    assert torch.allclose(observer.min_val, torch.tensor(2.0))
    assert torch.allclose(observer.max_val, torch.tensor(2.0))
    assert torch.allclose(scale, expected_scale)


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

    # New PerBlockMXObserver tests
    test_PerBlockMXObserver_buffer_reuse()
    test_PerBlockMXObserver_different_shapes()
    test_PerBlockMXObserver_buffer_reuse_with_growing_shapes()
    if torch.cuda.is_available():
        test_PerBlockMXObserver_cuda_buffer_reuse()
        if torch.cuda.device_count() >= 2:
            test_PerBlockMXObserver_multi_gpu_buffers()
    test_PerBlockMXObserver_large_tensor_fallback()
    test_PerBlockMXObserver_buffer_disabled()
    test_PerBlockMXObserver_different_dtypes()
    test_PerBlockMXObserver_empty_input()
    test_PerBlockMXObserver_calculate_qparams()
    # Correctness validation tests
    test_PerBlockMXObserver_buffer_reuse_vs_original()
    test_PerBlockMXObserver_numerical_stability()
    test_PerBlockMXObserver_qparams_equivalence()
    torch.cuda.empty_cache()
