#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
import quark
from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver, PerChannelMinMaxObserver
from quark.torch.quantization import FP4PerGroupSpec, FP8E4M3PerTensorSpec, OCP_MXFP4Spec
from quark.torch.export.nn.modules.realquantizer import SequentialRealQuantizer, StaticScaledRealQuantizer
from quark.testing import skip_if_no_gpu

@skip_if_no_gpu
def test_fp4_per_group_fp8_per_tensor_scale_real_quantize():
    fp4_real_quantizer = StaticScaledRealQuantizer(
        qspec=FP4PerGroupSpec(group_size=5, is_dynamic=False).to_quantization_spec(),
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=torch.float32,
        device=torch.device("cuda"),
        scale_shape=[10, 2],
        zero_point_shape=None)
    scale1 = torch.tensor([[4.0000, 5.5000],
                           [7.5000, 10.0000],
                           [6.5000, 2.2500],
                           [1.7500, 0.7500],
                           [1.3750, 5.5000],
                           [3.5000, 1.2500],
                           [6.0000, 0.5625],
                           [1.0000, 1.1250],
                           [1.2500, 3.5000],
                           [0.5625, 6.0000]], device='cuda', dtype=torch.float8_e4m3fn)
    fp4_real_quantizer.scale = scale1

    fp8_qspec = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()
    fp8_qspec.is_scale_quant = True
    fp8_real_quantizer = StaticScaledRealQuantizer(
        qspec=fp8_qspec,
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=torch.float32,
        device=torch.device("cuda"),
        scale_shape=[1],
        zero_point_shape=None)
    scale2 = torch.tensor([13.3567], device="cuda")
    fp8_real_quantizer.scale = scale2

    fp4_fp8_quantizer = SequentialRealQuantizer(fp4_real_quantizer, fp8_real_quantizer)
    input_tensor = torch.tensor([[23, 129, 202, 45, 88],
                                 [241, 175, 15, 193, 158],
                                 [92, 37, 240, 121, 50],
                                 [3, 212, 67, 142, 179],
                                 [234, 10, 189, 105, 246],
                                 [81, 152, 220, 53, 166],
                                 [7, 131, 28, 199, 74],
                                 [160, 115, 238, 39, 208],
                                 [96, 181, 62, 147, 224],
                                 [19, 173, 84, 227, 107]], dtype=torch.uint8, device="cuda")
    output_tensor = fp4_fp8_quantizer(input_tensor)

    x = fp4_real_quantizer.unpack_tensor(input_tensor)
    fp4_scale, fp4_zero_point = fp4_real_quantizer.unpack_params()
    fp8_scale, fp8_zero_point = fp8_real_quantizer.unpack_params()

    fp4_scale = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        fp8_real_quantizer.qspec.dtype.value, fp4_scale.to(fp8_real_quantizer.float_dtype), fp8_scale,
        fp8_zero_point, fp8_real_quantizer.qspec.ch_axis, fp8_real_quantizer.qspec.group_size,
        fp8_real_quantizer.qspec.qscheme.value)

    golden_tensor = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        fp4_real_quantizer.qspec.dtype.value, x.to(fp4_real_quantizer.float_dtype), fp4_scale,
        fp4_zero_point, fp4_real_quantizer.qspec.ch_axis, fp4_real_quantizer.qspec.group_size,
        fp4_real_quantizer.qspec.qscheme.value)

    assert torch.equal(output_tensor, golden_tensor)

@skip_if_no_gpu
def test_fp8_int4_perchannel_quantize():
    DEFAULT_FP8_PER_TENSOR_SYM_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                                       qscheme=QSchemeType.per_tensor,
                                                       observer_cls=PerTensorMinMaxObserver,
                                                       symmetric=True,
                                                       scale_type=ScaleType.float,
                                                       round_method=RoundType.half_even,
                                                       is_dynamic=False)
    DEFAULT_INT4_PER_CHANNEL_SYM_SPEC = QuantizationSpec(dtype=Dtype.int4,
                                                         qscheme=QSchemeType.per_channel,
                                                         observer_cls=PerChannelMinMaxObserver,
                                                         symmetric=True,
                                                         scale_type=ScaleType.float,
                                                         round_method=RoundType.half_even,
                                                         ch_axis=0,
                                                         is_dynamic=False)

    fp8_real_quantizer = StaticScaledRealQuantizer(
        qspec=DEFAULT_FP8_PER_TENSOR_SYM_SPEC,
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=torch.float32,
        device=torch.device("cuda"),
        scale_shape=[1],
        zero_point_shape=None
    )
    scale1 = torch.tensor([13.3567], device="cuda")
    fp8_real_quantizer.scale = scale1

    int4_real_quantizer = StaticScaledRealQuantizer(
        qspec=DEFAULT_INT4_PER_CHANNEL_SYM_SPEC,
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=torch.float32,
        device=torch.device("cuda"),
        scale_shape=[10],
        zero_point_shape=[10]
    )
    scale2 = torch.tensor([4.0624, 5.7138, 7.5172, 9.7354, 6.5853, 2.2768, 1.7770, 0.7387, 1.3623, 5.6237], device="cuda")
    int4_real_quantizer.scale = scale2
    zero_point2 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device="cuda")
    int4_real_quantizer.zero_point = zero_point2

    fp8_int4_quantizer = SequentialRealQuantizer(fp8_real_quantizer, int4_real_quantizer)
    input_tensor = torch.tensor([[-2147483648, 2147483647],
                                 [-123456789, 987654321],
                                 [-9999999, 88888888],
                                 [-42, 42],
                                 [20230101, -20230101],
                                 [10000000, -100000000],
                                 [7654321, -876543210],
                                 [-1234567, 123456789],
                                 [33333333, -444444444],
                                 [0, -2147483647]], dtype=torch.int32, device="cuda")
    output_tensor = fp8_int4_quantizer(input_tensor)

    x = int4_real_quantizer.unpack_tensor(input_tensor)
    int4_scale, int4_zero_point = int4_real_quantizer.unpack_params()
    int4_dequant = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        int4_real_quantizer.qspec.dtype.value, x.to(int4_real_quantizer.float_dtype), int4_scale,
        int4_zero_point, int4_real_quantizer.qspec.ch_axis, int4_real_quantizer.qspec.group_size,
        int4_real_quantizer.qspec.qscheme.value)

    fp8_unpack = fp8_real_quantizer.unpack_tensor(int4_dequant)
    fp8_scale, fp8_zero_point = fp8_real_quantizer.unpack_params()
    golden_tensor = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        fp8_real_quantizer.qspec.dtype.value, fp8_unpack.to(fp8_real_quantizer.float_dtype), fp8_scale,
        fp8_zero_point, fp8_real_quantizer.qspec.ch_axis, fp8_real_quantizer.qspec.group_size,
        fp8_real_quantizer.qspec.qscheme.value)

    assert torch.equal(output_tensor, golden_tensor)

@skip_if_no_gpu
def test_e8m0_scale_pack_unpack():
    fp4_e8m0_quantizer = StaticScaledRealQuantizer(
        qspec=OCP_MXFP4Spec(is_dynamic=False, ch_axis=-1).to_quantization_spec(),
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=torch.float32,
        device=torch.device("cuda"),
        scale_shape=[2, 2],
        zero_point_shape=None)

    scale_float = torch.tensor([[1.0, 0.5], [2.0, 0.25]], device="cuda", dtype=torch.float32)
    fp4_e8m0_quantizer.scale = scale_float
    fp4_e8m0_quantizer.maybe_convert_and_transpose_scale()

    golden_scale = torch.tensor([[127, 126], [128, 125]], device="cuda", dtype=torch.uint8)
    assert torch.equal(fp4_e8m0_quantizer.scale, golden_scale)

    scale, _ = fp4_e8m0_quantizer.unpack_params()
    assert torch.equal(scale, scale_float)
