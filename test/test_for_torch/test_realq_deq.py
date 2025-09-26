#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import huggingface_hub
import pytest
import torch

import quark.torch.kernel
from quark.shares.utils.testing_utils import torch_device
from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import (
    PerChannelMinMaxObserver,
    PerGroupMinMaxObserver,
    PerTensorMinMaxObserver,
)
from quark.torch.quantization.utils import calculate_qmin_qmax

input_tensor = torch.tensor(
    [
        [0.6191, 0.7309, 0.3578, 0.8316, 0.5851, 0.2182, 0.9903, 0.2811, 0.0726, 0.3206],
        [0.7251, 0.3405, 0.0916, 0.1509, 0.7608, 0.1274, 0.8781, 0.1523, 0.4676, 0.9383],
        [0.6402, 0.9676, 0.4585, 0.7609, 0.9124, 0.3023, 0.7049, 0.8500, 0.4799, 0.1849],
        [0.8534, 0.9870, 0.0381, 0.0680, 0.2642, 0.8987, 0.6815, 0.8725, 0.9522, 0.0837],
        [0.1448, 0.5640, 0.0564, 0.5648, 0.5244, 0.7106, 0.2336, 0.4107, 0.7222, 0.7721],
        [0.3156, 0.2894, 0.5143, 0.9863, 0.4379, 0.8379, 0.1246, 0.1268, 0.3579, 0.4660],
        [0.5908, 0.8586, 0.8974, 0.5856, 0.5473, 0.2334, 0.1024, 0.9793, 0.8940, 0.2232],
        [0.9779, 0.0856, 0.7413, 0.0988, 0.6105, 0.6858, 0.5405, 0.7468, 0.7344, 0.7093],
        [0.3523, 0.9880, 0.0434, 0.8566, 0.3969, 0.2502, 0.3937, 0.0752, 0.1865, 0.8141],
        [0.0093, 0.2886, 0.2003, 0.2778, 0.7022, 0.0261, 0.7897, 0.2308, 0.6554, 0.3933],
    ]
)


def process_int_per_tensor_qdq(quantization_spec, device, scale, zero_point):
    global input_tensor
    input_tensor = input_tensor.to(device)

    realq_res = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        quantization_spec.dtype.value,
        input_tensor,
        scale.to(device),
        zero_point.to(device),
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        -128,
        127,
        getattr(quantization_spec.round_method, "value", None),
        getattr(quantization_spec.qscheme, "value", None),
    )
    deq_res = quark.torch.kernel.dequantize(
        quantization_spec.dtype.value,
        realq_res,
        scale.to(device),
        zero_point.to(device),
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        getattr(quantization_spec.qscheme, "value", None),
    )
    scale = scale.to(torch.float)
    golden_tensor = torch.fake_quantize_per_tensor_affine(  # type: ignore[attr-defined]
        input_tensor, scale.to(device), zero_point.to(device).to(torch.int), -128, 127
    )

    assert torch.equal(deq_res, golden_tensor)


def test_int_per_tensor_quantize():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=False,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    scale = torch.tensor([13.3567])
    zero_point = torch.tensor([2])
    process_int_per_tensor_qdq(DEFAULT_INT8_PER_TENSOR_SYM_SPEC, torch_device, scale, zero_point)


def test_int_per_tensor_quantize_2():
    tensor_path = huggingface_hub.hf_hub_download("amd-quark/test-qdq", "tensor.pt")
    scale_path = huggingface_hub.hf_hub_download("amd-quark/test-qdq", "tensor_scale.pt")
    zero_point_path = huggingface_hub.hf_hub_download("amd-quark/test-qdq", "tensor_zero_point.pt")

    tensor = torch.load(tensor_path, weights_only=True)
    zero_point = torch.load(zero_point_path, weights_only=True)
    scale = torch.load(scale_path, weights_only=True)

    dtype = "int8"
    ch_axis = None
    group_size = None
    round_method = 8  # half_even
    qscheme = "per_tensor"
    quant_min, quant_max = calculate_qmin_qmax(Dtype.int8)

    tensor_qdq_ref = quark.torch.kernel.scaled_fake_quantize(
        dtype,
        tensor,
        scale,
        zero_point.to(torch.int),
        ch_axis,
        group_size,
        quant_min,
        quant_max,
        round_method,
        qscheme,
        "None",
    )

    tensor_q = quark.torch.kernel.scaled_real_quantize(
        dtype, tensor, scale, zero_point, ch_axis, group_size, quant_min, quant_max, round_method, qscheme
    )
    tensor_qdq = quark.torch.kernel.dequantize(dtype, tensor_q, scale, zero_point, ch_axis, group_size, qscheme).to(
        torch.float32
    )

    assert torch.equal(tensor_qdq_ref, tensor_qdq)


# This is to test dtype conversion of scale and zeropoint
def test_int_per_tensor_quantize_dtype_conversion():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=False,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    scale = torch.tensor([13])
    zero_point = torch.tensor([2.0])
    process_int_per_tensor_qdq(DEFAULT_INT8_PER_TENSOR_SYM_SPEC, torch_device, scale, zero_point)


def process_int_per_channel_qdq(quantization_spec, device, scale, zero_point):
    global input_tensor
    input_tensor = input_tensor.to(device)

    realq_res = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        quantization_spec.dtype.value,
        input_tensor,
        scale.to(device),
        zero_point.to(device),
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        -128,
        127,
        getattr(quantization_spec.round_method, "value", None),
        getattr(quantization_spec.qscheme, "value", None),
    )

    deq_res = quark.torch.kernel.dequantize(
        quantization_spec.dtype.value,
        realq_res,
        scale.to(device),
        zero_point.to(device),
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        getattr(quantization_spec.qscheme, "value", None),
    )

    scale = scale.to(torch.float)
    golden_tensor = torch.fake_quantize_per_channel_affine(  # type: ignore[attr-defined]
        input_tensor, scale.to(device), zero_point.to(device).to(torch.int), 1, -128, 127
    )
    assert torch.equal(deq_res, golden_tensor)


def test_int_per_channel_quantize():
    DEFAULT_INT8_PER_CHANNEL_SYM_SPEC = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_channel,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=False,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        ch_axis=1,
        is_dynamic=False,
    )
    scale = torch.tensor([4.0624, 5.7138, 7.5172, 9.7354, 6.5853, 2.2768, 1.7770, 0.7387, 1.3623, 5.6237])
    zero_point = torch.tensor([2, 3, 2, 1, 4, 1, 0, 2, 6, 1])
    process_int_per_channel_qdq(DEFAULT_INT8_PER_CHANNEL_SYM_SPEC, torch_device, scale, zero_point)


# This is to test dtype conversion of scale and zeropoint
def test_int_per_channel_quantize_dtype_conversion():
    DEFAULT_INT8_PER_CHANNEL_SYM_SPEC = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_channel,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=False,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        ch_axis=1,
        is_dynamic=False,
    )
    scale = torch.tensor([4, 5, 7, 9, 6, 2, 1, 0, 1, 5])
    zero_point = torch.tensor([2.0, 3.0, 2.0, 1.0, 4.0, 1.0, 0.0, 2.0, 6.0, 1.0])
    process_int_per_channel_qdq(DEFAULT_INT8_PER_CHANNEL_SYM_SPEC, torch_device, scale, zero_point)


def process_int_per_group_qdq(quantization_spec, device, scale, zero_point):
    global input_tensor
    input_tensor = input_tensor.to(device)

    realq_res = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        quantization_spec.dtype.value,
        input_tensor,
        scale.to(device),
        zero_point.to(device),
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        -8,
        7,
        getattr(quantization_spec.round_method, "value", None),
        getattr(quantization_spec.qscheme, "value", None),
    )
    deq_res = quark.torch.kernel.dequantize(
        quantization_spec.dtype.value,
        realq_res,
        scale.to(device),
        zero_point.to(device),
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        getattr(quantization_spec.qscheme, "value", None),
    )

    golden_tensor = torch.fake_quantize_per_channel_affine(  # type: ignore[attr-defined]
        input_tensor.view(-1, quantization_spec.group_size),
        scale.to(device).view(-1),
        zero_point.to(device).to(torch.int).view(-1),
        0,
        -8,
        7,
    )

    golden_tensor = golden_tensor.reshape(deq_res.shape)
    assert torch.equal(deq_res, golden_tensor)


def test_int_per_group_quantize():
    DEFAULT_INT8_PER_GROUP_SYM_SPEC = QuantizationSpec(
        dtype=Dtype.int4,
        qscheme=QSchemeType.per_group,
        observer_cls=PerGroupMinMaxObserver,
        symmetric=False,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        ch_axis=1,
        is_dynamic=False,
        group_size=5,
    )

    scale = torch.tensor(
        [
            [0.0033, 0.0038],
            [0.0039, 0.0039],
            [0.0018, 0.0035],
            [0.0033, 0.0039],
            [0.0036, 0.0028],
            [0.0035, 0.0033],
            [0.0039, 0.0031],
            [0.0034, 0.0038],
            [0.0037, 0.0035],
            [0.0037, 0.0032],
        ]
    )
    zero_point = torch.tensor(
        [[-7, -8], [-8, -2], [-8, -8], [-8, -6], [-2, -8], [-8, -3], [-4, -8], [-8, -8], [-4, -8], [-8, -8]],
        dtype=torch.int32,
    )
    process_int_per_group_qdq(DEFAULT_INT8_PER_GROUP_SYM_SPEC, torch_device, scale, zero_point)


def process_fp8_per_tensor_qdq(quantization_spec, device, scale, max_norm):
    global input_tensor
    input_tensor = input_tensor.to(device)

    quark_tensor = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
        quantization_spec.dtype.value,
        input_tensor,
        scale.to(device),
        None,
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        -max_norm,
        max_norm,
        getattr(quantization_spec.round_method, "value", None),
        getattr(quantization_spec.qscheme, "value", None),
        None,
    )

    realq_res = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        quantization_spec.dtype.value,
        input_tensor,
        scale.to(device),
        None,
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        -max_norm,
        max_norm,
        getattr(quantization_spec.round_method, "value", None),
        getattr(quantization_spec.qscheme, "value", None),
    )
    deq_res = quark.torch.kernel.dequantize(
        quantization_spec.dtype.value,
        realq_res,
        scale.to(device),
        None,
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        getattr(quantization_spec.qscheme, "value", None),
    )

    assert torch.equal(deq_res, quark_tensor)


@pytest.mark.parametrize(
    "dtype,max_norm",
    [
        (Dtype.fp8_e4m3, 448),
        (Dtype.fp8_e5m2, 57344),
    ],
)
def test_fp8_per_tensor_quantize(dtype, max_norm):
    DEFAULT_FP8_PER_TENSOR_SYM_SPEC = QuantizationSpec(
        dtype=dtype,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    scale = torch.tensor([13.3567])
    process_fp8_per_tensor_qdq(DEFAULT_FP8_PER_TENSOR_SYM_SPEC, torch_device, scale, max_norm)


def process_fp8_per_channel_qdq(quantization_spec, device, scale, max_norm):
    global input_tensor
    input_tensor = input_tensor.to(device)

    quark_tensor = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
        quantization_spec.dtype.value,
        input_tensor,
        scale.to(device),
        None,
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        -max_norm,
        max_norm,
        getattr(quantization_spec.round_method, "value", None),
        getattr(quantization_spec.qscheme, "value", None),
        None,
    )

    realq_res = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        quantization_spec.dtype.value,
        input_tensor,
        scale.to(device),
        None,
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        -max_norm,
        max_norm,
        getattr(quantization_spec.round_method, "value", None),
        getattr(quantization_spec.qscheme, "value", None),
    )
    deq_res = quark.torch.kernel.dequantize(
        quantization_spec.dtype.value,
        realq_res,
        scale.to(device),
        None,
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        getattr(quantization_spec.qscheme, "value", None),
    )

    assert torch.equal(deq_res, quark_tensor)


@pytest.mark.parametrize(
    "dtype,max_norm",
    [
        (Dtype.fp8_e4m3, 448),
        (Dtype.fp8_e5m2, 57344),
    ],
)
def test_fp8_per_channel_quantize(dtype, max_norm):
    DEFAULT_FP8_PER_CHANNEL_SYM_SPEC = QuantizationSpec(
        dtype=dtype,
        qscheme=QSchemeType.per_channel,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        ch_axis=1,
        is_dynamic=False,
    )
    scale = torch.tensor([4.0624, 5.7138, 7.5172, 9.7354, 6.5853, 2.2768, 1.7770, 0.7387, 1.3623, 5.6237])
    process_fp8_per_channel_qdq(DEFAULT_FP8_PER_CHANNEL_SYM_SPEC, torch_device, scale, max_norm)


# This test is to improve code coverage for nonpositive values of ch_axis
@pytest.mark.parametrize("dtype,max_norm,ch_axis", [(Dtype.fp8_e4m3, 448, i) for i in range(0, -3, -1)])
def test_fp8_per_channel_quantize_axis_range(dtype, max_norm, ch_axis):
    DEFAULT_FP8_PER_CHANNEL_SYM_SPEC = QuantizationSpec(
        dtype=dtype,
        qscheme=QSchemeType.per_channel,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        ch_axis=ch_axis,
        is_dynamic=False,
    )
    scale = torch.tensor([4.0624, 5.7138, 7.5172, 9.7354, 6.5853, 2.2768, 1.7770, 0.7387, 1.3623, 5.6237])
    process_fp8_per_channel_qdq(DEFAULT_FP8_PER_CHANNEL_SYM_SPEC, torch_device, scale, max_norm)


def process_fp16_per_channel_qdq(quantization_spec, device, scale, max_norm):
    global input_tensor
    input_tensor = input_tensor.to(device)

    quark_tensor = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
        quantization_spec.dtype.value,
        input_tensor,
        scale.to(device),
        None,
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        -max_norm,
        max_norm,
        getattr(quantization_spec.round_method, "value", None),
        getattr(quantization_spec.qscheme, "value", None),
        None,
    )

    realq_res = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        quantization_spec.dtype.value,
        input_tensor,
        scale.to(device),
        None,
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        -max_norm,
        max_norm,
        getattr(quantization_spec.round_method, "value", None),
        getattr(quantization_spec.qscheme, "value", None),
    )
    deq_res = quark.torch.kernel.dequantize(
        quantization_spec.dtype.value,
        realq_res,
        scale.to(device),
        None,
        quantization_spec.ch_axis,
        quantization_spec.group_size,
        getattr(quantization_spec.qscheme, "value", None),
    )

    assert torch.equal(deq_res, quark_tensor)


@pytest.mark.parametrize("dtype,max_norm", [(Dtype.float16, 448), (Dtype.bfloat16, 448)])
def test_fp16_per_channel_quantize_axis_range(dtype, max_norm):
    DEFAULT_FP16_PER_CHANNEL_SYM_SPEC = QuantizationSpec(
        dtype=dtype,
        qscheme=QSchemeType.per_channel,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        ch_axis=1,
        is_dynamic=False,
    )
    scale = torch.tensor([4.0624, 5.7138, 7.5172, 9.7354, 6.5853, 2.2768, 1.7770, 0.7387, 1.3623, 5.6237])
    process_fp16_per_channel_qdq(DEFAULT_FP16_PER_CHANNEL_SYM_SPEC, torch_device, scale, max_norm)
