#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import quark.torch.kernel  # noqa

import pytest
import torch
from torch import ops
from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
from quark.torch.quantization.config.config import QTensorConfig
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver, PerChannelMinMaxObserver
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, SequentialQuantize
from quark.torch.kernel import quant_fp8_e4m3, dequant_fp8_e4m3, quant_fp8_e5m2, dequant_fp8_e5m2
from quark.shares.utils.testing_utils import torch_device

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


def process_int_per_tensor_quantize(quantization_spec, device, scale, zero_point):
    fake_quantize = FakeQuantizeBase.get_fake_quantize(quantization_spec)
    fake_quantize.observer_enabled = torch.tensor([1], dtype=torch.uint8)
    fake_quantize.scale = scale.to(device)
    fake_quantize.zero_point = zero_point.to(device)
    global input_tensor
    input_tensor = input_tensor.to(device)
    output_tensor = fake_quantize(input_tensor)
    golden_tensor = torch.fake_quantize_per_tensor_affine(  # type: ignore[attr-defined]
        input_tensor, fake_quantize.scale, fake_quantize.zero_point.to(torch.int), -128, 127
    )
    assert torch.equal(output_tensor, golden_tensor)


def test_int_per_tensor_quantize():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QTensorConfig(
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
    process_int_per_tensor_quantize(DEFAULT_INT8_PER_TENSOR_SYM_SPEC, torch_device, scale, zero_point)


def process_int_per_channel_quantize(quantization_spec, device, scale, zero_point):
    fake_quantize = FakeQuantizeBase.get_fake_quantize(quantization_spec)
    fake_quantize.observer_enabled = torch.tensor([1], dtype=torch.uint8)
    fake_quantize.scale = scale.to(device)
    fake_quantize.zero_point = zero_point.to(device)
    global input_tensor
    input_tensor = input_tensor.to(device)
    output_tensor = fake_quantize(input_tensor)
    golden_tensor = torch.fake_quantize_per_channel_affine(  # type: ignore[attr-defined]
        input_tensor, fake_quantize.scale, fake_quantize.zero_point.to(torch.int), 1, -128, 127
    )
    assert torch.equal(output_tensor, golden_tensor)


def test_int_per_channel_quantize():
    DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QTensorConfig(
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
    process_int_per_channel_quantize(DEFAULT_INT8_PER_TENSOR_SYM_SPEC, torch_device, scale, zero_point)


def process_fp8_per_tensor_quantize(quantization_spec, device, scale, zero_point, torch_dtype, max_norm):
    fake_quantize = FakeQuantizeBase.get_fake_quantize(quantization_spec)
    fake_quantize.observer_enabled = torch.tensor([1], dtype=torch.uint8)
    fake_quantize.scale = scale.to(device)
    fake_quantize.zero_point = zero_point.to(device)
    global input_tensor
    input_tensor = input_tensor.to(device)
    output_tensor = fake_quantize(input_tensor)

    input_origin_type = input_tensor.dtype
    golden_tensor = input_tensor / fake_quantize.scale
    golden_tensor = torch.clamp(golden_tensor, min=-max_norm, max=max_norm)
    golden_tensor = golden_tensor.to(torch_dtype).to(input_origin_type) * fake_quantize.scale

    assert torch.equal(output_tensor, golden_tensor)


@pytest.mark.parametrize(
    "dtype,max_norm,torch_dtype",
    [
        (Dtype.fp8_e4m3, 448, torch.float8_e4m3fn),
        (Dtype.fp8_e5m2, 57344, torch.float8_e5m2),
    ],
)
def test_fp8_per_tensor_quantize(dtype, max_norm, torch_dtype):
    DEFAULT_FP8_PER_TENSOR_SYM_SPEC = QTensorConfig(
        dtype=dtype,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    scale = torch.tensor([13.3567])
    zero_point = torch.tensor([0])
    process_fp8_per_tensor_quantize(
        DEFAULT_FP8_PER_TENSOR_SYM_SPEC, torch_device, scale, zero_point, torch_dtype, max_norm
    )


@pytest.mark.parametrize(
    "quant_func,dequant_func,qdq_op,scale",
    [
        (quant_fp8_e4m3, dequant_fp8_e4m3, ops.quark.quant_dequant_fp8_e4m3, None),
        (quant_fp8_e4m3, dequant_fp8_e4m3, ops.quark.quant_dequant_fp8_e4m3, 1.0),
        (quant_fp8_e5m2, dequant_fp8_e5m2, ops.quark.quant_dequant_fp8_e5m2, None),
        (quant_fp8_e5m2, dequant_fp8_e5m2, ops.quark.quant_dequant_fp8_e5m2, 1.0),
    ],
)
def test_fp8_qdq_functions(quant_func, dequant_func, qdq_op, scale):
    w = torch.nn.Parameter(input_tensor, requires_grad=True)
    wq = quant_func(w, scale)
    wr = dequant_func(wq, scale)
    loss = torch.nn.functional.l1_loss(wr, input_tensor.to(torch.float16))
    assert loss < 0.05
    loss.backward()
    assert w.grad is not None
    wd = qdq_op(input_tensor).to(torch.float16)
    assert torch.all(torch.isclose(wd, wr))


def process_fp4_per_group_fp8_per_tensor_scale_quantize(quantization_spec, device, scale1, scale2):
    fake_quantize = SequentialQuantize(quantization_spec, device)
    # fake_quantize.observer_enabled = torch.tensor([1], dtype=torch.uint8)
    fake_quantize.enable_observer()

    fake_quantize[0].scale = scale1.to(device)
    fake_quantize[1].scale = scale2.to(device)
    global input_tensor
    input_tensor = input_tensor.to(device)
    input_dtype = input_tensor.dtype
    output_tensor = fake_quantize(input_tensor)

    fake_quantize[0].scale = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
        "fp8_e4m3",
        fake_quantize[0].scale,
        fake_quantize[1].scale,
        None,
        fake_quantize[1].ch_axis,
        fake_quantize[1].group_size,
        -448,
        448,
        fake_quantize[1].round_method,
        "per_tensor",
        None,
    )
    golden_tensor = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
        "fp4",
        input_tensor,
        fake_quantize[0].scale,
        None,
        fake_quantize[0].ch_axis,
        fake_quantize[0].group_size,
        -6,
        6,
        fake_quantize[0].round_method,
        "per_group",
        None,
    )

    golden_tensor = golden_tensor.to(input_dtype)
    assert torch.equal(output_tensor, golden_tensor)


def test_fp4_per_group_fp8_per_tensor_scale_quantize():
    from quark.torch.quantization import FP4PerGroupSpec, FP8E4M3PerTensorSpec, ScaleQuantSpec

    FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC = ScaleQuantSpec(
        first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=5, is_dynamic=False),
        second_stage=FP8E4M3PerTensorSpec(is_dynamic=False),
    ).to_quantization_spec()

    scale1 = torch.tensor(
        [
            [4.0624, 5.7138],
            [7.5172, 9.7354],
            [6.5853, 2.2768],
            [1.7770, 0.7387],
            [1.3623, 5.6237],
            [3.5421, 1.2345],
            [5.7524, 0.5751],
            [0.9876, 1.0987],
            [1.2345, 3.5421],
            [0.5751, 5.7524],
        ]
    )
    scale2 = torch.tensor([13.3567])
    process_fp4_per_group_fp8_per_tensor_scale_quantize(
        FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC, torch_device, scale1, scale2
    )


def process_fp8_int4_perchannel_quantize(quantization_spec, device, scale1, scale2, zero_point2):
    fake_quantize = SequentialQuantize(quantization_spec, device)
    # fake_quantize.observer_enabled = torch.tensor([1], dtype=torch.uint8)
    fake_quantize.enable_observer()

    fake_quantize[0].scale = scale1.to(device)
    fake_quantize[1].scale = scale2.to(device)
    fake_quantize[1].zero_point = zero_point2.to(device)
    global input_tensor
    input_dtype = input_tensor.dtype
    input_tensor = input_tensor.to(device)
    output_tensor = fake_quantize(input_tensor)

    golden_tensor = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        "fp8_e4m3",
        input_tensor,
        fake_quantize[0].scale,
        None,
        fake_quantize[0].ch_axis,
        fake_quantize[0].group_size,
        -448,
        448,
        fake_quantize[0].round_method,
        "per_tensor",
    )
    golden_tensor = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        "int4",
        golden_tensor,
        fake_quantize[1].scale,
        fake_quantize[1].zero_point.to(torch.int),
        fake_quantize[1].ch_axis,
        fake_quantize[1].group_size,
        -8,
        7,
        fake_quantize[1].round_method,
        "per_channel",
    )

    golden_tensor = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        "int4",
        golden_tensor,
        fake_quantize[1].scale,
        fake_quantize[1].zero_point.to(torch.int),
        fake_quantize[1].ch_axis,
        fake_quantize[1].group_size,
        "per_channel",
    )
    golden_tensor = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        "fp8_e4m3",
        golden_tensor,
        fake_quantize[0].scale,
        None,
        fake_quantize[0].ch_axis,
        fake_quantize[0].group_size,
        "per_tensor",
    )
    golden_tensor = golden_tensor.to(input_dtype)
    assert torch.equal(output_tensor, golden_tensor)


def test_fp8_int4_perchannel_quantize():
    from quark.torch.quantization import ProgressiveSpec

    DEFAULT_FP8_PER_TENSOR_SYM_SPEC = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    DEFAULT_INT4_PER_CHANNEL_SYM_SPEC = QTensorConfig(
        dtype=Dtype.int4,
        qscheme=QSchemeType.per_channel,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        ch_axis=0,
        is_dynamic=False,
    )

    FP4_PER_TENSOR_INT4_PER_CHANNEL_SPEC = ProgressiveSpec(
        first_stage=DEFAULT_FP8_PER_TENSOR_SYM_SPEC, second_stage=DEFAULT_INT4_PER_CHANNEL_SYM_SPEC
    ).to_quantization_spec()

    scale1 = torch.tensor([13.3567])
    scale2 = torch.tensor([4.0624, 5.7138, 7.5172, 9.7354, 6.5853, 2.2768, 1.7770, 0.7387, 1.3623, 5.6237])
    zero_point2 = torch.tensor([0])
    process_fp8_int4_perchannel_quantize(
        FP4_PER_TENSOR_INT4_PER_CHANNEL_SPEC, torch_device, scale1, scale2, zero_point2
    )


def test_int3_per_tensor_quantize():
    DEFAULT_INT3_PER_TENSOR_SPEC = QTensorConfig(
        dtype=Dtype.int3,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )

    fake_quantize = FakeQuantizeBase.get_fake_quantize(DEFAULT_INT3_PER_TENSOR_SPEC)
    fake_quantize.observer_enabled = torch.tensor([1], dtype=torch.uint8)

    scale = torch.tensor([0.25])
    zero_point = torch.tensor([0])
    fake_quantize.scale = scale.to(torch_device)
    fake_quantize.zero_point = zero_point.to(torch_device)

    int3_input = torch.tensor([[-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, -1.0]], dtype=torch.float32)
    int3_input = int3_input.to(torch_device)

    output_tensor = fake_quantize(int3_input)

    golden_tensor = quark.torch.kernel.scaled_fake_quantize(
        "int3",
        int3_input,
        fake_quantize.scale,
        fake_quantize.zero_point,
        None,
        None,
        -4,
        3,
        fake_quantize.round_method,
        "per_tensor",
        None,
    )

    assert torch.equal(output_tensor, golden_tensor)


def test_int3_per_channel_quantize():
    DEFAULT_INT3_PER_CHANNEL_SPEC = QTensorConfig(
        dtype=Dtype.int3,
        qscheme=QSchemeType.per_channel,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        ch_axis=1,
        is_dynamic=False,
    )

    fake_quantize = FakeQuantizeBase.get_fake_quantize(DEFAULT_INT3_PER_CHANNEL_SPEC)
    fake_quantize.observer_enabled = torch.tensor([1], dtype=torch.uint8)

    int3_input = torch.tensor(
        [[-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, -1.0], [-0.8, -0.4, 0.2, 0.3, 0.6, 0.9, 0.8, -0.6]],
        dtype=torch.float32,
    )
    int3_input = int3_input.to(torch_device)

    scale = torch.tensor([4.0624, 5.7138, 7.5172, 9.7354, 6.5853, 2.2768, 1.7770, 0.7387, 1.3623, 5.6237])
    zero_point = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
    fake_quantize.scale = scale.to(torch_device)
    fake_quantize.zero_point = zero_point.to(torch_device)

    output_tensor = fake_quantize(int3_input)

    golden_tensor = quark.torch.kernel.scaled_fake_quantize(
        "int3",
        int3_input,
        fake_quantize.scale,
        fake_quantize.zero_point,
        1,
        None,
        -4,
        3,
        fake_quantize.round_method,
        "per_channel",
        None,
    )

    assert torch.equal(output_tensor, golden_tensor)


def test_int3_per_group_quantize():
    from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver

    DEFAULT_INT3_PER_GROUP_SPEC = QTensorConfig(
        dtype=Dtype.int3,
        qscheme=QSchemeType.per_group,
        observer_cls=PerGroupMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        ch_axis=1,
        is_dynamic=False,
        group_size=4,
    )

    fake_quantize = FakeQuantizeBase.get_fake_quantize(DEFAULT_INT3_PER_GROUP_SPEC)
    fake_quantize.observer_enabled = torch.tensor([1], dtype=torch.uint8)

    scale = torch.tensor([[0.25, 0.3], [0.28, 0.32]], dtype=torch.float32)
    zero_point = torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)
    fake_quantize.scale = scale.to(torch_device)
    fake_quantize.zero_point = zero_point.to(torch_device)

    int3_input = torch.tensor(
        [[-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, -1.0], [-0.8, -0.4, 0.2, 0.3, 0.6, 0.9, 0.8, -0.6]],
        dtype=torch.float32,
    )
    int3_input = int3_input.to(torch_device)

    output_tensor = fake_quantize(int3_input)

    golden_tensor = quark.torch.kernel.scaled_fake_quantize(
        "int3",
        int3_input,
        fake_quantize.scale,
        fake_quantize.zero_point,
        None,
        4,
        -4,
        3,
        fake_quantize.round_method,
        "per_group",
        None,
    )

    assert torch.equal(output_tensor, golden_tensor)


if __name__ == "__main__":
    test_int3_per_tensor_quantize()
