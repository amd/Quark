#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import quark.torch.kernel  # noqa
from quark.torch.utils.pack import create_pack_method
import pytest

torch.manual_seed(42)

@pytest.mark.parametrize("qscheme", [
    "per_group"
])
def test_pack_unpack(qscheme):
    # change num for test
    N = 6
    M = 128 * N

    dtype_list = ["int4", "uint4", "int8", "uint8", "other"]

    int32_int4_tensor1 = torch.randint(-8, 7, (M, N * 3))
    int32_int4_tensor2 = torch.randint(-8, 7, (M,))

    int32_uint4_tensor1 = torch.randint(0, 15, (M, N * 3))
    int32_uint4_tensor2 = torch.randint(0, 15, (M,))

    int32_int8_tensor1 = torch.randint(-2 ** 7, 2 ** 7 - 1, (M, N * 3))
    int32_int8_tensor2 = torch.randint(-2 ** 7, 2 ** 7 - 1, (M,))

    int32_uint8_tensor1 = torch.randint(0, 2 ** 8 - 1, (M, N * 3))
    int32_uint8_tensor2 = torch.randint(0, 2 ** 8 - 1, (M,))

    other_tensor1 = torch.randn(M, N * 3)
    other_tensor2 = torch.randn(M,)


    tensor_list = [int32_int4_tensor1, int32_int4_tensor2, int32_uint4_tensor1, int32_uint4_tensor2, int32_int8_tensor1, int32_int8_tensor2, int32_uint8_tensor1, int32_uint8_tensor2, other_tensor1, other_tensor2]

    for i in range(len(tensor_list)):
        pack_method = create_pack_method(qscheme, dtype_list[i // 2])
        for reorder_or_not in [True, False]:
            packed_tensor = pack_method.pack(to_pack=tensor_list[i], reorder=reorder_or_not)
            unpacked_tensor = pack_method.unpack(packed_tensor, reorder=reorder_or_not)
            assert torch.equal(tensor_list[i], unpacked_tensor)

    bad_tensor = torch.randint(-8, 7, (128, 128 * 3, 128))
    pack_method = create_pack_method(qscheme, "uint4")

    try:
        packed_tensor = pack_method.pack(to_pack=bad_tensor, reorder=True)
    except ValueError as e:
        assert str(e) == "Pack: Only supports tensors with dimensions not greater than 2."
    else:
        raise ValueError("ValueError of pack is not raised")

    try:
        unpacked_tensor = pack_method.unpack(bad_tensor, reorder=False)
    except ValueError as e:
        assert str(e) == "Unpack: Only supports tensors with dimensions not greater than 2."
    else:
        raise ValueError("ValueError of Unpack is not raised")

@pytest.mark.parametrize("ndim", [pytest.param(ndim, id=f"ndim={ndim}") for ndim in [2, 3]])
@pytest.mark.parametrize("ch_axis", [pytest.param(ch_axis, id=f"ch_axis={ch_axis}") for ch_axis in [-1, 0, 1, -2]])
def test_pack_unpack_fp4(ndim: int, ch_axis: int):
    qscheme = "per_group"
    dtype = "fp4"
    group_size = 32
    round_method = 8
    quant_min = -6
    quant_max = 6

    shape = [256 // 2**i for i in range(ndim)]
    param = torch.rand(shape) * 10 - 5

    ch_axis_plus = ch_axis
    if ch_axis < 0:
        ch_axis_plus = ndim + ch_axis

    scale_shape = [256 // 2**i for i in range(ndim)]
    scale_shape[ch_axis_plus] = 256 // 2**ch_axis_plus // group_size
    scale_shape = scale_shape[:ch_axis_plus + 1] + [1] + scale_shape[ch_axis_plus + 1:]

    scale = torch.ones(scale_shape).to(torch.float32)
    zero_point = torch.zeros(scale_shape).to(torch.int32)

    w_res = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        dtype, param, scale, zero_point, ch_axis, group_size, quant_min, quant_max,
        round_method, qscheme)

    assert w_res.shape == param.shape

    pack_method = create_pack_method(qscheme, dtype)

    packed_tensor = pack_method.pack(w_res, reorder=False)
    assert tuple(packed_tensor.shape) == (*w_res.shape[:-1], w_res.shape[-1] // 2)

    unpacked_tensor = pack_method.unpack(packed_tensor, reorder=False)

    assert torch.equal(w_res, unpacked_tensor)

    w_res_dequant = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        dtype,
        unpacked_tensor,
        scale,
        zero_point,
        ch_axis,
        group_size,
        qscheme
    )

    assert torch.equal(w_res, w_res_dequant)

@pytest.mark.parametrize("mx_element_dtype", [
    "fp4", "fp6_e2m3", "fp6_e3m2"
])
def test_pack_unpack_mxfp(mx_element_dtype):
    qscheme = "per_group"
    dtype = "mx"
    axis = 1
    block_size = 32
    param = torch.rand(256, 256) * 10 - 5
    w_res = quark.torch.kernel.non_scaled_real_quantize(  # type: ignore[attr-defined]
        param, dtype, mx_element_dtype, axis, block_size)

    pack_method = create_pack_method(qscheme, dtype, mx_element_dtype)
    packed_tensor = pack_method.pack(w_res, reorder=False)
    unpacked_tensor = pack_method.unpack(packed_tensor, reorder=False)
    assert torch.equal(w_res, unpacked_tensor)

def test_pack_int4_tensor_of_non_integer_size():
    qscheme = "per_group"
    dtype = "int4"
    ch_axis = 1
    group_size = 32
    round_method = 8
    quant_min = -6
    quant_max = 6

    param = torch.rand(257, 256) * 10 - 5
    scale = torch.ones(257, 8).to(torch.float32)
    zero_point = torch.zeros(257, 8).to(torch.int32)

    w_res = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        dtype, param, scale, zero_point, ch_axis, group_size, quant_min, quant_max,
        round_method, qscheme)
    pack_method = create_pack_method(qscheme, dtype)
    packed_tensor = pack_method.pack(w_res, reorder=False)
    unpacked_tensor = pack_method.unpack(packed_tensor, reorder=False, origin_packed_axis_size=w_res.shape[0])
    assert torch.equal(w_res, unpacked_tensor.to(w_res.dtype))

    test_one_dim_tensor = torch.randint(-8, 7, (257,), dtype=torch.int32)
    test_packed_tensor = pack_method.pack(test_one_dim_tensor, reorder=False)
    test_unpacked_tensor = pack_method.unpack(test_packed_tensor, reorder=False, origin_packed_axis_size=test_one_dim_tensor.shape[0])
    assert torch.equal(test_one_dim_tensor, test_unpacked_tensor.to(test_one_dim_tensor.dtype))


def test_pack_fp4_little_endian():
    qscheme = "per_group"
    dtype = "fp4"
    pack_method = create_pack_method(qscheme, dtype)
    tensor = torch.tensor([[0.5, 1.0, 1.5, 2.0],
                           [3.0, 4.0, 6.0, 0.5]], dtype=torch.float32)
    packed_tensor = pack_method.pack(tensor, reorder=False)
    golden_tensor = torch.tensor([[0x21, 0x43], [0x65, 0x17]], dtype=torch.uint8)
    assert torch.equal(packed_tensor, golden_tensor)

def test_unpack_fp4_little_endian():
    qscheme = "per_group"
    dtype = "fp4"
    pack_method = create_pack_method(qscheme, dtype)
    tensor = torch.tensor([[0x21, 0x43], [0x65, 0x17]], dtype=torch.uint8)
    unpacked_tensor = pack_method.unpack(tensor, reorder=False)
    golden_tensor = torch.tensor([[0.5, 1.0, 1.5, 2.0],
                                  [3.0, 4.0, 6.0, 0.5]], dtype=torch.float32)
    assert torch.equal(unpacked_tensor, golden_tensor)
