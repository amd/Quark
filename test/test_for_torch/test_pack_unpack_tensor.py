#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
import torch

import quark.torch.kernel  # noqa
from quark.torch.utils import create_pack_method

torch.manual_seed(42)


@pytest.mark.parametrize("qscheme", ["per_group"])
def test_pack_unpack(qscheme):
    # change num for test
    N = 6
    M = 128 * N

    dtype_list = ["int4", "uint4", "int8", "uint8", "other"]

    int32_int4_tensor1 = torch.randint(-8, 7, (M, N * 3))
    int32_int4_tensor2 = torch.randint(-8, 7, (M,))

    int32_uint4_tensor1 = torch.randint(0, 15, (M, N * 3))
    int32_uint4_tensor2 = torch.randint(0, 15, (M,))

    int32_int8_tensor1 = torch.randint(-(2**7), 2**7 - 1, (M, N * 3))
    int32_int8_tensor2 = torch.randint(-(2**7), 2**7 - 1, (M,))

    int32_uint8_tensor1 = torch.randint(0, 2**8 - 1, (M, N * 3))
    int32_uint8_tensor2 = torch.randint(0, 2**8 - 1, (M,))

    other_tensor1 = torch.randn(M, N * 3)
    other_tensor2 = torch.randn(
        M,
    )

    tensor_list = [
        int32_int4_tensor1,
        int32_int4_tensor2,
        int32_uint4_tensor1,
        int32_uint4_tensor2,
        int32_int8_tensor1,
        int32_int8_tensor2,
        int32_uint8_tensor1,
        int32_uint8_tensor2,
        other_tensor1,
        other_tensor2,
    ]

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
    scale_shape = scale_shape[: ch_axis_plus + 1] + [1] + scale_shape[ch_axis_plus + 1 :]

    scale = torch.ones(scale_shape).to(torch.float32)
    zero_point = torch.zeros(scale_shape).to(torch.int32)

    w_res = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        dtype, param, scale, zero_point, ch_axis, group_size, quant_min, quant_max, round_method, qscheme
    )

    assert w_res.shape == param.shape

    pack_method = create_pack_method(qscheme, dtype)

    packed_tensor = pack_method.pack(w_res, reorder=False)
    assert tuple(packed_tensor.shape) == (*w_res.shape[:-1], w_res.shape[-1] // 2)

    unpacked_tensor = pack_method.unpack(packed_tensor, reorder=False)

    assert torch.equal(w_res, unpacked_tensor)

    w_res_dequant = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        dtype, unpacked_tensor, scale, zero_point, ch_axis, group_size, qscheme
    )

    assert torch.equal(w_res, w_res_dequant)


@pytest.mark.parametrize("mx_element_dtype", ["fp4", "fp6_e2m3", "fp6_e3m2"])
def test_pack_unpack_mxfp(mx_element_dtype):
    qscheme = "per_group"
    dtype = "mx"
    axis = 1
    block_size = 32
    param = torch.rand(256, 256) * 10 - 5
    w_res = quark.torch.kernel.non_scaled_real_quantize(  # type: ignore[attr-defined]
        param, dtype, mx_element_dtype, axis, block_size
    )

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
        dtype, param, scale, zero_point, ch_axis, group_size, quant_min, quant_max, round_method, qscheme
    )
    pack_method = create_pack_method(qscheme, dtype)
    packed_tensor = pack_method.pack(w_res, reorder=False)
    unpacked_tensor = pack_method.unpack(packed_tensor, reorder=False, origin_packed_axis_size=w_res.shape[0])
    assert torch.equal(w_res, unpacked_tensor.to(w_res.dtype))

    test_one_dim_tensor = torch.randint(-8, 7, (257,), dtype=torch.int32)
    test_packed_tensor = pack_method.pack(test_one_dim_tensor, reorder=False)
    test_unpacked_tensor = pack_method.unpack(
        test_packed_tensor, reorder=False, origin_packed_axis_size=test_one_dim_tensor.shape[0]
    )
    assert torch.equal(test_one_dim_tensor, test_unpacked_tensor.to(test_one_dim_tensor.dtype))


def test_pack_fp4_little_endian():
    qscheme = "per_group"
    dtype = "fp4"
    pack_method = create_pack_method(qscheme, dtype)
    tensor = torch.tensor([[0.5, 1.0, 1.5, 2.0], [3.0, 4.0, 6.0, 0.5]], dtype=torch.float32)
    packed_tensor = pack_method.pack(tensor, reorder=False)
    golden_tensor = torch.tensor([[0x21, 0x43], [0x65, 0x17]], dtype=torch.uint8)
    assert torch.equal(packed_tensor, golden_tensor)


def test_unpack_fp4_little_endian():
    qscheme = "per_group"
    dtype = "fp4"
    pack_method = create_pack_method(qscheme, dtype)
    tensor = torch.tensor([[0x21, 0x43], [0x65, 0x17]], dtype=torch.uint8)
    unpacked_tensor = pack_method.unpack(tensor, reorder=False)
    golden_tensor = torch.tensor([[0.5, 1.0, 1.5, 2.0], [3.0, 4.0, 6.0, 0.5]], dtype=torch.float32)
    assert torch.equal(unpacked_tensor, golden_tensor)


# TODO: test e2m3.
@pytest.mark.parametrize("dtype", ["fp6_e3m2"])
def test_pack_fp6(dtype: str):
    qscheme = "per_group"
    pack_method = create_pack_method(qscheme, dtype)
    original_tensor = torch.zeros(1, 32, dtype=torch.float32)

    original_tensor[0][0] = 1.75  # 001111
    original_tensor[0][1] = 10.0  # 011001
    original_tensor[0][2] = -0.125  # 100010
    original_tensor[0][3] = 0.125  # 000010

    packed_tensor = pack_method.pack(original_tensor, reorder=False)

    golden_packed = torch.zeros(1, 24, dtype=torch.uint8)
    golden_packed[0][0] = 0b01001111  # 01.001111
    golden_packed[0][1] = 0b00100110  # 0010.0110
    golden_packed[0][2] = 0b00001010  # 000010.10

    assert torch.equal(packed_tensor, golden_packed)


@pytest.mark.parametrize("dtype", ["fp6_e3m2", "fp6_e2m3"])
def test_pack_unpack_fp6(dtype: str):
    qscheme = "per_group"
    pack_method = create_pack_method(qscheme, dtype)
    original_tensor = torch.zeros(1, 32)

    # values acceptable both for e2m3 and e3m2.
    original_tensor[0][0] = 1.75
    original_tensor[0][1] = 7.0
    original_tensor[0][2] = -0.125
    original_tensor[0][3] = 0.125
    original_tensor[0][4] = -6.0
    original_tensor[0][5] = 2.5
    original_tensor[0][6] = 3.0
    original_tensor[0][7] = 6.0
    original_tensor[0][8] = 6.0

    packed_tensor = pack_method.pack(original_tensor, reorder=False)
    unpacked_tensor = pack_method.unpack(packed_tensor, reorder=False)

    assert torch.equal(unpacked_tensor, original_tensor)


@pytest.mark.parametrize("qscheme", ["per_channel", "per_group"])
def test_pack_unpack_int3(qscheme):
    pack_method = create_pack_method(qscheme, dtype="int3")

    tensor_1d = torch.tensor([-4, -3, -2, -1, 0, 1, 2, 3], dtype=torch.int8)
    packed_1d = pack_method.pack(tensor_1d, reorder=True)
    unpacked_1d = pack_method.unpack(packed_1d, reorder=True)
    assert torch.equal(tensor_1d, unpacked_1d)

    tensor_2d = torch.tensor([[3, -2, 1, 0, -4, 2, -1, 3], [-1, 0, 2, -3, 1, -2, 0, 1]], dtype=torch.int8)
    packed_2d = pack_method.pack(tensor_2d, reorder=True)
    unpacked_2d = pack_method.unpack(packed_2d, reorder=True)
    assert torch.equal(tensor_2d, unpacked_2d)


@pytest.mark.parametrize("qscheme", ["per_channel", "per_group"])
def test_int3_nonmultiple8_should_throw_error(qscheme):
    pack_method = create_pack_method(qscheme, dtype="int3")

    tensor_not_mult8 = torch.tensor([3, -2, 1, 0, -4, 2, -1, 3, 1, 2], dtype=torch.int8)
    try:
        pack_method.pack(tensor_not_mult8)
    except NotImplementedError:
        pass
    else:
        raise NotImplementedError("Not Impl error should be raised for non-multiples of 8.")


@pytest.mark.parametrize("qscheme", ["per_channel", "per_group"])
def test_int3_random_data(qscheme):
    pack_method = create_pack_method(qscheme, dtype="int3")

    tensor_1d = torch.randint(-4, 4, (32,), dtype=torch.int8)
    packed_1d = pack_method.pack(tensor_1d, reorder=True)
    unpacked_1d = pack_method.unpack(packed_1d, reorder=True)
    assert torch.equal(tensor_1d, unpacked_1d)

    large_tensor_2d = torch.randint(-4, 4, (256, 512), dtype=torch.int8)
    packed = pack_method.pack(large_tensor_2d, reorder=True)
    unpacked = pack_method.unpack(packed, reorder=True)
    assert torch.equal(large_tensor_2d, unpacked)


@pytest.mark.parametrize("qscheme", ["per_channel", "per_group"])
def test_int3_pack_size(qscheme):
    pack_method = create_pack_method(qscheme, dtype="int3")

    large_tensor_2d = torch.randint(-4, 4, (256, 512), dtype=torch.int8)
    packed = pack_method.pack(large_tensor_2d, reorder=True)

    original_size = large_tensor_2d.numel()
    packed_size = packed.numel()
    expected_packed_size = (original_size * 3) // 8
    assert packed_size == expected_packed_size, f"Expected {expected_packed_size} bytes, got {packed_size}"


@pytest.mark.parametrize("qscheme", ["per_channel", "per_group"])
def test_int3_infer_tensor_shape(qscheme):
    pack_method = create_pack_method(qscheme, "int3")

    assert pack_method._infer_tensor_shape((8,)) == (3,)
    assert pack_method._infer_tensor_shape((16,)) == (6,)
    assert pack_method._infer_tensor_shape((32,)) == (12,)
    assert pack_method._infer_tensor_shape((4, 8)) == (4, 3)
    assert pack_method._infer_tensor_shape((2, 16)) == (2, 6)
    assert pack_method._infer_tensor_shape((3, 32)) == (3, 12)
