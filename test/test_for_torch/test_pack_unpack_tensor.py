#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import quark
from quark.torch.utils.pack import create_pack_method
import pytest

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
if __name__ == "__main__":
    test_pack_unpack("per_group")
