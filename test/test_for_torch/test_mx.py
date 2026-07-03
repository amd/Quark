#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import math
from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_testing_utils import run_torch_op_variants  # type: ignore[import-not-found]

from quark.common.utils.testing_utils import require_torch_cuda
from quark.torch import ModelQuantizer
from quark.torch.export.nn.modules import realquantizer
from quark.torch.kernel.hw_emulation.hw_emulation_interface import fake_quantize_mx
from quark.torch.quantization import (
    MX6Spec,
    MX9Spec,
    OCP_MXFP4Spec,
    OCP_MXFP6E2M3Spec,
    OCP_MXFP6E3M2Spec,
    OCP_MXFP8E4M3Spec,
    OCP_MXFP8E5M2Spec,
    OCP_MXINT8Spec,
    QConfig,
    QLayerConfig,
    QTensorConfig,
)
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerBlockMXObserver
from quark.torch.quantization.utils import get_dtype_params, reshape_to_blocks


class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc1 = nn.Linear(8 * 8, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


input_tensor = torch.ones(1, 3, 32, 32)


class SimpleDataset(Dataset):
    def __init__(self):
        return

    def __len__(self):
        return 2

    def __getitem__(self, _):
        return input_tensor.squeeze(0)


def create_quantize_run_simple_network(config: QConfig):
    model = SimpleNetwork()
    model(input_tensor)
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    quantizer = ModelQuantizer(config)
    quantized_model = quantizer.quantize_model(model, dataloader)
    quantized_model(input_tensor)


valid_configs = [
    ("mx", False, OCP_MXFP8E4M3Spec, None, True),
    ("mx", False, OCP_MXFP8E4M3Spec, None, False),
    ("mx", False, OCP_MXFP8E5M2Spec, None, True),
    ("mx", False, OCP_MXFP8E5M2Spec, None, False),
    ("mx", False, OCP_MXFP6E2M3Spec, None, True),
    ("mx", False, OCP_MXFP6E2M3Spec, None, False),
    ("mx", False, OCP_MXFP6E3M2Spec, None, True),
    ("mx", False, OCP_MXFP6E3M2Spec, None, False),
    ("mx", False, OCP_MXFP4Spec, None, True),
    ("mx", False, OCP_MXFP4Spec, None, False),
    ("mx", False, OCP_MXINT8Spec, None, True),
    ("mx", False, OCP_MXINT8Spec, None, False),
    ("mx6", False, MX6Spec, 16, True),
    ("mx6", False, MX6Spec, 16, False),
    ("mx9", False, MX9Spec, 16, True),
    ("mx9", False, MX9Spec, 16, False),
]


@pytest.mark.parametrize("dtype, is_dynamic, spec_class, group_size, weight_only", valid_configs)
def test_mx_valid_config_verification(dtype, is_dynamic, spec_class, group_size, weight_only):
    if dtype == "mx":
        partial_spec = partial(spec_class, is_dynamic=is_dynamic)

        linear_input_spec = spec_class(ch_axis=-1, is_dynamic=True).to_quantization_spec()
        conv_input_spec = spec_class(ch_axis=1, is_dynamic=True).to_quantization_spec()
    elif dtype == "mx6" or dtype == "mx9":
        partial_spec = partial(spec_class, block_size=group_size)

        linear_input_spec = spec_class(ch_axis=-1, block_size=group_size).to_quantization_spec()
        conv_input_spec = spec_class(ch_axis=1, block_size=group_size).to_quantization_spec()
    else:
        assert 0

    if weight_only:
        linear_config = QLayerConfig(weight=partial_spec(ch_axis=-1).to_quantization_spec())
        conv_config = QLayerConfig(weight=partial_spec(ch_axis=1).to_quantization_spec())
    else:
        linear_config = QLayerConfig(
            input_tensors=linear_input_spec,
            weight=partial_spec(ch_axis=-1).to_quantization_spec(),
        )

        conv_config = QLayerConfig(
            input_tensors=conv_input_spec,
            weight=partial_spec(ch_axis=1).to_quantization_spec(),
        )
    config = QConfig(
        global_quant_config=QLayerConfig(),
        layer_type_quant_config={nn.Linear: linear_config, nn.Conv2d: conv_config},
    )
    create_quantize_run_simple_network(config)


reshape_args = [
    (
        torch.Size([10, 10]),
        32,
        0,
        torch.Size([10, 1, 32]),
    ),  # the output shape is the same as the output is reshaped to have the blocked dimension last
    (torch.Size([10, 10]), 32, 1, torch.Size([10, 1, 32])),
    (torch.Size([10, 10]), 32, 2, None),  # axis is greater than number of axes in tensor
    (torch.Size([10, 10]), 5, 0, torch.Size([10, 2, 5])),  # block size smaller than axis so needs to be tiled - however
    (torch.Size([10, 10]), 5, 1, torch.Size([10, 2, 5])),
]


@pytest.mark.parametrize("tensor_shape, block_size, axis, expected_shape", reshape_args)
def test_mx_reshape_to_blocks(tensor_shape, block_size, axis, expected_shape):
    a = torch.ones(tensor_shape)

    if expected_shape is None:
        with pytest.raises(IndexError):
            reshaped = reshape_to_blocks(a, block_size, axis)
    else:
        reshaped = reshape_to_blocks(a, block_size, axis)
        assert reshaped.shape == expected_shape


def test_mx_reshape_to_blocks_axis0():
    a = torch.ones(10, 10)
    for row_idx in range(10):
        a[row_idx] = a[row_idx] * row_idx

    block_size = 10
    block_a = reshape_to_blocks(a, block_size, 0)

    assert block_a.shape == torch.Size([10, 1, block_size])

    for row_idx in range(10):
        for cell_idx in range(10):
            assert block_a[row_idx, 0, cell_idx] == cell_idx


def test_mx_reshape_to_blocks_axis1():
    a = torch.ones(10, 10)
    for row_idx in range(10):
        a[row_idx] = a[row_idx] * row_idx

    block_size = 10
    block_a = reshape_to_blocks(a, block_size, 1)

    assert block_a.shape == torch.Size([10, 1, block_size])

    for row_idx in range(10):
        for cell_idx in range(10):
            assert block_a[row_idx, 0, cell_idx] == row_idx


def test_mx_reshape_to_blocks_more_detail():
    a = torch.zeros(2, 10)
    for i in range(10):
        a[0, i] = -5 + i
        a[1, i] = 5 - i

    # 'a' should look like this
    # [
    #    [ -5, -4, -3, -2, -1, 0, 1, 2, 3, 4],
    #    [ 5, 4, 3, 2, 1, 0 , -1, -2, -3, -4]
    # ]
    block_size = 5
    reshaped_a = reshape_to_blocks(a, block_size, 1)
    # 'reshaped_a' should look like
    # [
    #    [[ -5, -4, -3, -2, -1], [0, 1, 2, 3, 4]],
    #    [[ 5, 4, 3, 2, 1], [0, -1, -2, -3, -4]]
    # ]
    assert reshaped_a.dim() == 3
    assert reshaped_a.shape[0] == 2
    assert reshaped_a.shape[1] == 2, "Incorrect number of block tiles"
    assert reshaped_a.shape[2] == block_size

    # first block
    for idx, val in enumerate([-5, -4, -3, -2, -1]):
        assert reshaped_a[0, 0, idx] == val

    # second block
    for idx, val in enumerate([0, 1, 2, 3, 4]):
        assert reshaped_a[0, 1, idx] == val

    # third block
    for idx, val in enumerate([5, 4, 3, 2, 1]):
        assert reshaped_a[1, 0, idx] == val

    # fourth block
    for idx, val in enumerate([0, -1, -2, -3, -4]):
        assert reshaped_a[1, 1, idx] == val


def create_4d_tensor_with_interesting_pattern():
    # let's create a tensor with a nice pattern we can inspect
    # tensor([[[[ 10.,  20.,  30.,  40.],
    #           [ 20.,  40.,  60.,  80.],
    #           [ 30.,  60.,  90., 120.]],
    #           [[110., 120., 130., 140.],
    #           [120., 140., 160., 180.],
    #           [130., 160., 190., 220.]]]])
    result = torch.zeros(1, 2, 3, 4)
    for x0 in range(result.shape[0]):
        for x1 in range(result.shape[1]):
            for x2 in range(result.shape[2]):
                for x3 in range(result.shape[3]):
                    result[x0, x1, x2, x3] = (x3 + 1.0) * (10 * (x2 + 1.0)) + 100.0 * x1
    return result


def test_per_block_simple_scale():
    spec = OCP_MXFP8E4M3Spec(ch_axis=-1, scale_calculation_mode="floor").to_quantization_spec()
    observer = PerBlockMXObserver(qspec=spec)

    a = torch.zeros(10, 10)
    for i in range(10):
        a[i, i] = -5 + i

    observer(a)

    _, _, emax = get_dtype_params(Dtype.fp8_e4m3)

    for i in range(10):
        amax = abs(-5 + i)
        if amax != 0:
            scale_val = math.pow(2.0, math.floor(math.log2(amax)) - emax)
        else:
            scale_val = observer.eps
        # these values should be directly representable by floating point so direct comparison is valid here
        scale, _ = observer.calculate_qparams()
        assert scale[i, 0] == scale_val


def test_per_block_scale_tiled():
    a = torch.zeros(2, 10)
    for i in range(10):
        a[0, i] = -5 + i
        a[1, i] = 5 - i

    # a should look like this
    # [
    #    [ -5, -4, -3, -2, -1, 0, 1, 2, 3, 4],
    #    [ 5, 4, 3, 2, 1, 0 , -1, -2, -3, -4]
    # ]
    spec = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        observer_cls=PerBlockMXObserver,
        symmetric=None,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        scale_format="e8m0",
        scale_calculation_mode="even",
        qscheme=QSchemeType.per_group,
        ch_axis=1,
        is_dynamic=True,
        group_size=5,
    )
    observer = PerBlockMXObserver(qspec=spec)
    observer(a)

    _, _, emax = get_dtype_params(Dtype.fp8_e4m3)
    scale, _ = observer.calculate_qparams()
    assert scale[0, 0] == math.pow(2.0, math.floor(math.log2(5)) - emax)
    assert scale[0, 1] == math.pow(2.0, math.floor(math.log2(4)) - emax)
    assert scale[1, 0] == math.pow(2.0, math.floor(math.log2(5)) - emax)
    assert scale[1, 1] == math.pow(2.0, math.floor(math.log2(4)) - emax)


per_block_to_quantize_mx = [
    ("int8", 1, 8),
    ("fp8_e4m3", 1, 8),
    ("fp8_e5m2", 1, 8),
    ("fp6_e3m2", 1, 8),
    ("fp6_e2m3", 1, 8),
    ("fp4", 1, 8),
]


@pytest.mark.parametrize("element_dtype, axis, block_size", per_block_to_quantize_mx)
def test_per_block_to_fake_quantize_mx(element_dtype, axis, block_size):
    x_orig = create_4d_tensor_with_interesting_pattern()
    run_torch_op_variants(
        lambda: fake_quantize_mx(
            x_orig, axis, block_size, mx_element_dtype=element_dtype, scale_calculation_mode="floor"
        ),
    )


@pytest.mark.parametrize(
    "torch_dtype,qdtype,axis,block_size,expected_output",
    [
        (
            torch.float32,
            "mx6",
            -1,
            16,
            torch.tensor(
                [
                    [
                        [[10.0, 20.0, 32.0, 40.0], [20.0, 40.0, 64.0, 80.0], [32.0, 60.0, 88.0, 120.0]],
                        [[112.0, 120.0, 128.0, 144.0], [128.0, 144.0, 160.0, 176.0], [128.0, 160.0, 192.0, 224.0]],
                    ]
                ]
            ),
        ),
        (
            torch.float32,
            "mx9",
            -1,
            16,
            torch.tensor(
                [
                    [
                        [[10.0, 20.0, 30.0, 40.0], [20.0, 40.0, 60.0, 80.0], [30.0, 60.0, 90.0, 120.0]],
                        [[110.0, 120.0, 130.0, 140.0], [120.0, 140.0, 160.0, 180.0], [130.0, 160.0, 190.0, 220.0]],
                    ]
                ]
            ),
        ),
        (
            torch.float16,
            "mx6",
            -1,
            16,
            torch.tensor(
                [
                    [
                        [[10.0, 20.0, 32.0, 40.0], [20.0, 40.0, 64.0, 80.0], [32.0, 60.0, 88.0, 120.0]],
                        [[112.0, 120.0, 128.0, 144.0], [128.0, 144.0, 160.0, 176.0], [128.0, 160.0, 192.0, 224.0]],
                    ]
                ],
                dtype=torch.float16,
            ),
        ),
        (
            torch.float16,
            "mx9",
            -1,
            16,
            torch.tensor(
                [
                    [
                        [[10.0, 20.0, 30.0, 40.0], [20.0, 40.0, 60.0, 80.0], [30.0, 60.0, 90.0, 120.0]],
                        [[110.0, 120.0, 130.0, 140.0], [120.0, 140.0, 160.0, 180.0], [130.0, 160.0, 190.0, 220.0]],
                    ]
                ],
                dtype=torch.float16,
            ),
        ),
        (
            torch.bfloat16,
            "mx6",
            -1,
            16,
            torch.tensor(
                [
                    [
                        [[10.0, 20.0, 32.0, 40.0], [20.0, 40.0, 64.0, 80.0], [32.0, 60.0, 88.0, 120.0]],
                        [[112.0, 120.0, 128.0, 144.0], [128.0, 144.0, 160.0, 176.0], [128.0, 160.0, 192.0, 224.0]],
                    ]
                ],
                dtype=torch.bfloat16,
            ),
        ),
        (
            torch.bfloat16,
            "mx9",
            -1,
            16,
            torch.tensor(
                [
                    [
                        [[10.0, 20.0, 30.0, 40.0], [20.0, 40.0, 60.0, 80.0], [30.0, 60.0, 90.0, 120.0]],
                        [[110.0, 120.0, 130.0, 140.0], [120.0, 140.0, 160.0, 180.0], [130.0, 160.0, 190.0, 220.0]],
                    ]
                ],
                dtype=torch.bfloat16,
            ),
        ),
    ],
)
def test_per_block_to_fake_quantize_mx6_mx9(torch_dtype, qdtype, axis, block_size, expected_output):
    x_orig = create_4d_tensor_with_interesting_pattern().to(torch_dtype)
    output_tensor = torch.ops.quark.non_scaled_fake_quantize(x_orig, qdtype, "", axis, block_size, "floor")
    assert torch.all(torch.isclose(output_tensor, expected_output))


def generate_test_case_input_normal():
    normal_input = torch.tensor(
        [
            [
                0,
                0,
                -53,
                -61,
                0,
                64,
                92,
                -60,
                0,
                0,
                99,
                67,
                0,
                0,
                41,
                -60,
                0,
                0,
                -54,
                67,
                0,
                -128,
                -111,
                67,
                0,
                -128,
                -123,
                67,
                0,
                0,
                -113,
                67,
            ],
            [
                0,
                64,
                11,
                -60,
                0,
                -128,
                -4,
                67,
                0,
                0,
                47,
                -61,
                0,
                0,
                124,
                -61,
                0,
                -128,
                1,
                -60,
                0,
                0,
                8,
                -61,
                0,
                0,
                6,
                -61,
                0,
                -128,
                119,
                68,
            ],
            [
                0,
                0,
                20,
                -61,
                0,
                -128,
                -50,
                67,
                0,
                0,
                -83,
                -61,
                0,
                -128,
                110,
                -60,
                0,
                0,
                59,
                68,
                0,
                64,
                91,
                -60,
                0,
                0,
                73,
                -61,
                0,
                0,
                -87,
                67,
            ],
            [
                0,
                0,
                -125,
                -61,
                0,
                -128,
                -38,
                67,
                0,
                -64,
                60,
                68,
                0,
                0,
                61,
                68,
                0,
                0,
                65,
                68,
                0,
                -64,
                63,
                68,
                0,
                0,
                114,
                68,
                0,
                64,
                119,
                -60,
            ],
            [
                0,
                0,
                -88,
                67,
                0,
                0,
                24,
                -62,
                0,
                64,
                22,
                68,
                0,
                0,
                34,
                -61,
                0,
                64,
                18,
                68,
                0,
                -128,
                43,
                -60,
                0,
                -128,
                24,
                68,
                0,
                0,
                -60,
                66,
            ],
            [
                0,
                -128,
                51,
                -60,
                0,
                64,
                111,
                68,
                0,
                -128,
                85,
                -60,
                0,
                64,
                118,
                -60,
                0,
                -128,
                14,
                -60,
                0,
                64,
                119,
                68,
                0,
                -128,
                65,
                -60,
                0,
                -64,
                45,
                68,
            ],
            [
                0,
                0,
                49,
                68,
                0,
                0,
                -114,
                -62,
                0,
                -64,
                87,
                68,
                0,
                0,
                -126,
                67,
                0,
                0,
                124,
                -61,
                0,
                -128,
                11,
                68,
                0,
                -128,
                59,
                -60,
                0,
                64,
                84,
                -60,
            ],
            [
                0,
                0,
                -58,
                -61,
                0,
                -128,
                118,
                -60,
                0,
                -128,
                -92,
                -61,
                0,
                -128,
                75,
                68,
                0,
                0,
                -86,
                -61,
                0,
                -128,
                -111,
                -61,
                0,
                0,
                13,
                67,
                0,
                0,
                68,
                68,
            ],
        ],
        dtype=torch.int8,
    )
    return normal_input


def generate_test_case_input_float():
    float_input = torch.tensor(
        [
            [
                45,
                42,
                125,
                63,
                -48,
                89,
                124,
                63,
                126,
                -17,
                81,
                63,
                69,
                23,
                108,
                63,
                -101,
                108,
                92,
                63,
                -68,
                -32,
                16,
                62,
                -88,
                -27,
                -109,
                61,
                78,
                -7,
                -101,
                62,
            ],
            [
                -125,
                65,
                84,
                63,
                -74,
                -5,
                122,
                63,
                26,
                -79,
                124,
                63,
                104,
                -85,
                -92,
                61,
                66,
                -26,
                109,
                63,
                -12,
                -74,
                -33,
                62,
                -16,
                110,
                53,
                63,
                48,
                10,
                -14,
                61,
            ],
            [
                -96,
                -116,
                53,
                62,
                -60,
                -32,
                116,
                63,
                125,
                19,
                39,
                63,
                -8,
                4,
                25,
                63,
                -67,
                50,
                3,
                63,
                64,
                -28,
                -127,
                60,
                88,
                -40,
                58,
                63,
                111,
                58,
                16,
                63,
            ],
            [
                71,
                33,
                22,
                63,
                -106,
                8,
                -18,
                62,
                52,
                53,
                -25,
                62,
                -52,
                80,
                5,
                62,
                74,
                -83,
                -68,
                62,
                32,
                93,
                -2,
                61,
                -80,
                17,
                108,
                63,
                -106,
                28,
                90,
                63,
            ],
            [
                47,
                29,
                40,
                63,
                84,
                86,
                -44,
                62,
                90,
                -89,
                109,
                63,
                -38,
                -51,
                104,
                63,
                -44,
                -21,
                46,
                63,
                126,
                -11,
                -62,
                62,
                -63,
                -2,
                8,
                63,
                -90,
                106,
                -7,
                62,
            ],
            [
                8,
                112,
                -107,
                61,
                -93,
                -123,
                91,
                63,
                -2,
                -73,
                13,
                63,
                -124,
                34,
                35,
                62,
                47,
                43,
                35,
                63,
                -42,
                91,
                -20,
                62,
                -90,
                65,
                -67,
                62,
                -24,
                88,
                84,
                63,
            ],
            [
                21,
                -98,
                87,
                63,
                55,
                26,
                1,
                63,
                69,
                127,
                52,
                63,
                -28,
                51,
                48,
                62,
                2,
                -97,
                -10,
                62,
                -43,
                74,
                61,
                63,
                68,
                75,
                -8,
                62,
                80,
                125,
                57,
                62,
            ],
            [
                -128,
                -70,
                23,
                60,
                -64,
                45,
                -81,
                61,
                115,
                -74,
                66,
                63,
                -47,
                -116,
                20,
                63,
                121,
                -122,
                63,
                63,
                -16,
                61,
                -125,
                62,
                64,
                97,
                -66,
                60,
                88,
                -118,
                61,
                62,
            ],
        ],
        dtype=torch.int8,
    )
    return float_input


def generate_test_case_input():
    test_data = {
        "normal": generate_test_case_input_normal(),
        "float": generate_test_case_input_float(),
        "zeros": torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [
                    0,
                    64,
                    11,
                    -60,
                    0,
                    -128,
                    -4,
                    67,
                    0,
                    0,
                    47,
                    -61,
                    0,
                    0,
                    124,
                    -61,
                    0,
                    -128,
                    1,
                    -60,
                    0,
                    0,
                    8,
                    -61,
                    0,
                    0,
                    6,
                    -61,
                    0,
                    -128,
                    119,
                    68,
                ],
                [
                    0,
                    0,
                    20,
                    -61,
                    0,
                    -128,
                    -50,
                    67,
                    0,
                    0,
                    -83,
                    -61,
                    0,
                    -128,
                    110,
                    -60,
                    0,
                    0,
                    59,
                    68,
                    0,
                    64,
                    91,
                    -60,
                    0,
                    0,
                    73,
                    -61,
                    0,
                    0,
                    -87,
                    67,
                ],
                [
                    0,
                    0,
                    -125,
                    -61,
                    0,
                    -128,
                    -38,
                    67,
                    0,
                    -64,
                    60,
                    68,
                    0,
                    0,
                    61,
                    68,
                    0,
                    0,
                    65,
                    68,
                    0,
                    -64,
                    63,
                    68,
                    0,
                    0,
                    114,
                    68,
                    0,
                    64,
                    119,
                    -60,
                ],
                [
                    0,
                    0,
                    -88,
                    67,
                    0,
                    0,
                    24,
                    -62,
                    0,
                    64,
                    22,
                    68,
                    0,
                    0,
                    34,
                    -61,
                    0,
                    64,
                    18,
                    68,
                    0,
                    -128,
                    43,
                    -60,
                    0,
                    -128,
                    24,
                    68,
                    0,
                    0,
                    -60,
                    66,
                ],
                [
                    0,
                    -128,
                    51,
                    -60,
                    0,
                    64,
                    111,
                    68,
                    0,
                    -128,
                    85,
                    -60,
                    0,
                    64,
                    118,
                    -60,
                    0,
                    -128,
                    14,
                    -60,
                    0,
                    64,
                    119,
                    68,
                    0,
                    -128,
                    65,
                    -60,
                    0,
                    -64,
                    45,
                    68,
                ],
                [
                    0,
                    0,
                    49,
                    68,
                    0,
                    0,
                    -114,
                    -62,
                    0,
                    -64,
                    87,
                    68,
                    0,
                    0,
                    -126,
                    67,
                    0,
                    0,
                    124,
                    -61,
                    0,
                    -128,
                    11,
                    68,
                    0,
                    -128,
                    59,
                    -60,
                    0,
                    64,
                    84,
                    -60,
                ],
                [
                    0,
                    0,
                    -58,
                    -61,
                    0,
                    -128,
                    118,
                    -60,
                    0,
                    -128,
                    -92,
                    -61,
                    0,
                    -128,
                    75,
                    68,
                    0,
                    0,
                    -86,
                    -61,
                    0,
                    -128,
                    -111,
                    -61,
                    0,
                    0,
                    13,
                    67,
                    0,
                    0,
                    68,
                    68,
                ],
            ],
            dtype=torch.int8,
        ),
        "nan": torch.tensor(
            [
                [
                    0,
                    0,
                    -53,
                    -61,
                    0,
                    0,
                    -64,
                    127,
                    0,
                    0,
                    99,
                    67,
                    0,
                    0,
                    41,
                    -60,
                    0,
                    0,
                    -54,
                    67,
                    0,
                    -128,
                    -111,
                    67,
                    0,
                    -128,
                    -123,
                    67,
                    0,
                    0,
                    -113,
                    67,
                ],
                [
                    0,
                    64,
                    11,
                    -60,
                    0,
                    -128,
                    -4,
                    67,
                    0,
                    0,
                    47,
                    -61,
                    0,
                    0,
                    124,
                    -61,
                    0,
                    -128,
                    1,
                    -60,
                    0,
                    0,
                    8,
                    -61,
                    0,
                    0,
                    6,
                    -61,
                    0,
                    -128,
                    119,
                    68,
                ],
                [
                    0,
                    0,
                    20,
                    -61,
                    0,
                    -128,
                    -50,
                    67,
                    0,
                    0,
                    -83,
                    -61,
                    0,
                    -128,
                    110,
                    -60,
                    0,
                    0,
                    59,
                    68,
                    0,
                    64,
                    91,
                    -60,
                    0,
                    0,
                    73,
                    -61,
                    0,
                    0,
                    -87,
                    67,
                ],
                [
                    0,
                    0,
                    -125,
                    -61,
                    0,
                    -128,
                    -38,
                    67,
                    0,
                    -64,
                    60,
                    68,
                    0,
                    0,
                    61,
                    68,
                    0,
                    0,
                    65,
                    68,
                    0,
                    -64,
                    63,
                    68,
                    0,
                    0,
                    114,
                    68,
                    0,
                    64,
                    119,
                    -60,
                ],
                [
                    0,
                    0,
                    -88,
                    67,
                    0,
                    0,
                    24,
                    -62,
                    0,
                    64,
                    22,
                    68,
                    0,
                    0,
                    34,
                    -61,
                    0,
                    64,
                    18,
                    68,
                    0,
                    -128,
                    43,
                    -60,
                    0,
                    -128,
                    24,
                    68,
                    0,
                    0,
                    -60,
                    66,
                ],
                [
                    0,
                    -128,
                    51,
                    -60,
                    0,
                    64,
                    111,
                    68,
                    0,
                    -128,
                    85,
                    -60,
                    0,
                    64,
                    118,
                    -60,
                    0,
                    -128,
                    14,
                    -60,
                    0,
                    64,
                    119,
                    68,
                    0,
                    -128,
                    65,
                    -60,
                    0,
                    -64,
                    45,
                    68,
                ],
                [
                    0,
                    0,
                    49,
                    68,
                    0,
                    0,
                    -114,
                    -62,
                    0,
                    -64,
                    87,
                    68,
                    0,
                    0,
                    -126,
                    67,
                    0,
                    0,
                    124,
                    -61,
                    0,
                    -128,
                    11,
                    68,
                    0,
                    -128,
                    59,
                    -60,
                    0,
                    64,
                    84,
                    -60,
                ],
                [
                    0,
                    0,
                    -58,
                    -61,
                    0,
                    -128,
                    118,
                    -60,
                    0,
                    -128,
                    -92,
                    -61,
                    0,
                    -128,
                    75,
                    68,
                    0,
                    0,
                    -86,
                    -61,
                    0,
                    -128,
                    -111,
                    -61,
                    0,
                    0,
                    13,
                    67,
                    0,
                    0,
                    68,
                    68,
                ],
            ],
            dtype=torch.int8,
        ),
        "inf": torch.tensor(
            [
                [
                    0,
                    0,
                    -53,
                    -61,
                    0,
                    0,
                    -128,
                    127,
                    0,
                    0,
                    99,
                    67,
                    0,
                    0,
                    41,
                    -60,
                    0,
                    0,
                    -54,
                    67,
                    0,
                    -128,
                    -111,
                    67,
                    0,
                    -128,
                    -123,
                    67,
                    0,
                    0,
                    -113,
                    67,
                ],
                [
                    0,
                    64,
                    11,
                    -60,
                    0,
                    -128,
                    -4,
                    67,
                    0,
                    0,
                    47,
                    -61,
                    0,
                    0,
                    124,
                    -61,
                    0,
                    -128,
                    1,
                    -60,
                    0,
                    0,
                    8,
                    -61,
                    0,
                    0,
                    6,
                    -61,
                    0,
                    -128,
                    119,
                    68,
                ],
                [
                    0,
                    0,
                    20,
                    -61,
                    0,
                    -128,
                    -50,
                    67,
                    0,
                    0,
                    -83,
                    -61,
                    0,
                    -128,
                    110,
                    -60,
                    0,
                    0,
                    59,
                    68,
                    0,
                    64,
                    91,
                    -60,
                    0,
                    0,
                    73,
                    -61,
                    0,
                    0,
                    -87,
                    67,
                ],
                [
                    0,
                    0,
                    -125,
                    -61,
                    0,
                    -128,
                    -38,
                    67,
                    0,
                    -64,
                    60,
                    68,
                    0,
                    0,
                    61,
                    68,
                    0,
                    0,
                    65,
                    68,
                    0,
                    -64,
                    63,
                    68,
                    0,
                    0,
                    114,
                    68,
                    0,
                    64,
                    119,
                    -60,
                ],
                [
                    0,
                    0,
                    -88,
                    67,
                    0,
                    0,
                    24,
                    -62,
                    0,
                    64,
                    22,
                    68,
                    0,
                    0,
                    34,
                    -61,
                    0,
                    64,
                    18,
                    68,
                    0,
                    -128,
                    43,
                    -60,
                    0,
                    -128,
                    24,
                    68,
                    0,
                    0,
                    -60,
                    66,
                ],
                [
                    0,
                    -128,
                    51,
                    -60,
                    0,
                    64,
                    111,
                    68,
                    0,
                    -128,
                    85,
                    -60,
                    0,
                    64,
                    118,
                    -60,
                    0,
                    -128,
                    14,
                    -60,
                    0,
                    64,
                    119,
                    68,
                    0,
                    -128,
                    65,
                    -60,
                    0,
                    -64,
                    45,
                    68,
                ],
                [
                    0,
                    0,
                    49,
                    68,
                    0,
                    0,
                    -114,
                    -62,
                    0,
                    -64,
                    87,
                    68,
                    0,
                    0,
                    -126,
                    67,
                    0,
                    0,
                    124,
                    -61,
                    0,
                    -128,
                    11,
                    68,
                    0,
                    -128,
                    59,
                    -60,
                    0,
                    64,
                    84,
                    -60,
                ],
                [
                    0,
                    0,
                    -58,
                    -61,
                    0,
                    -128,
                    118,
                    -60,
                    0,
                    -128,
                    -92,
                    -61,
                    0,
                    -128,
                    75,
                    68,
                    0,
                    0,
                    -86,
                    -61,
                    0,
                    -128,
                    -111,
                    -61,
                    0,
                    0,
                    13,
                    67,
                    0,
                    0,
                    68,
                    68,
                ],
            ],
            dtype=torch.int8,
        ),
        "maximum": torch.tensor(
            [
                [
                    0,
                    0,
                    -53,
                    -61,
                    -27,
                    -19,
                    -68,
                    106,
                    0,
                    0,
                    99,
                    67,
                    0,
                    0,
                    41,
                    -60,
                    0,
                    0,
                    -54,
                    67,
                    0,
                    -128,
                    -111,
                    67,
                    0,
                    -128,
                    -123,
                    67,
                    0,
                    0,
                    -113,
                    67,
                ],
                [
                    0,
                    64,
                    11,
                    -60,
                    0,
                    -128,
                    -4,
                    67,
                    0,
                    0,
                    47,
                    -61,
                    0,
                    0,
                    124,
                    -61,
                    0,
                    -128,
                    1,
                    -60,
                    0,
                    0,
                    8,
                    -61,
                    0,
                    0,
                    6,
                    -61,
                    0,
                    -128,
                    119,
                    68,
                ],
                [
                    0,
                    0,
                    20,
                    -61,
                    0,
                    -128,
                    -50,
                    67,
                    0,
                    0,
                    -83,
                    -61,
                    0,
                    -128,
                    110,
                    -60,
                    0,
                    0,
                    59,
                    68,
                    0,
                    64,
                    91,
                    -60,
                    0,
                    0,
                    73,
                    -61,
                    0,
                    0,
                    -87,
                    67,
                ],
                [
                    0,
                    0,
                    -125,
                    -61,
                    0,
                    -128,
                    -38,
                    67,
                    0,
                    -64,
                    60,
                    68,
                    0,
                    0,
                    61,
                    68,
                    0,
                    0,
                    65,
                    68,
                    0,
                    -64,
                    63,
                    68,
                    0,
                    0,
                    114,
                    68,
                    0,
                    64,
                    119,
                    -60,
                ],
                [
                    0,
                    0,
                    -88,
                    67,
                    0,
                    0,
                    24,
                    -62,
                    0,
                    64,
                    22,
                    68,
                    0,
                    0,
                    34,
                    -61,
                    0,
                    64,
                    18,
                    68,
                    0,
                    -128,
                    43,
                    -60,
                    0,
                    -128,
                    24,
                    68,
                    0,
                    0,
                    -60,
                    66,
                ],
                [
                    0,
                    -128,
                    51,
                    -60,
                    0,
                    64,
                    111,
                    68,
                    0,
                    -128,
                    85,
                    -60,
                    0,
                    64,
                    118,
                    -60,
                    0,
                    -128,
                    14,
                    -60,
                    0,
                    64,
                    119,
                    68,
                    0,
                    -128,
                    65,
                    -60,
                    0,
                    -64,
                    45,
                    68,
                ],
                [
                    0,
                    0,
                    49,
                    68,
                    0,
                    0,
                    -114,
                    -62,
                    0,
                    -64,
                    87,
                    68,
                    0,
                    0,
                    -126,
                    67,
                    0,
                    0,
                    124,
                    -61,
                    0,
                    -128,
                    11,
                    68,
                    0,
                    -128,
                    59,
                    -60,
                    0,
                    64,
                    84,
                    -60,
                ],
                [
                    0,
                    0,
                    -58,
                    -61,
                    0,
                    -128,
                    118,
                    -60,
                    0,
                    -128,
                    -92,
                    -61,
                    0,
                    -128,
                    75,
                    68,
                    0,
                    0,
                    -86,
                    -61,
                    0,
                    -128,
                    -111,
                    -61,
                    0,
                    0,
                    13,
                    67,
                    0,
                    0,
                    68,
                    68,
                ],
            ],
            dtype=torch.int8,
        ),
    }
    return test_data


def load_test_case_result():
    result = {
        "input": generate_test_case_input_normal(),
        "fp8_e4m3": {
            "torchao_result": torch.tensor(
                [
                    [-416.0, -896.0, 224.0, -704.0, 416.0, 288.0, 256.0, 288.0],
                    [-576.0, 512.0, -176.0, -256.0, -512.0, -128.0, -128.0, 896.0],
                    [-144.0, 416.0, -352.0, -896.0, 768.0, -896.0, -208.0, 352.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 896.0, -896.0],
                    [320.0, -40.0, 576.0, -160.0, 576.0, -704.0, 640.0, 96.0],
                    [-704.0, 896.0, -832.0, -896.0, -576.0, 896.0, -768.0, 704.0],
                    [704.0, -72.0, 832.0, 256.0, -256.0, 576.0, -768.0, -832.0],
                    [-384.0, -896.0, -320.0, 832.0, -352.0, -288.0, 144.0, 768.0],
                ],
                dtype=torch.float32,
            ),
            "MX_result": torch.tensor(
                [
                    [-416.0, -896.0, 224.0, -704.0, 416.0, 288.0, 256.0, 288.0],
                    [-576.0, 512.0, -176.0, -256.0, -512.0, -128.0, -128.0, 896.0],
                    [-144.0, 416.0, -352.0, -896.0, 768.0, -896.0, -208.0, 352.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 896.0, -896.0],
                    [320.0, -40.0, 576.0, -160.0, 576.0, -704.0, 640.0, 96.0],
                    [-704.0, 896.0, -832.0, -896.0, -576.0, 896.0, -768.0, 704.0],
                    [704.0, -72.0, 832.0, 256.0, -256.0, 576.0, -768.0, -832.0],
                    [-384.0, -896.0, -320.0, 832.0, -352.0, -288.0, 144.0, 768.0],
                ],
                dtype=torch.float32,
            ),
        },
        "fp8_e5m2": {
            "torchao_result": torch.tensor(
                [
                    [-384.0, -896.0, 224.0, -640.0, 384.0, 320.0, 256.0, 256.0],
                    [-512.0, 512.0, -160.0, -256.0, -512.0, -128.0, -128.0, 896.0],
                    [-160.0, 384.0, -320.0, -896.0, 768.0, -896.0, -192.0, 320.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 896.0, -896.0],
                    [320.0, -40.0, 640.0, -160.0, 640.0, -640.0, 640.0, 96.0],
                    [-768.0, 896.0, -896.0, -896.0, -512.0, 896.0, -768.0, 640.0],
                    [768.0, -64.0, 896.0, 256.0, -256.0, 512.0, -768.0, -896.0],
                    [-384.0, -896.0, -320.0, 768.0, -320.0, -320.0, 128.0, 768.0],
                ],
                dtype=torch.float32,
            ),
            "MX_result": torch.tensor(
                [
                    [-384.0, -896.0, 224.0, -640.0, 384.0, 320.0, 256.0, 256.0],
                    [-512.0, 512.0, -160.0, -256.0, -512.0, -128.0, -128.0, 896.0],
                    [-160.0, 384.0, -320.0, -896.0, 768.0, -896.0, -192.0, 320.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 896.0, -896.0],
                    [320.0, -40.0, 640.0, -160.0, 640.0, -640.0, 640.0, 96.0],
                    [-768.0, 896.0, -896.0, -896.0, -512.0, 896.0, -768.0, 640.0],
                    [768.0, -64.0, 896.0, 256.0, -256.0, 512.0, -768.0, -896.0],
                    [-384.0, -896.0, -320.0, 768.0, -320.0, -320.0, 128.0, 768.0],
                ],
                dtype=torch.float32,
            ),
        },
        "fp6_e3m2": {
            "torchao_result": torch.tensor(
                [
                    [-384.0, -896.0, 224.0, -640.0, 384.0, 320.0, 256.0, 256.0],
                    [-512.0, 512.0, -160.0, -256.0, -512.0, -128.0, -128.0, 896.0],
                    [-160.0, 384.0, -320.0, -896.0, 768.0, -896.0, -192.0, 320.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 896.0, -896.0],
                    [320.0, -40.0, 640.0, -160.0, 640.0, -640.0, 640.0, 96.0],
                    [-768.0, 896.0, -896.0, -896.0, -512.0, 896.0, -768.0, 640.0],
                    [768.0, -64.0, 896.0, 256.0, -256.0, 512.0, -768.0, -896.0],
                    [-384.0, -896.0, -320.0, 768.0, -320.0, -320.0, 128.0, 768.0],
                ],
                dtype=torch.float32,
            ),
            "MX_result": torch.tensor(
                [
                    [-384.0, -896.0, 224.0, -640.0, 384.0, 320.0, 256.0, 256.0],
                    [-512.0, 512.0, -160.0, -256.0, -512.0, -128.0, -128.0, 896.0],
                    [-160.0, 384.0, -320.0, -896.0, 768.0, -896.0, -192.0, 320.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 896.0, -896.0],
                    [320.0, -40.0, 640.0, -160.0, 640.0, -640.0, 640.0, 96.0],
                    [-768.0, 896.0, -896.0, -896.0, -512.0, 896.0, -768.0, 640.0],
                    [768.0, -64.0, 896.0, 256.0, -256.0, 512.0, -768.0, -896.0],
                    [-384.0, -896.0, -320.0, 768.0, -320.0, -320.0, 128.0, 768.0],
                ],
                dtype=torch.float32,
            ),
        },
        "fp6_e2m3": {
            "torchao_result": torch.tensor(
                [
                    [-416.0, -896.0, 224.0, -704.0, 416.0, 288.0, 256.0, 288.0],
                    [-576.0, 512.0, -176.0, -256.0, -512.0, -128.0, -128.0, 960.0],
                    [-144.0, 416.0, -352.0, -960.0, 768.0, -896.0, -208.0, 352.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 960.0, -960.0],
                    [320.0, -32.0, 576.0, -160.0, 576.0, -704.0, 640.0, 96.0],
                    [-704.0, 960.0, -832.0, -960.0, -576.0, 960.0, -768.0, 704.0],
                    [704.0, -64.0, 832.0, 256.0, -256.0, 576.0, -768.0, -832.0],
                    [-384.0, -960.0, -320.0, 832.0, -352.0, -288.0, 144.0, 768.0],
                ],
                dtype=torch.float32,
            ),
            "MX_result": torch.tensor(
                [
                    [-416.0, -896.0, 224.0, -704.0, 416.0, 288.0, 256.0, 288.0],
                    [-576.0, 512.0, -176.0, -256.0, -512.0, -128.0, -128.0, 960.0],
                    [-144.0, 416.0, -352.0, -960.0, 768.0, -896.0, -208.0, 352.0],
                    [-256.0, 448.0, 768.0, 768.0, 768.0, 768.0, 960.0, -960.0],
                    [320.0, -32.0, 576.0, -160.0, 576.0, -704.0, 640.0, 96.0],
                    [-704.0, 960.0, -832.0, -960.0, -576.0, 960.0, -768.0, 704.0],
                    [704.0, -64.0, 832.0, 256.0, -256.0, 576.0, -768.0, -832.0],
                    [-384.0, -960.0, -320.0, 832.0, -352.0, -288.0, 144.0, 768.0],
                ],
                dtype=torch.float32,
            ),
        },
        "fp4": {
            "torchao_result": torch.tensor(
                [
                    [-384.0, -768.0, 256.0, -768.0, 384.0, 256.0, 256.0, 256.0],
                    [-512.0, 512.0, -192.0, -256.0, -512.0, -128.0, -128.0, 768.0],
                    [-128.0, 384.0, -384.0, -768.0, 768.0, -768.0, -192.0, 384.0],
                    [-256.0, 384.0, 768.0, 768.0, 768.0, 768.0, 768.0, -768.0],
                    [384.0, -64.0, 512.0, -192.0, 512.0, -768.0, 512.0, 128.0],
                    [-768.0, 768.0, -768.0, -768.0, -512.0, 768.0, -768.0, 768.0],
                    [768.0, -64.0, 768.0, 256.0, -256.0, 512.0, -768.0, -768.0],
                    [-384.0, -768.0, -384.0, 768.0, -384.0, -256.0, 128.0, 768.0],
                ],
                dtype=torch.float32,
            ),
            "MX_result": torch.tensor(
                [
                    [-384.0, -768.0, 256.0, -768.0, 384.0, 256.0, 256.0, 256.0],
                    [-512.0, 512.0, -192.0, -256.0, -512.0, -128.0, -128.0, 768.0],
                    [-128.0, 384.0, -384.0, -768.0, 768.0, -768.0, -192.0, 384.0],
                    [-256.0, 384.0, 768.0, 768.0, 768.0, 768.0, 768.0, -768.0],
                    [384.0, -64.0, 512.0, -192.0, 512.0, -768.0, 512.0, 128.0],
                    [-768.0, 768.0, -768.0, -768.0, -512.0, 768.0, -768.0, 768.0],
                    [768.0, -64.0, 768.0, 256.0, -256.0, 512.0, -768.0, -768.0],
                    [-384.0, -768.0, -384.0, 768.0, -384.0, -256.0, 128.0, 768.0],
                ],
                dtype=torch.float32,
            ),
        },
    }
    return result


quark_mx_dtype_lst = [Dtype.fp8_e4m3, Dtype.fp8_e5m2, Dtype.fp6_e3m2, Dtype.fp6_e2m3, Dtype.fp4, Dtype.int8]


@pytest.mark.parametrize("quark_mx_dtype", quark_mx_dtype_lst)
def test_fake_quantize_mx(quark_mx_dtype):
    test_data = generate_test_case_input()

    def pipeline():
        outs = []
        for _, test_tensor in test_data.items():
            test_tensor = test_tensor.view(torch.float32)
            block_size = 32
            axis = 1
            mx_element_dtype = quark_mx_dtype
            _, _, emax = get_dtype_params(mx_element_dtype)

            block_x = reshape_to_blocks(test_tensor, block_size, axis)
            scale, _ = torch.max(torch.abs(block_x), dim=axis + 1, keepdim=True)
            scale = torch.pow(2, torch.floor(torch.log2(scale)) - emax)

            outs.append(
                fake_quantize_mx(
                    input_tensor=test_tensor.clone(),
                    scale=scale,
                    mx_element_dtype=mx_element_dtype,
                    axis=axis,
                    block_size=block_size,
                    scale_calculation_mode="floor",
                )
            )
        return outs

    run_torch_op_variants(pipeline)


quark_supported_elem_dtype = {
    "fp8_e4m3": Dtype.fp8_e4m3,
    "fp8_e5m2": Dtype.fp8_e5m2,
    "fp6_e3m2": Dtype.fp6_e3m2,
    "fp6_e2m3": Dtype.fp6_e2m3,
    "fp4": Dtype.fp4,
    "int8": Dtype.int8,
}
elem_dtype_lst = ["fp8_e4m3", "fp8_e5m2", "fp6_e3m2", "fp6_e2m3", "fp4"]


@pytest.mark.parametrize("elem_dtype", elem_dtype_lst)
def test_compare_quark_ao_mx_repo(elem_dtype):
    """
    compare the performance of quark, torchao, MX
    quark: 0.1.0+559df62f
    torchao: 0.3.1
    """
    result = load_test_case_result()
    test_tensor = result["input"]
    test_tensor = test_tensor.view(torch.float32)

    block_size, axis = 32, 1
    mx_element_dtype = quark_supported_elem_dtype[elem_dtype]
    _, _, emax = get_dtype_params(mx_element_dtype)
    block_x = reshape_to_blocks(test_tensor, block_size, axis)
    scale, _ = torch.max(torch.abs(block_x), dim=axis + 1, keepdim=True)
    scale = torch.pow(2, torch.floor(torch.log2(scale)) - emax)

    def pipeline():
        return fake_quantize_mx(
            input_tensor=test_tensor.clone(),
            scale=scale,
            mx_element_dtype=mx_element_dtype,
            axis=axis,
            block_size=block_size,
            scale_calculation_mode="floor",
        )

    quark_output_tensor = run_torch_op_variants(pipeline)
    torchao_result = result[elem_dtype]["torchao_result"]
    MX_result = result[elem_dtype]["MX_result"]

    max_diff_ao = torch.max(abs(quark_output_tensor - torchao_result))
    max_diff_MX = torch.max(abs(quark_output_tensor - MX_result))
    assert max_diff_ao == 0, f"The {elem_dtype} quantization result of quark and torchao is different"
    assert max_diff_MX == 0, f"The {elem_dtype} quantization result of quark and MX is different"


@require_torch_cuda
def test_realquantizer_pipline():
    torch.manual_seed(42)
    qspec = OCP_MXINT8Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
    qspec.mx_element_dtype = Dtype.fp4
    input_quantizer = realquantizer.get_real_quantizer(
        qspec=qspec, quantizer=None, reorder=False, real_quantized=False, float_dtype=Dtype.fp4, device="cuda"
    )
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    input_quantizer.to_real_quantize_params(x)
