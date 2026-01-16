# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Sequence
from typing import Any

import numpy as np
import onnx
from numpy import typing as npt
from ryzenai_dynamic_dispatch import bfp16

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape


def get_activation_params(activation_shape: Sequence[int]) -> tuple[int, int]:
    if len(activation_shape) == 4:
        yi = activation_shape[1]
        xi = activation_shape[2]
    else:
        raise ValueError("the iConv activation tensor should be 4 dimensions")
    return yi, xi


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("IConv_noqdq")
    conv = subgraph[0]

    activation_shape = onnx.helper.get_node_attr_value(conv, "input_shape")
    yi, xi = get_activation_params(activation_shape)
    ofm_shape = ryzenai_onnx_utils.matcher.get_shape(conv.output[0], extractor)
    assert is_static_shape(ofm_shape)
    yo, xo = get_activation_params(ofm_shape)
    weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[1], extractor)
    if weight.dtype != np.float32:
        return subgraph, [], []
    bias = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[2], extractor)
    op_type = "IConv_noqdq"
    input_shape = np.array([yi, xi, yo, xo])
    bfp = False
    bfp_in_shape = []
    try:
        bfp_in_shape = onnx.helper.get_node_attr_value(conv, "bfp16_shape_0")
        bfp = True
    except ValueError:
        bfp = False

    try:
        weight_bfp: npt.NDArray[Any] = bfp16.const_from_fp32_to_bfp16(
            # TODO(varunsh): for some reason, not having this "redundant" cast, results
            # in a incompatible function call to the nanobind interface
            weight.astype(np.float32),
            bias.astype(np.float32),
            op_type,
            input_shape,
            bfp,
        )
    except RuntimeError as e:
        print(e)
        return subgraph, [], []

    weight_bfp_name = conv.input[1]
    weight_bfp_tensor = onnx.helper.make_tensor(
        weight_bfp_name,
        onnx.TensorProto.UINT8,
        weight_bfp.shape,
        weight_bfp.tobytes(),
        True,
    )
    new_initializers = [weight_bfp_tensor]

    if bfp:
        bfp_in = onnx.helper.get_node_attr_value(conv, "bfp16_tensors")
        bfp_in.append(weight_bfp_name)

        new_node = onnx.helper.make_node(
            op_type,
            inputs=[conv.input[0], weight_bfp_name, conv.input[2]],
            outputs=conv.output,
            domain=domain,
            name=conv.name,
            input_shape=activation_shape,
            bfp16_tensors=bfp_in,
            bfp16_shape_0=bfp_in_shape,
            bfp16_shape_1=weight.shape,
        )

        return [new_node], new_initializers, []
    else:
        new_node = onnx.helper.make_node(
            op_type,
            inputs=[conv.input[0], weight_bfp_name, conv.input[2]],
            outputs=conv.output,
            domain=domain,
            name=conv.name,
            input_shape=activation_shape,
            bfp16_tensors=[weight_bfp_name],
            bfp16_shape_0=weight.shape,
        )

        return [new_node], new_initializers, []


PATTERN = ["IConv_noqdq([?,?,?], ?)"]
REPLACEMENT = replacement
