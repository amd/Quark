# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
from onnx.helper import make_attribute
from ryzenai_dynamic_dispatch import bfp16

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def get_activation_params(activation_shape: Sequence[int]) -> tuple[int, int]:
    if len(activation_shape) == 2:
        m = activation_shape[0]
        k = activation_shape[1]
    elif len(activation_shape) == 3:
        assert activation_shape[0] == 1
        m = activation_shape[1]
        k = activation_shape[2]
    else:
        raise ValueError("the MatMul activation tensor should be in 2 or 3 dimension")
    return m, k


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("MatMul_Transpose_noqdq")
    matmul = subgraph[0]

    assert len(matmul.input) == 3
    assert len(matmul.output) == 1

    activation_shape = onnx.helper.get_node_attr_value(matmul, "input_shape")
    m, k = get_activation_params(activation_shape)

    weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul.input[1], extractor)
    if weight.dtype != np.float32:
        return subgraph, [], []
    bias = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul.input[2], extractor)
    input_shape = np.array([m, weight.shape[0], weight.shape[1]])
    op_type = "MatMul_Transpose_noqdq"
    bfp = False
    bfp_out = False
    bfp_in_shape = []
    try:
        bfp_in_shape = onnx.helper.get_node_attr_value(matmul, "bfp16_shape_0")
        bfp = True
    except ValueError:
        bfp = False
    bfp_in_shape_1 = []
    try:
        bfp_in_shape_1 = onnx.helper.get_node_attr_value(matmul, "bfp16_shape_1")
        bfp_out = True
    except ValueError:
        bfp_out = False

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

    weight_bfp_name = matmul.input[1]
    weight_bfp_tensor = onnx.helper.make_tensor(
        weight_bfp_name,
        onnx.TensorProto.UINT8,
        weight_bfp.shape,
        weight_bfp.tobytes(),
        True,
    )
    new_initializers = [weight_bfp_tensor]
    matmul_node = onnx.helper.make_node(
        "MatMul_Transpose_noqdq",
        inputs=[matmul.input[0], weight_bfp_name, matmul.input[2]],
        outputs=matmul.output,
        domain=domain,
        name=matmul.name,
        input_shape=activation_shape,
    )
    if bfp:
        bfp_in = onnx.helper.get_node_attr_value(matmul, "bfp16_tensors")
        bfp_in.append(weight_bfp_name)
        matmul_node.attribute.append(make_attribute("bfp16_tensors", bfp_in))
        matmul_node.attribute.append(make_attribute("bfp16_shape_0", bfp_in_shape))
        if not bfp_out:
            matmul_node.attribute.append(make_attribute("bfp16_shape_1", weight.shape))
        else:
            matmul_node.attribute.append(make_attribute("bfp16_shape_1", bfp_in_shape_1))
            matmul_node.attribute.append(make_attribute("bfp16_shape_2", weight.shape))
    else:
        matmul_node.attribute.append(make_attribute("bfp16_tensors", [weight_bfp_name]))
        matmul_node.attribute.append(make_attribute("bfp16_shape_0", weight.shape))

    return [matmul_node], new_initializers, []


PATTERN = ["MatMul_Transpose_noqdq([?,?,?], ?)"]
REPLACEMENT = replacement
