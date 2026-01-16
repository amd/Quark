# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.utils
from ryzenai_onnx_utils.matcher import add_attribute
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype,
    add_cast_dtype_to_bfloat16,
)
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape


# add the KY and KX to distinguish the 1x1 and 3x3 conv, not only by input and output shape
def _get_matmulnbits_params(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> tuple[int, int, int, int, int, int]:
    if len(input_shape) == 2:
        BI, KI = input_shape
        MI = 1
    elif len(input_shape) == 3:
        BI, MI, KI = input_shape
    assert len(weight_shape) == 3
    NI, K1, K2 = weight_shape
    assert len(scale_shape) == 1 and scale_shape[0] == NI * K1
    if len(output_shape) == 2:
        BO, NO = output_shape
        MO = 1
    elif len(output_shape) == 3:
        BO, MO, NO = output_shape
    assert len(input_shape) == len(output_shape)
    assert BI == BO
    assert MI == MO
    assert NI == NO
    return (BI, MI, KI, NI, K1, K2)


def get_matmulnbits_params(
    matmulnbits: onnx.NodeProto, extractor: onnx.utils.Extractor
) -> tuple[int, int, int, int, int, int]:
    input_shape = ryzenai_onnx_utils.matcher.get_shape(matmulnbits.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(matmulnbits.input[1], extractor)
    scale_shape = ryzenai_onnx_utils.matcher.get_shape(matmulnbits.input[2], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(matmulnbits.output[0], extractor)
    assert (
        is_static_shape(input_shape)
        and is_static_shape(weight_shape)
        and is_static_shape(output_shape)
        and is_static_shape(scale_shape)
    )
    return _get_matmulnbits_params(input_shape, weight_shape, scale_shape, output_shape)


def is_matmulnbits_supported(matmulnbits: onnx.NodeProto, op_namespace: str, extractor: onnx.utils.Extractor) -> bool:
    if not ryzenai_onnx_utils.matcher.is_initializer(matmulnbits.input[1], extractor):
        return False
    wts_shape = ryzenai_onnx_utils.matcher.get_shape(matmulnbits.input[1], extractor)
    if len(wts_shape) != 3:
        return False
    # BI, MI, KI, NI, K1, K2
    supported_shapes = {
        "phi3.5": {
            (1, 577, 1024, 1024, 32, 16),  # count 23
            (1, 577, 1024, 4096, 32, 16),  # count 23
            (1, 577, 4096, 1024, 128, 16),  # count 23
        },
    }
    BI, MI, KI, NI, K1, K2 = get_matmulnbits_params(matmulnbits, extractor)
    return (BI, MI, KI, NI, K1, K2) in supported_shapes[op_namespace]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDMatMulNBits")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    (matmulnbits,) = subgraph

    assert len(matmulnbits.input) == 5 or len(matmulnbits.input) == 6
    assert len(matmulnbits.output) == 1
    has_bias = len(matmulnbits.input) == 6

    if not is_matmulnbits_supported(matmulnbits, op_namespace, extractor):
        return subgraph, [], None

    B, M, K, N, K1, K2 = get_matmulnbits_params(matmulnbits, extractor)
    tvis = []

    pre_cast_output = matmulnbits.input[0] + f".out{pass_id}"
    matmulnbits_input_shape = ryzenai_onnx_utils.matcher.get_shape(matmulnbits.input[0], extractor)
    matmulnbits_output_shape = ryzenai_onnx_utils.matcher.get_shape(matmulnbits.output[0], extractor)
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        matmulnbits.input[0],
        pre_cast_output,
        matmulnbits_input_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(matmulnbits.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi)

    new_inputs = [pre_cast_output, matmulnbits.input[1]]
    matmulnbits_output = matmulnbits.output[0] + f".out{pass_id}"
    op_type = "SDMatMulNBits"
    matmulnbits_node = onnx.helper.make_node(
        op_type,
        inputs=new_inputs,
        outputs=[matmulnbits_output],
        domain=domain,
        name=matmulnbits.name,
    )
    # need a leading one because of how the kernel is implemented
    add_attribute(matmulnbits_node, "input_shape", [B, M, K])
    add_attribute(matmulnbits_node, "in_dtypes", ["bfloat16", "bfloat16"])
    add_attribute(matmulnbits_node, "out_dtypes", ["bfloat16"])
    add_attribute(matmulnbits_node, "output_shape", [B, M, N])
    add_attribute(matmulnbits_node, "weight_shape", [N, K1, K2])
    add_attribute(matmulnbits_node, "scale_shape", [N * K1])
    if has_bias:
        add_attribute(matmulnbits_node, "bias_shape", [N])

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        matmulnbits.output[0] + f".out{pass_id}",
        matmulnbits.output[0],
        matmulnbits_output_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(matmulnbits.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast, matmulnbits_node, *post_cast], [], tvis


PATTERN = [["MatMulNBits([?,?,?,?,?,?], ?)"]]
REPLACEMENT = [replacement] * len(PATTERN)
