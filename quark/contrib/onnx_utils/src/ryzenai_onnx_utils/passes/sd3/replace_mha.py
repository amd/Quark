# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import math

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_qkv_matmul(reshape, transpose, label: str, extractor) -> bool:
    # if not ryzenai_onnx_utils.matcher.is_initializer(matmul.input[1], extractor):
    #     return False
    # if not ryzenai_onnx_utils.matcher.is_initializer(bias.input[0], extractor):
    #     return False
    # bias_shape = ryzenai_onnx_utils.matcher.get_shape(bias.input[0], extractor)
    # if len(bias_shape) != 1:
    #     return False
    # split head, from NTC to NTHP
    in_shape = ryzenai_onnx_utils.matcher.get_shape(reshape.input[0], extractor)
    out_shape = ryzenai_onnx_utils.matcher.get_shape(reshape.output[0], extractor)
    if (
        len(in_shape) not in [2, 3]
        or len(out_shape) != 4
        # or math.prod(in_shape[:-1]) != math.prod(out_shape[:-2])
        or math.prod([i for i in in_shape if isinstance(i, int)])
        != math.prod([o for o in out_shape if isinstance(o, int)])
    ):
        return False
    perm = onnx.helper.get_node_attr_value(transpose, "perm")
    if label in ["query", "value"]:
        if perm != [0, 2, 1, 3]:
            return False
    elif label == "key" and perm != [0, 2, 3, 1]:
        return False
    return True


def is_mul_scaler(mul, extractor):
    if not ryzenai_onnx_utils.matcher.is_initializer(mul.input[1], extractor):
        return False
    coeff = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul.input[1], extractor)
    return coeff.shape == () or coeff.shape == (1,)


def is_attn_matmul(matmul, extractor):
    q_shape, k_shape = ryzenai_onnx_utils.matcher.get_shapes(matmul.input, extractor)
    return len(q_shape) == len(k_shape) and q_shape[:-2] == k_shape[:-2]


def is_fused_attn_matmul(matmul, extractor):
    if onnx.helper.get_node_attr_value(matmul, "activation").lower().decode() != "softmax":
        return False
    q_shape, k_shape = ryzenai_onnx_utils.matcher.get_shapes(matmul.input, extractor)
    return len(q_shape) == len(k_shape) and q_shape[:-2] == k_shape[:-2]


def is_attn_softmax(softmax, extractor) -> bool:
    in_shape = ryzenai_onnx_utils.matcher.get_shape(softmax.input[0], extractor)
    axis = onnx.helper.get_node_attr_value(softmax, "axis")
    return axis in [-1, len(in_shape) - 1]


def is_hidden_matmul(matmul, transpose, reshape, extractor):
    attn_shape, v_shape = ryzenai_onnx_utils.matcher.get_shapes(matmul.input, extractor)
    if len(attn_shape) != len(v_shape):
        return False
    perm = onnx.helper.get_node_attr_value(transpose, "perm")
    if perm != [0, 2, 1, 3]:
        return False
    in_shape = ryzenai_onnx_utils.matcher.get_shape(reshape.input[0], extractor)
    out_shape = ryzenai_onnx_utils.matcher.get_shape(reshape.output[0], extractor)
    return (len(in_shape) == 4 and len(out_shape) == 3 and in_shape[:2] == out_shape[:2]) or (
        len(in_shape) == 4
        and len(out_shape) == 2
        and in_shape[0] * in_shape[1] == out_shape[0]
        and in_shape[2] * in_shape[3] == out_shape[1]
    )


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    if len(subgraph) == 13:
        (
            q_reshape,
            k_reshape,
            v_reshape,
            q_transpose,
            k_transpose,
            v_transpose,
            q_mul,
            k_mul,
            qk_matmul,
            softmax,
            h_matmul,
            h_transpose,
            h_reshape,
        ) = subgraph
        if False in [
            is_qkv_matmul(q_reshape, q_transpose, "query", extractor),  # query
            is_qkv_matmul(k_reshape, k_transpose, "key", extractor),  # key
            is_qkv_matmul(v_reshape, v_transpose, "value", extractor),  # value
            is_mul_scaler(q_mul, extractor),  # q_coeff
            is_mul_scaler(k_mul, extractor),  # k_coeff
            is_attn_matmul(qk_matmul, extractor),  # attn matmul
            is_attn_softmax(softmax, extractor),  # attn softmax
            is_hidden_matmul(h_matmul, h_transpose, h_reshape, extractor),  # hidden matmul
        ]:
            return subgraph, [], None
    elif len(subgraph) == 10:
        (
            q_reshape,
            k_reshape,
            v_reshape,
            q_transpose,
            k_transpose,
            v_transpose,
            qk_matmul,
            h_matmul,
            h_transpose,
            h_reshape,
        ) = subgraph
        q_mul = None
        k_mul = None
        softmax = None

        if False in [
            is_qkv_matmul(q_reshape, q_transpose, "query", extractor),  # query
            is_qkv_matmul(k_reshape, k_transpose, "key", extractor),  # key
            is_qkv_matmul(v_reshape, v_transpose, "value", extractor),  # value
            is_fused_attn_matmul(qk_matmul, extractor),  # attn matmul
            is_hidden_matmul(h_matmul, h_transpose, h_reshape, extractor),  # hidden matmul
        ]:
            return subgraph, [], None

    tvis = []
    tensors = []
    nodes = []

    if q_mul and k_mul:
        q_coeff = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(q_mul.input[1], extractor)

        k_coeff = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(k_mul.input[1], extractor)
        qk_coeff = q_coeff * k_coeff
    else:
        qk_coeff = onnx.helper.get_node_attr_value(qk_matmul, "alpha")

    proj_qk = ryzenai_onnx_utils.matcher.get_shape(qk_matmul.input[0], extractor)[-1]
    onnx_mha_scale = 1.0 / math.sqrt(proj_qk)

    h_reshape_out_shape = ryzenai_onnx_utils.matcher.get_shape(h_reshape.output[0], extractor)
    if len(h_reshape_out_shape) == 3:
        new_mha_output = h_reshape.output[0]
    else:
        h_reshape_in_shape = ryzenai_onnx_utils.matcher.get_shape(h_reshape.input[0], extractor)
        attn_out_tvi = onnx.helper.make_tensor_value_info(
            h_reshape.output[0] + "_3d",
            ryzenai_onnx_utils.matcher.get_dtype(h_reshape.output[0], extractor),
            (
                h_reshape_in_shape[0],
                h_reshape_in_shape[1],
                h_reshape_in_shape[2] * h_reshape_in_shape[3],
            ),
        )
        tvis.append(attn_out_tvi)
        new_mha_output = attn_out_tvi.name

        reshape_node, reshape_tvis, reshape_tensor = add_reshape(
            attn_out_tvi.name,
            h_reshape.output[0] + "_3dshape",
            h_reshape.output[0],
            ryzenai_onnx_utils.matcher.get_dtype(h_reshape.output[0], extractor),
            (
                h_reshape_in_shape[0],
                h_reshape_in_shape[1],
                h_reshape_in_shape[2] * h_reshape_in_shape[3],
            ),
            h_reshape_out_shape,
        )
        nodes.append(reshape_node)
        tvis.extend(reshape_tvis)
        tensors.append(reshape_tensor)

    new_mha_node = onnx.helper.make_node(
        "MultiHeadAttention",
        inputs=[
            q_reshape.input[0],
            k_reshape.input[0],
            v_reshape.input[0],
            # mha_bias_name,
        ],
        outputs=[new_mha_output],
        name=(qk_matmul.name + f"mha_{pass_id}").replace("query", ""),
        domain="com.microsoft",
    )
    num_heads = ryzenai_onnx_utils.matcher.get_shape(q_reshape.output[0], extractor)[-2]
    ryzenai_onnx_utils.matcher.add_attribute(new_mha_node, "num_heads", num_heads)
    ryzenai_onnx_utils.matcher.add_attribute(new_mha_node, "unidirectional", 0)
    if np.fabs(qk_coeff - onnx_mha_scale) > 1e-7:
        ryzenai_onnx_utils.matcher.add_attribute(new_mha_node, "scale", qk_coeff)
    nodes.append(new_mha_node)

    return nodes, tensors, tvis


PATTERN = [
    [
        "Reshape([?, ?], qr)",
        "Reshape([?, ?], kr)",
        "Reshape([?, ?], vr)",
        "Transpose([qr], qt)",
        "Transpose([kr], kt)",
        "Transpose([vr], vt)",
        "Mul([qt, ?], qm)",
        "Mul([kt, ?], km)",
        "MatMul([qm, km], beta)",
        "Softmax([beta], alpha)",
        "MatMul([alpha, vt], h)",
        "Transpose([h], ht)",
        "Reshape([ht, ?], hr)",
    ],
    [
        "Reshape([?, ?], qr)",
        "Reshape([?, ?], kr)",
        "Reshape([?, ?], vr)",
        "Transpose([qr], qt)",
        "Transpose([kr], kt)",
        "Transpose([vr], vt)",
        "FusedMatMulActivation([qt, kt], alpha)",
        "MatMul([alpha, vt], h)",
        "Transpose([h], ht)",
        "Reshape([ht, ?], hr)",
    ],
]

REPLACEMENT = [replacement] * len(PATTERN)
