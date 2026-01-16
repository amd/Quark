# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass fuses qkv matmuls preceding a GroupQueryAttention node into a single
MatMulNBits node with concatenated weights, scales, zero points and bias. If
the fuse_qkv attribute is set to false, the pass will skip the concatenation and
use a Concat node instead. The single input, whether from the concatenated MatMul
or Concat, is then fed into the GroupQueryAttention node.
"""

import contextlib

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs, StrictPassOutputArgs


def concatenate_initializers(
    q_matmul: onnx.NodeProto,
    k_matmul: onnx.NodeProto,
    v_matmul: onnx.NodeProto,
    index: int,
    extractor: onnx.utils.Extractor,
) -> tuple[onnx.TensorProto, onnx.ValueInfoProto]:
    q_np = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(q_matmul.input[index], extractor)
    k_np = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(k_matmul.input[index], extractor)
    v_np = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(v_matmul.input[index], extractor)

    qkv_np = np.concatenate((q_np, k_np, v_np), 0)
    qkv_np_name: str = q_matmul.input[index].replace("q_proj", "qkv_proj")
    qkv_np_name = qkv_np_name.replace("Add", "MatMulNBits")
    qkv_tensor = onnx.numpy_helper.from_array(qkv_np, qkv_np_name)
    tvi = onnx.helper.make_tensor_value_info(
        qkv_np_name, onnx.helper.np_dtype_to_tensor_dtype(qkv_np.dtype), qkv_np.shape
    )
    return qkv_tensor, tvi


def get_qkv_attribute(
    matmuls: tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto], attr_name: str, check_equal: bool
) -> onnx.AttributeProto:
    """
    Go through the qkv matmuls and extract the attribute values. Check if they're
    equal, except for the case of N where we add them together

    Args:
        matmuls (list[NodeProto]): q, k, v matmuls
        attr_name (str): Name of the attribute to get
        check_equal (bool): Check if attribute value across matmuls are equal
    """
    attr_value = None
    for matmul in matmuls:
        if attr_value is None:
            attr_value = onnx.helper.get_node_attr_value(matmul, attr_name)
        else:
            new_value = onnx.helper.get_node_attr_value(matmul, attr_name)
            if check_equal:
                assert attr_value == new_value
            else:
                # for N
                attr_value += new_value
    return onnx.helper.make_attribute(attr_name, attr_value)


def process_gqa_split_qkv(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
) -> StrictPassOutputArgs:
    q_matmul = subgraph[0]
    k_matmul = subgraph[1]
    v_matmul = subgraph[2]
    gqa = subgraph[3]

    # assuming matmulnbits have weights, scales, zero points and bias
    assert len(q_matmul.input) == len(k_matmul.input) == len(v_matmul.input)
    assert len(q_matmul.input) == 6
    assert q_matmul.input[0] == k_matmul.input[0] == v_matmul.input[0]

    new_nodes = []
    new_tvis = []

    qkv_output_name = q_matmul.output[0].replace("q_proj", "qkv_proj")

    q_matmul_shape = ryzenai_onnx_utils.matcher.get_shape(q_matmul.output[0], extractor)
    k_matmul_shape = ryzenai_onnx_utils.matcher.get_shape(k_matmul.output[0], extractor)
    v_matmul_shape = ryzenai_onnx_utils.matcher.get_shape(v_matmul.output[0], extractor)
    assert (
        isinstance(q_matmul_shape[-1], int)
        and isinstance(k_matmul_shape[-1], int)
        and isinstance(v_matmul_shape[-1], int)
    )
    concat_shape = list(q_matmul_shape)[:-1] + [q_matmul_shape[-1] + k_matmul_shape[-1] + v_matmul_shape[-1]]

    # Create Concat node
    concat_node = onnx.helper.make_node(
        "Concat",
        inputs=[q_matmul.output[0], k_matmul.output[0], v_matmul.output[0]],
        outputs=[qkv_output_name],
        axis=2,  # Concatenate along feature dimension (last axis)
    )

    # Create tensor value info for the output
    concat_tvi = onnx.helper.make_tensor_value_info(
        qkv_output_name,
        ryzenai_onnx_utils.matcher.get_dtype(v_matmul.output[0], extractor),
        concat_shape,
    )

    q_matmul.name = f"MatMulNBits_q_{pass_id}"
    k_matmul.name = f"MatMulNBits_k_{pass_id}"
    v_matmul.name = f"MatMulNBits_v_{pass_id}"

    new_nodes.append(q_matmul)
    new_nodes.append(k_matmul)
    new_nodes.append(v_matmul)
    new_nodes.append(concat_node)
    new_tvis.append(concat_tvi)

    gqa.input[0] = qkv_output_name
    gqa.input[1] = ""  # empty key input
    gqa.input[2] = ""  # empty value input

    new_nodes.append(gqa)

    return new_nodes, [], new_tvis


def process_gqa_merged_qkv(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
) -> StrictPassOutputArgs:
    q_matmul = subgraph[0]
    k_matmul = subgraph[1]
    v_matmul = subgraph[2]
    gqa = subgraph[3]

    # assuming matmulnbits have weights, scales, zero points and bias
    assert len(q_matmul.input) == len(k_matmul.input) == len(v_matmul.input)
    assert len(q_matmul.input) == 6
    has_bias = len(q_matmul.input) == 6
    assert q_matmul.input[0] == k_matmul.input[0] == v_matmul.input[0]

    qkv_weights_tensor, weights_tvi = concatenate_initializers(q_matmul, k_matmul, v_matmul, 1, extractor)
    qkv_scales_tensor, scales_tvi = concatenate_initializers(q_matmul, k_matmul, v_matmul, 2, extractor)
    qkv_zp_tensor, zp_tvi = concatenate_initializers(q_matmul, k_matmul, v_matmul, 3, extractor)

    new_nodes = []
    new_tvis = [weights_tvi, scales_tvi, zp_tvi]

    qkv_inputs = [
        q_matmul.input[0],
        qkv_weights_tensor.name,
        qkv_scales_tensor.name,
        qkv_zp_tensor.name,
    ]
    new_initializers = [
        qkv_weights_tensor,
        qkv_scales_tensor,
        qkv_zp_tensor,
    ]

    if has_bias:
        qkv_inputs.append("")  # empty g_idx
        qkv_bias_tensor, bias_tvi = concatenate_initializers(q_matmul, k_matmul, v_matmul, 5, extractor)
        new_tvis.append(bias_tvi)
        qkv_inputs.append(qkv_bias_tensor.name)
        new_initializers.append(qkv_bias_tensor)

    qkv_output_name = q_matmul.output[0].replace("q_proj", "qkv_proj")

    qkv_matmul = onnx.helper.make_node(
        "MatMulNBits",
        inputs=qkv_inputs,
        outputs=[qkv_output_name],
        domain=q_matmul.domain,
        name=f"MatMulNBits_{pass_id}",
    )

    with contextlib.suppress(ValueError):
        qkv_matmul.attribute.append(get_qkv_attribute((q_matmul, k_matmul, v_matmul), "accuracy_level", True))
    qkv_matmul.attribute.append(get_qkv_attribute((q_matmul, k_matmul, v_matmul), "bits", True))
    qkv_matmul.attribute.append(get_qkv_attribute((q_matmul, k_matmul, v_matmul), "block_size", True))
    qkv_matmul.attribute.append(get_qkv_attribute((q_matmul, k_matmul, v_matmul), "K", True))
    qkv_matmul.attribute.append(get_qkv_attribute((q_matmul, k_matmul, v_matmul), "N", False))
    new_nodes.append(qkv_matmul)

    output_shape = list(ryzenai_onnx_utils.matcher.get_shape(q_matmul.output[0], extractor))
    output_shape[-1] = onnx.helper.get_node_attr_value(qkv_matmul, "N")
    output_tvi = onnx.helper.make_tensor_value_info(
        qkv_output_name,
        ryzenai_onnx_utils.matcher.get_dtype(q_matmul.output[0], extractor),
        output_shape,
    )
    new_tvis.append(output_tvi)

    gqa.input[0] = qkv_output_name
    gqa.input[1] = ""  # empty key input
    gqa.input[2] = ""  # empty value input

    new_nodes.append(gqa)

    return new_nodes, new_initializers, new_tvis


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    fuse_qkv = params.get_bool_attr("fuse_qkv", True)
    if fuse_qkv:
        return process_gqa_merged_qkv(extractor, pass_id, subgraph)
    return process_gqa_split_qkv(extractor, pass_id, subgraph)


PATTERN = [
    "MatMulNBits([?,?,?,?,?], [a0])",  # q
    "MatMulNBits([?,?,?,?,?], [a1])",  # k
    "MatMulNBits([?,?,?,?,?], [a2])",  # v
    "GroupQueryAttention([a0,a1,a2,?,?,?,?,?,?], [?,?,?])",
]
REPLACEMENT = replacement
