# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from functools import reduce
from operator import mul
from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    copy_attributes,
    delete_attribute,
    find_nodes_by_input,
    get_attribute,
    get_dtype,
    get_shape,
    has_multiple_successors,
    set_attribute,
)
from ryzenai_onnx_utils.partitioner import get_dynamic_shape_candidate
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    WhiteboxBasePass,
    register_whitebox_pass,
)
from ryzenai_onnx_utils.passes.sd15.mha_to_sd_mha import make_sd_mha_node
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype,
)
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDGemmConcat")
class SDGemmConcatPass(WhiteboxBasePass):
    whitebox_flow_op_type = "GemmConcatTrans"
    force_whitelist = True

    # For SD3 SDGemmConcat
    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd3": {
                # mmdit seq-160
                ((2, 160, 1536), (1536, 1536), (2, 4096, 1536), (1536, 1536)),  # 1024
                ((2, 160, 1536), (1536, 1536), (2, 1024, 1536), (1536, 1536)),  # 512
                ((1, 160, 1536), (1536, 1536), (1, 4096, 1536), (1536, 1536)),  # 1024
                ((1, 160, 1536), (1536, 1536), (1, 1024, 1536), (1536, 1536)),  # 512
            },
        }
        input_shape_0 = tuple(check_shapes["input_shape"][0])
        input_shape_1 = tuple(check_shapes["input_shape"][1])
        weights_shape_0 = tuple(check_shapes["input_shape"][2])
        weights_shape_1 = tuple(check_shapes["input_shape"][4])
        return (
            input_shape_0,
            weights_shape_0,
            input_shape_1,
            weights_shape_1,
        ) in supported_shapes[op_namespace]

    # For SD3 SDGemmConcat
    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)),  # input_shape_0
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[1], extractor)),  # input_shape_1
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[2], extractor)),  # weights_shape_0
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[3], extractor)),  # bias_shape_0
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[4], extractor)),  # weights_shape_1
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[5], extractor)),  # bias_shape_1
            ],
            "output_shape": [
                list(ryzenai_onnx_utils.matcher.get_shape(node.output[0], extractor)),
            ],
        }
        return shape_lists


def is_supported_pattern(concat_node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> bool:
    if has_multiple_successors(concat_node.output[0], extractor.graph):
        return False

    axis = onnx.helper.get_node_attr_value(concat_node, "axis")
    concat_output_shape = get_shape(concat_node.output[0], extractor)
    axis = axis if axis >= 0 else axis + len(concat_output_shape)
    if axis != len(concat_output_shape) - 2:
        return False

    concat_output_nodes = find_nodes_by_input(concat_node.output[0], extractor.graph)
    return concat_output_nodes[0].op_type == "MultiHeadAttention"


def merge_attributes_flattened(
    nodes: list[onnx.NodeProto], target_node: onnx.NodeProto, merge_keys: list[str] | None = None
) -> None:
    if not nodes:
        return
    if merge_keys is None:
        merge_keys = []

    attr_keys_list = [{attr.name for attr in node.attribute} for node in nodes]
    if not all(keys == attr_keys_list[0] for keys in attr_keys_list):
        raise ValueError("Attribute names of nodes do not match. Cannot merge attributes.")

    attr_keys = list(attr_keys_list[0])

    for name in attr_keys:
        values = []
        for idx, node in enumerate(nodes):
            value = get_attribute(node, name)
            if name in merge_keys:
                values.append(value)
            else:
                indexed_name = f"{name}_{idx}"
                add_attribute(target_node, indexed_name, value)

        if name in merge_keys:
            if all(v == values[0] for v in values):
                add_attribute(target_node, name, values[0])
            else:
                raise ValueError(f"Cannot merge attribute '{name}': values are not equal: {values}")


def make_sd_gemm_concat_node(
    group: list[onnx.NodeProto],
    extractor: onnx.utils.Extractor,
    params: ryzenai_onnx_utils.ReplaceParams,
    pass_id: str,
    domain: str,
    op_namespace: str,
    num_heads: int,
) -> tuple[list[onnx.NodeProto] | None, list[onnx.TensorProto] | None, list[onnx.ValueInfoProto] | None]:
    concat_node = group[-1]
    if not is_supported_pattern(concat_node, extractor):
        return None, None, None

    gemm_nodes = group[:2]
    rms_nodes = []
    has_rn = False
    if len(group) > 5:
        has_rn = True
        rms_nodes = group[4:6]

    gemm_nodes_shape = [
        ryzenai_onnx_utils.matcher.get_shapes(gemm_nodes[0].input, extractor)[0],
        ryzenai_onnx_utils.matcher.get_shapes(gemm_nodes[1].input, extractor)[0],
    ]
    candidates = get_dynamic_shape_candidate(gemm_nodes_shape, params.attributes)
    # the smaller product first
    sorted_input_shapes = [sorted(candidate, key=lambda x: reduce(mul, x), reverse=False) for candidate in candidates]
    check_shapes = [candidates[idx] == sorted_input_shapes[idx] for idx in range(len(candidates))]
    assert all(flag == check_shapes[0] for flag in check_shapes)
    if not check_shapes[0]:
        gemm_nodes = [gemm_nodes[1], gemm_nodes[0]]
        if has_rn:
            rms_nodes = [rms_nodes[1], rms_nodes[0]]

    new_inputs = [
        gemm_nodes[0].input[0],
        gemm_nodes[1].input[0],
        gemm_nodes[0].input[1],
        gemm_nodes[0].input[2],
        gemm_nodes[1].input[1],
        gemm_nodes[1].input[2],
    ]

    op_type = "SDGemmConcat"
    if has_rn:
        new_inputs.append(rms_nodes[0].input[1])
        new_inputs.append(rms_nodes[1].input[1])
        op_type = "SDGemmRNConcat"

    new_output = concat_node.output[0] + f".out{pass_id}"
    gemm_concat_node = onnx.helper.make_node(
        op_type,
        inputs=new_inputs,
        outputs=[new_output],
        name=concat_node.name,
        domain=domain,
    )
    copy_attributes(concat_node, gemm_concat_node)
    set_attribute(gemm_concat_node, "trans_head", 1)
    add_attribute(gemm_concat_node, "head_num", num_heads)
    add_attribute(gemm_concat_node, "concat_axis", get_attribute(concat_node, "axis"))
    delete_attribute(gemm_concat_node, "axis")
    merge_attributes_flattened(
        gemm_nodes,
        gemm_concat_node,
        ["in_dtypes", "out_dtypes", "bias_enable"],
    )

    if has_rn:
        set_attribute(gemm_concat_node, "trans_head", 2)
        merge_attributes_flattened(
            rms_nodes,
            gemm_concat_node,
            ["axis", "epsilon"],
        )
        add_attribute(gemm_concat_node, "rmsnorm_axis", get_attribute(rms_nodes[0], "axis"))
        delete_attribute(gemm_concat_node, "axis")

    B, M, N = get_shape(concat_node.output[0], extractor)
    assert N % num_heads == 0
    new_output_shape = [B, num_heads, M, N // num_heads]

    cast_nodes, cast_tvi = add_cast_bfloat16_to_dtype(
        new_output,
        concat_node.output[0],
        new_output_shape,
        domain,
        get_dtype(concat_node.output[0], extractor),
    )

    return [gemm_concat_node] + cast_nodes, [], cast_tvi


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDGemmConcat")
    op_namespace = params.get_subgraph_op_namespace(subgraph)

    group_size = 5
    groups = [subgraph[i : i + group_size] for i in range(0, 15, group_size)]
    q_subgraph = groups[0]
    k_subgraph = groups[1]
    v_subgraph = groups[2]
    mha_node = subgraph[-1]

    assert len(mha_node.input) == 3
    assert len(mha_node.output) == 1

    num_heads = get_attribute(mha_node, "num_heads")
    tvis: list[onnx.ValueInfoProto] = []
    initializers: list[onnx.TensorProto] = []
    new_nodes: list[onnx.NodeProto] = []

    for group in [q_subgraph, k_subgraph, v_subgraph]:
        ret_nodes, ret_tensors, ret_tvis = make_sd_gemm_concat_node(
            group, extractor, params, pass_id, domain, op_namespace, num_heads
        )
        if ret_nodes is None:
            return subgraph, [], None

        assert ret_tensors is not None and ret_tvis is not None

        new_nodes.extend(ret_nodes)
        initializers.extend(ret_tensors)
        tvis.extend(ret_tvis)

    ret_nodes, _, ret_tvis = make_sd_mha_node(mha_node, pass_id, extractor, params, domain, op_namespace, "v2")
    if ret_nodes is None:
        return subgraph, [], None
    new_nodes.extend(ret_nodes)
    tvis.extend(ret_tvis)

    return new_nodes, initializers, tvis


PATTERN = [
    # Q branch
    "SDGemm([?, ?, ?], m1)",
    "SDGemm([?, ?, ?], m2)",
    "CastAvx([m1], c1)",
    "CastAvx([m2], c2)",
    "Concat([c1, c2], q)",
    # K branch
    "SDGemm([?, ?, ?], m3)",
    "SDGemm([?, ?, ?], m4)",
    "CastAvx([m3], c3)",
    "CastAvx([m4], c4)",
    "Concat([c3, c4], k)",
    # V branch
    "SDGemm([?, ?, ?], m5)",
    "SDGemm([?, ?, ?], m6)",
    "CastAvx([m5], c5)",
    "CastAvx([m6], c6)",
    "Concat([c5, c6], v)",
    # Attention
    "MultiHeadAttention([q,k,v], ?)",
]

REPLACEMENT = replacement
