# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces GQA + MatMulNBits subgraphs with a single GQO custom op.
"""

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as cast
from ryzenai_onnx_utils.typing import PassOutputArgs, SubPass


def add_casts(
    before_cast_inputs: list[str],
    before_cast_outputs: list[str],
    pass_id: str,
    domain: str,
    extractor: onnx.utils.Extractor,
    cast_kv_cache: bool,
) -> tuple[list[onnx.NodeProto], list[onnx.ValueInfoProto], list[str], list[str]]:
    new_nodes = []
    new_tvis = []

    pre_cast, pre_tvi = cast.add_cast_dtype_to_bfloat16_auto(before_cast_inputs[0], pass_id, domain, extractor)
    # this is used to match and remove casts in hybrid_llm_add_cast_attributes
    pre_cast[0].name += ".hybrid_llm_0"
    new_nodes.extend(pre_cast)
    new_tvis.extend(pre_tvi)
    new_inputs = [pre_cast[0].output[0]]
    if before_cast_inputs[1] != "":
        pre_cast1, pre_tvi1 = cast.add_cast_dtype_to_bfloat16_auto(before_cast_inputs[1], pass_id, domain, extractor)
        # this is used to match and remove casts in hybrid_llm_add_cast_attributes
        # pre_cast1[0].name += ".hybrid_llm_1"
        new_nodes.extend(pre_cast1)
        new_tvis.extend(pre_tvi1)
        pre_cast2, pre_tvi2 = cast.add_cast_dtype_to_bfloat16_auto(before_cast_inputs[2], pass_id, domain, extractor)
        # this is used to match and remove casts in hybrid_llm_add_cast_attributes
        # pre_cast2[0].name += ".hybrid_llm_2"
        new_nodes.extend(pre_cast2)
        new_tvis.extend(pre_tvi2)
        new_inputs.extend([pre_cast1[0].output[0], pre_cast2[0].output[0]])
    else:
        new_inputs.extend([*before_cast_inputs[1:3]])

    if cast_kv_cache:
        post_cast_0, post_tvi_0 = cast.add_cast_bfloat16_to_dtype_auto(
            before_cast_outputs[0], pass_id, domain, extractor
        )
        post_cast_0[0].name += ".hybrid_llm_1"
        new_nodes.extend(post_cast_0)
        new_tvis.extend(post_tvi_0)
        post_cast_1, post_tvi_1 = cast.add_cast_bfloat16_to_dtype_auto(
            before_cast_outputs[1], pass_id, domain, extractor
        )
        post_cast_1[0].name += ".hybrid_llm_1"
        new_nodes.extend(post_cast_1)
        new_tvis.extend(post_tvi_1)
    post_cast_2, post_tvi_2 = cast.add_cast_bfloat16_to_dtype_auto(before_cast_outputs[2], pass_id, domain, extractor)
    post_cast_2[0].name += ".hybrid_llm_1"
    new_nodes.extend(post_cast_2)
    new_tvis.extend(post_tvi_2)

    new_inputs.extend([*before_cast_inputs[3:]])

    if cast_kv_cache:
        new_outputs = [
            post_cast_0[0].input[0],
            post_cast_1[0].input[0],
            post_cast_2[0].input[0],
        ]
    else:
        new_outputs = [
            *before_cast_outputs[:2],
            post_cast_2[0].input[0],
        ]
    return new_nodes, new_tvis, new_inputs, new_outputs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("GQO")
    gqa = subgraph[-2]
    matmul = subgraph[-1]

    assert len(gqa.input) == 12, f"Got {len(gqa.input)} expected 12 inputs for GQA"
    assert len(matmul.input) == 6, f"Got {len(matmul.input)} expected 6 inputs for following matmul"

    separate_qkv = gqa.input[1] != "" and gqa.input[2] != ""
    if separate_qkv and len(subgraph) == 3:
        # only use the last pattern if using separate q, k, v inputs
        return subgraph, [], []

    if separate_qkv:
        key_name = gqa.input[1]
        value_name = gqa.input[2]
    else:
        # use placeholders assuming concatenated qkv input
        key_name = ""
        value_name = ""

    before_cast_inputs = [
        gqa.input[0],
        key_name,
        value_name,
        *gqa.input[3:],
        # matmul initializer
        *matmul.input[1:4],
        # skip empty group tensor index
        matmul.input[5],
    ]

    before_cast_outputs = [
        *gqa.output[1:],
        matmul.output[0],
    ]

    cast_kv_cache = params.get_bool_attr("cast_kv_cache", True)
    new_nodes, new_tvis, new_inputs, new_outputs = add_casts(
        before_cast_inputs,
        before_cast_outputs,
        pass_id,
        domain,
        extractor,
        cast_kv_cache,
    )

    new_node = onnx.helper.make_node(
        "GQO",
        inputs=new_inputs,
        outputs=new_outputs,
        name=f"gqo_{pass_id}",
        domain=domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(gqa, new_node)

    for attr in matmul.attribute:
        attr.name = f"o_proj_{attr.name}"
    ryzenai_onnx_utils.matcher.copy_attributes(matmul, new_node)

    head_size = ryzenai_onnx_utils.matcher.get_shape(new_node.input[3], extractor)[-1]
    if not separate_qkv:
        qkv_input_tvi = [x for x in new_tvis if x.name == new_node.input[0]][0]
        qkv_input_dim = ryzenai_onnx_utils.matcher.get_shape(qkv_input_tvi)[-1]
        kv_num_heads = onnx.helper.get_node_attr_value(new_node, "kv_num_heads")
        num_heads = onnx.helper.get_node_attr_value(new_node, "num_heads")
        assert head_size == qkv_input_dim / (num_heads + 2 * kv_num_heads)

    ryzenai_onnx_utils.matcher.add_attribute(new_node, "head_size", head_size)

    new_nodes.append(new_node)

    if subgraph[0].op_type == "MatMulNBits":
        # rename here in case base model already had qkv matmul fused
        subgraph[0].name = f"MatMulNBits_{pass_id}"
    if not separate_qkv:
        new_nodes.append(subgraph[0])

    return new_nodes, [], new_tvis


PATTERN = [
    SubPass(
        "GQO_Pre_Merged_QKV",
        [
            "MatMulNBits([?,?,?,?,?,?], [a1])",
            "GroupQueryAttention([a1,?,?,?,?,?,?,?,?], [a0,?,?])",
            "MatMulNBits([a0,?,?,?,?,?], [?])",
        ],
    ),
    SubPass(
        "GQO_Pre_Split_QKV",
        [
            "Concat([?,?,?], [a1])",
            "GroupQueryAttention([a1,?,?,?,?,?,?,?,?], [a0,?,?])",
            "MatMulNBits([a0,?,?,?,?,?], [?])",
        ],
    ),
    SubPass(
        "GQO_unsplit_QKV",
        [
            "GroupQueryAttention([?,?,?,?,?,?,?,?,?], [a0,?,?])",
            "MatMulNBits([a0,?,?,?,?,?], [?])",
        ],
    ),
]
REPLACEMENT = replacement
