# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces SSMLP and SSGMLP subgraphs with their respective custom ops.
"""

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as cast
from ryzenai_onnx_utils.typing import PassOutputArgs, SubPass


def get_before_cast_outputs(ssln_1: onnx.NodeProto) -> list[str]:
    new_outputs = [ssln_1.output[0]]
    if len(ssln_1.output) == 4:
        new_outputs.append(ssln_1.output[3])
    return new_outputs


def process_ssmlp(
    subgraph: list[onnx.NodeProto], extractor: onnx.utils.Extractor
) -> tuple[list[str], list[str], onnx.NodeProto, onnx.NodeProto, onnx.NodeProto, onnx.NodeProto]:
    ssln_0 = subgraph[0]
    gate_proj = subgraph[1]
    up_proj = subgraph[4]
    down_proj = subgraph[6]
    ssln_1 = subgraph[7]

    assert len(gate_proj.input) == 6
    assert len(up_proj.input) == 6
    assert len(down_proj.input) == 6

    assert len(ssln_0.output) == 4
    # the last SSLN has a single output
    assert len(ssln_1.output) in [1, 4]

    epsilon = onnx.helper.get_node_attr_value(ssln_0, "epsilon")
    epsilon_2 = onnx.helper.get_node_attr_value(ssln_1, "epsilon")
    assert epsilon == epsilon_2

    check_biases(gate_proj, up_proj, down_proj, extractor)

    before_cast_inputs = [
        *ssln_0.input,
        # matmul initializer
        *gate_proj.input[1:4],
        *up_proj.input[1:4],
        *down_proj.input[1:4],
        # last ssln initializer
        ssln_1.input[2],
    ]
    before_cast_outputs = get_before_cast_outputs(ssln_1)

    return (
        before_cast_inputs,
        before_cast_outputs,
        gate_proj,
        up_proj,
        down_proj,
        ssln_0,
    )


def process_ssgmlp(
    subgraph: list[onnx.NodeProto], extractor: onnx.utils.Extractor
) -> tuple[list[str], list[str], onnx.NodeProto, onnx.NodeProto, onnx.NodeProto, onnx.NodeProto]:
    if len(subgraph) == 9:
        sln_0 = subgraph[0]
        ssln_0 = subgraph[1]
        gate_proj = subgraph[2]
        up_proj = subgraph[3]
        down_proj = subgraph[6]
        sln_1 = subgraph[7]
        ssln_1 = subgraph[8]
    else:
        sln_0 = subgraph[1]
        ssln_0 = subgraph[2]
        gate_proj = subgraph[4]
        up_proj = subgraph[5]
        down_proj = subgraph[8]
        sln_1 = subgraph[10]
        ssln_1 = subgraph[11]
    first_node = subgraph[0]
    last_node = subgraph[-1]
    assert len(gate_proj.input) == 6
    assert len(up_proj.input) == 6
    assert len(down_proj.input) == 6

    assert len(ssln_0.input) == 3

    assert len(ssln_0.output) == 4
    # the last SSLN has a single output
    assert len(ssln_1.output) in [1, 4]

    epsilon = onnx.helper.get_node_attr_value(sln_0, "epsilon")
    epsilon_1 = onnx.helper.get_node_attr_value(ssln_0, "epsilon")
    epsilon_2 = onnx.helper.get_node_attr_value(sln_1, "epsilon")
    epsilon_3 = onnx.helper.get_node_attr_value(ssln_1, "epsilon")
    assert epsilon == epsilon_1 == epsilon_2 == epsilon_3

    check_biases(gate_proj, up_proj, down_proj, extractor)

    before_cast_inputs = [
        ssln_0.input[0],
        first_node.input[0],
        sln_0.input[1],
        ssln_0.input[2],
        # matmul initializer
        *gate_proj.input[1:4],
        *up_proj.input[1:4],
        *down_proj.input[1:4],
        sln_1.input[1],
        # last ssln initializer
        ssln_1.input[2],
    ]
    before_cast_outputs = get_before_cast_outputs(ssln_1)
    before_cast_outputs[0] = last_node.output[0]

    return (
        before_cast_inputs,
        before_cast_outputs,
        gate_proj,
        up_proj,
        down_proj,
        ssln_0,
    )


def check_biases(
    gate_proj: onnx.NodeProto,
    up_proj: onnx.NodeProto,
    down_proj: onnx.NodeProto,
    extractor: onnx.utils.Extractor,
) -> None:
    # current SSMLP/SSGMLP custom ops requires no biases
    gate_bias = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gate_proj.input[5], extractor)
    assert not gate_bias.any()
    up_bias = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(up_proj.input[5], extractor)
    assert not up_bias.any()
    down_bias = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(down_proj.input[5], extractor)
    assert not down_bias.any()


def add_casts(
    before_cast_inputs: list[str],
    before_cast_outputs: list[str],
    pass_id: str,
    domain: str,
    extractor: onnx.utils.Extractor,
) -> tuple[list[onnx.NodeProto], list[onnx.ValueInfoProto], list[str], list[str]]:
    new_nodes = []
    new_tvis = []

    pre_cast_0, pre_tvi_0 = cast.add_cast_dtype_to_bfloat16_auto(before_cast_inputs[0], pass_id, domain, extractor)
    # this is used to match and remove casts in hybrid_llm_add_cast_attributes
    pre_cast_0[0].name += ".hybrid_llm_0"
    new_nodes.extend(pre_cast_0)
    new_tvis.extend(pre_tvi_0)

    pre_cast_1, pre_tvi_1 = cast.add_cast_dtype_to_bfloat16_auto(before_cast_inputs[1], pass_id, domain, extractor)
    pre_cast_1[0].name += ".hybrid_llm_0"
    new_nodes.extend(pre_cast_1)
    new_tvis.extend(pre_tvi_1)

    post_cast_0, post_tvi_0 = cast.add_cast_bfloat16_to_dtype_auto(before_cast_outputs[0], pass_id, domain, extractor)
    post_cast_0[0].name += ".hybrid_llm_1"
    new_nodes.extend(post_cast_0)
    new_tvis.extend(post_tvi_0)

    if len(before_cast_outputs) > 1:
        post_cast_1, post_tvi_1 = cast.add_cast_bfloat16_to_dtype_auto(
            before_cast_outputs[1], pass_id, domain, extractor
        )
        post_cast_1[0].name += ".hybrid_llm_1"
        new_tvis.extend(post_tvi_1)
        new_nodes.extend(post_cast_1)

    new_inputs = [
        pre_cast_0[0].output[0],
        pre_cast_1[0].output[0],
        *before_cast_inputs[2:],
    ]
    new_outputs = [post_cast_0[0].input[0]]
    if len(before_cast_outputs) > 1:
        new_outputs.append(post_cast_1[0].input[0])

    return new_nodes, new_tvis, new_inputs, new_outputs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    is_ssmlp = len(subgraph) == 8

    if is_ssmlp:
        (
            before_cast_inputs,
            before_cast_outputs,
            gate_proj,
            up_proj,
            down_proj,
            ssln_0,
        ) = process_ssmlp(subgraph, extractor)
        op_type = "SSMLP"
        node_name = f"ssmlp_{pass_id}"
    else:
        (
            before_cast_inputs,
            before_cast_outputs,
            gate_proj,
            up_proj,
            down_proj,
            ssln_0,
        ) = process_ssgmlp(subgraph, extractor)
        op_type = "SSGMLP"
        node_name = f"ssgmlp_{pass_id}"
    domain = params.get_domain(op_type)

    new_nodes, new_tvis, new_inputs, new_outputs = add_casts(
        before_cast_inputs, before_cast_outputs, pass_id, domain, extractor
    )

    new_node = onnx.helper.make_node(op_type, inputs=new_inputs, outputs=new_outputs, name=node_name, domain=domain)

    epsilon = onnx.helper.get_node_attr_value(ssln_0, "epsilon")
    ryzenai_onnx_utils.matcher.add_attribute(new_node, "epsilon", epsilon)

    for attr in gate_proj.attribute:
        attr.name = f"gate_{attr.name}"
    ryzenai_onnx_utils.matcher.copy_attributes(gate_proj, new_node)

    for attr in up_proj.attribute:
        attr.name = f"up_{attr.name}"
    ryzenai_onnx_utils.matcher.copy_attributes(up_proj, new_node)

    for attr in down_proj.attribute:
        attr.name = f"down_{attr.name}"
    ryzenai_onnx_utils.matcher.copy_attributes(down_proj, new_node)

    if not is_ssmlp:
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "has_gelu", True)
    new_nodes.append(new_node)

    return new_nodes, [], new_tvis


PATTERN = [
    SubPass(
        "SSMLP",
        [
            "SkipSimplifiedLayerNormalization([?,?,?], [a0,?,?,a1])",
            "MatMulNBits([a0,?,?,?], [a2])",
            "Sigmoid(a2, a3)",
            "Mul([a2,a3], a4)",
            "MatMulNBits([a0,?,?,?], [a5])",
            "Mul([a4,a5], a6)",
            "MatMulNBits([a6,?,?,?], [a7])",
            "SkipSimplifiedLayerNormalization([a1,a7,?], [?,?,?,?])",
        ],
    ),
    SubPass(
        "SSGMLP",
        [
            "SimplifiedLayerNormalization([?,?],a2)",
            "SkipSimplifiedLayerNormalization([?,a2,?],[a5,?,?,a7])",
            "MatMulNBits([a5,?,?,?],a11)",
            "MatMulNBits([a5,?,?,?],a15)",
            "Gelu(a11,a16)",
            "Mul([a16,a15],a17)",
            "MatMulNBits([a17,?,?,?],a21)",
            "SimplifiedLayerNormalization([a21,?],a23)",
            "SkipSimplifiedLayerNormalization([a7,a23,?],[?,?,?,?])",
        ],
    ),
    SubPass(
        "SSGMLPCast",
        [
            "Cast(?,a0)",
            "SimplifiedLayerNormalization([a0,?],a2)",
            "SkipSimplifiedLayerNormalization([?,a2,?],[a5,?,?,a7])",
            "Cast(a5,a55)",
            "MatMulNBits([a55,?,?,?],a11)",
            "MatMulNBits([a55,?,?,?],a15)",
            "Gelu(a11,a16)",
            "Mul([a16,a15],a17)",
            "MatMulNBits([a17,?,?,?],a21)",
            "Cast(a21,a211)",
            "SimplifiedLayerNormalization([a211,?],a23)",
            "SkipSimplifiedLayerNormalization([a7,a23,?],[a24,?,?,?])",
            "Cast(a24,?)",
        ],
    ),
]
REPLACEMENT = replacement
