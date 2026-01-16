# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import add_attribute, copy_attributes
from ryzenai_onnx_utils.transform.cast import (
    add_cast_to_bf16,
    add_cast_to_float,
)
from ryzenai_onnx_utils.typing import PassOutputArgs
from ryzenai_onnx_utils.utils import float_numpy_to_bfloat_tensor


def is_resize_supported(a_shape: tuple[int | str, ...], op_namespace: str) -> bool:
    supported_shapes = {
        "sdxlt": {
            (64, 64, 512),
        }
    }

    if len(a_shape) == 4:
        if a_shape[0] != 1:
            return False
        a_shape = a_shape[1:]
    elif len(a_shape) != 3:
        return False

    return a_shape in supported_shapes[op_namespace]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("GroupNorm_noqdq")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    ln_node = subgraph[3]
    mul_node = subgraph[5]
    add_node = subgraph[6]
    init_transpose = subgraph[0]
    lone_cast_node = subgraph[8]
    rm_transpose = subgraph[9]

    top_cast_node = ryzenai_onnx_utils.matcher.find_nodes_by_output(init_transpose.input[0], extractor.graph)[0]
    lone_cast_node.input[0] = top_cast_node.output[0]

    bot_node = ryzenai_onnx_utils.matcher.find_nodes_by_input(rm_transpose.output[0], extractor.graph)[0]
    ix = 0
    for index, i in enumerate(bot_node.input):
        if i == rm_transpose.output[0]:
            ix = index
            break
    bot_node.input[ix] = lone_cast_node.output[0]
    input_list = []
    input_list.append(init_transpose.input[0])
    (input_shape,) = ryzenai_onnx_utils.matcher.get_shapes(input_list, extractor)
    output_list = []
    output_list.append(subgraph[7].output[0])
    (output_shape,) = ryzenai_onnx_utils.matcher.get_shapes(output_list, extractor)

    if not is_resize_supported(input_shape, op_namespace):
        return subgraph, [], None

    flatten = True
    gamma_name = mul_node.input[1]
    gamma_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gamma_name, extractor)
    gamma = float_numpy_to_bfloat_tensor(gamma_f, gamma_name, flatten)
    beta_name = add_node.input[1]
    beta_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(beta_name, extractor)
    beta = float_numpy_to_bfloat_tensor(beta_f, beta_name, flatten)

    initializers = [gamma, beta]
    tvis = []

    pre_cast_output = init_transpose.input[0] + f".out_{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_to_bf16(init_transpose.input[0], pre_cast_output, input_shape, domain)
    tvis.extend(pre_cast_tvi)

    new_inputs = [pre_cast_output, gamma.name, beta.name]
    tr_node_output = subgraph[7].output[0] + f".out_{pass_id}"
    dd_node = onnx.helper.make_node(
        "GroupNorm_noqdq",
        inputs=new_inputs,
        outputs=[tr_node_output],
        domain=domain,
        name=f"GroupNorm_noqdq_{pass_id}",
    )
    copy_attributes(ln_node, dd_node)
    add_attribute(dd_node, "activation", 0)
    add_attribute(dd_node, "channels_last", 1)
    add_attribute(dd_node, "groups", 32)

    post_cast, post_cast_tvi = add_cast_to_float(tr_node_output, subgraph[7].output[0], output_shape, domain)
    tvis.extend(post_cast_tvi)

    return [*pre_cast, dd_node, *post_cast, lone_cast_node], initializers, tvis


PATTERN = [
    "Transpose(?, u0)",
    "Reshape(u0,a0)",
    "Reshape(a0,b0)",
    "InstanceNormalization(b0,c0)",
    "Reshape(c0,d0)",
    "Mul(d0,e0)",
    "Add(e0,f0)",
    "Transpose(f0, ?)",
    "CastAvx(u0, g0)",
    "Transpose(g0, ?)",
]
REPLACEMENT = replacement
