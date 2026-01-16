# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    skiplayernorm = subgraph[0]

    new_nodes, new_tvis = [], []

    # standard inputs: [input, skip, gamma, beta, bias]
    input_name = skiplayernorm.input[0]
    skip_name = skiplayernorm.input[1]
    gamma = skiplayernorm.input[2]
    beta = skiplayernorm.input[3]
    bias = skiplayernorm.input[4] if len(skiplayernorm.input) > 4 else None

    add_output = skiplayernorm.output[-1] if len(skiplayernorm.output) == 4 else skiplayernorm.name + "_add_skip"
    layernorm_output = skiplayernorm.output[:3]

    add_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(skiplayernorm.input[0], extractor)
    add_out_shape = ryzenai_onnx_utils.matcher.get_shape(skiplayernorm.input[0], extractor)

    if bias:
        add1_output = skiplayernorm.name + "_add_bias"
        # 1. create Add: input + bias (exists)
        add1 = onnx.helper.make_node(
            "Add",
            inputs=[input_name, bias],
            outputs=[add1_output],
            name=skiplayernorm.name + "_add_bias",
        )
        add1_out_tvi = onnx.helper.make_tensor_value_info(add1_output, add_out_dtype, add_out_shape)
        new_nodes.append(add1)
        new_tvis.append(add1_out_tvi)
        # 2. create Add: input + skip
        add2 = onnx.helper.make_node(
            "Add",
            inputs=[add1_output, skip_name],
            outputs=[add_output],
            name=skiplayernorm.name + "_add_skip",
        )
        add2_out_tvi = onnx.helper.make_tensor_value_info(add_output, add_out_dtype, add_out_shape)
        new_nodes.append(add2)
        new_tvis.append(add2_out_tvi)
    else:
        # 1. create Add: input + skip
        add1 = onnx.helper.make_node(
            "Add",
            inputs=[input_name, skip_name],
            outputs=[add_output],
            name=skiplayernorm.name + "_add_skip",
        )
        add1_out_tvi = onnx.helper.make_tensor_value_info(add_output, add_out_dtype, add_out_shape)
        new_nodes.append(add1)
        new_tvis.append(add1_out_tvi)

    # 3. create LayerNormalization
    layernorm = onnx.helper.make_node(
        "LayerNormalization",
        inputs=[add_output, gamma, beta],
        outputs=layernorm_output,
        name=skiplayernorm.name + "_layernorm",
    )
    # Keep epsilon if exists
    for attr in skiplayernorm.attribute:
        if attr.name == "epsilon":
            layernorm.attribute.extend([attr])
    new_nodes.append(layernorm)

    return new_nodes, [], new_tvis


PATTERN = ["SkipLayerNormalization(?, ?)"]
REPLACEMENT = replacement
