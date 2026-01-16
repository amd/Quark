# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs
from ryzenai_onnx_utils.utils import float_numpy_to_bfloat_tensor


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("Concat_noqdq")
    layer_norm = subgraph[0]

    assert len(layer_norm.input) == 3
    assert len(layer_norm.output) == 1

    # layer_norm_successors = ryzenai_onnx_utils.matcher.find_nodes_by_input(
    #     layer_norm.output[0], extractor.graph
    # )
    # if len(layer_norm_successors) > 1 or ryzenai_onnx_utils.matcher.is_output_edge(
    #     layer_norm.output[0], extractor.graph
    # ):
    #     return subgraph, [], None

    input_shape = ryzenai_onnx_utils.matcher.get_shape(extractor.vimap[layer_norm.input[0]])
    output_shape = ryzenai_onnx_utils.matcher.get_shape(extractor.vimap[layer_norm.output[0]])

    # m, k, n = get_matmul_params(shapes[matmul.name])
    flatten = True
    gamma_name = layer_norm.input[1]
    gamma_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gamma_name, extractor)
    gamma = float_numpy_to_bfloat_tensor(gamma_f, gamma_name, flatten)
    beta_name = layer_norm.input[2]
    beta_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(beta_name, extractor)
    beta = float_numpy_to_bfloat_tensor(beta_f, beta_name, flatten)
    initializers = [gamma, beta]

    tvis = []

    pre_cast_output = layer_norm.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_to_bf16(layer_norm.input[0], pre_cast_output, input_shape, domain)
    tvis.extend(pre_cast_tvi)

    new_inputs = [pre_cast_output, gamma.name, beta.name]
    matmul_output = layer_norm.output[0] + f".out{pass_id}"
    layernorm_node = onnx.helper.make_node(
        "LayerNorm_noqdq",
        inputs=new_inputs,
        outputs=[matmul_output],
        domain=domain,
        name=layer_norm.name,
    )

    post_cast, post_cast_tvi = add_cast_to_float(
        layer_norm.output[0] + f".out{pass_id}",
        layer_norm.output[0],
        output_shape,
        domain,
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast, layernorm_node, *post_cast], initializers, tvis


PATTERN = ["LayerNormalization([?,?,?],?)"]
REPLACEMENT = replacement
