# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import math

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape
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

    input_shape = ryzenai_onnx_utils.matcher.get_shape(extractor.vimap[layer_norm.input[0]])
    output_shape = ryzenai_onnx_utils.matcher.get_shape(extractor.vimap[layer_norm.output[0]])
    assert is_static_shape(output_shape)

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
    layernorm_output = layer_norm.output[0] + f".out{pass_id}"
    bfp_out_size = int(math.prod(output_shape) / 8 * 9)
    output_tvi = onnx.helper.make_tensor_value_info(layernorm_output, onnx.TensorProto.UINT8, [1, bfp_out_size])
    layernorm_node = onnx.helper.make_node(
        "LayerNorm_noqdq",
        inputs=new_inputs,
        outputs=[layernorm_output],
        domain=domain,
        name=layer_norm.name,
        bfp16_tensors=[layernorm_output],
        bfp16_shape_0=output_shape,
    )
    tvis.extend([output_tvi])

    bfp_to_bf_input_tvi = onnx.helper.make_tensor_value_info(
        layernorm_output, onnx.TensorProto.UINT8, [1, bfp_out_size]
    )
    bfp_to_bf_output = layer_norm.output[0] + f".out{pass_id}_bfp_bf"
    bfp_to_bf_output_tvi = onnx.helper.make_tensor_value_info(bfp_to_bf_output, onnx.TensorProto.BFLOAT16, output_shape)
    bfp_to_bf_tvi_v = [bfp_to_bf_input_tvi, bfp_to_bf_output_tvi]
    tvis.extend(bfp_to_bf_tvi_v)

    # Dummy node which will be removed in scope of simplify_bfps
    bfp_to_bf_node = onnx.helper.make_node(
        "BFP16_to_BF16",
        inputs=[layernorm_output],
        outputs=[bfp_to_bf_output],
        domain=domain,
        name=f"bfp16_to_bf16_{pass_id}",
        bfp16_tensors=[layernorm_output],
        bfp16_shape_0=output_shape,
    )

    post_cast, post_cast_tvi = add_cast_to_float(
        bfp_to_bf_output,
        layer_norm.output[0],
        output_shape,
        domain,
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast, layernorm_node, bfp_to_bf_node, *post_cast], initializers, tvis


PATTERN = ["LayerNormalization([?,?,?],?)"]
REPLACEMENT = replacement
