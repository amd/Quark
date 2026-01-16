# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import copy_attributes
from ryzenai_onnx_utils.transform.cast import (
    add_cast_to_bf16,
    add_cast_to_float,
    add_cast_to_float_and_add_silu,
)
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape
from ryzenai_onnx_utils.utils import float_numpy_to_bfloat_tensor


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("Concat_noqdq")
    groupnorm = subgraph[0]
    assert len(groupnorm.input) == 3
    assert len(groupnorm.output) == 1

    try:
        activation = onnx.helper.get_node_attr_value(groupnorm, "activation")
    except ValueError:
        activation = 0

    try:
        channels_last = onnx.helper.get_node_attr_value(groupnorm, "channels_last")
    except ValueError:
        channels_last = 1

    input_shape = ryzenai_onnx_utils.matcher.get_shape(groupnorm.input[0], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(groupnorm.output[0], extractor)
    assert is_static_shape(input_shape) and is_static_shape(output_shape)
    [_batch, n, m, k] = input_shape
    flatten = True
    gamma_name = groupnorm.input[1]
    gamma_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gamma_name, extractor)
    gamma = float_numpy_to_bfloat_tensor(gamma_f, gamma_name, flatten)
    beta_name = groupnorm.input[2]
    beta_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(beta_name, extractor)
    beta = float_numpy_to_bfloat_tensor(beta_f, beta_name, flatten)
    initializers = [gamma, beta]

    tvis = []

    pre_cast_output = groupnorm.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_to_bf16(groupnorm.input[0], pre_cast_output, input_shape, domain)
    new_inputs = []
    tvis.extend(pre_cast_tvi)
    if channels_last == 0:
        transpose_output_in = pre_cast_output + f".out{pass_id}"
        transpose_in, transpose_tvi_in = add_transpose(
            f"Transpose_{pass_id}",
            pre_cast_output,
            transpose_output_in,
            onnx.TensorProto.BFLOAT16,
            [1, n, m, k],
            [1, m, k, n],
            [0, 2, 3, 1],
        )
        tvis.extend(transpose_tvi_in)
        new_inputs.append(transpose_output_in)
    else:
        new_inputs = [pre_cast_output]
    new_inputs.extend([gamma.name, beta.name])

    groupnorm_output = groupnorm.output[0] + f".out{pass_id}"
    groupnorm_node = onnx.helper.make_node(
        "GroupNorm_noqdq",
        inputs=new_inputs,
        outputs=[groupnorm_output],
        domain=domain,
        name=groupnorm.name,
    )
    copy_attributes(groupnorm, groupnorm_node)
    if activation == 1:
        index = int(pass_id.split("_")[-1])
        post_cast, post_cast_tvi = add_cast_to_float_and_add_silu(
            groupnorm_output,
            groupnorm.output[0],
            output_shape,
            index,
            channels_last,
            domain,
        )
        tvis.extend(post_cast_tvi)

        if channels_last == 0:
            return (
                [*pre_cast, transpose_in, groupnorm_node, *post_cast],
                initializers,
                tvis,
            )
        else:
            return [*pre_cast, groupnorm_node, *post_cast], initializers, tvis
    else:
        post_cast, post_cast_tvi = add_cast_to_float(groupnorm_output, groupnorm.output[0], output_shape, domain)
        tvis.extend(post_cast_tvi)
        if channels_last == 0:
            return (
                [*pre_cast, transpose_in, groupnorm_node, *post_cast],
                initializers,
                tvis,
            )
        else:
            return [*pre_cast, groupnorm_node, *post_cast], initializers, tvis


PATTERN = ["GroupNorm([?,?,?], ?)"]
REPLACEMENT = replacement
