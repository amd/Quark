# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    act_domain = params.get_domain("ActMatmul_noqdq")
    softmax_domain = params.get_domain("Concat_noqdq")
    mha = subgraph[0]

    assert len(mha.input) == 3  # there's the redundant qkv in the input as well
    assert len(mha.output) == 1

    initializers: list[onnx.TensorProto] = []

    tvis = []
    input_shape_0, input_shape_1, input_shape_2 = ryzenai_onnx_utils.matcher.get_shapes(mha.input, extractor)
    (output_shape,) = ryzenai_onnx_utils.matcher.get_shapes(mha.output, extractor)

    suffix = f"{pass_id}"
    pre_cast_output_q = mha.input[0] + f":.out{suffix}"
    pre_cast_output_k = mha.input[1] + f":.out{suffix}"
    pre_cast_output_v = mha.input[2] + f":.out{suffix}"
    # transpose_output_v = pre_cast_output_v + f".out{suffix}"
    pre_cast_q, pre_cast_tvi_q = add_cast_to_bf16(mha.input[0], pre_cast_output_q, input_shape_0, act_domain)
    pre_cast_k, pre_cast_tvi_k = add_cast_to_bf16(mha.input[1], pre_cast_output_k, input_shape_1, act_domain)
    pre_cast_v, pre_cast_tvi_v = add_cast_to_bf16(mha.input[2], pre_cast_output_v, input_shape_2, act_domain)
    # transpose_v, transpose_tvi_v = add_transpose(
    #     pre_cast_output_v,
    #     transpose_output_v,
    #     onnx.TensorProto.BFLOAT16,
    #     [1, mv, nv],
    #     [1, nv, mv],
    #     [0, 2, 1],
    # )
    tvis.extend(pre_cast_tvi_q)
    tvis.extend(pre_cast_tvi_k)
    tvis.extend(pre_cast_tvi_v)
    # tvis.extend(transpose_tvi_v)

    actmatmul_sm_inputs = [pre_cast_output_q, pre_cast_output_k]
    actmatmul_sm_output = mha.output[0] + f".out{suffix}"
    act_sm_out_shape = [1, input_shape_0[1], input_shape_1[1]]
    output_tvi = onnx.helper.make_tensor_value_info(actmatmul_sm_output, onnx.TensorProto.BFLOAT16, act_sm_out_shape)
    tvis.extend([output_tvi])
    actmatmul_sm_node = onnx.helper.make_node(
        "ActMatmul_noqdq",
        inputs=actmatmul_sm_inputs,
        outputs=[actmatmul_sm_output],
        domain=act_domain,
        name=f"ActMatmul_{suffix}_0",
    )
    softmax_inputs = [actmatmul_sm_output]
    softmax_output = actmatmul_sm_output + f".out{suffix}"
    softmax_out_shape = [1, input_shape_0[1], input_shape_1[1]]
    output_tvi_softmax = onnx.helper.make_tensor_value_info(
        softmax_output, onnx.TensorProto.BFLOAT16, softmax_out_shape
    )
    tvis.extend([output_tvi_softmax])
    softmax_node = onnx.helper.make_node(
        "Softmax_noqdq",
        inputs=softmax_inputs,
        outputs=[softmax_output],
        domain=softmax_domain,
        name=f"Softmax_{suffix}",
    )
    actmatmul_inputs = [softmax_output, pre_cast_output_v]
    actmatmul_output = softmax_output + f".out{suffix}"
    output_tvi_sm = onnx.helper.make_tensor_value_info(actmatmul_output, onnx.TensorProto.BFLOAT16, output_shape)
    tvis.extend([output_tvi_sm])
    actmatmul_node = onnx.helper.make_node(
        "ActMatmul_noqdq",
        inputs=actmatmul_inputs,
        outputs=[actmatmul_output],
        domain=act_domain,
        name=f"ActMatmul_{suffix}_1",
    )

    post_cast, post_cast_tvi = add_cast_to_float(actmatmul_output, mha.output[0], output_shape, act_domain)
    tvis.extend(post_cast_tvi)

    return (
        [
            *pre_cast_q,
            *pre_cast_k,
            *pre_cast_v,
            actmatmul_sm_node,
            softmax_node,
            actmatmul_node,
            *post_cast,
        ],
        initializers,
        tvis,
    )


PATTERN = ["MultiHeadAttention([?,?,?], ?)"]
REPLACEMENT = replacement
