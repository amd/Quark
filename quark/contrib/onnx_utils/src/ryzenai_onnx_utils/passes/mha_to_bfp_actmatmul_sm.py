# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import math

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape


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

    mha_pre_k_matmul = ryzenai_onnx_utils.matcher.find_nodes_by_output(mha.input[1], extractor.graph)

    # Annotate the v transpose to add transpose in later pass.
    mha_pre_k_matmul[0].name += "_(MatMul_transpose)"
    initializers: list[onnx.TensorProto] = []

    tvis = []
    input_shape_0, input_shape_1, input_shape_2 = ryzenai_onnx_utils.matcher.get_shapes(mha.input, extractor)
    (output_shape,) = ryzenai_onnx_utils.matcher.get_shapes(mha.output, extractor)
    assert (
        is_static_shape(input_shape_0)
        and is_static_shape(input_shape_1)
        and is_static_shape(input_shape_2)
        and is_static_shape(output_shape)
    )

    suffix = f"{pass_id}"
    pre_cast_output_q = mha.input[0] + f":.out{suffix}"
    pre_cast_output_k = mha.input[1] + f":.out{suffix}"
    pre_cast_output_v = mha.input[2] + f":.out{suffix}"
    # transpose_output_v = pre_cast_output_v + f".out{suffix}"
    pre_cast_q, pre_cast_tvi_q = add_cast_to_bf16(mha.input[0], pre_cast_output_q, input_shape_0, act_domain)
    pre_cast_k, pre_cast_tvi_k = add_cast_to_bf16(mha.input[1], pre_cast_output_k, input_shape_1, act_domain)
    pre_cast_v, pre_cast_tvi_v = add_cast_to_bf16(mha.input[2], pre_cast_output_v, [1, 512, 4096], act_domain)
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

    bfp_inputs_q = [pre_cast_output_q]
    bfp_inputs_k = [pre_cast_output_k]
    bfp_inputs_v = [pre_cast_output_v]

    bfp_output_q = mha.input[0] + f".out{pass_id}.bfp_post"
    bfp_output_k = mha.input[1] + f".out{pass_id}.bfp_post"
    bfp_output_v = mha.input[2] + f".out{pass_id}.bfp_post"
    input_tvi_q = onnx.helper.make_tensor_value_info(pre_cast_output_q, onnx.TensorProto.BFLOAT16, input_shape_0)
    bfp_out_size_q = int(math.prod(input_shape_0) / 8 * 9)
    output_tvi_q = onnx.helper.make_tensor_value_info(bfp_output_q, onnx.TensorProto.UINT8, [1, bfp_out_size_q])
    bfp_tvi_q = [input_tvi_q, output_tvi_q]
    tvis.extend(bfp_tvi_q)

    input_tvi_k = onnx.helper.make_tensor_value_info(pre_cast_output_k, onnx.TensorProto.BFLOAT16, input_shape_1)
    bfp_out_size_k = int(math.prod(input_shape_1) / 8 * 9)
    output_tvi_k = onnx.helper.make_tensor_value_info(bfp_output_k, onnx.TensorProto.UINT8, [1, bfp_out_size_k])
    bfp_tvi_k = [input_tvi_k, output_tvi_k]
    tvis.extend(bfp_tvi_k)
    input_tvi_v = onnx.helper.make_tensor_value_info(pre_cast_output_v, onnx.TensorProto.BFLOAT16, [1, 512, 4096])
    bfp_out_size_v = int(math.prod(input_shape_2) / 8 * 9)
    output_tvi_v = onnx.helper.make_tensor_value_info(bfp_output_v, onnx.TensorProto.UINT8, [1, bfp_out_size_v])
    bfp_tvi_v = [input_tvi_v, output_tvi_v]
    tvis.extend(bfp_tvi_v)

    bfp16_node_q = onnx.helper.make_node(
        "BF16_to_BFP16",
        inputs=bfp_inputs_q,
        outputs=[bfp_output_q],
        domain=act_domain,
        name=f"bf16_to_bfp16_{pass_id}_q",
        bfp16_tensors=[bfp_output_q],
        bfp16_shape_0=input_shape_0,
    )
    bfp16_node_k = onnx.helper.make_node(
        "BF16_to_BFP16",
        inputs=bfp_inputs_k,
        outputs=[bfp_output_k],
        domain=act_domain,
        name=f"bf16_to_bfp16_{pass_id}_k",
        bfp16_tensors=[bfp_output_k],
        bfp16_shape_0=input_shape_1,
    )
    bfp16_node_v = onnx.helper.make_node(
        "BF16_to_BFP16",
        inputs=bfp_inputs_v,
        outputs=[bfp_output_v],
        domain=act_domain,
        name=f"bf16_to_bfp16_{pass_id}_v",
        bfp16_tensors=[bfp_output_v],
        bfp16_shape_0=input_shape_2,
    )

    actmatmul_sm_inputs = [bfp_output_q, bfp_output_k]
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
        bfp16_tensors=[bfp_output_q, bfp_output_k],
        bfp16_shape_0=input_shape_0,
        bfp16_shape_1=input_shape_1,
    )
    softmax_inputs = [actmatmul_sm_output]
    softmax_output = actmatmul_sm_output + f".out{suffix}_softmax"
    softmax_out_shape = [1, input_shape_0[1], input_shape_1[1]]
    # bfp_out_size_sm = int(math.prod(softmax_out_shape) / 8 * 9)
    # output_tvi_softmax = onnx.helper.make_tensor_value_info(
    #    softmax_output, onnx.TensorProto.BFLOAT16, softmax_out_shape
    # )
    # tvis.extend([output_tvi_softmax])
    softmax_node = onnx.helper.make_node(
        "Softmax_noqdq",
        inputs=softmax_inputs,
        outputs=[softmax_output],
        domain=softmax_domain,
        name=f"Softmax_{suffix}",
    )
    bfp_output_act = softmax_output + f".out{pass_id}.bfp16"
    input_tvi = onnx.helper.make_tensor_value_info(softmax_output, onnx.TensorProto.BFLOAT16, softmax_out_shape)
    bfp_out_size_sm = int(math.prod(softmax_out_shape) / 8 * 9)
    output_tvi = onnx.helper.make_tensor_value_info(bfp_output_act, onnx.TensorProto.UINT8, [1, bfp_out_size_sm])
    bfp_tvi = [input_tvi, output_tvi]
    tvis.extend(bfp_tvi)
    bfp16_node = onnx.helper.make_node(
        "BF16_to_BFP16",
        inputs=[softmax_output],
        outputs=[bfp_output_act],
        domain=act_domain,
        name=f"bf16_to_bfp16_{pass_id}_act",
        bfp16_tensors=[bfp_output_act],
        bfp16_shape_0=softmax_out_shape,
    )
    actmatmul_inputs = [bfp_output_act, bfp_output_v]
    actmatmul_output = softmax_output + f".out{suffix}"
    bfp_out_size_act = int(math.prod(output_shape) / 8 * 9)
    output_tvi_sm = onnx.helper.make_tensor_value_info(actmatmul_output, onnx.TensorProto.UINT8, [1, bfp_out_size_act])
    tvis.extend([output_tvi_sm])
    actmatmul_node = onnx.helper.make_node(
        "ActMatmul_noqdq",
        inputs=actmatmul_inputs,
        outputs=[actmatmul_output],
        domain=act_domain,
        name=f"ActMatmul_{suffix}_1",
        bfp16_tensors=[bfp_output_act, bfp_output_v, actmatmul_output],
        bfp16_shape_0=softmax_out_shape,
        bfp16_shape_1=input_shape_2,
        bfp16_shape_2=output_shape,
    )

    bfp_to_bf_input_tvi = onnx.helper.make_tensor_value_info(
        actmatmul_output, onnx.TensorProto.UINT8, [1, bfp_out_size_act]
    )
    bfp_to_bf_output = mha.output[0] + f".out{pass_id}_bfp_bf"
    bfp_to_bf_output_tvi = onnx.helper.make_tensor_value_info(bfp_to_bf_output, onnx.TensorProto.BFLOAT16, output_shape)
    bfp_to_bf_tvi_v = [bfp_to_bf_input_tvi, bfp_to_bf_output_tvi]
    tvis.extend(bfp_to_bf_tvi_v)

    # Dummy node which will be removed in scope of simplify_bfps
    bfp_to_bf_node = onnx.helper.make_node(
        "BFP16_to_BF16",
        inputs=[actmatmul_output],
        outputs=[bfp_to_bf_output],
        domain=act_domain,
        name=f"bfp16_to_bf16_{pass_id}",
        bfp16_tensors=[actmatmul_output],
        bfp16_shape_0=output_shape,
    )

    post_cast, post_cast_tvi = add_cast_to_float(bfp_to_bf_output, mha.output[0], output_shape, act_domain)
    tvis.extend(post_cast_tvi)

    return (
        [
            *pre_cast_q,
            *pre_cast_k,
            *pre_cast_v,
            bfp16_node_q,
            bfp16_node_k,
            bfp16_node_v,
            actmatmul_sm_node,
            softmax_node,
            bfp16_node,
            actmatmul_node,
            bfp_to_bf_node,
            *post_cast,
        ],
        initializers,
        tvis,
    )


PATTERN = ["MultiHeadAttention([?,?,?], ?)"]
REPLACEMENT = replacement
