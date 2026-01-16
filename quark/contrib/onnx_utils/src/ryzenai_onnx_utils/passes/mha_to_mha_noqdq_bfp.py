# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.passes.bfp_matmul_noqdq import get_matmul_params
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_mha_supported_by_kernel(mq: int, nq: int, mk: int, nk: int, mv: int, nv: int, op_namespace: str) -> bool:
    supported_shapes = {
        "sdxlt": {
            (1024, 640, 1024, 640, 1024, 640),
            (1024, 640, 77, 640, 77, 640),
            (256, 1280, 256, 1280, 256, 1280),
            (256, 1280, 77, 1280, 77, 1280),
        }
    }
    return (mq, nq, mk, nk, mv, nv) in supported_shapes[op_namespace]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("MHA_noqdq")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    mha = subgraph[0]

    assert len(mha.input) == 4  # there's the redundant qkv_bias in the input as well
    assert len(mha.output) == 1

    mha_pre_q_matmul = ryzenai_onnx_utils.matcher.find_nodes_by_output(mha.input[0], extractor.graph)
    mha_pre_k_matmul = ryzenai_onnx_utils.matcher.find_nodes_by_output(mha.input[1], extractor.graph)
    mha_pre_v_matmul = ryzenai_onnx_utils.matcher.find_nodes_by_output(mha.input[2], extractor.graph)
    if len(mha_pre_q_matmul) != 1 or mha_pre_q_matmul[0].op_type != "MatMul":
        return subgraph, [], None
    if len(mha_pre_k_matmul) != 1 or mha_pre_k_matmul[0].op_type != "MatMul":
        return subgraph, [], None
    if len(mha_pre_v_matmul) != 1 or mha_pre_v_matmul[0].op_type != "MatMul":
        return subgraph, [], None

    mq, kq, nq, mq_pad = get_matmul_params(mha_pre_q_matmul[0], extractor)
    mk, kk, nk, mk_pad = get_matmul_params(mha_pre_k_matmul[0], extractor)
    mv, kv, nv, mv_pad = get_matmul_params(mha_pre_v_matmul[0], extractor)

    if not is_mha_supported_by_kernel(mq, nq, mk, nk, mv, nv, op_namespace):
        return subgraph, [], None

    # Annotate the v transpose to add transpose in later pass.
    mha_pre_v_matmul[0].name += "_(MatMul_transpose)"

    initializers: list[onnx.TensorProto] = []

    tvis = []

    pre_cast_output_q = mha_pre_q_matmul[0].name + f":.out{pass_id}.bfp_pre"
    pre_cast_output_k = mha_pre_k_matmul[0].name + f":.out{pass_id}.bfp_pre"
    pre_cast_output_v = mha_pre_v_matmul[0].name + f":.out{pass_id}.bfp_pre"
    pre_cast_q, pre_cast_tvi_q = add_cast_to_bf16(mha.input[0], pre_cast_output_q, [1, mq, nq], domain)
    pre_cast_k, pre_cast_tvi_k = add_cast_to_bf16(mha.input[1], pre_cast_output_k, [1, mk, nk], domain)
    pre_cast_v, pre_cast_tvi_v = add_cast_to_bf16(mha.input[2], pre_cast_output_v, [1, nv, mv], domain)
    tvis.extend(pre_cast_tvi_q)
    tvis.extend(pre_cast_tvi_k)
    tvis.extend(pre_cast_tvi_v)

    bfp_inputs_q = [pre_cast_output_q]
    bfp_inputs_k = [pre_cast_output_k]
    bfp_inputs_v = [pre_cast_output_v]

    bfp_output_q = mha_pre_q_matmul[0].name + f".out{pass_id}.bfp_post"
    bfp_output_k = mha_pre_k_matmul[0].name + f".out{pass_id}.bfp_post"
    bfp_output_v = mha_pre_v_matmul[0].name + f".out{pass_id}.bfp_post"

    input_tvi_q = onnx.helper.make_tensor_value_info(pre_cast_output_q, onnx.TensorProto.BFLOAT16, [1, mq, nq])
    output_tvi_q = onnx.helper.make_tensor_value_info(
        bfp_output_q, onnx.TensorProto.UINT8, [1, int(mq_pad * nq / 8 * 9)]
    )
    bfp_tvi_q = [input_tvi_q, output_tvi_q]
    tvis.extend(bfp_tvi_q)

    input_tvi_k = onnx.helper.make_tensor_value_info(pre_cast_output_k, onnx.TensorProto.BFLOAT16, [1, mk, nk])
    output_tvi_k = onnx.helper.make_tensor_value_info(
        bfp_output_k, onnx.TensorProto.UINT8, [1, int(mk_pad * nk / 8 * 9)]
    )
    bfp_tvi_k = [input_tvi_k, output_tvi_k]
    tvis.extend(bfp_tvi_k)

    input_tvi_v = onnx.helper.make_tensor_value_info(pre_cast_output_v, onnx.TensorProto.BFLOAT16, [1, nv, mv])
    output_tvi_v = onnx.helper.make_tensor_value_info(
        bfp_output_v, onnx.TensorProto.UINT8, [1, int(mv_pad * nv / 8 * 9)]
    )
    bfp_tvi_v = [input_tvi_v, output_tvi_v]
    tvis.extend(bfp_tvi_v)

    bfp16_node_q = onnx.helper.make_node(
        "BF16_to_BFP16",
        inputs=bfp_inputs_q,
        outputs=[bfp_output_q],
        domain=domain,
        name=f"bf16_to_bfp16_{pass_id}_q",
        bfp16_tensors=[bfp_output_q],
        bfp16_shape_0=[1, mq_pad, nq],
    )
    bfp16_node_k = onnx.helper.make_node(
        "BF16_to_BFP16",
        inputs=bfp_inputs_k,
        outputs=[bfp_output_k],
        domain=domain,
        name=f"bf16_to_bfp16_{pass_id}_k",
        bfp16_tensors=[bfp_output_k],
        bfp16_shape_0=[1, mk_pad, nk],
    )
    bfp16_node_v = onnx.helper.make_node(
        "BF16_to_BFP16",
        inputs=bfp_inputs_v,
        outputs=[bfp_output_v],
        domain=domain,
        name=f"bf16_to_bfp16_{pass_id}_v",
        bfp16_tensors=[bfp_output_v],
        bfp16_shape_0=[1, nv, mv_pad],
    )

    new_inputs = [bfp_output_q, bfp_output_k, bfp_output_v]
    mha_output = mha.output[0] + f".out{pass_id}"
    output_tvi = onnx.helper.make_tensor_value_info(mha_output, onnx.TensorProto.UINT8, [1, int(mq * nv / 8 * 9)])
    mha_node = onnx.helper.make_node(
        "MHA_noqdq",
        inputs=new_inputs,
        outputs=[mha_output],
        domain=domain,
        name=mha.name,
        bfp16_tensors=[bfp_output_q, bfp_output_k, bfp_output_v, mha_output],
        bfp16_shape_0=[1, mq_pad, nq],
        bfp16_shape_1=[1, mk_pad, nk],
        bfp16_shape_2=[1, nv, mv_pad],
        bfp16_shape_3=[1, mq, nv],
    )
    tvis.extend([output_tvi])

    bfp_to_bf_input_tvi = onnx.helper.make_tensor_value_info(
        mha_output, onnx.TensorProto.UINT8, [1, int(mq * nv / 8 * 9)]
    )
    bfp_to_bf_output = mha.output[0] + f".out{pass_id}_bfp_bf"
    bfp_to_bf_output_tvi = onnx.helper.make_tensor_value_info(bfp_to_bf_output, onnx.TensorProto.BFLOAT16, [1, mq, nv])
    bfp_to_bf_tvi_v = [bfp_to_bf_input_tvi, bfp_to_bf_output_tvi]
    tvis.extend(bfp_to_bf_tvi_v)

    # Dummy node which will be removed in scope of simplify_bfps
    bfp_to_bf_node = onnx.helper.make_node(
        "BFP16_to_BF16",
        inputs=[mha_output],
        outputs=[bfp_to_bf_output],
        domain=domain,
        name=f"bfp16_to_bf16_{pass_id}",
        bfp16_tensors=[mha_output],
        bfp16_shape_0=[1, mq, nv],
    )

    post_cast, post_cast_tvi = add_cast_to_float(
        bfp_to_bf_output,
        mha.output[0],
        [1, mq, nv],
        domain,
    )
    tvis.extend(post_cast_tvi)

    return (
        [
            *pre_cast_q,
            *pre_cast_k,
            *pre_cast_v,
            bfp16_node_q,
            bfp16_node_k,
            bfp16_node_v,
            mha_node,
            bfp_to_bf_node,
            *post_cast,
        ],
        initializers,
        tvis,
    )


PATTERN = ["MultiHeadAttention([?,?,?,?], ?)"]
REPLACEMENT = replacement
