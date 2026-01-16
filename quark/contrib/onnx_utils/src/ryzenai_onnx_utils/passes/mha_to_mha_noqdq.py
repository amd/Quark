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

    # Filter only self-attention, of which the predecessor of q_matmul should be a LayerNorm
    # mha_pre_k_matmul_input = mha_pre_k_matmul[0].input[0]
    # mha_pre_k_matmul_predecessor = ryzenai_onnx_utils.matcher.find_nodes_by_output(
    #     mha_pre_k_matmul_input, extractor.graph
    # )
    # if len(mha_pre_k_matmul_predecessor) == 0:
    #     return subgraph, [], None

    mq, kq, nq, mq_pad = get_matmul_params(mha_pre_q_matmul[0], extractor)
    mk, kk, nk, mk_pad = get_matmul_params(mha_pre_k_matmul[0], extractor)
    mv, kv, nv, mv_pad = get_matmul_params(mha_pre_v_matmul[0], extractor)

    if not is_mha_supported_by_kernel(mq, nq, mk, nk, mv, nv, op_namespace):
        return subgraph, [], None

    # Annotate the v transpose to add transpose in later pass.
    mha_pre_v_matmul[0].name += "_(MatMul_transpose)"

    initializers: list[onnx.TensorProto] = []

    tvis = []

    pre_cast_output_q = mha_pre_q_matmul[0].name + f":.out{pass_id}"
    pre_cast_output_k = mha_pre_k_matmul[0].name + f":.out{pass_id}"
    pre_cast_output_v = mha_pre_v_matmul[0].name + f":.out{pass_id}"
    # transpose_output_v = pre_cast_output_v + f".out{pass_id}"
    pre_cast_q, pre_cast_tvi_q = add_cast_to_bf16(mha.input[0], pre_cast_output_q, [1, mq, nq], domain)
    pre_cast_k, pre_cast_tvi_k = add_cast_to_bf16(mha.input[1], pre_cast_output_k, [1, mk, nk], domain)
    pre_cast_v, pre_cast_tvi_v = add_cast_to_bf16(mha.input[2], pre_cast_output_v, [1, nv, mv], domain)
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

    new_inputs = [pre_cast_output_q, pre_cast_output_k, pre_cast_output_v]
    mha_output = mha.output[0] + f".out{pass_id}"
    mha_node = onnx.helper.make_node(
        "MHA_noqdq",
        inputs=new_inputs,
        outputs=[mha_output],
        domain=domain,
        name=mha.name,
    )

    post_cast, post_cast_tvi = add_cast_to_float(
        mha.output[0] + f".out{pass_id}",
        mha.output[0],
        [1, mq, nv],
        domain,
    )
    tvis.extend(post_cast_tvi)

    return (
        [*pre_cast_q, *pre_cast_k, *pre_cast_v, mha_node, *post_cast],
        initializers,
        tvis,
    )


PATTERN = ["MultiHeadAttention([?,?,?,?], ?)"]
REPLACEMENT = replacement
