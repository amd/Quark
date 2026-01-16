# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_self_attention_with_attn_bias(extractor, mha):
    if len(mha.input) != 3:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(mha.input[1], extractor):
        return False
    return ryzenai_onnx_utils.matcher.is_initializer(mha.input[2], extractor)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (mha,) = subgraph
    if not is_self_attention_with_attn_bias(extractor, mha):
        return subgraph, [], None

    mha_weights, mha_bias = mha.input[1], mha.input[2]
    input_shape = ryzenai_onnx_utils.matcher.get_shape(mha.input[0], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(mha.input[0], extractor)
    attn_wts_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mha.input[1], extractor)
    attn_bias_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mha.input[2], extractor)
    assert len(attn_bias_data) % 3 == 0
    len_bias = len(attn_bias_data) // 3

    q_wts_tvi = onnx.helper.make_tensor_value_info(mha_weights + "_query", dtype, [attn_wts_data.shape[0], len_bias])
    k_wts_tvi = onnx.helper.make_tensor_value_info(mha_weights + "_key", dtype, [attn_wts_data.shape[0], len_bias])
    v_wts_tvi = onnx.helper.make_tensor_value_info(mha_weights + "_value", dtype, [attn_wts_data.shape[0], len_bias])

    q_wts_tensor = onnx.helper.make_tensor(
        q_wts_tvi.name,
        dtype,
        [attn_wts_data.shape[0], len_bias],
        attn_wts_data[:, :len_bias].tobytes(),
        True,
    )
    k_wts_tensor = onnx.helper.make_tensor(
        k_wts_tvi.name,
        dtype,
        [attn_wts_data.shape[0], len_bias],
        attn_wts_data[:, len_bias : 2 * len_bias].tobytes(),
        True,
    )
    v_wts_tensor = onnx.helper.make_tensor(
        v_wts_tvi.name,
        dtype,
        [attn_wts_data.shape[0], len_bias],
        attn_wts_data[:, 2 * len_bias : 3 * len_bias].tobytes(),
        True,
    )

    q_mm_tvi = onnx.helper.make_tensor_value_info(
        mha.input[0] + "_mm_query",
        dtype,
        [input_shape[0], input_shape[1], attn_wts_data.shape[0]],
    )
    k_mm_tvi = onnx.helper.make_tensor_value_info(
        mha.input[0] + "_mm_key",
        dtype,
        [input_shape[0], input_shape[1], attn_wts_data.shape[0]],
    )
    v_mm_tvi = onnx.helper.make_tensor_value_info(
        mha.input[0] + "_mm_value",
        dtype,
        [input_shape[0], input_shape[1], attn_wts_data.shape[0]],
    )

    q_mm = onnx.helper.make_node(
        "MatMul",
        inputs=[mha.input[0], q_wts_tvi.name],
        outputs=[q_mm_tvi.name],
        name=mha.input[0] + "_MatMul_query",
    )
    k_mm = onnx.helper.make_node(
        "MatMul",
        inputs=[mha.input[0], k_wts_tvi.name],
        outputs=[k_mm_tvi.name],
        name=mha.input[0] + "_MatMul_key",
    )
    v_mm = onnx.helper.make_node(
        "MatMul",
        inputs=[mha.input[0], v_wts_tvi.name],
        outputs=[v_mm_tvi.name],
        name=mha.input[0] + "_MatMul_value",
    )

    q_bias_tvi = onnx.helper.make_tensor_value_info(mha_bias + "_query", dtype, [len_bias])
    k_bias_tvi = onnx.helper.make_tensor_value_info(mha_bias + "_key", dtype, [len_bias])
    v_bias_tvi = onnx.helper.make_tensor_value_info(mha_bias + "_value", dtype, [len_bias])

    q_bias_tensor = onnx.helper.make_tensor(
        q_bias_tvi.name,
        dtype,
        [len_bias],
        attn_bias_data[:len_bias].tobytes(),
        True,
    )
    k_bias_tensor = onnx.helper.make_tensor(
        k_bias_tvi.name,
        dtype,
        [len_bias],
        attn_bias_data[len_bias : 2 * len_bias].tobytes(),
        True,
    )
    v_bias_tensor = onnx.helper.make_tensor(
        v_bias_tvi.name,
        dtype,
        [len_bias],
        attn_bias_data[2 * len_bias : 3 * len_bias].tobytes(),
        True,
    )

    qo_bias_tvi = onnx.helper.make_tensor_value_info(
        mha.input[0] + "_bias_query",
        dtype,
        [input_shape[0], input_shape[1], attn_wts_data.shape[0]],
    )
    ko_bias_tvi = onnx.helper.make_tensor_value_info(
        mha.input[0] + "_bias_key",
        dtype,
        [input_shape[0], input_shape[1], attn_wts_data.shape[0]],
    )
    vo_bias_tvi = onnx.helper.make_tensor_value_info(
        mha.input[0] + "_bias_value",
        dtype,
        [input_shape[0], input_shape[1], attn_wts_data.shape[0]],
    )

    q_add = onnx.helper.make_node(
        "Add",
        inputs=[q_mm.output[0], q_bias_tvi.name],
        outputs=[qo_bias_tvi.name],
        name=mha.input[0] + "_bias_add_query",
    )
    k_add = onnx.helper.make_node(
        "Add",
        inputs=[k_mm.output[0], k_bias_tvi.name],
        outputs=[ko_bias_tvi.name],
        name=mha.input[0] + "_bias_add_key",
    )
    v_add = onnx.helper.make_node(
        "Add",
        inputs=[v_mm.output[0], v_bias_tvi.name],
        outputs=[vo_bias_tvi.name],
        name=mha.input[0] + "_bias_add_value",
    )

    new_mha_node = onnx.helper.make_node(
        "MultiHeadAttention",
        inputs=[qo_bias_tvi.name, ko_bias_tvi.name, vo_bias_tvi.name],
        outputs=[mha.output[0]],
        name=mha.name,
        domain="com.microsoft",
    )
    ryzenai_onnx_utils.matcher.copy_attributes(mha, new_mha_node)
    # num_heads = ryzenai_onnx_utils.matcher.get_shape(q_reshape.output[0], extractor)[-2]
    # ryzenai_onnx_utils.matcher.add_attribute(new_mha_node, "num_heads", num_heads)
    # ryzenai_onnx_utils.matcher.add_attribute(new_mha_node, "unidirectional", 0)
    # if np.fabs(q_coeff * k_coeff - onnx_mha_scale) > 1e-7:
    #     ryzenai_onnx_utils.matcher.add_attribute(
    #         new_mha_node, "scale", q_coeff * k_coeff
    #     )

    return (
        [q_mm, k_mm, v_mm, q_add, k_add, v_add, new_mha_node],
        [
            q_wts_tensor,
            k_wts_tensor,
            v_wts_tensor,
            q_bias_tensor,
            k_bias_tensor,
            v_bias_tensor,
        ],
        [
            q_wts_tvi,
            k_wts_tvi,
            v_wts_tvi,
            q_mm_tvi,
            k_mm_tvi,
            v_mm_tvi,
            q_bias_tvi,
            k_bias_tvi,
            v_bias_tvi,
            qo_bias_tvi,
            ko_bias_tvi,
            vo_bias_tvi,
        ],
    )


PATTERN = ["Attention([?,?,?], ?)"]
REPLACEMENT = replacement
