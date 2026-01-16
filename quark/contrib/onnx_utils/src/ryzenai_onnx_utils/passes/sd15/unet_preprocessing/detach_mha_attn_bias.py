# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_mha_with_attn_bias(extractor, mha):
    return len(mha.input) > 3 and ryzenai_onnx_utils.matcher.is_initializer(mha.input[-1], extractor)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (mha,) = subgraph
    if not is_mha_with_attn_bias(extractor, mha):
        return subgraph, [], None

    mha_bias = mha.input[-1]
    dtype = ryzenai_onnx_utils.matcher.get_dtype(mha.input[-1], extractor)
    attn_bias_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mha.input[-1], extractor)
    assert len(attn_bias_data) % 3 == 0

    if np.all(attn_bias_data == 0):
        del mha.input[3]
        return subgraph, [], None

    len_bias = len(attn_bias_data) // 3
    q_bias_tvi = onnx.helper.make_tensor_value_info(mha_bias + "_query", dtype, [len_bias])
    k_bias_tvi = onnx.helper.make_tensor_value_info(mha_bias + "_key", dtype, [len_bias])
    v_bias_tvi = onnx.helper.make_tensor_value_info(mha_bias + "_value", dtype, [len_bias])

    q_tensor = onnx.helper.make_tensor(q_bias_tvi.name, dtype, [len_bias], attn_bias_data[:len_bias])
    k_tensor = onnx.helper.make_tensor(k_bias_tvi.name, dtype, [len_bias], attn_bias_data[len_bias : 2 * len_bias])
    v_tensor = onnx.helper.make_tensor(v_bias_tvi.name, dtype, [len_bias], attn_bias_data[2 * len_bias : 3 * len_bias])

    qo_bias_tvi = onnx.helper.make_tensor_value_info(mha_bias + "_q_bias", dtype, [len_bias])
    ko_bias_tvi = onnx.helper.make_tensor_value_info(mha_bias + "_k_bias", dtype, [len_bias])
    vo_bias_tvi = onnx.helper.make_tensor_value_info(mha_bias + "_v_bias", dtype, [len_bias])

    q_add = onnx.helper.make_node(
        "Add",
        inputs=[mha.input[0], q_bias_tvi.name],
        outputs=[qo_bias_tvi.name],
        name=mha.input[0] + "_bias_add",
    )
    k_add = onnx.helper.make_node(
        "Add",
        inputs=[mha.input[1], k_bias_tvi.name],
        outputs=[ko_bias_tvi.name],
        name=mha.input[1] + "_bias_add",
    )
    v_add = onnx.helper.make_node(
        "Add",
        inputs=[mha.input[2], v_bias_tvi.name],
        outputs=[vo_bias_tvi.name],
        name=mha.input[2] + "_bias_add",
    )

    mha.input[0] = qo_bias_tvi.name
    mha.input[1] = ko_bias_tvi.name
    mha.input[2] = vo_bias_tvi.name
    del mha.input[3]

    return (
        [q_add, k_add, v_add, mha],
        [q_tensor, k_tensor, v_tensor],
        [q_bias_tvi, k_bias_tvi, v_bias_tvi],
    )


PATTERN = ["MultiHeadAttention([?,?,?,?], ?)"]
REPLACEMENT = replacement
