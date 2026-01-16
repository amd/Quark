# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import ml_dtypes
import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as cast
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    node = subgraph[0]
    domain = params.get_domain("CastAvx")

    if not ryzenai_onnx_utils.matcher.is_initializer(node.input[0], extractor):
        return subgraph, [], None

    init_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node.input[0], extractor)
    if not np.issubdtype(init_data.dtype, np.floating) or init_data.dtype == ml_dtypes.bfloat16:
        return subgraph, [], None

    init_data = init_data.astype(ml_dtypes.bfloat16)
    bf16_name = node.input[0]
    bf16_tensor = onnx.helper.make_tensor(
        bf16_name, onnx.TensorProto.BFLOAT16, init_data.shape, init_data.tobytes(), True
    )

    new_nodes = []
    new_tvis = []

    post_cast, post_cast_tvi = cast.add_cast_bfloat16_to_dtype_auto(node.output[0], pass_id, domain, extractor)
    new_nodes.extend(post_cast)
    new_tvis.extend(post_cast_tvi)

    new_node = onnx.helper.make_node(
        node.op_type,
        [bf16_name, node.input[1]],
        [post_cast[0].input[0]],
        node.name,
        domain=node.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(node, new_node)
    new_nodes.append(new_node)

    return new_nodes, [bf16_tensor], new_tvis


PATTERN = ["Gather([?, ?], ?)"]
REPLACEMENT = replacement
