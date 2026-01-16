# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    node = subgraph[0]

    initializers = [7, 8]  # sin/cos cache

    new_nodes = []
    new_inputs = list(node.input)
    new_initializers = []

    for initializer_index in initializers:
        initializer_name = node.input[initializer_index]
        if not ryzenai_onnx_utils.matcher.is_initializer(initializer_name, extractor):
            continue

        init_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(initializer_name, extractor)

        if not np.issubdtype(init_data.dtype, np.floating) or init_data.dtype == np.float16:
            continue

        init_data = init_data.astype(np.float16)
        fp16_tensor = onnx.helper.make_tensor(
            initializer_name + ".fp16",
            onnx.TensorProto.FLOAT16,
            init_data.shape,
            init_data.tobytes(),
            True,
        )
        new_inputs[initializer_index] = fp16_tensor.name
        new_initializers.append(fp16_tensor)

    new_node = onnx.helper.make_node(
        node.op_type,
        new_inputs,
        node.output,
        node.name,
        domain=node.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(node, new_node)
    new_nodes.append(new_node)

    return new_nodes, new_initializers, None


PATTERN = ["GQO([?,?,?,?,?,?,?,?], [?,?,?])"]
REPLACEMENT = replacement
