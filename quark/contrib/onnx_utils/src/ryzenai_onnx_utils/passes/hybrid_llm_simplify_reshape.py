# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    cast_0 = subgraph[0]
    reshape_node = subgraph[1]
    cast_1 = subgraph[2]

    new_inputs = [
        cast_0.input[0],
        reshape_node.input[1],
    ]

    new_node = onnx.helper.make_node(
        "Reshape",
        inputs=new_inputs,
        outputs=cast_1.output,
        name=reshape_node.name,
        domain=reshape_node.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(reshape_node, new_node)

    return [new_node], [], None


REPLACEMENT = replacement
PATTERN = ["CastAvx(?, a1)", "Reshape([a1,?], a2)", "CastAvx(a2, ?)"]
