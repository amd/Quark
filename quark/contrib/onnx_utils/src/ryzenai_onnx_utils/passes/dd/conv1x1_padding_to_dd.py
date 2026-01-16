# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.transform import build_dd_node
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("Conv1x1_noqdq")
    conv_node = subgraph[0]

    # go back to the original Conv1x1_noqdq node
    node = onnx.helper.make_node(
        "Conv1x1_noqdq",
        inputs=conv_node.input,
        outputs=conv_node.output,
        domain=domain,
        name=conv_node.name,
    )
    node.attribute.extend(conv_node.attribute)

    dd_node = build_dd_node(extractor, [node], params, op_name="DynamicDispatchConvPadded")

    return [dd_node], [], None


REPLACEMENT = replacement
PATTERN = [
    "Conv1x1_padding_noqdq([?,?,?], ?)",
]
