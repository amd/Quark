# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (unsqueeze_node,) = subgraph
    input_shape = ryzenai_onnx_utils.matcher.get_shape(unsqueeze_node.input[0], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(unsqueeze_node.output[0], extractor)
    unsqueeze_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(unsqueeze_node.output[0], extractor)
    reshape_node, reshape_tvis, reshape_tensor = add_reshape(
        input_name=unsqueeze_node.input[0],
        shape_name=unsqueeze_node.name + f".shape{pass_id}",
        output_name=unsqueeze_node.output[0],
        dtype=unsqueeze_out_dtype,
        in_shape=input_shape,
        out_shape=output_shape,
    )

    return [reshape_node], [reshape_tensor], reshape_tvis


PATTERN = ["Unsqueeze([?],?)"]
REPLACEMENT = replacement
