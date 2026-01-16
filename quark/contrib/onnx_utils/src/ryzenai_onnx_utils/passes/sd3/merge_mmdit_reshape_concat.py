# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_mmdit_concat(reshape_left, reshape_right, concat, extractor):
    if ryzenai_onnx_utils.matcher.has_multiple_successors(
        reshape_left.output[0], extractor.graph
    ) or ryzenai_onnx_utils.matcher.has_multiple_successors(reshape_right.output[0], extractor.graph):
        return False
    in_shape_left = ryzenai_onnx_utils.matcher.get_shape(reshape_left.input[0], extractor)
    in_shape_right = ryzenai_onnx_utils.matcher.get_shape(reshape_right.input[0], extractor)
    out_shape_left = ryzenai_onnx_utils.matcher.get_shape(reshape_left.output[0], extractor)
    out_shape_right = ryzenai_onnx_utils.matcher.get_shape(reshape_right.output[0], extractor)
    if (
        len(in_shape_left) not in [2, 3]
        or len(in_shape_right) not in [2, 3]
        or len(out_shape_left) != 4
        or len(out_shape_right) != 4
    ):
        return False
    if (
        in_shape_left[0] != out_shape_left[0]
        or in_shape_left[1] != out_shape_left[1]
        or in_shape_left[-1] != out_shape_left[2] * out_shape_left[3]
        or in_shape_right[0] != out_shape_right[0]
        or in_shape_right[1] != out_shape_right[1]
        or in_shape_right[-1] != out_shape_right[2] * out_shape_right[3]
    ):
        return False
    axis = onnx.helper.get_node_attr_value(concat, "axis")
    return axis == 1


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (reshape_left, reshape_right, concat) = subgraph
    if not is_mmdit_concat(reshape_left, reshape_right, concat, extractor):
        return subgraph, [], None

    tvis = []

    in_shape0 = ryzenai_onnx_utils.matcher.get_shape(reshape_left.input[0], extractor)
    in_shape1 = ryzenai_onnx_utils.matcher.get_shape(reshape_right.input[0], extractor)
    output_shape = list(in_shape0)
    if isinstance(in_shape0[-2], int) and isinstance(in_shape1[-2], int):
        output_shape[-2] = in_shape0[-2] + in_shape1[-2]
    else:
        output_shape[-2] = "+".join([str(in_shape0[-2]), str(in_shape1[-2])])
    concat_output_tvi = onnx.helper.make_tensor_value_info(
        concat.output[0] + "_3d", onnx.TensorProto.FLOAT, output_shape
    )
    tvis.append(concat_output_tvi)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(concat.output[0], extractor)

    concat_node = onnx.helper.make_node(
        "Concat",
        inputs=[reshape_left.input[0], reshape_right.input[0]],
        outputs=[concat_output_tvi.name],
        name=concat.name + "_3d",
        axis=-2,
        domain=concat.domain,
    )

    reshape_node, reshape_tvis, reshape_initialiers = add_reshape(
        concat_output_tvi.name,
        concat_output_tvi.name + "_shape",
        concat.output[0],
        dtype,
        output_shape,
        ryzenai_onnx_utils.matcher.get_shape(concat.output[0], extractor),
    )
    tvis.extend(reshape_tvis)

    return [concat_node, reshape_node], [reshape_initialiers], tvis


PATTERN = ["Reshape([?,?], t1)", "Reshape([?,?], t2)", "Concat([t1,t2], t3)"]

REPLACEMENT = replacement
