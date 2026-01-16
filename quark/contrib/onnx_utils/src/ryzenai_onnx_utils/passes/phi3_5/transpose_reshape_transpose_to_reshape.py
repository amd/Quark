# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_paired_transpose(
    transpose0: onnx.NodeProto, transpose1: onnx.NodeProto, extractor: onnx.utils.Extractor
) -> bool:
    shape0 = ryzenai_onnx_utils.matcher.get_shape(transpose0.input[0], extractor)
    shape1 = ryzenai_onnx_utils.matcher.get_shape(transpose1.output[0], extractor)
    if len(shape0) != 4 or len(shape1) != 3:
        return False
    assert isinstance(shape0[0], int) and isinstance(shape0[2], int)
    if shape0[0] == shape1[0] and shape0[1] * shape0[2] == shape1[1] and shape0[3] == shape1[2]:
        return True
    raise NotImplementedError("Not handled")


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    transpose0, shape0, slice0, concat0, reshape0, transpose1 = subgraph
    if not is_paired_transpose(transpose0, transpose1, extractor):
        return subgraph, [], None

    input_name = transpose0.input[0]
    output_name = transpose1.output[0]
    dtype = ryzenai_onnx_utils.matcher.get_dtype(transpose0.input[0], extractor)
    in_shape = ryzenai_onnx_utils.matcher.get_shape(transpose0.input[0], extractor)
    out_shape = ryzenai_onnx_utils.matcher.get_shape(transpose1.output[0], extractor)
    shape_name = transpose1.output[0] + "_shape"

    input_tvi = onnx.helper.make_tensor_value_info(input_name, dtype, in_shape)
    output_tvi = onnx.helper.make_tensor_value_info(output_name, dtype, out_shape)
    shape_tvi = onnx.helper.make_tensor_value_info(shape_name, onnx.TensorProto.INT64, [len(out_shape)])

    node = onnx.helper.make_node(
        "Reshape",
        inputs=[input_name, shape_name],
        outputs=[output_name],
    )

    shape_tensor = onnx.helper.make_tensor(shape_name, onnx.TensorProto.INT64, [len(out_shape)], [0, -1, out_shape[2]])

    return [node], [shape_tensor], [input_tvi, output_tvi, shape_tvi]


PATTERN = [
    "Transpose([?],b0)",
    "Shape([b0], b1)",
    "Slice([b1, ?, ?, ?], b2)",
    "Concat([b2, ?], b3)",
    "Reshape([b0, b3],b4)",
    "Transpose([b4],[b5])",
]
REPLACEMENT = replacement
