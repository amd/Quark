# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (
        shape0,
        gather0,
        split,
        squeeze_n,
        squeeze_c,
        squeeeze_h,
        squeeze_w,
        mul_hw,
        unsqueeze_n,
        unsqueeze_c,
        unsqueeze_hw,
        concat_nhwc,
        reshape_3d,
        reshape_2,
        shape1,
        gather1,
        gather2,
        unsqueeze_n2,
        unsqueeze_h2,
        unsqueeze_w2,
        concat_4d,
        transpose_3d,
        reshape_4d,
    ) = subgraph

    nodes = [shape0]
    tvis = []
    tensors = []
    in_shape = ryzenai_onnx_utils.matcher.get_shape(shape0.input[0], extractor)
    out_shape = (in_shape[0], in_shape[1] + "*" + in_shape[2], in_shape[3])
    new_reshape3d_node, new_reshape3d_tvis, new_reshape3d_tensor = add_reshape(
        reshape_3d.input[0],
        reshape_3d.input[0] + "_shape_3d",
        reshape_3d.output[0],
        ryzenai_onnx_utils.matcher.get_dtype(reshape_3d.output[0], extractor),
        in_shape,
        out_shape,
    )
    nodes.append(new_reshape3d_node)
    tensors.append(new_reshape3d_tensor)
    tvis.extend(new_reshape3d_tvis)

    reshape_4d_tvi = onnx.helper.make_tensor_value_info(
        transpose_3d.output[0] + "_4d",
        ryzenai_onnx_utils.matcher.get_dtype(transpose_3d.output[0], extractor),
        in_shape,
    )
    tvis.append(reshape_4d_tvi)

    new_reshape4d_node = onnx.helper.make_node(
        "Reshape",
        inputs=[transpose_3d.input[0], shape0.output[0]],
        outputs=[reshape_4d_tvi.name],
    )
    nodes.append(new_reshape4d_node)

    transpose_out_shape = [in_shape[0], in_shape[3], in_shape[1], in_shape[2]]
    new_transpose_node, transpose_tvis = add_transpose(
        transpose_3d.name + "_4d",
        reshape_4d_tvi.name,
        reshape_4d.output[0],
        ryzenai_onnx_utils.matcher.get_dtype(transpose_3d.output[0], extractor),
        in_shape,
        transpose_out_shape,
        [0, 3, 1, 2],
    )
    nodes.append(new_transpose_node)
    tvis.extend(transpose_tvis)

    return nodes, tensors, tvis


PATTERN = [
    "Shape(?,b0)",
    "Gather([b0,?],b1)",
    "Split([b1],[b2,b3,b4,b5])",
    "Squeeze([b2,?],b6)",
    "Squeeze([b3,?],b7)",
    "Squeeze([b4,?],b8)",
    "Squeeze([b5,?],b9)",
    "Mul([b8,b9],b10)",
    "Unsqueeze([b6],b11)",
    "Unsqueeze([b7],b12)",
    "Unsqueeze([b10],b13)",
    "Concat([b11,b13,b12],b14)",
    "Reshape([?,b14],b15)",
    "Reshape([?,?],b16)",
    "Shape([b16],b17)",
    "Gather([b17,?], b18)",
    "Gather([b18,?],b19)",
    "Unsqueeze([b19],b20)",
    "Unsqueeze([b8,?],b21)",
    "Unsqueeze([b9,?],b22)",
    "Concat([b20,b12,b21,b22],b23)",
    "Transpose([?],b24)",
    "Reshape([b24,b23],b25)",
]
REPLACEMENT = replacement
