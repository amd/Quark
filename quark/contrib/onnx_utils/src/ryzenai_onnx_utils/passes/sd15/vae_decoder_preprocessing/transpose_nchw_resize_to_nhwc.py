# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import copy_attributes
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_nchw_resize(resize: int, extractor):
    resize_in_shape = ryzenai_onnx_utils.matcher.get_shape(resize.input[0], extractor)
    resize_out_shape = ryzenai_onnx_utils.matcher.get_shape(resize.output[0], extractor)
    scale = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(resize.input[2], extractor)
    if len(resize_in_shape) != 4 or len(resize_out_shape) != 4:
        return False
    return scale[0] == 1 and scale[1] == 1


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (resize,) = subgraph

    if not is_nchw_resize(resize, extractor):
        return subgraph, [], None

    tvis = []
    resize_in_shape = ryzenai_onnx_utils.matcher.get_shape(resize.input[0], extractor)
    resize_out_shape = ryzenai_onnx_utils.matcher.get_shape(resize.output[0], extractor)
    pre_name = resize.input[0] + f".nhwc{pass_id}"
    resize_dtype = ryzenai_onnx_utils.matcher.get_dtype(resize.input[0], extractor)
    pre_transpose_node, pre_resize_tvis = add_transpose(
        node_name=resize.name + f".pre_transpose{pass_id}",
        input_name=resize.input[0],
        output_name=pre_name,
        dtype=resize_dtype,
        shape_prev=resize_in_shape,
        shape_after=[
            resize_in_shape[0],
            resize_in_shape[2],
            resize_in_shape[3],
            resize_in_shape[1],
        ],
        perm_vec=[0, 2, 3, 1],
    )
    tvis.extend(pre_resize_tvis)

    new_scale_name = resize.input[2] + f".nhwc{pass_id}"
    scale = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(resize.input[2], extractor)
    scale_dtype = ryzenai_onnx_utils.matcher.get_dtype(resize.input[2], extractor)
    scale_tvis = onnx.helper.make_tensor_value_info(new_scale_name, scale_dtype, [len(scale)])
    scale_tensor = onnx.helper.make_tensor(
        new_scale_name,
        scale_dtype,
        [len(scale)],
        [scale[0], scale[2], scale[3], scale[1]],
    )
    initializers = [scale_tensor]
    tvis.append(scale_tvis)

    new_resize_out_name = resize.output[0] + f".nhwc{pass_id}"
    new_resize_node = onnx.helper.make_node(
        "Resize",
        inputs=[pre_transpose_node.output[0], resize.input[1], new_scale_name],
        outputs=[new_resize_out_name],
        name=resize.name,
    )
    copy_attributes(resize, new_resize_node)
    resize_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(resize.output[0], extractor)
    post_transpose_node, post_resize_tvis = add_transpose(
        node_name=resize.name + f".post_transpose{pass_id}",
        input_name=new_resize_node.output[0],
        output_name=resize.output[0],
        dtype=resize_out_dtype,
        shape_prev=[
            resize_out_shape[0],
            resize_out_shape[2],
            resize_out_shape[3],
            resize_out_shape[1],
        ],
        shape_after=resize_out_shape,
        perm_vec=[0, 3, 1, 2],
    )
    tvis.extend(post_resize_tvis)

    return (
        [pre_transpose_node, new_resize_node, post_transpose_node],
        initializers,
        tvis,
    )


PATTERN = ["Resize([?,?],?)"]
REPLACEMENT = replacement
