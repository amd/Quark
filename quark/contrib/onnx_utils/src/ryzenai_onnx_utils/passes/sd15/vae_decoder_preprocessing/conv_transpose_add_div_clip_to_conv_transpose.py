# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_scalar_op(node, extractor):
    if len(node.input) != 2:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(node.input[1], extractor):
        return False
    value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node.input[1], extractor)
    return value.shape == (1,) or value.shape == ()


def is_op_chain(node_list, extractor):
    return all(not ryzenai_onnx_utils.matcher.has_multiple_successors(node, extractor.graph) for node in node_list)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    conv, transpose, add, div, clip = subgraph

    if False in [
        is_scalar_op(add, extractor),
        is_scalar_op(div, extractor),
        is_op_chain(subgraph, extractor),
    ]:
        return subgraph, [], None

    tvis = []
    wts_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[1], extractor)
    bias_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[2], extractor)

    add_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add.input[1], extractor)
    div_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(div.input[1], extractor)

    fused_wts_data = wts_data / div_data
    fused_bias_data = (bias_data + add_data) / div_data
    wts_init = ryzenai_onnx_utils.matcher.get_initializer(conv.input[1], extractor)
    wts_init.raw_data = fused_wts_data.tobytes()
    bias_init = ryzenai_onnx_utils.matcher.get_initializer(conv.input[2], extractor)
    bias_init.raw_data = fused_bias_data.tobytes()

    clip_output_name = clip.output[0] + "_nhwc" + f"_{pass_id}"
    conv_out_shape = ryzenai_onnx_utils.matcher.get_shape(conv.output[0], extractor)
    conv_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(conv.output[0], extractor)
    clip_output_tvi = onnx.helper.make_tensor_value_info(clip_output_name, conv_out_dtype, conv_out_shape)
    tvis.append(clip_output_tvi)
    new_clip = onnx.helper.make_node(
        "Clip",
        inputs=[conv.output[0], clip.input[1], clip.input[2]],
        outputs=[clip_output_name],
        domain=clip.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(clip, new_clip)

    perm = onnx.helper.get_node_attr_value(transpose, "perm")
    transpose_out_shape = ryzenai_onnx_utils.matcher.get_shape(transpose.output[0], extractor)
    new_transpose, transpose_tvis = add_transpose(
        transpose.name,
        clip_output_name,
        clip.output[0],
        conv_out_dtype,
        conv_out_shape,
        transpose_out_shape,
        perm,
    )
    tvis.extend(transpose_tvis)

    return [conv, new_clip, new_transpose], [], tvis


PATTERN = [
    "NhwcConv([?,?,?],b0)",
    "Transpose([b0],b1)",
    "Add([b1,?],b2)",
    "Div([b2,?], b3)",
    "Clip([b3,?,?], b4)",
]
REPLACEMENT = replacement
