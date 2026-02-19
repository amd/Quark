# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.typing import PassOutputArgs


def check_pattern(transpose_0, transpose_1, conv) -> bool:
    supported_pattern = {
        "transpose_0_perm": [0, 1, 3, 5, 2, 4],
        "transpose_1_perm": [0, 2, 3, 1],
        "conv_dilations": [1, 1],
        "conv_group": 1,
        "conv_kernel_shape": [1, 1],
        "conv_strides": [1, 1],
        "conv_pads": [0, 0, 0, 0],
    }

    transpose_0_perm = ryzenai_onnx_utils.matcher.get_attribute(transpose_0, "perm")
    transpose_1_perm = ryzenai_onnx_utils.matcher.get_attribute(transpose_1, "perm")
    conv_dilations = ryzenai_onnx_utils.matcher.get_attribute(conv, "dilations")
    conv_group = ryzenai_onnx_utils.matcher.get_attribute(conv, "group")
    conv_kernel_shape = ryzenai_onnx_utils.matcher.get_attribute(conv, "kernel_shape")
    conv_strides = ryzenai_onnx_utils.matcher.get_attribute(conv, "strides")
    conv_pads = ryzenai_onnx_utils.matcher.get_attribute(conv, "pads")

    return (
        (transpose_0_perm == supported_pattern["transpose_0_perm"])
        and (transpose_1_perm == supported_pattern["transpose_1_perm"])
        and (conv_dilations == supported_pattern["conv_dilations"])
        and (conv_group == supported_pattern["conv_group"])
        and (conv_kernel_shape == supported_pattern["conv_kernel_shape"])
        and (conv_strides == supported_pattern["conv_strides"])
        and (conv_pads == supported_pattern["conv_pads"])
    )


def is_supported_pattern(extractor, subgraph) -> bool:
    (
        reshape_0,
        transpose0,
        reshape_0_0,
        transpose_0_1,
        reshape_0_2,
        transpose_0_3,
        conv_0_4,
        reshape_1_0,
        transpose_1_1,
        reshape_1_2,
        transpose_1_3,
        conv_1_4,
    ) = subgraph

    return check_pattern(transpose_0_1, transpose_0_3, conv_0_4) and check_pattern(
        transpose_1_1, transpose_1_3, conv_1_4
    )


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (
        reshape_0,
        transpose0,
        reshape_0_0,
        transpose_0_1,
        reshape_0_2,
        transpose_0_3,
        conv_0_4,
        reshape_1_0,
        transpose_1_1,
        reshape_1_2,
        transpose_1_3,
        conv_1_4,
    ) = subgraph

    if not is_supported_pattern(extractor, subgraph):
        return subgraph, [], None

    conv_0_4_wts, conv_0_4_bias = conv_0_4.input[1], conv_0_4.input[2]
    conv_0_4_wts_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv_0_4_wts, extractor)
    conv_1_4_wts, conv_1_4_bias = conv_1_4.input[1], conv_1_4.input[2]
    conv_1_4_wts_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv_1_4_wts, extractor)

    ###################################################################
    # case1:
    # (768, 1, 1, 12288) -> (768, 768', 4, 4) => (0, 2, 3, 1) => (768, 4, 4, 768')
    # case2:
    # (768, 1, 1, 3072) -> (768, 768', 2, 2) => (0, 2, 3, 1) => (768, 2, 2, 768')
    ###################################################################
    # transform wts data
    reshape_0_0_out_shape = ryzenai_onnx_utils.matcher.get_shape(
        reshape_0_0.output[0], extractor
    )  # (2, 768, 4, 4, 4, 4)
    reshape_1_0_out_shape = ryzenai_onnx_utils.matcher.get_shape(
        reshape_1_0.output[0], extractor
    )  # (2, 768, 8, 2, 8, 2)
    conv_0_4_shape = (
        conv_0_4_wts_data.shape[0],
        int(conv_0_4_wts_data.shape[-1] / (reshape_0_0_out_shape[3] * reshape_0_0_out_shape[5])),
        reshape_0_0_out_shape[3],
        reshape_0_0_out_shape[5],
    )
    conv_1_4_shape = (
        conv_1_4_wts_data.shape[0],
        int(conv_1_4_wts_data.shape[-1] / (reshape_1_0_out_shape[3] * reshape_1_0_out_shape[5])),
        reshape_1_0_out_shape[3],
        reshape_1_0_out_shape[5],
    )
    conv_0_4_wts_data_new = conv_0_4_wts_data.reshape(conv_0_4_shape).transpose(0, 2, 3, 1)
    conv_1_4_wts_data_new = conv_1_4_wts_data.reshape(conv_1_4_shape).transpose(0, 2, 3, 1)

    new_nodes, new_inits, new_tvis = [], [], []

    # left
    new_reshape_0, new_reshape_tvis_0, new_reshape_init_0 = add_reshape(
        input_name=reshape_0.input[0],
        shape_name=reshape_0.input[1] + f"_0_{pass_id}",
        output_name=reshape_0.output[0] + f"_0_{pass_id}",
        dtype=ryzenai_onnx_utils.matcher.get_dtype(reshape_0.output[0], extractor),
        in_shape=ryzenai_onnx_utils.matcher.get_shape(reshape_0.input[0], extractor),
        out_shape=ryzenai_onnx_utils.matcher.get_shape(reshape_0.output[0], extractor),
        name=reshape_0.name + f"_0_{pass_id}",
    )
    new_nodes.append(new_reshape_0)
    new_tvis.extend(new_reshape_tvis_0)
    new_inits.append(new_reshape_init_0)

    conv_0_4_wts_name_new = conv_0_4_wts + f"_{pass_id}"
    conv_0_4_wts_shape_new = (conv_0_4_shape[0], conv_0_4_shape[2], conv_0_4_shape[3], conv_0_4_shape[1])
    conv_0_4_wts_dtype_new = ryzenai_onnx_utils.matcher.get_dtype(conv_0_4_wts, extractor)
    conv_0_4_wts_tvi = onnx.helper.make_tensor_value_info(
        conv_0_4_wts_name_new, conv_0_4_wts_dtype_new, conv_0_4_wts_shape_new
    )
    conv_0_4_wts_tensor = onnx.helper.make_tensor(
        conv_0_4_wts_name_new, conv_0_4_wts_dtype_new, conv_0_4_wts_shape_new, conv_0_4_wts_data_new.tobytes(), True
    )

    new_conv_0 = onnx.helper.make_node(
        op_type="NhwcConv",
        inputs=[reshape_0.output[0] + f"_0_{pass_id}", conv_0_4_wts_name_new, conv_0_4_bias],
        outputs=conv_0_4.output,
        name=conv_0_4.name + f"_{pass_id}",
    )
    new_nodes.append(new_conv_0)
    new_tvis.append(conv_0_4_wts_tvi)
    new_inits.append(conv_0_4_wts_tensor)
    ryzenai_onnx_utils.matcher.copy_attributes(conv_0_4, new_conv_0)
    ryzenai_onnx_utils.matcher.set_attribute(
        new_conv_0, "kernel_shape", [reshape_0_0_out_shape[3], reshape_0_0_out_shape[5]]
    )
    ryzenai_onnx_utils.matcher.set_attribute(
        new_conv_0, "strides", [reshape_0_0_out_shape[3], reshape_0_0_out_shape[5]]
    )

    # right
    new_reshape_1, new_reshape_tvis_1, new_reshape_init_1 = add_reshape(
        input_name=reshape_0.input[0],
        shape_name=reshape_0.input[1] + f"_1_{pass_id}",
        output_name=reshape_0.output[0] + f"_1_{pass_id}",
        dtype=ryzenai_onnx_utils.matcher.get_dtype(reshape_0.output[0], extractor),
        in_shape=ryzenai_onnx_utils.matcher.get_shape(reshape_0.input[0], extractor),
        out_shape=ryzenai_onnx_utils.matcher.get_shape(reshape_0.output[0], extractor),
        name=reshape_0.name + f"_1_{pass_id}",
    )
    new_nodes.append(new_reshape_1)
    new_tvis.extend(new_reshape_tvis_1)
    new_inits.append(new_reshape_init_1)

    conv_1_4_wts_name_new = conv_1_4_wts + f"_{pass_id}"
    conv_1_4_wts_shape_new = (conv_1_4_shape[0], conv_1_4_shape[2], conv_1_4_shape[3], conv_1_4_shape[1])
    conv_1_4_wts_dtype_new = ryzenai_onnx_utils.matcher.get_dtype(conv_1_4_wts, extractor)
    conv_1_4_wts_tvi = onnx.helper.make_tensor_value_info(
        conv_1_4_wts_name_new, conv_1_4_wts_dtype_new, conv_1_4_wts_shape_new
    )
    conv_1_4_wts_tensor = onnx.helper.make_tensor(
        conv_1_4_wts_name_new, conv_1_4_wts_dtype_new, conv_1_4_wts_shape_new, conv_1_4_wts_data_new.tobytes(), True
    )

    new_conv_1 = onnx.helper.make_node(
        op_type="NhwcConv",
        inputs=[reshape_0.output[0] + f"_1_{pass_id}", conv_1_4_wts_name_new, conv_1_4_bias],
        outputs=conv_1_4.output,
        name=conv_1_4.name + f"_{pass_id}",
    )
    new_nodes.append(new_conv_1)
    new_tvis.append(conv_1_4_wts_tvi)
    new_inits.append(conv_1_4_wts_tensor)
    ryzenai_onnx_utils.matcher.copy_attributes(conv_1_4, new_conv_1)
    ryzenai_onnx_utils.matcher.set_attribute(
        new_conv_1, "kernel_shape", [reshape_1_0_out_shape[3], reshape_1_0_out_shape[5]]
    )
    ryzenai_onnx_utils.matcher.set_attribute(
        new_conv_1, "strides", [reshape_1_0_out_shape[3], reshape_1_0_out_shape[5]]
    )

    return new_nodes, new_inits, new_tvis


PATTERN = [
    "Reshape([?], b0)",
    "Transpose([b0], b1)",
    "Reshape([b1,?], b2)",
    "Transpose([b2], b3)",
    "Reshape([b3,?], b4)",
    "Transpose([b4], b5)",
    "NhwcConv([b5,?,?], ?)",
    "Reshape([b1,?], b6)",
    "Transpose([b6], b7)",
    "Reshape([b7,?], b8)",
    "Transpose([b8], b9)",
    "NhwcConv([b9,?,?], ?)",
]
REPLACEMENT = replacement
