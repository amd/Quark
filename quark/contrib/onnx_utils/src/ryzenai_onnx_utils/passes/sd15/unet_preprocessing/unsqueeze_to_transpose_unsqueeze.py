# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx
from onnx import numpy_helper

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs


def get_constant_value(extractor, const_node):
    for node in extractor.graph.node:
        if node.output[0] == const_node and node.op_type == "Constant":
            attr = [attr for attr in node.attribute if attr.name == "value"][0]
            value_tensor = numpy_helper.to_array(attr.t)
            return value_tensor


def is_supported(extractor, unsqueeze1, unsqueeze2):
    unsqueeze0_input_shape = ryzenai_onnx_utils.matcher.get_shape(unsqueeze1.input[0], extractor)
    unsqueeze1_input_shape = ryzenai_onnx_utils.matcher.get_shape(unsqueeze2.input[0], extractor)
    if len(unsqueeze0_input_shape) != 2 or len(unsqueeze1_input_shape) != 3:
        return False
    axes0 = get_constant_value(extractor, unsqueeze1.input[1])
    axes1 = get_constant_value(extractor, unsqueeze2.input[1])
    return axes0 == 2 and axes1 == 3


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (
        unsqueeze1,
        unsqueeze2,
        conv,
        add,
    ) = subgraph
    if not is_supported(extractor, unsqueeze1, unsqueeze2):
        return subgraph, [], None
    unsqueeze1_out_shape = ryzenai_onnx_utils.matcher.get_shape(unsqueeze1.output[0], extractor)
    unsqueeze1_out_shape = [-1 if isinstance(shape, str) else shape for shape in unsqueeze1_out_shape]
    unsqueeze2_out_shape = ryzenai_onnx_utils.matcher.get_shape(unsqueeze2.output[0], extractor)
    unsqueeze2_out_shape = [-1 if isinstance(shape, str) else shape for shape in unsqueeze2_out_shape]
    unsqueeze1_trans_out_shape = [
        unsqueeze1_out_shape[0],
        unsqueeze1_out_shape[2],
        unsqueeze1_out_shape[1],
    ]
    unsqueeze2_trans_out_shape = [
        unsqueeze2_out_shape[0],
        unsqueeze2_out_shape[2],
        unsqueeze2_out_shape[3],
        unsqueeze2_out_shape[1],
    ]
    new_nodes = []
    new_tvis = []
    new_inits = []
    # create unsqueeze1
    unsqueeze_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(unsqueeze1.output[0], extractor)
    unsqueeze_axes_dtype = ryzenai_onnx_utils.matcher.get_dtype(unsqueeze1.input[1], extractor)
    unsqueeze1_trans_out_tvi = onnx.helper.make_tensor_value_info(
        f"{unsqueeze1.output[0]}_{pass_id}",
        unsqueeze_out_dtype,
        unsqueeze1_trans_out_shape,
    )
    new_tvis.append(unsqueeze1_trans_out_tvi)
    unsqueeze1_axes_init = onnx.helper.make_tensor(f"{unsqueeze1.input[1]}_{pass_id}", unsqueeze_axes_dtype, [1], [1])
    new_inits.append(unsqueeze1_axes_init)
    unsqueeze1_node = onnx.helper.make_node(
        unsqueeze1.op_type,
        inputs=(unsqueeze1.input[0], unsqueeze1_axes_init.name),
        outputs=[unsqueeze1_trans_out_tvi.name],
        name=unsqueeze1.name,
    )
    new_nodes.append(unsqueeze1_node)
    # create unsqueeze2
    unsqueeze2_trans_out_tvi = onnx.helper.make_tensor_value_info(
        f"{unsqueeze2.output[0]}_{pass_id}",
        unsqueeze_axes_dtype,
        unsqueeze2_trans_out_shape,
    )
    new_tvis.append(unsqueeze2_trans_out_tvi)
    unsqueeze2_axes_init = onnx.helper.make_tensor(f"{unsqueeze2.input[1]}_{pass_id}", unsqueeze_axes_dtype, [1], [2])
    new_inits.append(unsqueeze2_axes_init)
    unsqueeze2_node = onnx.helper.make_node(
        unsqueeze2.op_type,
        inputs=(unsqueeze1_node.output[0], unsqueeze2_axes_init.name),
        outputs=[unsqueeze2_trans_out_tvi.name],
        name=unsqueeze2.name,
    )
    new_nodes.append(unsqueeze2_node)
    # create trans node
    trans1_name = f"{unsqueeze2.name} + {pass_id}_trans"
    trans1_node, trans1_tvis = add_transpose(
        trans1_name,
        unsqueeze2_node.output[0],
        unsqueeze2.output[0],
        unsqueeze_out_dtype,
        unsqueeze2_trans_out_shape,
        unsqueeze2_out_shape,
        [0, 3, 1, 2],
    )
    new_tvis.extend(trans1_tvis)
    new_nodes.append(trans1_node)

    # add transpose before conv
    conv_input_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[0], extractor)
    conv_input_shape = [-1 if isinstance(shape, str) else shape for shape in conv_input_shape]
    conv_input_dtype = ryzenai_onnx_utils.matcher.get_dtype(conv.input[0], extractor)
    conv_input_trans_shape = [
        conv_input_shape[0],
        conv_input_shape[2],
        conv_input_shape[3],
        conv_input_shape[1],
    ]
    trans2_name = f"{conv.input[0]}_{pass_id}_trans"
    trans2_node, trans2_tvis = add_transpose(
        trans2_name,
        conv.input[0],
        f"{conv.input[0]}_{pass_id}_trans",
        conv_input_dtype,
        conv_input_shape,
        conv_input_trans_shape,
        [0, 2, 3, 1],
    )
    new_tvis.extend(trans2_tvis)
    new_nodes.append(trans2_node)

    # NhwcConv weights tensor
    wgts = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[1], extractor)
    np_transpose = np.transpose(wgts, [0, 2, 3, 1])
    wgts_dtype = onnx.helper.np_dtype_to_tensor_dtype(np_transpose.dtype)
    wgts_trans_init = onnx.helper.make_tensor(conv.input[1], wgts_dtype, np_transpose.shape, np_transpose)
    new_inits.append(wgts_trans_init)
    # NhwcConv output tensor
    conv_out_shape = ryzenai_onnx_utils.matcher.get_shape(conv.output[0], extractor)
    conv_out_shape = [-1 if isinstance(shape, str) else shape for shape in conv_out_shape]
    conv_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(conv.output[0], extractor)
    nhwc_conv_out_shape = [
        conv_out_shape[0],
        conv_out_shape[2],
        conv_out_shape[3],
        conv_out_shape[1],
    ]
    nhwc_conv_out_tvi = onnx.helper.make_tensor_value_info(
        f"{conv.output[0]}_{pass_id}",
        conv_out_dtype,
        nhwc_conv_out_shape,
    )
    new_tvis.append(nhwc_conv_out_tvi)
    nhwc_conv_node = onnx.helper.make_node(
        "NhwcConv",
        inputs=[trans2_node.output[0], wgts_trans_init.name, conv.input[2]],
        outputs=[nhwc_conv_out_tvi.name],
        name=conv.name,
        domain="com.microsoft",
    )
    ryzenai_onnx_utils.matcher.copy_attributes(conv, nhwc_conv_node)
    new_nodes.append(nhwc_conv_node)
    # add transpose after nhwc conv
    trans3_name = f"{conv.output[0]}_{pass_id}_trans"
    trans3_node, trans3_tvis = add_transpose(
        trans3_name,
        nhwc_conv_node.output[0],
        conv.output[0],
        conv_out_dtype,
        nhwc_conv_out_shape,
        conv_out_shape,
        [0, 3, 1, 2],
    )
    new_nodes.append(trans3_node)
    new_tvis.extend(trans3_tvis)
    # create add
    add_node = onnx.helper.make_node(
        add.op_type,
        inputs=[trans3_node.output[0], trans1_node.output[0]],
        outputs=add.output,
        name=add.name,
    )
    new_nodes.append(add_node)
    return new_nodes, new_inits, new_tvis


PATTERN = [
    "Unsqueeze([?, ?],a0)",
    "Unsqueeze([a0, ?],a1)",
    "Conv([?, ?, ?],a2)",
    "Add([a2, a1], a3)",
]
REPLACEMENT = replacement
