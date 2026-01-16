# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces the CPU Conv operator with a custom operator version.
"""

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    input_transpose = subgraph[0]
    concat = subgraph[1]
    slice = subgraph[2]
    output_transpose = subgraph[3]
    conv = subgraph[4]
    conv_slice = subgraph[5]

    new_nodes = []
    new_initializers = []
    new_tvis = []

    global_input_name = input_transpose.input[0]
    global_output_name = output_transpose.output[0]

    ryzenai_onnx_utils.matcher.set_attribute(
        concat, "axis", int(ryzenai_onnx_utils.matcher.get_attribute(concat, "axis")) - 1
    )

    new_nodes.append(concat)

    const_1, const_1_tvi = ryzenai_onnx_utils.matcher.get_integer_const_by_value(1, extractor)
    if const_1 is not None and const_1_tvi is not None:
        new_nodes.append(const_1)
        new_tvis.append(const_1_tvi)
    new_slice = onnx.helper.make_node(
        "Slice",
        inputs=[*slice.input[:-1], "const_1"],
        outputs=slice.output,
        name=slice.name,
        domain=slice.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(slice, new_slice)
    new_nodes.append(new_slice)

    input_tvi = ryzenai_onnx_utils.matcher.get_tvi(input_transpose.input[0], extractor)
    new_input_shape = list(ryzenai_onnx_utils.matcher.get_shape(input_transpose.output[0], extractor))
    new_input_shape[1] += 1  # add one for mul in fusion
    extractor.graph.input.remove(input_tvi)
    new_input_tvi = onnx.helper.make_tensor_value_info(global_input_name, onnx.TensorProto.BFLOAT16, new_input_shape)
    extractor.graph.input.append(new_input_tvi)
    new_tvis.append(new_input_tvi)

    const_0, const_0_tvi = ryzenai_onnx_utils.matcher.get_integer_const_by_value(0, extractor)
    if const_0 is not None and const_0_tvi is not None:
        new_nodes.append(const_0)
        new_tvis.append(const_0_tvi)

    const_3, const_3_tvi = ryzenai_onnx_utils.matcher.get_integer_const_by_value(3, extractor)
    if const_3 is not None and const_3_tvi is not None:
        new_nodes.append(const_3)
        new_tvis.append(const_3_tvi)

    # print(input_transpose.output)
    new_slice = onnx.helper.make_node(
        "Slice",
        inputs=[global_input_name, "const_0", "const_3", "const_1"],
        outputs=[input_transpose.output[0] + ".bf16"],
        name=f"{slice.name}_input_{pass_id}",
        domain=slice.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(slice, new_slice)
    new_nodes.append(new_slice)

    old_output_shape = ryzenai_onnx_utils.matcher.get_shape(output_transpose.input[0], extractor)
    input_slice_cast, input_slice_cast_tvi = ryzenai_onnx_utils.transform.cast.add_cast_bfloat16_to_dtype(
        input_transpose.output[0] + ".bf16",
        input_transpose.output[0],
        old_output_shape,
        params.get_domain("CastAvx"),
        onnx.TensorProto.FLOAT,
    )
    new_nodes.extend(input_slice_cast)
    new_tvis.extend(input_slice_cast_tvi)

    output_tvi = ryzenai_onnx_utils.matcher.get_tvi(output_transpose.output[0], extractor)

    new_output_shape = list(ryzenai_onnx_utils.matcher.get_shape(output_transpose.input[0], extractor))
    new_output_shape[1] += 1  # add one for mul in fusion
    extractor.graph.output.remove(output_tvi)
    new_tvi = onnx.helper.make_tensor_value_info(global_output_name, onnx.TensorProto.BFLOAT16, new_output_shape)
    new_tvis.append(new_tvi)
    extractor.graph.output.append(new_tvi)

    if "pad_tensor" not in extractor.wmap:
        # padd from [1, 3, 2048] to [1, 4, 2048]
        pad_tensor = onnx.helper.make_tensor(
            "pad_tensor",
            onnx.TensorProto.INT64,
            [6],
            [0, 0, 0, 0, 1, 0],
        )
        new_initializers.append(pad_tensor)

    output_pad_cast, output_pad_cast_tvi = ryzenai_onnx_utils.transform.cast.add_cast_dtype_to_bfloat16(
        global_output_name + ".fp32",
        global_output_name,
        new_output_shape,
        params.get_domain("CastAvx"),
        onnx.TensorProto.FLOAT,
    )
    new_nodes.extend(output_pad_cast)
    new_tvis.extend(output_pad_cast_tvi)

    pad = onnx.helper.make_node(
        "Pad",
        inputs=[output_transpose.input[0], "pad_tensor"],
        outputs=[output_pad_cast[0].input[0]],
        name=f"pad_{pass_id}",
    )
    new_nodes.append(pad)

    # new_nodes.append(
    #     onnx.helper.make_node(
    #         "Identity", inputs=[global_output_name], outputs=[global_mul_output], name=f"identity_{pass_id}"
    #     )
    # )

    weights = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[1], extractor)
    weights_transpose = np.transpose(weights, (0, 2, 1))
    weights_tensor = onnx.numpy_helper.from_array(weights_transpose, conv.input[1] + ".t")
    new_initializers.append(weights_tensor)

    # add conv+slice
    new_conv = onnx.helper.make_node(
        "Conv",
        inputs=[conv.input[0], conv.input[1] + ".t"],
        outputs=[conv_slice.output[0]],
        name=conv.name + f"_{pass_id}",
        domain=params.get_domain(conv.op_type),
    )
    ryzenai_onnx_utils.matcher.copy_attributes(conv, new_conv)
    new_nodes.append(new_conv)

    ryzenai_onnx_utils.matcher.set_opset(extractor.model, "ai.onnx", 21)

    return new_nodes, new_initializers, new_tvis


REPLACEMENT = replacement
PATTERN = [
    "Transpose(?, transpose_out)",
    "Concat([transpose_out, ?], concat_out)",
    "Slice([concat_out, ?, ?, ?], slice_out)",
    "Transpose(slice_out, ?)",
    "NhwcConv([concat_out, ?], conv_out)",
    "Slice([conv_out, ?, ?, ?], ?)",
]
