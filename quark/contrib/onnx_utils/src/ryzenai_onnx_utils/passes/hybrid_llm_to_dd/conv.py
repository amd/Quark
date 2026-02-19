# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as cast
import ryzenai_onnx_utils.utils
from ryzenai_onnx_utils.strategy_builder import MladfVersion
from ryzenai_onnx_utils.typing import PassOutputArgs


def replace_tvi(old_tvi: str, new_tvi: onnx.ValueInfoProto, graph: onnx.GraphProto):
    for index, output_tvi in enumerate(graph.input):
        if output_tvi.name == old_tvi:
            graph.input.remove(graph.input[index])
            graph.input.insert(index, new_tvi)
            break

    for index, output_tvi in enumerate(graph.output):
        if output_tvi.name == old_tvi:
            graph.output.remove(graph.output[index])
            graph.output.insert(index, new_tvi)
            break


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("LiquidAILFM2MulConv1d")
    mul = subgraph[0]
    input_transpose = subgraph[1]
    concat = subgraph[2]
    conv = subgraph[3]
    slice = subgraph[4]
    output_transpose = subgraph[6]

    global_input_name = input_transpose.input[0]
    global_output_name = output_transpose.output[0]
    assert ryzenai_onnx_utils.matcher.is_input_edge(global_input_name, extractor.graph)
    assert ryzenai_onnx_utils.matcher.is_output_edge(global_output_name, extractor.graph), global_output_name

    conv_input = global_input_name
    conv_output = global_output_name

    new_nodes = []
    new_initializers = []
    new_tvis = []

    shape = ryzenai_onnx_utils.matcher.get_shape(concat.output[0], extractor)
    new_global_input_tvi = onnx.helper.make_tensor_value_info(
        conv_input,
        onnx.TensorProto.BFLOAT16,
        shape,
    )
    replace_tvi(conv_input, new_global_input_tvi, extractor.graph)
    new_tvis.append(new_global_input_tvi)

    new_global_output_tvi = onnx.helper.make_tensor_value_info(
        conv_output,
        onnx.TensorProto.BFLOAT16,
        shape,
    )
    replace_tvi(conv_output, new_global_output_tvi, extractor.graph)
    new_tvis.append(new_global_output_tvi)

    input_cast, input_cast_tvis = cast.add_cast_dtype_to_bfloat16_auto(mul.input[0], pass_id, domain, extractor)
    new_nodes.extend(input_cast)
    new_tvis.extend(input_cast_tvis)

    input_cast_1, input_cast_1_tvis = cast.add_cast_dtype_to_bfloat16_auto(mul.input[1], pass_id, domain, extractor)
    new_nodes.extend(input_cast_1)
    new_tvis.extend(input_cast_1_tvis)

    output_cast, output_cast_tvis = cast.add_cast_bfloat16_to_dtype_auto(slice.output[0], pass_id, domain, extractor)
    new_nodes.extend(output_cast)
    new_tvis.extend(output_cast_tvis)

    op_version = MladfVersion(params.attributes["mladf_version"])
    lora = params.get_bool_attr("lora", False)
    if lora:
        op_version = MladfVersion.FLAT

    # TODO(varunsh): this is assuming the pass_id will be separated by underscores
    # get the count of how many passes have run and use that to offset the indices below
    # this is also assuming the order of the subpasses below
    # +1 for sin_cos cache
    match_index = int(pass_id.split("_")[-1]) + 1

    weights = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[1], extractor)
    weights_tensor = ryzenai_onnx_utils.utils.float_numpy_to_bfloat_tensor(
        np.transpose(weights, (2, 0, 1)), conv.input[1] + ".t"
    )
    new_initializers.append(weights_tensor)

    new_conv = onnx.helper.make_node(
        "LiquidAILFM2MulConv1d",
        inputs=[input_cast[0].output[0], input_cast_1[0].output[0], conv_input, conv.input[1] + ".t"],
        outputs=[conv_output, output_cast[0].input[0]],
        name=conv.name,
        domain=domain,
    )
    pdi_id = int(params.attributes.get("pdi_id", 0))
    if pdi_id != 0:
        ryzenai_onnx_utils.matcher.add_attribute(new_conv, "pdi_id", int(pdi_id))
    ryzenai_onnx_utils.matcher.add_attribute(new_conv, "op_version", str(op_version))
    ryzenai_onnx_utils.matcher.add_attribute(new_conv, "in_dtypes", ["bfloat16", "bfloat16", "bfloat16", "bfloat16"])
    ryzenai_onnx_utils.matcher.add_attribute(new_conv, "out_dtypes", ["bfloat16", "bfloat16"])
    external_buffers = []
    external_buffers.extend([2, 1, match_index, match_index])  # global conv input
    external_buffers.extend([4, 1, match_index, match_index])  # global conv output
    ryzenai_onnx_utils.matcher.add_attribute(new_conv, "external_buffers", external_buffers)
    new_nodes.append(new_conv)

    return new_nodes, new_initializers, new_tvis


PATTERN = [
    "Mul(?, a0)",
    "Transpose([?], a1)",
    "Concat([a1, a0], a2)",
    "NhwcConv([a2,?], a3)",
    "Slice(a3, ?)",
    "Slice(a2, a5)",
    "Transpose(a5, ?)",
]
REPLACEMENT = replacement
