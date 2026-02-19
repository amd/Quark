# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass pads the input_ids to a fixed length for prefill fusion. The input is
padded at runtime based on the values in the DD metajson.
"""

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform import hybrid_llm
from ryzenai_onnx_utils.typing import PassOutputArgs, ShapeType, SubPass


def add_pad(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    node = subgraph[0]
    io_to_pad = hybrid_llm.get_input_ids_name(extractor.graph, params.attributes)
    new_name = f"{io_to_pad}_padded"

    if node.op_type == "Gather":
        if node.input[1] != io_to_pad:
            return subgraph, [], None
        input0 = node.input[0]
        input1 = new_name
    else:
        if node.input[0] != io_to_pad:
            return subgraph, [], None
        input0 = new_name
        input1 = node.input[1]
    new_nodes = []
    new_tvis = []

    domain = params.get_domain("DynamicPad")
    new_node = onnx.helper.make_node(
        "DynamicPad",
        inputs=[io_to_pad],
        outputs=[new_name],
        name=io_to_pad + "_padding",
        domain=domain,
    )
    if node.op_type == "Gather":
        output_shape: ShapeType = [1, "sequence_length_padded"]
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "input_shape", ",".join(str(x) for x in output_shape))
    else:
        shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
        output_shape: ShapeType = [1, "sequence_length_padded", shape[2]]
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "input_shape", ",".join(str(x) for x in output_shape))
    new_nodes.append(new_node)

    new_node = onnx.helper.make_node(
        node.op_type,
        inputs=[input0, input1],
        outputs=node.output,
        name=node.name,
        domain=node.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(node, new_node)
    new_nodes.append(new_node)
    new_tvis.append(onnx.helper.make_tensor_value_info(new_name, onnx.TensorProto.INT64, output_shape))

    tvis_to_add = []
    for index, tvi in enumerate(extractor.graph.value_info):
        shape = ryzenai_onnx_utils.matcher.get_shape(tvi)
        dtype = tvi.type.tensor_type.elem_type
        if "sequence_length" in shape:
            new_shape = tuple("sequence_length_padded" if x == "sequence_length" else x for x in shape)
            tvis_to_add.append((index, onnx.helper.make_tensor_value_info(tvi.name, dtype, new_shape)))

    for index, tvi in reversed(tvis_to_add):
        del extractor.graph.value_info[index]
        extractor.graph.value_info.append(tvi)
        if tvi.name in extractor.vimap:
            extractor.vimap[tvi.name] = tvi

    return new_nodes, [], new_tvis


REPLACEMENT = add_pad
PATTERN = [
    SubPass("Gather", ["Gather([?,?], ?)"]),
    SubPass("SimplifiedLayerNormalization", ["SimplifiedLayerNormalization([?,?], ?)"]),
]
