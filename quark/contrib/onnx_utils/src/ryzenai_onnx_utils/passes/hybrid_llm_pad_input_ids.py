# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass pads the input_ids to a fixed length for prefill fusion. The input is
padded at runtime based on the values in the DD metajson.
"""

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs, ShapeType


def add_pad(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    gather = subgraph[0]
    io_to_pad = "input_ids"

    if gather.input[1] != io_to_pad:
        return subgraph, [], None

    new_nodes = []
    new_tvis = []

    domain = params.get_domain("DynamicPad")
    new_name = f"{io_to_pad}_padded"
    new_node = onnx.helper.make_node(
        "DynamicPad",
        inputs=[io_to_pad],
        outputs=[new_name],
        name="input_ids_padding",
        domain=domain,
    )

    output_shape: ShapeType = [1, "sequence_length_padded"]
    ryzenai_onnx_utils.matcher.add_attribute(new_node, "input_shape", ",".join(str(x) for x in output_shape))

    new_nodes.append(new_node)

    new_gather = onnx.helper.make_node(
        gather.op_type,
        inputs=[gather.input[0], new_name],
        outputs=gather.output,
        name=gather.name,
        domain=gather.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(gather, new_gather)
    new_nodes.append(new_gather)
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
PATTERN = ["Gather([?,?], ?)"]
