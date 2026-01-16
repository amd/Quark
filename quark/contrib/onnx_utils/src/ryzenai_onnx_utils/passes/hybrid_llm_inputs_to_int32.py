# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass converts attention mask and position ids inputs to int32 if they are
not already int32. This is required for OGA pipeline execution, which has
assumptions on what the input dtypes are.
"""

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.typing import PatternType


def to_int32(input_name: str, new_name: str, extractor: onnx.utils.Extractor) -> None:
    if not ryzenai_onnx_utils.matcher.is_input_edge(input_name, extractor.graph):
        return

    original_dtype = ryzenai_onnx_utils.matcher.get_dtype(input_name, extractor)
    if original_dtype == onnx.TensorProto.INT32:
        return

    child_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(input_name, extractor.graph)

    cast_tvi = ryzenai_onnx_utils.matcher.build_tvi(input_name, extractor, name=new_name)
    new_tvi = ryzenai_onnx_utils.matcher.build_tvi(input_name, extractor, dtype=onnx.TensorProto.INT32)
    ryzenai_onnx_utils.matcher.remove_graph_inputs([input_name], extractor.graph)
    extractor.graph.input.append(new_tvi)

    new_subgraph: list[onnx.NodeProto] = []
    for node in child_nodes:
        new_inputs = [inp if inp != input_name else new_name for inp in node.input]
        new_node = onnx.helper.make_node(
            node.op_type,
            inputs=new_inputs,
            outputs=node.output,
            domain=node.domain,
        )
        ryzenai_onnx_utils.matcher.copy_attributes(node, new_node)
        new_subgraph.append(new_node)

    cast_node = onnx.helper.make_node(
        "Cast",
        inputs=[input_name],
        outputs=[new_name],
        to=original_dtype,
    )
    new_subgraph.append(cast_node)

    ryzenai_onnx_utils.matcher.replace_subgraph(child_nodes, new_subgraph, extractor.graph, False)
    ryzenai_onnx_utils.matcher.replace_tvis([new_tvi, cast_tvi], extractor)


@global_pass
def inputs_to_int32(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    to_int32("attention_mask", "attention_mask_cast", extractor)
    to_int32("position_ids", "position_ids_cast", extractor)


REPLACEMENT = inputs_to_int32
PATTERN: PatternType = []
