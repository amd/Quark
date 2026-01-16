# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs, ShapeType

from .hybrid_llm_gqa_flatmha import add_attention_mask


def add_pad(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    io_to_slice = "attention_mask"

    reduce_sum = subgraph[0]
    if reduce_sum.input[0] != io_to_slice:
        return subgraph, [], None

    new_nodes = []
    new_tvis = []

    attention_mask_exists = ryzenai_onnx_utils.matcher.find_nodes_by_output(
        "attention_mask_const_uint", extractor.graph
    )
    if not attention_mask_exists:
        attn_mask_nodes, attn_mask_tvis = add_attention_mask()
        new_nodes.extend(attn_mask_nodes)
        new_tvis.extend(attn_mask_tvis)
    cast_out = onnx.helper.make_tensor_value_info("attention_mask_const_int64", onnx.TensorProto.INT64, [1])

    cast_node = onnx.helper.make_node(
        "Cast", inputs=["attention_mask_const_uint"], outputs=["attention_mask_const_int64"], to=onnx.TensorProto.INT64
    )

    new_nodes.append(cast_node)
    new_tvis.append(cast_out)
    slice_nodes, slice_inits, slice_tvis, _ = ryzenai_onnx_utils.matcher.make_slice_nodes(
        io_to_slice,
        [("", 0, 1)],
        # TODO(varunsh): this name is set in hybrid_llm_gqa_flatmha pass
        {"": "attention_mask_const_int64"},
        onnx.TensorProto.INT64,
        output_shape=[1, f"{io_to_slice}_sliced"],
    )
    io_to_pad = slice_nodes[0].output[0]

    new_nodes.extend(slice_nodes)
    new_tvis.extend(slice_tvis)
    new_initializers = slice_inits

    domain = params.get_domain("DynamicPad")
    new_name = f"{io_to_slice}_padded"
    new_node = onnx.helper.make_node(
        "DynamicPad",
        inputs=[io_to_pad],
        outputs=[new_name],
        name=f"{io_to_slice}_padding",
        domain=domain,
    )

    output_shape: ShapeType = [1, f"{io_to_slice}_padded"]
    ryzenai_onnx_utils.matcher.add_attribute(new_node, "input_shape", ",".join(str(x) for x in output_shape))

    new_nodes.append(new_node)

    new_tvis.append(onnx.helper.make_tensor_value_info(new_name, onnx.TensorProto.INT64, output_shape))
    new_nodes.append(reduce_sum)

    return new_nodes, new_initializers, new_tvis


REPLACEMENT = add_pad
PATTERN = ["ReduceSum([?,?], ?)"]
