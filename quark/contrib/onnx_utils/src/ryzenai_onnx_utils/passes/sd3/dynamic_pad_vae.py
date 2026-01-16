# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx
from onnx import helper

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    get_dtype,
    get_shapes,
    replace_tvis,
)
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.passes.sd3.dynamic_pad_mmdit import (
    make_arith_node,
    make_dynamic_pad_node,
    make_shape_gather_nodes,
)


@global_pass
def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    if not params.attributes.get("dynamic_shape_graph", False):
        return

    graph = extractor.graph
    input_names = [x.name for x in graph.input]
    output_names = [x.name for x in graph.output]

    dyn_input_names = []
    input_shapes = get_shapes(input_names, extractor)
    for i in range(len(input_shapes)):
        if any(dim in {"h", "w", "height", "width"} for dim in input_shapes[i]):
            dyn_input_names.append(input_names[i])

    new_nodes = []
    new_tvis = []
    initializers = []

    for dyn_input_name in dyn_input_names:
        ret_node, ret_tvi, ret_output_name = make_dynamic_pad_node(dyn_input_name, extractor)
        new_nodes.append(ret_node)
        new_tvis.append(ret_tvi)
        for node in extractor.model.graph.node:
            for idx, inp in enumerate(node.input):
                if inp == dyn_input_name:
                    node.input[idx] = ret_output_name

    is_vae_encoder = dyn_input_names[0] == "sample"
    is_vae_decoder = dyn_input_names[0] == "latent_sample"

    gather_nodes, gather_inits, gather_tvis, gather_outputs = make_shape_gather_nodes(
        dyn_input_names[0], [("h", 2), ("w", 3)]
    )
    new_nodes.extend(gather_nodes)
    new_tvis.extend(gather_tvis)
    initializers.extend(gather_inits)

    for output_name in output_names:
        dtype = get_dtype(output_name, extractor)
        original_output_name = output_name
        internal_output_name = f"{original_output_name}_padded"
        for node in graph.node:
            for idx, out in enumerate(node.output):
                if out == original_output_name:
                    node.output[idx] = internal_output_name

        op_type = "Mul" if is_vae_decoder else "Div" if is_vae_encoder else "Mul"
        arith_h_node, arith_h_init, arith_h_tvi, arith_h_out_name = make_arith_node(gather_outputs["h"], op_type)
        arith_w_node, arith_w_init, arith_w_tvi, arith_w_out_name = make_arith_node(gather_outputs["w"], op_type)
        new_nodes.extend([arith_h_node, arith_w_node])
        initializers.extend([arith_h_init, arith_w_init])
        new_tvis.extend([arith_h_tvi, arith_w_tvi])

        slice_nodes, slice_inits, slice_tvis, _ = ryzenai_onnx_utils.matcher.make_slice_nodes(
            input_tensor_name=internal_output_name,
            dims=[("h", 0, 2), ("w", 0, 3)],
            slice_ends={"h": arith_h_out_name, "w": arith_w_out_name},
            dtype=get_dtype(original_output_name, extractor),
        )
        new_nodes.extend(slice_nodes)
        new_tvis.extend(slice_tvis)
        initializers.extend(slice_inits)
        new_tvis.append(
            helper.make_tensor_value_info(
                internal_output_name,
                dtype,
                None,
            )
        )
        slice_nodes[-1].output[0] = original_output_name

    for init in initializers:
        extractor.wmap[init.name] = init
    extractor.model.graph.node.extend(new_nodes)
    replace_tvis(new_tvis, extractor)


PATTERN = []
REPLACEMENT = replacement
