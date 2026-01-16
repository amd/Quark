# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx
from onnx import helper, numpy_helper

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    get_dtype,
    get_shape,
    get_shapes,
    replace_tvis,
)
from ryzenai_onnx_utils.passes import global_pass

from .dynamic_pad_mmdit import make_arith_node, make_dynamic_pad_node, make_shape_gather_nodes

# def make_concat_reshape_nodes(
#     input_tensor_name: str,
#     dim_inputs: list[str],
#     output_tensor_name: str,
#     dtype: int,
# ) -> tuple[list[onnx.NodeProto], list[onnx.ValueInfoProto]]:
#     """
#     Build [Concat] + [Reshape] nodes.

#     Args:
#       input_tensor: Tensor to reshape.
#       dim_inputs: List of dim input names to concat as new shape.
#       output_tensor: Output tensor name after reshape.
#       dtype: dtype for input_tensor.

#     Returns:
#       nodes: [Concat, Reshape]
#       tvis: [Concat output, Reshape output]
#     """

#     shape_name = f"{output_tensor_name}_shape"

#     new_nodes, initializers, tvis = [], [], []
#     if dim_inputs:
#         concat_node = helper.make_node(
#             "Concat",
#             inputs=dim_inputs,
#             outputs=[shape_name],
#             name=f"{output_tensor_name}_ConcatShape",
#             axis=0,
#         )
#         concat_tvi = helper.make_tensor_value_info(
#             shape_name, onnx.TensorProto.INT64, [len(dim_inputs)]
#         )
#         new_nodes.append(concat_node)
#         tvis.append(concat_tvi)
#     else:
#         reshape_shape_tensor = numpy_helper.from_array(
#             np.array([1], dtype=np.int64), name=shape_name
#         )
#         reshape_shape_tvi = helper.make_tensor_value_info(
#             shape_name, onnx.TensorProto.INT64, reshape_shape_tensor.dims
#         )
#         initializers.append(reshape_shape_tensor)
#         tvis.append(reshape_shape_tvi)

#     reshape_node = helper.make_node(
#         "Reshape",
#         inputs=[input_tensor_name, shape_name],
#         outputs=[output_tensor_name],
#         name=f"{output_tensor_name}_Reshape",
#     )
#     reshape_tvi = helper.make_tensor_value_info(
#         output_tensor_name, dtype, None if dim_inputs else [1]
#     )
#     new_nodes.append(reshape_node)
#     tvis.append(reshape_tvi)

#     return new_nodes, initializers, tvis


# def make_safe_reshape(in_name: str):
#     shape_name = f"{in_name}_safeshape"
#     out_name = f"{in_name}_safe"
#     tensor = numpy_helper.from_array(np.array([1], dtype=np.int64), name=shape_name)
#     node = helper.make_node("Reshape", [in_name, shape_name], [out_name], name=f"{out_name}_Node")
#     tvis = [
#         helper.make_tensor_value_info(shape_name, onnx.TensorProto.INT64, [1]),
#         helper.make_tensor_value_info(out_name, onnx.TensorProto.INT64, [1]),
#     ]
#     return [node], [tensor], tvis, out_name


def make_concat_reshape_nodes(
    input_tensor_name: str,
    dim_inputs: list[str],
    output_tensor_name: str,
    dtype: int,
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto], list[onnx.ValueInfoProto]]:
    """
    Build [Concat] + [Reshape] nodes.
    All dim_inputs will be auto-squeezed to shape [1] if needed.
    """

    shape_name = f"{output_tensor_name}_shape"
    new_nodes, initializers, tvis = [], [], []
    reshape_input_names: dict[str, str] = {}

    safe_dim_inputs = []
    for inp in dim_inputs:
        if inp in reshape_input_names:
            safe_out = reshape_input_names[inp]
        else:
            reshape_shape_name = f"{inp}_safe_shape"
            reshape_out_name = f"{inp}_safe"

            reshape_shape_tensor = numpy_helper.from_array(
                np.array([1], dtype=np.int64),
                name=reshape_shape_name,
            )
            reshape_shape_tvi = helper.make_tensor_value_info(reshape_shape_name, onnx.TensorProto.INT64, [1])

            reshape_node = helper.make_node(
                "Reshape",
                inputs=[inp, reshape_shape_name],
                outputs=[reshape_out_name],
                name=f"{reshape_out_name}_SafeReshape",
            )
            reshape_tvi = helper.make_tensor_value_info(reshape_out_name, onnx.TensorProto.INT64, [1])

            safe_out = reshape_out_name
            reshape_input_names[inp] = reshape_out_name

            initializers.append(reshape_shape_tensor)
            tvis.extend([reshape_shape_tvi, reshape_tvi])
            new_nodes.append(reshape_node)

        safe_dim_inputs.append(safe_out)

    concat_node = helper.make_node(
        "Concat",
        inputs=safe_dim_inputs,
        outputs=[shape_name],
        name=f"{output_tensor_name}_ConcatShape",
        axis=0,
    )
    concat_tvi = helper.make_tensor_value_info(shape_name, onnx.TensorProto.INT64, [len(safe_dim_inputs)])
    new_nodes.append(concat_node)
    tvis.append(concat_tvi)

    reshape_node = helper.make_node(
        "Reshape",
        inputs=[input_tensor_name, shape_name],
        outputs=[output_tensor_name],
        name=f"{output_tensor_name}_Reshape",
    )
    reshape_tvi = helper.make_tensor_value_info(output_tensor_name, dtype, None)
    new_nodes.append(reshape_node)
    tvis.append(reshape_tvi)

    return new_nodes, initializers, tvis


@global_pass
def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    if not params.attributes.get("dynamic_shape_graph", False):
        return

    graph = extractor.model.graph

    input_tensors = graph.input
    output_tensors = graph.output

    input_names = [input_tensor.name for input_tensor in input_tensors]
    output_names = [output_tensor.name for output_tensor in output_tensors]
    dyn_input_names, controlnet_input_names = [], []
    input_shapes = get_shapes(input_names, extractor)
    for i in range(len(input_shapes)):
        if any(dim in {"h", "w", "height", "width"} for dim in input_shapes[i]):
            dyn_input_names.append(input_names[i])
        if any(dim == "state_dim1" for dim in input_shapes[i]):
            controlnet_input_names.append(input_names[i])

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

    gather_nodes, gather_inits, gather_tvis, gather_outputs = make_shape_gather_nodes(
        dyn_input_names[0], [("h", 2), ("w", 3)]
    )
    new_nodes.extend(gather_nodes)
    new_tvis.extend(gather_tvis)
    initializers.extend(gather_inits)

    if controlnet_input_names:
        op_type = "Div"
        arith_h_node, arith_h_init, arith_h_tvi, arith_h_out_name = make_arith_node(gather_outputs["h"], op_type, 2)
        arith_w_node, arith_w_init, arith_w_tvi, arith_w_out_name = make_arith_node(gather_outputs["w"], op_type, 2)
        new_nodes.extend([arith_h_node, arith_w_node])
        initializers.extend([arith_h_init, arith_w_init])
        new_tvis.extend([arith_h_tvi, arith_w_tvi])

        for controlnet_input_name in controlnet_input_names:
            dtype = get_dtype(controlnet_input_name, extractor)
            (
                controlnet_gather_nodes,
                controlnet_gather_inits,
                controlnet_gather_tvis,
                controlnet_gather_outputs,
            ) = make_shape_gather_nodes(controlnet_input_name, [("b", 0), ("c", 2)])
            new_nodes.extend(controlnet_gather_nodes)
            initializers.extend(controlnet_gather_inits)
            new_tvis.extend(controlnet_gather_tvis)

            shape_inputs = [
                controlnet_gather_outputs["b"],
                arith_h_out_name,
                arith_w_out_name,
                controlnet_gather_outputs["c"],
            ]
            reshape_output_name = f"{controlnet_input_name}_reshaped"
            controlnet_shape_nodes, controlner_shape_inits, controlnet_shape_tvis = make_concat_reshape_nodes(
                controlnet_input_name, shape_inputs, reshape_output_name, dtype
            )
            new_nodes.extend(controlnet_shape_nodes)
            initializers.extend(controlner_shape_inits)
            new_tvis.extend(controlnet_shape_tvis)

            controlnet_shape = get_shape(controlnet_input_name, extractor)
            dyn_pad_input_shape = [controlnet_shape[0], "floor(w/2)", "floor(h/2)", controlnet_shape[-1]]
            dyn_pad_dtype = get_dtype(controlnet_input_name, extractor)
            pad_node, pad_tvi, pad_output_name = make_dynamic_pad_node(
                reshape_output_name, extractor, input_shape=dyn_pad_input_shape, dtype=dyn_pad_dtype
            )
            new_nodes.append(pad_node)
            new_tvis.append(pad_tvi)

            # pad_shape_tensor = numpy_helper.from_array(
            #     np.array([0, -1, 3], dtype=np.int64),
            #     name=f"{pad_output_name}_reshape_shape"
            # )
            # pad_shape_tvi = helper.make_tensor_value_info(
            #     f"{pad_output_name}_reshape_shape", onnx.TensorProto.INT64,pad_shape_tensor.dims
            # )
            # pad_reshape_output_name = f"{pad_output_name}_reshaped"
            # pad_reshape_node = helper.make_node(
            #     "Reshape",
            #     inputs=[pad_output_name, pad_shape_tensor.name],
            #     outputs=[pad_reshape_output_name],
            #     name=f"{pad_output_name}_Reshape",
            # )
            # pad_reshape_tvi = helper.make_tensor_value_info(
            #     pad_reshape_output_name, dtype, None
            # )
            # new_nodes.append(pad_reshape_node)
            # initializers.append(pad_shape_tensor)
            # new_tvis.extend([pad_shape_tvi, pad_reshape_tvi])

            pad_gather_nodes, pad_gather_inits, pad_gather_tvis, pad_gather_outputs = make_shape_gather_nodes(
                pad_output_name, [("h", 1), ("w", 2)]
            )
            new_nodes.extend(pad_gather_nodes)
            new_tvis.extend(pad_gather_tvis)
            initializers.extend(pad_gather_inits)

            mul_output_name = f"{pad_gather_outputs['h']}_mul_{pad_gather_outputs['w']}"
            mul_node = helper.make_node(
                "Mul",
                [pad_gather_outputs["h"], pad_gather_outputs["w"]],
                [mul_output_name],
                name=f"Mul_{pad_gather_outputs['h']}_{pad_gather_outputs['w']}",
            )
            mul_tvi = helper.make_tensor_value_info(mul_output_name, onnx.TensorProto.INT64, None)
            new_nodes.append(mul_node)
            new_tvis.append(mul_tvi)

            final_shape_inputs = [
                controlnet_gather_outputs["b"],
                mul_output_name,
                controlnet_gather_outputs["c"],
            ]
            final_reshape_output_name = f"{pad_output_name}_reshaped"
            final_shape_nodes, final_shape_inits, final_shape_tvis = make_concat_reshape_nodes(
                pad_output_name,
                final_shape_inputs,
                final_reshape_output_name,
                dtype,
            )
            new_nodes.extend(final_shape_nodes)
            initializers.extend(final_shape_inits)
            new_tvis.extend(final_shape_tvis)

            for node in extractor.model.graph.node:
                for idx, inp in enumerate(node.input):
                    if inp == controlnet_input_name:
                        node.input[idx] = final_reshape_output_name

    is_controlnet_model = len(dyn_input_names) == 2
    if is_controlnet_model:
        op_type = "Div"
        (
            final_arith_h_node,
            final_arith_h_init,
            final_arith_h_tvi,
            final_arith_h_out_name,
        ) = make_arith_node(gather_outputs["h"], op_type, 2)
        (
            final_arith_w_node,
            final_arith_w_init,
            final_arith_w_tvi,
            final_arith_w_out_name,
        ) = make_arith_node(gather_outputs["w"], op_type, 2)
        # new_nodes.extend([final_arith_h_node, final_arith_w_node])
        # initializers.extend([final_arith_h_init, final_arith_w_init])
        # new_tvis.extend([final_arith_h_tvi, final_arith_w_tvi])

        mul_output_name = f"{final_arith_h_out_name}_mul_{final_arith_w_out_name}"
        mul_node = helper.make_node(
            "Mul",
            [final_arith_h_out_name, final_arith_w_out_name],
            [mul_output_name],
            name=f"Mul_{final_arith_h_out_name}_{final_arith_w_out_name}",
        )
        mul_tvi = helper.make_tensor_value_info(mul_output_name, onnx.TensorProto.INT64, None)
        new_nodes.extend([final_arith_h_node, final_arith_w_node, mul_node])
        initializers.extend([final_arith_h_init, final_arith_w_init])
        new_tvis.extend([final_arith_h_tvi, final_arith_w_tvi, mul_tvi])

        # div_h_reshape_output_name = "/pos_embed/Div_output_0_reshaped"
        # div_w_reshape_output_name = "/pos_embed/Div_1_output_0_reshaped"
        # div_h_reshape_nodes, div_h_reshape_inits, div_h_reshape_tvis  = make_concat_reshape_nodes("/pos_embed/Div_output_0",[], div_h_reshape_output_name, onnx.TensorProto.INT64)
        # div_w_reshape_nodes, div_w_reshape_inits, div_w_reshape_tvis = make_concat_reshape_nodes("/pos_embed/Div_1_output_0",[], div_w_reshape_output_name, onnx.TensorProto.INT64)
        # new_nodes.extend(div_h_reshape_nodes + div_w_reshape_nodes)
        # initializers.extend(div_h_reshape_inits + div_w_reshape_inits)
        # new_tvis.extend(div_h_reshape_tvis+ div_w_reshape_tvis)

        # arith_h_node, arith_h_init, arith_h_tvi, arith_h_out_name = make_arith_node(
        #     "/pos_embed/Gather_output_0", op_type, 2
        # )
        # arith_w_node, arith_w_init, arith_w_tvi, arith_w_out_name = make_arith_node(
        #     "/pos_embed/Gather_1_output_0", op_type, 2
        # )
        # new_nodes.extend([arith_h_node, arith_w_node])
        # initializers.extend([arith_h_init, arith_w_init])
        # new_tvis.extend([arith_h_tvi, arith_w_tvi])

        for idx_out, output_name in enumerate(output_names):
            dtype = get_dtype(output_name, extractor)
            original_output_name = output_name
            internal_output_name = f"{original_output_name}_padded"
            for node in graph.node:
                for idx, out in enumerate(node.output):
                    if out == original_output_name:
                        node.output[idx] = internal_output_name

            (
                controlnet_output_gather_nodes,
                controlnet_output_gather_inits,
                controlnet_output_gather_tvis,
                controlnet_output_gather_outputs,
            ) = make_shape_gather_nodes(internal_output_name, [("b", 0), ("c", 2)])
            new_nodes.extend(controlnet_output_gather_nodes)
            initializers.extend(controlnet_output_gather_inits)
            new_tvis.extend(controlnet_output_gather_tvis)

            shape_inputs = [
                controlnet_output_gather_outputs["b"],
                "/pos_embed/Div_output_0",
                "/pos_embed/Div_1_output_0",
                # div_h_reshape_output_name,
                # div_w_reshape_output_name,
                controlnet_output_gather_outputs["c"],
            ]
            reshape_output_name = f"{internal_output_name}_reshaped"
            (
                controlnet_output_shape_nodes,
                controlner_output_shape_inits,
                controlnet_output_shape_tvis,
            ) = make_concat_reshape_nodes(internal_output_name, shape_inputs, reshape_output_name, dtype)
            new_nodes.extend(controlnet_output_shape_nodes)
            initializers.extend(controlner_output_shape_inits)
            new_tvis.extend(controlnet_output_shape_tvis)

            slice_nodes, slice_inits, slice_tvis, slice_output_name = ryzenai_onnx_utils.matcher.make_slice_nodes(
                input_tensor_name=reshape_output_name,
                dims=[("h", 0, 1), ("w", 0, 2)],
                slice_ends={"h": final_arith_h_out_name, "w": final_arith_w_out_name},
                dtype=dtype,
            )
            new_nodes.extend(slice_nodes)
            new_tvis.extend(slice_tvis)
            initializers.extend(slice_inits)
            # slice_nodes[-1].output[0] = original_output_name

            # final_shape_tensor = numpy_helper.from_array(
            #     np.array([0, -1, 0], dtype=np.int64),
            #     name=f"{slice_output_name}_reshape_shape"
            # )
            # final_shape_tvi = helper.make_tensor_value_info(
            #     f"{slice_output_name}_reshape_shape", onnx.TensorProto.INT64, final_shape_tensor.dims
            # )
            # final_reshape_node = helper.make_node(
            #     "Reshape",
            #     inputs=[slice_output_name, final_shape_tensor.name],
            #     outputs=[original_output_name],
            #     name=f"{slice_output_name}_Reshape",
            # )
            # # final_reshape_tvi = helper.make_tensor_value_info(
            # #     slice_output_name, dtype, None
            # # )
            # new_nodes.append(final_reshape_node)
            # initializers.append(final_shape_tensor)
            # new_tvis.append(final_shape_tvi)

            output_shape_inputs = [
                controlnet_output_gather_outputs["b"],
                mul_output_name,
                controlnet_output_gather_outputs["c"],
            ]
            (
                final_output_shape_nodes,
                final_output_shape_inits,
                final_output_shape_tvis,
            ) = make_concat_reshape_nodes(slice_output_name, output_shape_inputs, original_output_name, dtype)
            new_nodes.extend(final_output_shape_nodes)
            initializers.extend(final_output_shape_inits)
            new_tvis.extend(final_output_shape_tvis)

            # debug_tvi_0 = add_debug_output(new_nodes, new_tvis, slice_output_name,dtype)
            # debug_tvi_1 = add_debug_output(new_nodes, new_tvis, final_output_shape_nodes[-2].output[0])
            # extractor.model.graph.output.extend([final_output_shape_tvis[-1], slice_tvis[-1]])

            output_tensors[idx_out].name = original_output_name
    else:
        assert len(output_tensors) == 1
        original_output_name = output_tensors[0].name
        internal_output_name = f"{original_output_name}_padded"
        for node in graph.node:
            for idx, out in enumerate(node.output):
                if out == original_output_name:
                    node.output[idx] = internal_output_name

        slice_nodes, slice_inits, slice_tvis, _ = ryzenai_onnx_utils.matcher.make_slice_nodes(
            input_tensor_name=internal_output_name,
            dims=[("h", 0, 2), ("w", 0, 3)],
            slice_ends=gather_outputs,
            dtype=get_dtype(original_output_name, extractor),
        )
        new_nodes.extend(slice_nodes)
        new_tvis.extend(slice_tvis)
        initializers.extend(slice_inits)
        new_tvis.append(
            helper.make_tensor_value_info(
                internal_output_name,
                get_dtype(original_output_name, extractor),
                None,
            )
        )
        slice_nodes[-1].output[0] = original_output_name
        output_tensors[0].name = original_output_name

    for init in initializers:
        extractor.wmap[init.name] = init
    extractor.model.graph.node.extend(new_nodes)
    replace_tvis(new_tvis, extractor)


PATTERN = []
REPLACEMENT = replacement
