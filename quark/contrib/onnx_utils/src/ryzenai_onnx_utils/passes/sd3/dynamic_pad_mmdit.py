# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx
from onnx import helper, numpy_helper

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    get_dtype,
    get_shape,
    get_shapes,
    replace_tvis,
)
from ryzenai_onnx_utils.passes import global_pass


def make_dynamic_pad_node(dyn_input_name, extractor, input_shape=None, dtype=None):
    output_name = f"{dyn_input_name}_padded"
    pad_node = helper.make_node(
        "DynamicPad",
        inputs=[dyn_input_name],
        outputs=[output_name],
        name=output_name,
        domain="com.ryzenai",
    )
    if dtype is None:
        dtype = get_dtype(dyn_input_name, extractor)
    if input_shape is None:
        input_shape = get_shape(dyn_input_name, extractor)

    tvi = helper.make_tensor_value_info(output_name, dtype, None)
    if all(isinstance(dim, int) for dim in input_shape):
        add_attribute(pad_node, "input_shape", input_shape)
    else:
        add_attribute(pad_node, "input_shape", ",".join(str(x) for x in input_shape))

    return pad_node, tvi, output_name


def make_shape_gather_nodes(
    input_tensor_name: str,
    dim_specs: list[tuple[str, int]],
):
    """
    Build [Shape] + [Gather] nodes to extract dynamic dims.

    Args:
      input_tensor_name: input tensor name.
      dim_specs: list of (label, index), e.g. [("b", 0), ("h", 2), ("w", 3)].

    Returns:
      nodes: Shape and Gather nodes.
      initializers: Gather indices.
      tvis: Shape and Gather tvis.
      outputs: {label: gather output name}
    """
    new_nodes = []
    initializers = []
    new_tvis = []
    outputs = {}

    shape_out = f"{input_tensor_name}_shape"
    shape_node = helper.make_node(
        "Shape",
        [input_tensor_name],
        [shape_out],
        name=f"{input_tensor_name}_Shape",
    )
    shape_tvi = helper.make_tensor_value_info(shape_out, onnx.TensorProto.INT64, None)
    new_nodes.append(shape_node)
    new_tvis.append(shape_tvi)

    for label, idx in dim_specs:
        index_const = numpy_helper.from_array(
            np.array([idx], dtype=np.int64), name=f"{input_tensor_name}_indices_{label}"
        )
        gather_out = f"{shape_out}_dim_{label}"
        gather_node = helper.make_node(
            "Gather",
            [shape_out, index_const.name],
            [gather_out],
            name=f"{input_tensor_name}_Gather_{label.upper()}",
        )
        index_const_tvi = helper.make_tensor_value_info(index_const.name, onnx.TensorProto.INT64, index_const.dims)
        gather_tvi = helper.make_tensor_value_info(gather_out, onnx.TensorProto.INT64, None)
        new_nodes.append(gather_node)
        new_tvis.extend([index_const_tvi, gather_tvi])
        initializers.append(index_const)
        outputs[label] = gather_out

    return new_nodes, initializers, new_tvis, outputs


def make_arith_node(
    input_name: str,
    op_type: str = "Mul",  # "Mul" or "Div"
    factor_value: int = 8,
) -> tuple[onnx.NodeProto, onnx.TensorProto, onnx.ValueInfoProto, str]:
    """
    Build Mul or Div node: input * factor or input / factor.

    Args:
      input_name: input tensor name.
      op_type: "Mul" or "Div".
      factor_value: constant factor.

    Returns:
      arith_node: Mul or Div node.
      factor_const: factor constant tensor.
      arith_tvi: output tensor value info.
      arith_output_name: output tensor name.
    """
    assert op_type in {"Mul", "Div"}, f"Unsupported op_type: {op_type}"

    safe_input = input_name.replace("/", "_")

    factor_init_name = f"{safe_input}_factor_{factor_value}"
    factor_init = numpy_helper.from_array(np.array([factor_value], dtype=np.int64), name=factor_init_name)

    arith_output_name = f"{safe_input}_{op_type.lower()}_{factor_value}"
    arith_node_name = f"{op_type}_{safe_input}_{factor_value}"

    arith_node = helper.make_node(
        op_type,
        [input_name, factor_init.name],
        [arith_output_name],
        name=arith_node_name,
    )

    arith_tvi = helper.make_tensor_value_info(arith_output_name, onnx.TensorProto.INT64, None)

    return arith_node, factor_init, arith_tvi, arith_output_name


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

    input_names = [x.name for x in graph.input]
    output_names = [x.name for x in graph.output]
    dyn_input_names, controlnet_input_names = [], []
    input_shapes = get_shapes(input_names, extractor)
    for i in range(len(input_shapes)):
        if any(dim in {"h", "w", "height", "width"} for dim in input_shapes[i]):
            dyn_input_names.append(input_names[i])
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

    is_controlnet_model = len(dyn_input_names) == 2

    if not is_controlnet_model:
        gather_nodes, gather_inits, gather_tvis, gather_outputs = make_shape_gather_nodes(
            dyn_input_names[0], [("h", 2), ("w", 3)]
        )
        new_nodes.extend(gather_nodes)
        new_tvis.extend(gather_tvis)
        initializers.extend(gather_inits)

        for output_name in output_names:
            original_output_name = output_name
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

    for init in initializers:
        extractor.wmap[init.name] = init
    extractor.model.graph.node.extend(new_nodes)
    replace_tvis(new_tvis, extractor)


PATTERN = []
REPLACEMENT = replacement
