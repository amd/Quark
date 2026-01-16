# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_elwadd_supported(a_shape: tuple[int | str, ...], b_shape: tuple[int | str, ...], op_namespace: str) -> bool:
    supported_shapes = {
        "sdxlt": {
            # Unet
            (1024, 640),
            (256, 1280),
            # (1, 1280),
            (64, 64, 320),
            (32, 32, 640),
            (16, 16, 1280),
            # VAE decoder
            (64, 64, 512),
            (128, 128, 512),
            (256, 256, 256),
            (512, 512, 128),
        }
    }

    if a_shape != b_shape:
        return False

    if len(a_shape) == 3 or len(a_shape) == 4:
        if a_shape[0] != 1:
            return False
        a_shape = a_shape[1:]
    elif len(a_shape) != 2:
        return False

    return a_shape in supported_shapes[op_namespace]


# +start:
def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    """
    The replacement function is called on every matched subgraph specified by the pattern.
    Then for each subgraph, it will do subgraph replacement as defined. A new subgraph

    Args:
        extractor (onnx.utils.Extractor): Allows access to the original model using
            extractor.model
        pass_id (str): a unique string to identify this pass on this particular subgraph
        subgraph (list[onnx.NodeProto]): a list of Nodes in the same order as your pattern.
            This is the subgraph you matched with your pattern that you want to replace
        params (ryzenai_onnx_utils.ReplaceParams): additional parameters for more arguments
            passed from the top level to each pass

    Returns:
        (list[onnx.NodeProto], list[onnx.TensorProto], Optional[onnx.ValueInfoProto]): Returned
            values represent the new subgraph, new tensors to add, and new tensor_value_info
            objects to add
    """

    """
    ONNX operators are defined in a domain. For new ops we insert into the ONNX
    model, we need a domain for them. This domain can be specified in the
    partition strategy on an operator level and is retrieved here.
    """
    domain = params.get_domain("ElwAdd_noqdq")
    """
    We also assign ONNX operators to a namespace. This namespace value is used
    to determine whether an operator is supported for offload as different
    operators we add may have different shapes they support. This namespace is
    set in the partition strategy on an operator level and is retrieved here.
    """
    op_namespace = params.get_op_namespace("Add")
    """
    Since the pattern is a single Add operator, the node at index 0 will be an Add node
    """
    add_node = subgraph[0]

    """
    For some ops, you may want to do some validation on the incoming nodes
    """
    assert len(add_node.input) == 2
    assert len(add_node.output) == 1

    input_shape_0, input_shape_1 = ryzenai_onnx_utils.matcher.get_shapes(add_node.input, extractor)
    (output_shape,) = ryzenai_onnx_utils.matcher.get_shapes(add_node.output, extractor)

    if ryzenai_onnx_utils.matcher.get_initializers(add_node.input, extractor, False):
        """
        For this particular pass, it requires two activations so if there is one constant
        then skip modifications by returning the same subgraph back with empty tensors and tvis
        """
        return subgraph, [], None

    if not is_elwadd_supported(input_shape_0, input_shape_1, op_namespace):
        """
        For this particular pass, if the shape is not supported by the new operator, then skip
        modifications by returning the same subgraph back with empty tensors and tvis
        """
        return subgraph, [], None

    """
    Otherwise, modify the subgraph. In this case, the original Add node is replaced
    by a bfloat16 ElwAdd_noqdq node. To maintain the original semantics of the
    graph, casts to and from bfloat16 from and to float32 are added around the
    node. Redundant casts are replaced in later pass.

    In some passes, you may create new Tensors or initializers. These have to be
    returned from the replacement function as well so they get added into the
    graph.

    TVIs (TensorValueInfo) represent the type and shape of new edges you insert
    into the graph. These are important for inferring the correct shapes and
    performing shape checks on nodes. If you create new nodes/edges, create
    appropriate TVIs and return them.
    """

    tvis = []

    pre_cast_output_0 = add_node.input[0] + f".out_{pass_id}"
    pre_cast_0, pre_cast_tvi_0 = add_cast_to_bf16(add_node.input[0], pre_cast_output_0, input_shape_0, domain)
    tvis.extend(pre_cast_tvi_0)

    pre_cast_output_1 = add_node.input[1] + f".out_{pass_id}"
    pre_cast_1, pre_cast_tvi_1 = add_cast_to_bf16(add_node.input[1], pre_cast_output_1, input_shape_1, domain)
    tvis.extend(pre_cast_tvi_1)

    new_inputs = [pre_cast_output_0, pre_cast_output_1]
    add_node_output = add_node.output[0] + f".out_{pass_id}"
    elwadd_node = onnx.helper.make_node(
        "ElwAdd_noqdq",
        inputs=new_inputs,
        outputs=[add_node_output],
        domain=domain,
        name=add_node.name,
    )

    post_cast, post_cast_tvi = add_cast_to_float(add_node_output, add_node.output[0], output_shape, domain)
    tvis.extend(post_cast_tvi)

    return [*pre_cast_0, *pre_cast_1, elwadd_node, *post_cast], [], tvis


"""
The REPLACEMENT keyword is special and it should be set to the name of the replacement function
to use.
"""
REPLACEMENT = replacement
"""
The PATTERN keyword is special and it's set to the pattern to match on. This is a list of
patterns that could be as short as one operator or thousands. You can use the "match"
functionality in ryzenai_onnx_utils to create a valid pattern from an ONNX graph.
"""
PATTERN = ["Add([?,?],?)"]
# -end:
