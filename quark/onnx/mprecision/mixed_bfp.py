#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from quark.shares.utils.log import ScreenLogger
import numpy as np
import onnx

from onnx import onnx_pb as onnx_proto
from quark.onnx.quant_utils import (BFPFIX_OP_NAME, COP_DOMAIN, QUANT_OP_TYPES, DEQUANT_OP_TYPES, ONNXQuantizedModel,
                                    infer_custom_op_shape)
from typing import List, Dict, Any

logger = ScreenLogger(__name__)

NONLINEAR_OP_TYPE = ["Relu", "LeakyRelu", "PRelu"]
QDQ_OP_TYPE = QUANT_OP_TYPES + DEQUANT_OP_TYPES


def mixed_bfp_kernel(model: onnx.ModelProto, target_op_type: List[str], bfp_attrs: Dict[str, Any]) -> Any:
    parser = ONNXQuantizedModel(model)

    input_qdqs = []  # These input QDQs will be replaced
    added_fns = {}  # These fix neurons will be added

    # For weight and bias, replace DQ or QDQ with BFPFixNeuron
    for node in model.graph.node:
        if node.op_type not in target_op_type:
            continue

        for tensor_index, tensor_name in enumerate(node.input):
            if node.op_type != "MatMul" and tensor_index == 0:
                continue

            dq, q = parser._find_node_input_qdq(node, tensor_name)
            if dq is None:
                logger.debug(f"Node {node.name} input#{tensor_index} has no input DQ")
                continue
            elif dq in input_qdqs:
                logger.debug(f"Node {node.name} input#{tensor_index} has a shared input DQ")
                continue

            init_name = q.input[0] if q is not None else dq.input[0]

            init = parser.onnx_model.get_initializer(init_name)
            if init is None:
                if node.op_type == "MatMul":
                    logger.debug(f"{node.op_type} {node.name} input#{tensor_index} is an activation")
                else:
                    logger.warning(f"Node {node.name} input#{tensor_index} is not an initializer")
                continue

            if q is None:  # If Q was folded, restore floating point weight/bias
                init_data = onnx.numpy_helper.to_array(init)

                scale_init = parser.onnx_model.get_initializer(dq.input[1])
                scale_data = onnx.numpy_helper.to_array(scale_init)

                zp_init = parser.onnx_model.get_initializer(dq.input[2])
                zp_data = onnx.numpy_helper.to_array(zp_init)

                if zp_init.data_type == onnx_proto.TensorProto.BFLOAT16:
                    logger.warning(f"Node {node.name} input#{tensor_index} can't fold Q")
                    continue  # For BFloat16 QDQ, Q can't be folded here

                new_init_data = init_data - zp_data
                new_init_data = new_init_data.astype(np.float32)
                new_init_data = new_init_data * scale_data

                init_name = init_name + "_dequantized"
                new_init = onnx.numpy_helper.from_array(new_init_data, init_name)

                parser.onnx_model.add_initializer(new_init)

            fn_name = node.name + "_BFPFixNeuron_" + str(tensor_index)
            fn_node = onnx.helper.make_node(
                BFPFIX_OP_NAME,
                [init_name],
                [dq.output[0]],
                fn_name,
                axis=None,
                domain=COP_DOMAIN,
            )
            for k, v in bfp_attrs.items():
                if k == 'convert_to_bfloat_before_bfp':
                    v = 0
                elif k == "axis":
                    if node.op_type == "MatMul" or tensor_index == 2:
                        v = 0
                fn_node.attribute.append(onnx.helper.make_attribute(k, v))

            input_qdqs.append(dq)
            if q is not None:
                input_qdqs.append(q)

            added_fns[init.name] = fn_node

    # For input activation, just insert BFPFixNeurons
    for node in model.graph.node:
        if node.op_type not in target_op_type:
            continue

        for tensor_index, tensor_name in enumerate(node.input):
            if node.op_type != "MatMul" and tensor_index != 0:
                continue

            dq, q = parser._find_node_input_qdq(node, tensor_name)
            if dq is None or q is None:
                logger.debug(f"Node {node.name} input#{tensor_index} has no input Q/DQ")
                continue

            if dq in input_qdqs or q in input_qdqs:
                logger.debug(f"Node {node.name} input#{tensor_index} has a shared input Q/DQ")
                continue

            insert_tensor_name = dq.output[0]

            fn_name = node.name + "_BFPFixNeuron_" + str(tensor_index)
            fn_output_name = fn_name + "_output"
            fn_node = onnx.helper.make_node(
                BFPFIX_OP_NAME,
                [insert_tensor_name],
                [fn_output_name],
                fn_name,
                axis=None,
                domain=COP_DOMAIN,
            )
            for k, v in bfp_attrs.items():
                if k == "axis":
                    if node.op_type == "ReduceMean":
                        v = 3
                fn_node.attribute.append(onnx.helper.make_attribute(k, v))

            parser.onnx_model.replace_input_of_all_nodes(insert_tensor_name, fn_output_name)
            added_fns[insert_tensor_name] = fn_node

    # For output activation, insert BFPFixNeurons and should skip non-linear nodes
    """
    for node in model.graph.node:
        if node.op_type not in target_op_type:
            continue

        children = parser.in_name_to_nodes[node.output[0]]
        if len(children) == 1 and children[0].op_type in NONLINEAR_OP_TYPE:
            child_node = children[0]
        else:
            child_node = node

        if child_node.output[0] not in added_fns:
            insert_tensor_name = child_node.output[0]

            fn_name = child_node.output[0] + "_BFPFixNeuron"
            fn_output_name = fn_name + "_output"
            fn_node = onnx.helper.make_node(
                BFPFIX_OP_NAME,
                [insert_tensor_name],
                [fn_output_name],
                fn_name,
                axis=None,
                domain=COP_DOMAIN,
            )
            for k, v in bfp_attrs.items():
                if k == "axis":
                    if node.op_type == "ReduceMean":
                        v = 3
                fn_node.attribute.append(onnx.helper.make_attribute(k, v))

            parser.onnx_model.replace_input_of_all_nodes(insert_tensor_name, fn_output_name)
            added_fns[insert_tensor_name] = fn_node
    """

    parser.onnx_model.remove_nodes(input_qdqs)
    parser.onnx_model.add_nodes(added_fns.values())

    parser.onnx_model.clean_initializers()
    parser.onnx_model.topological_sort()

    return parser.onnx_model.model


def mixed_bfp_postprocess(model: onnx.ModelProto, target_op_type: List[str]) -> Any:
    parser = ONNXQuantizedModel(model)

    # step1. Fixed the issue that some quantized nodes with non-bfp
    # were connected to the BFPFixNeuron's output
    for node in model.graph.node:
        if node.op_type != BFPFIX_OP_NAME:
            continue

        for tensor_index, tensor_name in enumerate(node.input):
            if tensor_index != 0:
                continue

            if node.input[0] not in parser.out_name_to_node:
                continue

            parent = parser.out_name_to_node[node.input[0]]
            if parent.op_type not in QDQ_OP_TYPE:
                continue

            children = parser.in_name_to_nodes[node.output[0]]
            if len(children) == 1:
                continue

            for child in children:
                if child.op_type not in target_op_type:
                    parser.onnx_model.replace_node_input(child, node.output[0], node.input[0])
                    logger.debug(f"Refined {child.op_type} {child.name}'s input")

    # step2. Refine the BFPFixNeurons-Q-DQ-BFPFixNeurons structure
    remove_nodes = []

    for node in model.graph.node:
        if node.op_type != BFPFIX_OP_NAME:
            continue

        for tensor_index, tensor_name in enumerate(node.input):
            if tensor_index != 0:
                continue

            dq, q = parser._find_node_input_qdq(node, tensor_name)
            if dq is None or q is None:
                continue

            if q.input[0] not in parser.out_name_to_node:
                continue

            if len(parser.onnx_model.get_children(dq)) > 1:
                logger.debug(f"{dq.name} has multiple consumption nodes")
                continue

            parent = parser.out_name_to_node[q.input[0]]
            if parent.op_type != BFPFIX_OP_NAME:
                continue

            remove_nodes.append(parent)
            remove_nodes.append(q)
            remove_nodes.append(dq)

            parser.onnx_model.replace_input_of_all_nodes(dq.output[0], parent.input[0])

            logger.debug(f"Refined BFPFixNeurons-Q-DQ-BFPFixNeurons at {parent.input[0]}")

    parser.onnx_model.remove_nodes(remove_nodes)

    parser.onnx_model.clean_initializers()
    parser.onnx_model.topological_sort()

    return infer_custom_op_shape(parser.onnx_model.model)


def mixed_bfp(model: onnx.ModelProto, target_op_type: List[str], bfp_attrs: Dict[str, Any]) -> Any:
    mixed_model = mixed_bfp_kernel(model, target_op_type, bfp_attrs)
    return mixed_bfp_postprocess(mixed_model, target_op_type)
