#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# pylint: disable=g-explicit-length-test
"""Utility functions."""

import onnx
import enum
import copy
from quark.shares.utils.log import ScreenLogger
import numpy as np
from typing import Any, Union, Dict, List
from onnx import (TensorProto, ModelProto, NodeProto, ValueInfoProto, TensorShapeProto)
from google.protobuf import text_format
from onnxruntime.quantization.onnx_model import ONNXModel

logger = ScreenLogger(__name__)


def get_tensor_value(initializer: TensorProto) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Convert TensorProto to numpy array."""
    return onnx.numpy_helper.to_array(initializer)


def generate_initializer(tensor_array: np.ndarray[Any, np.dtype[np.float32]], dtype: Any, name: str) -> TensorProto:
    """Generate initializers from numpy array."""
    tensor = tensor_array.astype(dtype)
    init = onnx.numpy_helper.from_array(tensor, name)
    return init


def save_model(model: ModelProto, path: str, as_text: bool = False) -> None:
    """Save onnx model to disk."""
    if as_text:
        with open(path, 'w') as f:
            f.write(text_format.MessageToString(model))
    else:
        onnx.save(model, path)


class SharedNodesHelper(object):

    class NodeType(enum.Enum):
        NODE = 1
        INITIALIZER = 2
        INPUT = 3

    @staticmethod
    def _node_type(node: Union[NodeProto, TensorProto, ValueInfoProto]) -> NodeType:
        """Returns whether the node is a node or initializer."""
        if isinstance(node, onnx.NodeProto):
            return SharedNodesHelper.NodeType.NODE
        elif isinstance(node, onnx.TensorProto):
            return SharedNodesHelper.NodeType.INITIALIZER
        elif isinstance(node, onnx.ValueInfoProto):
            return SharedNodesHelper.NodeType.INPUT
        else:
            raise ValueError('Unknown node type for node: {}'.format(node))

    @staticmethod
    def _add_node_init(model: ModelProto, node_init_to_add: Union[NodeProto, TensorProto, ValueInfoProto]) -> None:
        """Add the node or initializer/input to the model."""
        if SharedNodesHelper._node_type(node_init_to_add) == SharedNodesHelper.NodeType.NODE:
            for node in model.graph.node:
                if node.name == node_init_to_add.name:
                    logger.info('Node `{}` is already in model, skip adding it.'.format(node.name))
                    return
            new_node = model.graph.node.add()
            new_node.CopyFrom(node_init_to_add)
        elif SharedNodesHelper._node_type(node_init_to_add) == SharedNodesHelper.NodeType.INITIALIZER:
            for init in model.graph.initializer:
                if init.name == node_init_to_add.name:
                    logger.info('Initializer `{}` is already in model, skip adding it.'.format(init.name))
                    return
            new_init = model.graph.initializer.add()
            new_init.CopyFrom(node_init_to_add)
        elif SharedNodesHelper._node_type(node_init_to_add) == SharedNodesHelper.NodeType.INPUT:
            for inp in model.graph.input:
                if inp.name == node_init_to_add.name:
                    logger.info('Input `{}` is already in model, skip adding it.'.format(inp.name))
                    return
            new_input = model.graph.input.add()
            new_input.CopyFrom(node_init_to_add)

    @staticmethod
    def _map_name_to_node(model: ModelProto) -> dict[str, NodeProto]:
        """Returns a dict of name to node.

        Returns:
            {node.name: node}
        """
        name_to_node_map = {}
        for node in model.graph.node:
            name_to_node_map[SharedNodesHelper._get_node_name(node)] = node
        return name_to_node_map

    @staticmethod
    def _map_name_to_init(model: ModelProto) -> dict[str, TensorProto]:
        """Returns a dict of name to initializer.

        Returns:
            {initializer.name: initializer}
        """
        name_to_init_map = {}
        for init in model.graph.initializer:
            name_to_init_map[init.name] = init
        return name_to_init_map

    @staticmethod
    def _map_name_to_input(model: ModelProto) -> dict[str, ValueInfoProto]:
        """Returns a dict of name to input.

        Returns:
            {initializer.name: initializer}
        """
        name_to_input_map = {}
        for inp in model.graph.input:
            name_to_input_map[inp.name] = inp
        return name_to_input_map

    @staticmethod
    def _map_tensor_to_producer(model: ModelProto) -> dict[str, NodeProto]:
        """Returns a dict of tensor to its producer node.

        Returns:
            {tensor.name: producer_node}
        """
        tensor_to_producer_map = {}
        for node in model.graph.node:
            for output_tensor in node.output:
                tensor_to_producer_map[output_tensor] = node

        for init in model.graph.initializer:
            tensor_to_producer_map[init.name] = init

        for inp in model.graph.input:
            tensor_to_producer_map[inp.name] = inp
        return tensor_to_producer_map

    @staticmethod
    def _map_tensor_to_consumer(model: ModelProto) -> Dict[str, List[NodeProto]]:
        """Returns a dict of tensor to its consumer nodes.

        Returns:
            {tensor.name: [consumer_nodes]}
        """
        tensor_to_consumer_map = {}
        for node in model.graph.node:
            for input_tensor in node.input:
                if input_tensor not in tensor_to_consumer_map:
                    tensor_to_consumer_map[input_tensor] = [node]
                else:
                    tensor_to_consumer_map[input_tensor].append(node)
        return tensor_to_consumer_map

    @staticmethod
    def _get_node_name(node: NodeProto) -> str:
        return node.name


def copy_shared_nodes(model: ModelProto) -> ModelProto:
    helper = SharedNodesHelper()

    # Rename all nodes
    type_idx = {}
    for node in model.graph.node:
        if node.op_type not in type_idx:
            type_idx[node.op_type] = 1
        node.name = node.op_type + '_' + str(type_idx[node.op_type])
        logger.info('Add node name: ', node.name)
        type_idx[node.op_type] += 1

    onnx.save(model, 'node_name_added.onnx')

    modified_flag = True
    while modified_flag:
        modified_flag = False
        name_to_node_map: Dict[str, onnx.NodeProto] = helper._map_name_to_node(model)
        name_to_init_map: Dict[str, onnx.TensorProto] = helper._map_name_to_init(model)
        name_to_input_map: Dict[str, onnx.ValueInfoProto] = helper._map_name_to_input(model)
        tensor_to_producer_map: Dict[str, NodeProto] = helper._map_tensor_to_producer(model)

        tensor_to_consumer_map = {}
        new_nodes = {}
        node_inputs_to_rename = []
        for node in model.graph.node:
            for inp_id, input_tensor in enumerate(node.input):
                if input_tensor not in tensor_to_consumer_map:
                    tensor_to_consumer_map[input_tensor] = [node]
                else:
                    producer = tensor_to_producer_map[input_tensor]
                    new_node = copy.deepcopy(producer)
                    idx = len(tensor_to_consumer_map[input_tensor])
                    new_node.name = new_node.name + '_' + str(idx)

                    if isinstance(producer, onnx.TensorProto):
                        new_nodes[new_node.name] = new_node
                        logger.info('Need to update init: ', node.name, inp_id, new_node.name)
                        node_inputs_to_rename.append((node.name, inp_id, new_node.name))
                        tensor_to_consumer_map[input_tensor].append(node)
                    elif isinstance(producer, onnx.NodeProto):
                        if producer.op_type in ["DequantizeLinear"]:
                            new_node.output[0] = new_node.name + '_out'
                            new_nodes[new_node.name] = new_node
                            logger.info('Need to update: ', node.name, inp_id, new_node.output[0])
                            node_inputs_to_rename.append((node.name, inp_id, new_node.output[0]))
                            tensor_to_consumer_map[input_tensor].append(node)
                    else:
                        pass

        for name, new_node in new_nodes.items():
            helper._add_node_init(model, new_node)

        name_to_node_map = helper._map_name_to_node(model)

        for (node_name, inp_id, new_node_name) in node_inputs_to_rename:
            modified_flag = True
            node = name_to_node_map[node_name]
            logger.info('Update node input', node_name, inp_id, new_node_name)
            node.input[inp_id] = new_node_name

    return model


def clean_initializer_in_input(model: ModelProto) -> ModelProto:

    if model.ir_version < 4:
        logger.warning("Initilizer should be included in input domain if the model ir_version is below 4.")
        logger.warning("The mode ir_version will be set as 4")
        model.ir_version = 4

    inputs = model.graph.input
    input_name_dict = {}
    for inp in inputs:
        input_name_dict[inp.name] = inp

    for init in model.graph.initializer:
        if init.name in input_name_dict:
            model.graph.input.remove(input_name_dict[init.name])

    return model


def get_shape_list(shape: TensorShapeProto) -> List[Union[int, str]]:
    shape_list = []
    for d in shape.dim:
        if d.HasField("dim_value"):
            shape_list.append(d.dim_value)
        elif d.HasField("dim_param"):
            shape_list.append(d.dim_param)
        else:
            shape_list.append("?")
    return shape_list


def convert_nchw_to_nhwc(model: ModelProto) -> Any:
    temp_model = clean_initializer_in_input(model)
    onnx_model = ONNXModel(temp_model)

    node_name_list = []
    for node in onnx_model.graph().node:
        node_name_list.append(node.name)

    for inp in onnx_model.graph().input:
        shape_list = get_shape_list(inp.type.tensor_type.shape)

        if len(shape_list) != 4:
            logger.warning(f"Expected 4-dimension input shape but got {shape_list}, skip the nchw to nhwc conversion.")
            continue

        C, H, W = shape_list[1:]
        if not all(isinstance(_, int) for _ in [C, H, W]):
            logger.warning(f"Expected integer input shape but got [{C}, {H}, {W}], skip the nchw to nhwc conversion.")
            continue

        if not (int(H) > int(C) and int(W) > int(C)):
            logger.warning(
                f"Expected H,W > C but got [{C}, {H}, {W}]. Please confirm whether the input model is in NCHW format")

        inp.type.tensor_type.shape.dim[1].dim_value = H
        inp.type.tensor_type.shape.dim[2].dim_value = W
        inp.type.tensor_type.shape.dim[3].dim_value = C

        transpose_name = inp.name + "_transpose"
        count = 1
        while transpose_name in node_name_list:
            transpose_name += "_" + str(count)
            count += 1
        inp_transpose_node = onnx.helper.make_node("Transpose", [inp.name], [transpose_name],
                                                   name=transpose_name,
                                                   perm=[0, 3, 1, 2])
        onnx_model.replace_input_of_all_nodes(inp.name, transpose_name)
        onnx_model.add_node(inp_transpose_node)

    for out in onnx_model.graph().output:
        shape_list = get_shape_list(out.type.tensor_type.shape)

        if len(shape_list) != 4:
            logger.info(
                f"Expected 4-dimension output shape but got {shape_list}, skip the nchw to nhwc conversion for output {out}."
            )
            continue

        C, H, W = shape_list[1:]
        if not all(isinstance(_, int) for _ in [C, H, W]):
            logger.warning(
                f"Expected integer output shape but got [{C}, {H}, {W}], skip the nchw to nhwc conversion for output {out}."
            )
            continue

        if not (int(H) > int(C) and int(W) > int(C)):
            logger.warning(
                f"Expected H,W > C but got [{C}, {H}, {W}], Please confirm whether the output {out} is in NCHW format")

        out.type.tensor_type.shape.dim[1].dim_value = H
        out.type.tensor_type.shape.dim[2].dim_value = W
        out.type.tensor_type.shape.dim[3].dim_value = C

        transpose_name = out.name + "_transpose"
        count = 1
        while transpose_name in node_name_list:
            transpose_name += "_" + str(count)
            count += 1
        out_transpose_node = onnx.helper.make_node("Transpose", [out.name], [transpose_name],
                                                   name=transpose_name,
                                                   perm=[0, 2, 3, 1])
        onnx_model.add_node(out_transpose_node)
        last_node: Union[NodeProto, Any] = None
        penultimate_node: Union[NodeProto, Any] = None
        for node in onnx_model.graph().node:
            if node.output[0] == out.name:
                last_node = node
                logger.debug('last_node name :`{}` .'.format(last_node.name))
        for node in onnx_model.graph().node:
            if node.output[0] == last_node.input[0]:
                penultimate_node = node
                logger.debug('penultimate_node name :`{}` .'.format(penultimate_node.name))
        if (last_node.op_type == "DequantizeLinear" and penultimate_node.op_type == "QuantizeLinear"):
            quantize_linear_name = out_transpose_node.name + "_QuantizeLinear"
            transpose_QuantizeLinear = onnx.helper.make_node(
                op_type=penultimate_node.op_type,
                inputs=[out_transpose_node.output[0], penultimate_node.input[1], penultimate_node.input[2]],
                outputs=[quantize_linear_name],
                name=out_transpose_node.name + "_QuantizeLinear",
                domain=penultimate_node.domain)
            out_transpose_node.output[0] = quantize_linear_name
            onnx_model.graph().node.extend([transpose_QuantizeLinear])
            dequantize_linear_name = out_transpose_node.name + "_DequantizeLinear"
            transpose_DequantizeLinear = onnx.helper.make_node(
                op_type=last_node.op_type,
                inputs=[transpose_QuantizeLinear.output[0], last_node.input[1], last_node.input[2]],
                outputs=[dequantize_linear_name],
                name=out_transpose_node.name + "_DequantizeLinear",
                domain=last_node.domain)
            onnx_model.graph().node.extend([transpose_DequantizeLinear])
            transpose_DequantizeLinear.output[0] = transpose_name
            out.name = dequantize_linear_name
        else:
            out.name = transpose_name
    onnx_model.topological_sort()
    return onnx_model.model
