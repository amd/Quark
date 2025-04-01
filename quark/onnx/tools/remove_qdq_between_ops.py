#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Remove QuantizeLinear (q) and DequantizeLinear (dq) nodes between specified operator pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from typing import Any, List, Tuple, Optional, Union

import onnx
from onnx import ModelProto, NodeProto
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def remove_qdq_between_ops(model: ModelProto, between_ops: Union[list[Tuple[str, str]], Any]) -> Any:
    """
    Modify an ONNX quantized model to remove q and dq ops between specified operation pairs.
    Start from `lower_op` nodes and traverse upwards to `upper_op`.

    :param model: The input ONNX model to be modified.
    :param between_ops: A list of tuples where each tuple contains two operator types.
                        The function will look for `q` and `dq` nodes between these pairs.
    :return: The modified ONNX model with the specified q and dq nodes removed.
    """

    try:

        nodes = model.graph.node

        nodes_to_remove = []
        edges_to_reconnect = []

        for upper_op_type, lower_op_type in between_ops:
            for lower_node in nodes:
                if lower_node.op_type == lower_op_type:
                    lower_inputs = lower_node.input

                    for input_name in lower_inputs:
                        dq_node = find_node_by_output(nodes, input_name)
                        if dq_node and dq_node.op_type == "DequantizeLinear":
                            dq_input = dq_node.input[0]

                            q_node = find_node_by_output(nodes, dq_input)
                            if q_node and q_node.op_type == "QuantizeLinear":
                                q_input = q_node.input[0]

                                upper_node = find_node_by_output(nodes, q_input)
                                if upper_node and upper_node.op_type == upper_op_type:
                                    nodes_to_remove.extend([q_node, dq_node])
                                    edges_to_reconnect.append((upper_node.output[0], lower_node, input_name))

        for node in nodes_to_remove:
            nodes.remove(node)

        for upper_node_output, lower_node, original_input in edges_to_reconnect:
            for i, lower_inp in enumerate(lower_node.input):
                if lower_inp == original_input:
                    lower_node.input[i] = upper_node_output

        onnx_model = ONNXModel(model)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()
        logger.info(f"Removed QuantizeLinear & DequantizeLinear operations: {between_ops}.")

        return onnx_model.model

    except Exception as e:
        logger.warning(f"Unable to remove QuantizeLinear & DequantizeLinear operations: {between_ops}. Exception: {e}")


def find_node_by_output(nodes: List[NodeProto], output_name: str) -> Optional[NodeProto]:
    """
    Find a node that produces the specified output.

    :param nodes: List of nodes to search.
    :param output_name: The output name to match.
    :return: The node that matches the output or None if not found.
    """
    for node in nodes:
        if output_name in node.output:
            return node
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, default="", help="input onnx model file path.")
    parser.add_argument('--between_ops',
                        type=lambda s: [tuple(item.split(',')) for item in s.split(';')],
                        help="List of operation pairs to match, formatted as 'Op1,Op2;Op3,Op4;Op5,Op6'.")
    parser.add_argument("--output_model", type=str, default="", help="output onnx model file path.")
    FLAGS, uparsed = parser.parse_known_args()

    if not os.path.isfile(FLAGS.input_model):
        print("Input model file '{}' does not exist!".format(FLAGS.input_model))
        print(
            "Usage: python -m quark.onnx.tools.remove_qdq_between_ops --input_model INPUT_MODEL_PATH --between_ops 'Conv,Relu;Conv,LeakyRelu;Conv,PRelu;Mul,Add' --output_model OUTPUT_MODEL_PATH."
        )
        exit()

    model = onnx.load_model(FLAGS.input_model)
    converted_model = remove_qdq_between_ops(model, FLAGS.between_ops)
    onnx.save(converted_model, FLAGS.output_model)
    logger.info('Conversion Finished!')
    logger.info('Converted model saved in: {}'.format(FLAGS.output_model))


if __name__ == '__main__':
    main()
