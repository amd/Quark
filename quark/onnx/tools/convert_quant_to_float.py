#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert quantized model to FP32 model.
'''

import onnx
from onnx import numpy_helper, ModelProto
import numpy as np
from onnxruntime.quantization.onnx_model import ONNXModel
from argparse import ArgumentParser, Namespace

from typing import Dict, Tuple


def convert_initializers_to_float(model: ModelProto, initializers_to_convert: Dict[str, Dict[str, str]]) -> ModelProto:
    """
    Convert integer initializers used by DequantizeLinear nodes to float initializers.
    """
    initializer_map = {init.name: init for init in model.graph.initializer}

    new_initializers = []
    for init in model.graph.initializer:
        if init.name in initializers_to_convert:
            int_data = numpy_helper.to_array(init).astype(np.float32)
            scale = numpy_helper.to_array(initializer_map[initializers_to_convert[init.name]['scale']])
            zero_point = numpy_helper.to_array(initializer_map[initializers_to_convert[init.name]['zero_point']])
            # Convert to float
            float_data = (int_data - zero_point) * scale
            new_init = numpy_helper.from_array(float_data, init.name)
            new_initializers.append(new_init)
        else:
            new_initializers.append(init)

    # Replace initializers in the graph
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)

    return model


def remove_quantize_dequantize_nodes(model: ModelProto) -> Tuple[ModelProto, Dict[str, Dict[str, str]]]:
    nodes_to_remove = []
    initializers_to_convert = {}
    node_output_map = {}
    for node in model.graph.node:
        node_output_map[node.output[0]] = node

    for node in model.graph.node:
        if node.op_type == 'DequantizeLinear':
            input_name = node.input[0]
            output_name = node.output[0]

            # Check if the input is an initializer
            is_initializer = any(init.name == input_name for init in model.graph.initializer)
            if is_initializer:
                scale_name = node.input[1]
                zero_point_name = node.input[2]
                initializers_to_convert[input_name] = {'scale': scale_name, 'zero_point': zero_point_name}

            nodes_to_remove.append(node)
        elif node.op_type == 'QuantizeLinear':
            nodes_to_remove.append(node)
    all_graph_ouput_name = [output.name for output in model.graph.output]
    # Remove nodes and update connections
    for node in nodes_to_remove:
        input_name = node.input[0]
        output_name = node.output[0]

        # Update all nodes that take the output of the current node as input
        for other_node in model.graph.node:
            for idx, input_name_in_other_node in enumerate(other_node.input):
                if input_name_in_other_node == output_name:
                    other_node.input[idx] = input_name

        # Remove the node from the graph
        model.graph.node.remove(node)

        # Update graph outputs
        if output_name in all_graph_ouput_name:
            last_node = node_output_map[node.input[0]]
            last_node.output[0] = output_name
    return model, initializers_to_convert


def convert_quant_to_float(quant_model_path: str, float_model_path: str) -> None:
    # Load the ONNX model
    model_path = quant_model_path
    model = onnx.load(model_path)

    # Remove QuantizeLinear and DequantizeLinear nodes
    model, initializers_to_convert = remove_quantize_dequantize_nodes(model)

    # Convert initializers to float if needed
    if initializers_to_convert:
        model = convert_initializers_to_float(model, initializers_to_convert)

    # Save the modified model
    topo_model = ONNXModel(model)
    topo_model.topological_sort()
    topo_model.clean_initializers()
    output_model = topo_model.model
    onnx.save_model(model, float_model_path, save_as_external_data=True)
    print(f"Converted model saved to {float_model_path}")


def parse_args() -> Namespace:
    usage_str = "python -m quark.onnx.tools.convert_quant_to_float.py --input [INPUT_PATH] --output [OUTPUT_PATH]"
    parser = ArgumentParser("convert_quant_to_float", usage=usage_str)
    parser.add_argument("input", type=str, help="input onnx model path")
    parser.add_argument("output", type=str, help="output onnx model path")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    convert_quant_to_float(args.input, args.output)
