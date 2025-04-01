#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Print names and quantity of A16W8 and A8W8 Conv ConvTranspose and Gemm.
'''
import onnx
from onnxruntime.quantization.onnx_model import ONNXModel
from argparse import ArgumentParser, Namespace
from typing import Tuple, List


def parse_args() -> Namespace:
    parser = ArgumentParser("PrintA16W8A8W8Nodes")
    parser.add_argument("input", type=str)
    args, _ = parser.parse_known_args()
    return args


def a16w8_a8w8_nodes(input_model_path: str) -> Tuple[List[str], List[str]]:

    def _has_input_and_output(node: onnx.NodeProto) -> bool:
        return bool(node.input) and bool(node.output)

    try:
        ComputeOperations = ('Conv', 'ConvTranspose', 'Gemm')
        quantized_nodes_dict = {}
        int8_count = 0
        int16_count = 0
        int8_node_name_list = []
        int16_node_name_list = []
        quantized_model = onnx.load(input_model_path)
        onnx_model = ONNXModel(quantized_model)

        for node in onnx_model.model.graph.node:
            if node.op_type == "QuantizeLinear" or node.op_type == "DequantizeLinear":
                continue
            if not _has_input_and_output(node):
                continue
            inp = node.input[0]
            out = node.output[0]
            quantized_flag = False
            for inp_node in onnx_model.model.graph.node:
                if not _has_input_and_output(inp_node):
                    continue
                if inp_node.output[0] == inp:
                    if inp_node.op_type == "DequantizeLinear":
                        quantized_flag = True
            for out_node in onnx_model.model.graph.node:
                if not _has_input_and_output(out_node):
                    continue
                if out_node.input[0] == out:
                    if out_node.op_type == "QuantizeLinear":
                        quantized_flag = True
            if node.op_type in ComputeOperations and quantized_flag:
                if node.op_type not in quantized_nodes_dict:
                    quantized_nodes_dict[node.op_type] = 0
                quantized_nodes_dict[node.op_type] += 1
                for node_tmp in onnx_model.model.graph.node:
                    if node_tmp.op_type == "DequantizeLinear" and len(
                            node_tmp.output) == 1 and node_tmp.output[0] == inp:
                        for init in onnx_model.model.graph.initializer:
                            if init.name == node_tmp.input[2]:
                                if init.data_type == 3:
                                    int8_count += 1
                                    int8_node_name_list.append(node.name)
                                elif init.data_type == 5:
                                    int16_node_name_list.append(node.name)
                                    int16_count += 1
                                else:
                                    pass

        return int8_node_name_list, int16_node_name_list

    except Exception as e:
        return [], []


def print_a16w8_a8w8_nodes(args: Namespace) -> None:
    input_model_path = args.input
    int8_node_name_list, int16_node_name_list = a16w8_a8w8_nodes(input_model_path)
    print("int8 activation node names:")
    for int8_node_name in int8_node_name_list:
        print(int8_node_name)
    print("int16 activation node names:")
    for int16_node_name in int16_node_name_list:
        print(int16_node_name)
    print("int8 activation node count: ", len(int8_node_name_list))
    print("int16 activation node count: ", len(int16_node_name_list))


if __name__ == '__main__':
    args = parse_args()
    print_a16w8_a8w8_nodes(args)
