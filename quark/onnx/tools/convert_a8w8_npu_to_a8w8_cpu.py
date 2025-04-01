#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert the A8W8_NPU model to A8W8_CPU model.

    Example : python -m quark.onnx.tools.convert_a8w8_npu_to_a8w8_cpu --input [INPUT_PATH] --output [OUTPUT_PATH]

'''

import onnx
import onnxruntime
import numpy as np
from argparse import ArgumentParser, Namespace

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def parse_args() -> Namespace:
    usage_str = "python -m quark.onnx.tools.convert_a8w8_npu_to_a8w8_cpu --input [INPUT_PATH] --output [OUTPUT_PATH]"
    parser = ArgumentParser("convert_a8w8_npu_to_a8w8_cpu", usage=usage_str)
    parser.add_argument("input", type=str, help="input onnx model path")
    parser.add_argument("output", type=str, help="output onnx model path")
    args, _ = parser.parse_known_args()
    return args


def convert_a8w8_npu_to_a8w8_cpu(model: onnx.ModelProto) -> onnx.ModelProto:
    conv_node_list = []
    for node in model.graph.node:
        if node.op_type == "Conv":
            conv_node_list.append(node)
    conv_qdq_node_list = []

    for conv_node in conv_node_list:
        if len(conv_node.input) != 3:
            continue
        tmp_node_list = []
        for node in model.graph.node:
            if len(node.output) == 1 and node.output[0] == conv_node.input[0]:
                tmp_node_list.append(node)
            if len(node.output) == 1 and node.output[0] == conv_node.input[1]:
                tmp_node_list.append(node)
            if len(node.output) == 1 and node.output[0] == conv_node.input[2]:
                tmp_node_list.append(node)
        conv_qdq_node_list.append(tmp_node_list[::-1])

    for tmp_node_list in conv_qdq_node_list:
        act_qdq_node = tmp_node_list[0]
        weights_qdq_node = tmp_node_list[1]
        bias_qdq_node = tmp_node_list[2]
        for init in model.graph.initializer:
            if init.name == bias_qdq_node.input[0]:
                bias_x_init = init
                bias_x = np.reshape(np.frombuffer(bias_x_init.raw_data, dtype=np.int8), bias_x_init.dims)
            if init.name == bias_qdq_node.input[1]:
                bias_scale_init = init
                bias_scale = bias_scale_init.float_data[0]
            if init.name == bias_qdq_node.input[2]:
                bias_zp_init = init
                bias_zp = bias_zp_init.int32_data[0]
            if init.name == act_qdq_node.input[1]:
                act_scale_init = init
                act_scale = act_scale_init.float_data[0]
            if init.name == weights_qdq_node.input[1]:
                weights_scale_init = init
                weights_scale = weights_scale_init.float_data[0]
        new_bias_scale = act_scale * weights_scale
        new_bias_x = bias_x * bias_scale / new_bias_scale
        new_bias_zp = 0
        int32_bias_x = np.array(new_bias_x, dtype=np.int32)
        int32_bias_scale = np.array(new_bias_scale, dtype=np.float32)
        int32_bias_zp = np.array(0, dtype=np.int32)
        for init in model.graph.initializer:
            if init.name == bias_qdq_node.input[2]:
                model.graph.initializer.remove(init)
        for init in model.graph.initializer:
            if init.name == bias_qdq_node.input[1]:
                model.graph.initializer.remove(init)
        for init in model.graph.initializer:
            if init.name == bias_qdq_node.input[0]:
                model.graph.initializer.remove(init)
        int32_bias_x_tensor = onnx.numpy_helper.from_array(int32_bias_x, bias_qdq_node.input[0])
        int32_bias_scale_tensor = onnx.numpy_helper.from_array(int32_bias_scale, bias_qdq_node.input[1])
        int32_bias_zp_tensor = onnx.numpy_helper.from_array(int32_bias_zp, bias_qdq_node.input[2])
        model.graph.initializer.extend([int32_bias_x_tensor, int32_bias_scale_tensor, int32_bias_zp_tensor])
    return model


if __name__ == '__main__':
    args = parse_args()
    try:
        ort_session = onnxruntime.InferenceSession(args.input, providers=['CPUExecutionProvider'])
    except Exception as e:
        raise RuntimeError(f"Invalid input model got, please check the input model. ONNX Runtime Error: \n{e}")
    input_model = onnx.load(args.input)
    output_model = convert_a8w8_npu_to_a8w8_cpu(input_model)
    onnx.save(output_model, args.output)
    print(f"Converted the A8W8_NPU model {args.input} to the A8W8_CPU model {args.output}.")
