#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert s8s8 to u8s8.
'''

import numpy as np
import onnx
from onnx import numpy_helper, ModelProto
from onnxruntime.quantization.onnx_model import ONNXModel
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser("S8S8ToU8S8Converter")
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args, _ = parser.parse_known_args()
    return args


def convert_s8s8_to_u8s8(model: ModelProto) -> ModelProto:
    onnx_model = ONNXModel(model)
    tensor_to_init_dict = {}
    for init in onnx_model.model.graph.initializer:
        tensor_to_init_dict[init.name] = init

    uint8_zp128_init = onnx.TensorProto()
    uint8_zp128_init.name = 'uint8_zp128'
    uint8_zp128_init.data_type = onnx.TensorProto.UINT8
    uint8_zp128_init.raw_data = np.array(128, dtype=np.uint8).tobytes()
    onnx_model.add_initializer(uint8_zp128_init)

    for node in onnx_model.model.graph.node:
        if node.op_type == "DequantizeLinear":
            dq_input = node.input[0]
            if dq_input in tensor_to_init_dict:
                continue
            else:
                is_weight = False
                for n in onnx_model.model.graph.node:
                    if len(n.output) > 0 and n.output[0] == dq_input and n.op_type == "QuantizeLinear" and n.input[
                            0] in tensor_to_init_dict:
                        is_weight = True
                if not is_weight and tensor_to_init_dict[
                        node.input[2]].data_type == onnx.TensorProto.INT8 and numpy_helper.to_array(
                            tensor_to_init_dict[node.input[2]]).tolist() == 0:
                    node.input[2] = 'uint8_zp128'

        elif node.op_type == "QuantizeLinear":
            if node.input[0] not in tensor_to_init_dict and tensor_to_init_dict[
                    node.input[2]].data_type == onnx.TensorProto.INT8 and numpy_helper.to_array(
                        tensor_to_init_dict[node.input[2]]).tolist() == 0:
                node.input[2] = 'uint8_zp128'

    onnx_model.clean_initializers()

    return onnx_model.model  # type: ignore


def convert(args: Namespace) -> None:

    model_s8s8 = onnx.load(args.input)
    model_u8s8 = convert_s8s8_to_u8s8(model_s8s8)

    onnx.save(model_u8s8, args.output)
    print(f"Convert the s8s8 model {args.input} to the u8s8 model {args.output}.")


if __name__ == '__main__':
    args = parse_args()
    convert(args)
