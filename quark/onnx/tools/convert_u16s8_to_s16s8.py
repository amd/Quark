#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert u16s8 to s16s8.
'''

import numpy as np
import onnx
from onnx import numpy_helper, ModelProto
from onnxruntime.quantization.onnx_model import ONNXModel
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser("U16S8ToS16S8Converter")
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args, _ = parser.parse_known_args()
    return args


def convert_u16s8_to_s16s8(model: ModelProto) -> ModelProto:
    onnx_model = ONNXModel(model)
    tensor_to_init_dict = {}
    for init in onnx_model.model.graph.initializer:
        tensor_to_init_dict[init.name] = init

    int16_zp0_init = onnx.TensorProto()
    int16_zp0_init.name = 'int16_zp0'
    int16_zp0_init.data_type = onnx.TensorProto.INT16
    int16_zp0_init.raw_data = np.array(0, dtype=np.int16).tobytes()
    onnx_model.add_initializer(int16_zp0_init)

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
                        node.input[2]].data_type == onnx.TensorProto.UINT16 and numpy_helper.to_array(
                            tensor_to_init_dict[node.input[2]]).tolist() in [32767, 32768]:
                    node.input[2] = 'int16_zp0'

        elif node.op_type == "QuantizeLinear":
            if node.input[0] not in tensor_to_init_dict and tensor_to_init_dict[
                    node.input[2]].data_type == onnx.TensorProto.UINT16 and numpy_helper.to_array(
                        tensor_to_init_dict[node.input[2]]).tolist() in [32767, 32768]:
                node.input[2] = 'int16_zp0'

    onnx_model.clean_initializers()
    assert isinstance(onnx_model.model, ModelProto), "The converted model is not of type ModelProto"
    return onnx_model.model


def convert(args: Namespace) -> None:

    model_u16s8 = onnx.load(args.input)
    model_s16s8 = convert_u16s8_to_s16s8(model_u16s8)

    onnx.save(model_s16s8, args.output)
    print(f"Convert the u16s8 model {args.input} to the s16s8 model {args.output}.")


if __name__ == '__main__':
    args = parse_args()
    convert(args)
