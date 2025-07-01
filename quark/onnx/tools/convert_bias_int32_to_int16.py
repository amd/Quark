#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert the quantized model with int32 bias to one with int16 bias.

    Example : python -m quark.onnx.tools.convert_bias_int32_to_int16 --input_model_path [INPUT_MODEL_PATH] --output_model_path [OUTPUT_MODEL_PATH]

'''

import onnx
import onnxruntime
import numpy as np
from typing import Tuple
from argparse import ArgumentParser, Namespace
from onnx import numpy_helper
from onnxruntime.quantization.onnx_model import ONNXModel
from quark.onnx.tools.convert_opset_version import convert_opset_version
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def parse_args() -> Namespace:
    usage_str = "python -m quark.onnx.tools.convert_bias_int32_to_int16 --input_model_path [INPUT_MODEL_PATH] --output_model_path [OUTPUT_MODEL_PATH]"
    parser = ArgumentParser("convert_bias_int32_to_int16", usage=usage_str)
    parser.add_argument("--input_model_path", type=str, help="input onnx model path")
    parser.add_argument("--output_model_path", type=str, help="output onnx model path")
    parser.add_argument('--save_as_external_data', action='store_true')
    args, _ = parser.parse_known_args()
    return args


def convert_bias_int32_to_int16(model: onnx.ModelProto) -> Tuple[onnx.ModelProto, bool]:
    opset_version = model.opset_import[0].version
    if opset_version < 21:
        model = convert_opset_version(model, 21)
    opset_version = model.opset_import[0].version
    if opset_version == 21:
        logger.info(
            "The opset version of the model has been automatically upgraded to 21; otherwise, ONNXRuntime can not infer models with int16 bias."
        )
    elif opset_version < 21:
        logger.warning(
            "Please upgrade the opset version of the model to 21 or higher; otherwise, ONNXRuntime can not infer models with int16 bias."
        )

    flag = False

    tensor_to_producer_dict = {}
    for node in model.graph.node:
        for output in node.output:
            tensor_to_producer_dict[output] = node

    onnx_model = ONNXModel(model)
    for node in onnx_model.model.graph.node:
        if node.op_type in [
                "Conv", "ConvTranspose", "Gemm", "LayerNormalization", "InstanceNormalization", "BatchNormalization"
        ] and len(node.input) > 2:
            if node.input[2] in tensor_to_producer_dict:
                bias_dq_node = tensor_to_producer_dict[node.input[2]]
                if len(bias_dq_node.input) == 3:
                    for init in onnx_model.model.graph.initializer:
                        if init.name == bias_dq_node.input[0] or init.name == bias_dq_node.input[2]:
                            if init.data_type == 6:
                                int32_tensor = numpy_helper.to_array(init)
                                int16_tensor = np.clip(int32_tensor, -32768, 32767).astype(np.int16)
                                new_init = numpy_helper.from_array(int16_tensor, name=init.name)
                                init.CopyFrom(new_init)
                                flag = True
    onnx_model.clean_initializers()
    onnx_model.topological_sort()
    return onnx_model.model, flag


if __name__ == '__main__':
    args = parse_args()
    try:
        ort_session = onnxruntime.InferenceSession(args.input_model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        raise RuntimeError(f"Invalid input model got, please check the input model. ONNX Runtime Error: \n{e}")
    input_model = onnx.load(args.input_model_path)
    output_model, flag = convert_bias_int32_to_int16(input_model)
    if flag:
        onnx.save(output_model, args.output_model_path, save_as_external_data=args.save_as_external_data)
        logger.info(
            f"Converted the quantized model {args.input_model_path} with int32 bias to quantized model with int16 bias {args.output_model_path}."
        )
    else:
        logger.info(f"There is no int32 bias in this quantized model {args.input_model_path}")
