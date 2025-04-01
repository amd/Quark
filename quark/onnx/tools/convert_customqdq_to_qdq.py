#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Convert Custom QDQ to QDQ."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import onnx
import os
import argparse
from typing import Any


def convert_customqdq_to_qdq(model: onnx.ModelProto) -> Any:
    """Convert Custom QDQ to Standard QDQ.
    :param model: source model
    :return: converted model
    """
    from onnxruntime.quantization.onnx_model import ONNXModel
    OpMapping = {"VitisQuantizeLinear": "QuantizeLinear", "VitisDequantizeLinear": "DequantizeLinear"}
    OpDomain = "com.microsoft"  # Q/DQ of this domain supports 16bit
    OpQuantType = (onnx.TensorProto.INT8, onnx.TensorProto.UINT8, onnx.TensorProto.INT16, onnx.TensorProto.UINT16,
                   onnx.TensorProto.INT32)

    onnx_model = ONNXModel(model)

    for node in onnx_model.model.graph.node:
        if node.op_type not in OpMapping:
            continue

        zp_init = onnx_model.get_initializer(node.input[2])
        if zp_init is not None and zp_init.data_type in OpQuantType:
            node.op_type = OpMapping[node.op_type]
            node.domain = OpDomain
        else:
            print(f"Skipped node {node.name} because its quant_type is"
                  f" {zp_init.data_type}")  # type_to_name[zp_init.data_type]

    return onnx_model.model


def custom_ops_infer_shapes(model: onnx.ModelProto) -> Any:
    """Generate value info for output tensors of custom ops.
    :param model: source model
    :return: converted model
    """
    CustomOps = ("VitisQuantizeLinear", "VitisDequantizeLinear", "VitisInstanceNormalization")

    has_customop = False

    for node in model.graph.node:
        if node.op_type in CustomOps:
            has_customop = True
            break

    if has_customop:
        from quark.onnx.quant_utils import infer_custom_op_shape as infer_shape
        print("Infer tensor's shape to generate value info for custom ops")
        return infer_shape(model)

    return model


def run_main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, default="", help="input onnx model file path.")
    parser.add_argument("--output_model", type=str, default="", help="output onnx model file path.")
    FLAGS, uparsed = parser.parse_known_args()

    if not os.path.isfile(FLAGS.input_model):
        print("Input model file '{}' does not exist!".format(FLAGS.input_model))
        print(
            "Usage: python -m quark.onnx.tools.convert_customqdq_to_qdq --input_model INPUT_MODEL_PATH --output_model OUTPUT_MODEL_PATH."
        )
        exit()

    model = onnx.load_model(FLAGS.input_model)
    converted_model = convert_customqdq_to_qdq(model)
    converted_model = custom_ops_infer_shapes(converted_model)
    onnx.save(converted_model, FLAGS.output_model)
    print('Conversion Finished!')
    print('Converted model saved in: {}'.format(FLAGS.output_model))


if __name__ == '__main__':
    run_main()
