#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Convert Quark extended custom ops to deprecated Vitis custom ops, or vice versa."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import onnx
import os
import argparse
from typing import Any, Dict

NEW_DOMAIN = "com.amd.quark"
OLD_DOMAIN = "com.vai.quantize"

NAME_MAPPING = {
    "ExtendedQuantizeLinear": "VitisQuantizeLinear",
    "ExtendedDequantizeLinear": "VitisDequantizeLinear",
    "ExtendedInstanceNormalization": "VitisInstanceNormalization",
    "ExtendedLSTM": "VitisLSTM",
    "BFPQuantizeDequantize": "BFPFixNeuron",
    "MXQuantizeDequantize": "MXFixNeuron"
}


def convert_custom_ops(model: onnx.ModelProto, domain: str, mapping: Dict[str, str]) -> Any:
    from onnxruntime.quantization.onnx_model import ONNXModel

    onnx_model = ONNXModel(model)

    converted_num = 0
    for node in onnx_model.model.graph.node:
        if node.op_type not in mapping:
            continue

        node.domain = domain
        node.op_type = mapping[node.op_type]

        converted_num += 1

    if converted_num > 0:
        onnx_model.set_opset_import(domain, 1)

    return onnx_model.model


def run_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, default="", help="input onnx model file path.")
    parser.add_argument("--output_model", type=str, default="", help="output onnx model file path.")
    parser.add_argument("--external_data", type=bool, default=False, help="load and save model with external data.")
    parser.add_argument("--reverse_conversion", type=bool, default=False, help="convert old ops to new ones.")
    FLAGS, uparsed = parser.parse_known_args()

    if not os.path.isfile(FLAGS.input_model):
        print("Input model file '{}' does not exist!".format(FLAGS.input_model))
        print(
            "Usage: python -m quark.onnx.tools.convert_custom_ops --input_model INPUT_MODEL_PATH --output_model OUTPUT_MODEL_PATH."
        )
        exit()

    if FLAGS.reverse_conversion:
        domain = NEW_DOMAIN
        mapping = {v: k for k, v in NAME_MAPPING.items()}
    else:
        domain = OLD_DOMAIN
        mapping = NAME_MAPPING

    model = onnx.load_model(FLAGS.input_model, load_external_data=FLAGS.external_data)
    converted_model = convert_custom_ops(model, domain, mapping)
    onnx.save_model(converted_model, FLAGS.output_model, save_as_external_data=FLAGS.external_data)

    print('Conversion Finished!')
    print('Converted model saved in: {}'.format(FLAGS.output_model))


if __name__ == '__main__':
    run_main()
