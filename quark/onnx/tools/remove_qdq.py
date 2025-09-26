#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Remove QDQ operators."""

import argparse
import os
from typing import Any

import onnx

from quark.onnx.optimizations import convert_transforms_pipeline


def remove_qdq(model: onnx.ModelProto) -> Any:
    """Remove QDQ operators.
    :param model: source model
    :return: converted model
    """
    convert_pipeline = convert_transforms_pipeline.RemoveQDQTransformsPipeline()
    converted_model, _ = convert_pipeline.apply(model, candidate_nodes=None, node_metadata=None)
    return converted_model


def run_main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, default="", help="input onnx model file path.")
    parser.add_argument("--output_model", type=str, default="", help="output onnx model file path.")
    FLAGS, uparsed = parser.parse_known_args()

    if not os.path.isfile(FLAGS.input_model):
        print(f"Input model file '{FLAGS.input_model}' does not exist!")
        print(
            "Usage: python -m quark.onnx.tools.remove_qdq --input_model INPUT_MODEL_PATH --output_model OUTPUT_MODEL_PATH."
        )
        exit()

    model = onnx.load_model(FLAGS.input_model)
    converted_model = remove_qdq(model)
    onnx.save(converted_model, FLAGS.output_model)
    print("Conversion Finished!")
    print(f"Converted model saved in: {FLAGS.output_model}")


if __name__ == "__main__":
    run_main()
