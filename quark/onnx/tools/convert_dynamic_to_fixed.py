#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Convert dynamic to fixed shape."""
import argparse
import os
import pathlib
import sys

import onnx

from onnxruntime.tools.onnx_model_utils import (fix_output_shapes, make_input_shape_fixed)
from typing import Dict, List


def get_input_shapes(onnx_model: onnx.ModelProto) -> Dict[str, List[int]]:
    input_shapes = {}
    for input_info in onnx_model.graph.input:
        input_name = input_info.name
        input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
        input_shapes[input_name] = input_shape
    return input_shapes


def get_output_shapes(onnx_model: onnx.ModelProto) -> Dict[str, List[int]]:
    output_shapes = {}
    for output_info in onnx_model.graph.output:
        output_name = output_info.name
        output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
        output_shapes[output_name] = output_shape
    return output_shapes


def fix_shapes(model: onnx.ModelProto, fix_shapes_config: str) -> onnx.ModelProto:
    name_shape_list = [item.strip() for item in fix_shapes_config.split(";")]
    names = []
    shapes = []
    for name_shape_item in name_shape_list:
        name_shape = [item.strip() for item in name_shape_item.rsplit(":", 1)]
        names.append(name_shape[0])
        if name_shape[1].startswith('[') and name_shape[1].endswith(']'):
            shapes.append([int(dim) for dim in name_shape[1][1:-1].split(',')])
        else:
            print("Has Error: Plase Check the input shape format. like: 'input_1:[1,224,224,3];input_2:[1,96,96,3]'")
            exit(-1)

    for name, shape in zip(names, shapes):
        make_input_shape_fixed(model.graph, name, shape)

    # update the output shapes to make them fixed if possible.
    fix_output_shapes(model)
    return model


def convert_dynamic_to_fix() -> None:
    parser = argparse.ArgumentParser(
        f"{os.path.basename(__file__)}:{convert_dynamic_to_fix.__name__}",
        description="""
                                     Assign a fixed value to a input shape
                                     Provide input_name and input_shape. like: 'input_1:[1,224,224,3];input_2:[1,96,96,3]'""",
    )

    parser.add_argument(
        "--fix_shapes",
        type=str,
        required=False,
        help=
        "Model input name&input_shape to replace shape of. Provide fix_shapes if name specified. like: 'input_1:[1,224,224,3];input_2:[1.96.96.3]'",
    )

    parser.add_argument("input_model", type=pathlib.Path, help="Provide path to ONNX model to update.")
    parser.add_argument("output_model", type=pathlib.Path, help="Provide path to write updated ONNX model to.")

    args = parser.parse_args()

    if (not args.fix_shapes):
        print("Invalid usage.")
        parser.print_help()
        sys.exit(-1)

    print("The dynamic model is:", str(args.input_model.resolve(strict=True)))
    model = onnx.load(str(args.input_model.resolve(strict=True)))
    input_shapes = get_input_shapes(model)
    for input_name, shape in input_shapes.items():
        print("The origin input_name:", input_name, ":", shape)

    output_shapes = get_output_shapes(model)
    for output_name, shape in output_shapes.items():
        print("The origin output_name:", output_name, ":", shape)
    model = fix_shapes(model, args.fix_shapes)
    input_shapes = get_input_shapes(model)
    for input_name, shape in input_shapes.items():
        print("Fixed input_name:", input_name, ":", shape)

    output_shapes = get_output_shapes(model)
    for output_name, shape in output_shapes.items():
        print("Fixed output_name:", output_name, ":", shape)

    onnx.save(model, str(args.output_model.resolve()))
    print("The output model is:", str(args.output_model.resolve()))


if __name__ == '__main__':
    convert_dynamic_to_fix()
