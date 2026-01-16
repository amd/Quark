#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Modification Copyright (c) 2024 Advanced Micro Devices, Inc.
# Licensed under the MIT License.

"""
This is a copy of a similar function in onnxruntime. The original function does
not work for large models.
"""

import argparse
import os
import pathlib
import sys

import onnx
from onnxruntime.tools.onnx_model_utils import (
    is_fixed_size_tensor,
    make_dim_param_fixed,
    make_input_shape_fixed,
)


def fix_output_shapes(model_path: str) -> onnx.ModelProto:
    """
    Update the output shapesof a model where the input shape/s were made fixed, if possible.
    This is mainly to make the model usage clearer if the output shapes can be inferred from the new input shapes.
    :param model: Model that had input shapes fixed.
    """

    # get a version of the model with shape inferencing info in it. this will provide fixed output shapes if possible.
    model = onnx.load(model_path)
    onnx.shape_inference.infer_shapes_path(model_path)
    # onnx.checker.check_model(model_path)
    m2 = onnx.load(model_path)

    for idx, o in enumerate(model.graph.output):
        if not is_fixed_size_tensor(o):
            new_o = m2.graph.output[idx]
            if is_fixed_size_tensor(new_o):
                o.type.tensor_type.shape.CopyFrom(new_o.type.tensor_type.shape)

    return m2


def make_dynamic_shape_fixed_helper() -> None:
    parser = argparse.ArgumentParser(
        f"{os.path.basename(__file__)}:{make_dynamic_shape_fixed_helper.__name__}",
        description="""
                                     Assign a fixed value to a dim_param or input shape
                                     Provide either dim_param and dim_value or input_name and input_shape.""",
    )

    parser.add_argument(
        "--dim_param",
        type=str,
        required=False,
        nargs="*",
        help="Symbolic parameter name. Provide dim_value if specified.",
    )
    parser.add_argument(
        "--dim_value",
        type=int,
        required=False,
        nargs="*",
        help="Value to replace dim_param with in the model. Must be > 0.",
    )
    parser.add_argument(
        "--input_name",
        type=str,
        required=False,
        help="Model input name to replace shape of. Provide input_shape if specified.",
    )
    parser.add_argument(
        "--input_shape",
        type=lambda x: [int(i) for i in x.split(",")],
        required=False,
        help="Shape to use for input_shape. Provide comma separated list for the shape. "
        "All values must be > 0. e.g. --input_shape 1,3,256,256",
    )
    parser.add_argument(
        "--external-data",
        action="store",
        help="If nonempty, save with external data with this name",
    )

    parser.add_argument("input_model", type=pathlib.Path, help="Provide path to ONNX model to update.")
    parser.add_argument(
        "output_model",
        type=pathlib.Path,
        help="Provide path to write updated ONNX model to.",
    )

    args = parser.parse_args()

    if (
        (args.dim_param and args.input_name)
        or (not args.dim_param and not args.input_name)
        # or (args.dim_param and (not args.dim_value or args.dim_value < 1))
        or (args.input_name and (not args.input_shape or any(value < 1 for value in args.input_shape)))
    ):
        print("Invalid usage.")
        parser.print_help()
        sys.exit(-1)

    assert len(args.dim_param) == len(args.dim_value)

    input_model_path = str(args.input_model.resolve(strict=True))
    output_model_path = str(args.output_model)
    model = onnx.load(input_model_path)

    if args.dim_param:
        for param, value in zip(args.dim_param, args.dim_value, strict=True):
            make_dim_param_fixed(model.graph, param, value)
    else:
        make_input_shape_fixed(model.graph, args.input_name, args.input_shape)

    external_data_name = args.external_data
    save_as_external = bool(external_data_name)
    external_data_path: pathlib.Path = args.output_model.parent / external_data_name
    onnx.save_model(
        model,
        output_model_path,
        save_as_external_data=save_as_external,
        location=external_data_name,
    )

    # update the output shapes to make them fixed if possible.
    new_model = fix_output_shapes(output_model_path)

    # converted_model = onnx.version_converter.convert_version(new_model, 17)

    external_data_path.unlink()
    onnx.save_model(
        new_model,
        output_model_path,
        save_as_external_data=save_as_external,
        location=external_data_name,
    )


if __name__ == "__main__":
    make_dynamic_shape_fixed_helper()
