#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Remove initializers from input and upgrdte ir_version if it is blow 4.
'''
import onnx
import argparse
from argparse import Namespace


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser("RemoveInitializerFromInput")
    parser.add_argument("--input", required=True, help="input model", type=str)
    parser.add_argument("--output", required=True, help="output model", type=str)
    args, _ = parser.parse_known_args()
    return args


def remove_initializer_from_input(args: Namespace) -> None:
    model = onnx.load(args.input)
    if model.ir_version < 4:
        print("Model with ir_version below 4 requires to include initilizer in graph input, change ir_version to 7")
        model.ir_version = 7

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, args.output)
    print(f"The model of removing intializers from input is saved at {args.output}.")


if __name__ == "__main__":
    args = parse_args()
    remove_initializer_from_input(args)
