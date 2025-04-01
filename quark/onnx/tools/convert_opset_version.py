#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert the opset version of input model.

:param input: the path of input model
:param target_opset: the target opset version
:param output: the path of output model

Use the convert_opset_version to convert a model's opset version:

```
python convert_opset_version.py --input $INPUT_ONNX_MODEL_PATH --target_opset &TARGET_OPSET_VERSION --output $OUTPUT_ONNX_MODEL_PATH
```

'''

import onnx
from onnx import version_converter
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser("ConvertOpsetVersion")
    parser.add_argument("input", type=str)
    parser.add_argument("target_opset", type=int, default=20)
    parser.add_argument("output", type=str)
    args, _ = parser.parse_known_args()
    return args


def convert_opset_version(model: onnx.ModelProto, target_opset: int) -> onnx.ModelProto:
    opset_version = model.opset_import[0].version
    print(f"The current model's opset version is: {opset_version}")
    converted_model = version_converter.convert_version(model, target_opset)
    return converted_model


if __name__ == '__main__':
    args = parse_args()
    model = onnx.load(args.input)
    converted_model = convert_opset_version(model, args.target_opset)
    onnx.save(converted_model, args.output)
    print(f"Convert the model {args.input} to the model {args.output} with opset version {args.target_opset}")
