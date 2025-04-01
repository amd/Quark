#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert tensor float type in the ONNX ModelProto input to tensor float16.

:param model: ONNX ModelProto object
:param disable_shape_infer: Type/shape information is needed for conversion to work.
                            Set to True only if the model already has type/shape information for all tensors.
:return: converted ONNX ModelProto object

Examples:

::

    Example 1: Convert ONNX ModelProto object:
    import float16
    new_onnx_model = float16.convert_float_to_float16(onnx_model)

    Example 2: Convert ONNX model binary file:
    import onnx
    import float16
    onnx_model = onnx.load_model('model.onnx')
    new_onnx_model = float16.convert_float_to_float16(onnx_model)
    onnx.save_model(new_onnx_model, 'new_model.onnx')

Use the convert_float32_to_float16.py to convert a float32 model to a float16 model:

```
python convert_fp32_to_fp16.py --input $FLOAT_32_ONNX_MODEL_PATH --output $FLOAT_16_ONNX_MODEL_PATH
```

The conversion from float32 models to float16 models may result in
the generation of unnecessary operations such as casts in the model.
It is recommended to use onnx-simplifier to remove these redundant nodes.
'''

import onnx
from . import float16
import onnxsim
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser("float32Converter")
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--keep_io_types", action='store_true')
    parser.add_argument('--disable_shape_infer', action='store_true')
    parser.add_argument('--save_as_external_data', action='store_true')
    parser.add_argument('--all_tensors_to_one_file', action='store_true')
    parser.add_argument('--not_simplify', action='store_true')
    args, _ = parser.parse_known_args()
    return args


def convert(args: Namespace) -> None:

    model = onnx.load(args.input)
    model_fp16 = float16.convert_float_to_float16(model,
                                                  keep_io_types=args.keep_io_types,
                                                  disable_shape_infer=args.disable_shape_infer)
    if args.not_simplify:
        model_simp = model_fp16
    else:
        try:
            model_simp, check = onnxsim.simplify(model_fp16)
            assert check, "Simplified ONNX model could not be validated"
        except Exception as e:
            print(f"Fail to Simplify ONNX model because of {e}.")
            model_simp = model_fp16

    onnx.save(model_simp,
              args.output,
              save_as_external_data=args.save_as_external_data,
              all_tensors_to_one_file=args.all_tensors_to_one_file)
    print(f"Convert the float32 model {args.input} to the float16 model {args.output}.")


if __name__ == '__main__':
    args = parse_args()
    convert(args)
