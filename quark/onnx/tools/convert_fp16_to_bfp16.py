#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert tensor float16 type in the ONNX ModelProto input to tensor bfp16.

Use the convert_fp16_to_bfp16.py to convert a float16 model to a bfp16 model:

```
python convert_fp16_to_bfp16.py --input $FLOAT_16_ONNX_MODEL_PATH --output $BFP_16_ONNX_MODEL_PATH
```

'''

import copy
from argparse import ArgumentParser, Namespace
from quark.onnx.quantization.api import ModelQuantizer
from quark.onnx.quantization.config.config import Config
from quark.onnx.quantization.config.custom_config import BFP16_CONFIG


def parse_args() -> Namespace:
    parser = ArgumentParser("FP16TOBFP16Converter")
    parser.add_argument("input", type=str, required=True)
    parser.add_argument("output", type=str, required=True)
    parser.add_argument('--save_as_external_data', action='store_true')
    args, _ = parser.parse_known_args()
    return args


def convert_fp16_to_bfp16(input_model_path: str, output_model_path: str) -> None:
    config_copy = copy.deepcopy(BFP16_CONFIG)
    config_copy.extra_options["UseRandomData"] = True
    if args.save_as_external_data:
        config_copy.use_external_data_format = True
    config_copy.extra_options['QuantizeFP16'] = True
    config_copy.extra_options['UseFP32Scale'] = True
    quant_config = Config(global_quant_config=config_copy)
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(input_model_path, output_model_path, None)


def convert(args: Namespace) -> None:
    convert_fp16_to_bfp16(args.input, args.output)
    print(f"Convert the float16 model {args.input} to the bfp16 model {args.output}.")


if __name__ == '__main__':
    args = parse_args()
    convert(args)
