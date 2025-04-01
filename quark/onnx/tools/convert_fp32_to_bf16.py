#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert tensor float type in the ONNX ModelProto input to tensor bfloat16.

Use the convert_fp32_to_bf16.py to convert a float32 model to a bfloat16 model:

```
python convert_fp32_to_bf16.py --input $FLOAT_32_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH
```

'''

import copy
import onnx
from onnx import onnx_pb as onnx_proto
from argparse import ArgumentParser, Namespace
from quark.onnx.quant_utils import convert_to_bf16
from quark.onnx.quantization.config.custom_config import BF16_CONFIG
from quark.onnx.quantization.config.config import Config
from quark.onnx.quantization.api import ModelQuantizer


def parse_args() -> Namespace:
    parser = ArgumentParser("FP32TOBF16Converter")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--format", type=str, required=False, default='with_cast')
    parser.add_argument('--save_as_external_data', action='store_true')
    args, _ = parser.parse_known_args()
    return args


def convert(args: Namespace) -> None:

    if args.format not in ("bf16", "vitisqdq", "with_cast", "simulate_bf16"):
        raise ValueError(
            f"The param {args.format} is invalid. Please set the param format as bf16 or vitisqdq or with_cast or simulate_bf16. The default value is with_cast."
        )

    if args.format == "bf16":
        fp32_model = onnx.load(args.input)
        qType = onnx_proto.TensorProto.BFLOAT16
        bf16_model = convert_to_bf16(fp32_model, qType)
        onnx.save(bf16_model, args.output, save_as_external_data=args.save_as_external_data)
        print(f"Convert the float32 model {args.input} to the bfloat16 model {args.output}.")

    else:
        config_copy = copy.deepcopy(BF16_CONFIG)
        config_copy.extra_options["UseRandomData"] = True
        if args.save_as_external_data:
            config_copy.use_external_data_format = True
        if args.format == "with_cast":
            config_copy.extra_options['BF16QDQToCast'] = True
        if args.format == "simulate_bf16":
            config_copy.extra_options['EnableVaimlBF16'] = True
        quant_config = Config(global_quant_config=config_copy)
        quantizer = ModelQuantizer(quant_config)
        quantizer.quantize_model(args.input, args.output, None)


if __name__ == '__main__':
    args = parse_args()
    convert(args)
