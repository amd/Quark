#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert tensor float16 type in the ONNX ModelProto input to tensor bfloat16.

Use the convert_fp16_to_bf16.py to convert a float16 model to a bfloat16 model:

```
python convert_fp16_to_bf16.py --input $FLOAT_16_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH
```

'''
import tempfile
import copy
import onnx
from . import float16
from pathlib import Path
from typing import Dict, List
from onnx import onnx_pb as onnx_proto
from argparse import ArgumentParser, Namespace
from quark.onnx.quant_utils import convert_to_bf16
from quark.onnx.quantization.config.custom_config import BF16_CONFIG
from quark.onnx.quantization.config.config import Config
from quark.onnx.quantization.api import ModelQuantizer
from onnx.onnx_ml_pb2 import ModelProto, NodeProto


def parse_args() -> Namespace:
    parser = ArgumentParser("FP16TOBF16Converter")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--format", type=str, required=False, default='with_cast')
    parser.add_argument('--save_as_external_data', action='store_true')
    args, _ = parser.parse_known_args()
    return args


def add_fp16_input_output_cast(model: ModelProto) -> ModelProto:
    add_node_list = []

    input_list = []
    for input_tensor in model.graph.input:
        if input_tensor.type.tensor_type.elem_type == 1:
            input_tensor.type.tensor_type.elem_type = 10
            input_list.append(input_tensor.name)
    for node in model.graph.node:
        for i in range(len(node.input)):
            input_ = node.input[i]
            if input_ in input_list:
                node.input[i] = input_ + "_cast"
                cast_node = onnx.helper.make_node("Cast",
                                                  inputs=[input_],
                                                  outputs=[input_ + "_cast"],
                                                  to=onnx_proto.TensorProto.FLOAT)
                add_node_list.append(cast_node)

    input_to_node: Dict[str, List[NodeProto]] = {}
    for node in model.graph.node:
        for input_ in node.input:
            if input_ not in input_to_node:
                input_to_node[input_] = []
            input_to_node[input_].append(node)

    output_list = []
    for output_tensor in model.graph.output:
        if output_tensor.type.tensor_type.elem_type == 1:
            output_tensor.type.tensor_type.elem_type = 10
            output_list.append(output_tensor.name)
    for node in model.graph.node:
        for i in range(len(node.output)):
            output_ = node.output[i]
            if output_ in output_list:
                node.output[i] = output_ + "_cast"
                cast_node = onnx.helper.make_node("Cast",
                                                  inputs=[output_ + "_cast"],
                                                  outputs=[output_],
                                                  to=onnx_proto.TensorProto.FLOAT16)
                add_node_list.append(cast_node)
                if output_ in input_to_node:
                    for after_node in input_to_node[output_]:
                        for j in range(len(after_node.input)):
                            input_ = after_node.input[j]
                            if input_ == output_:
                                after_node.input[j] = output_ + "_cast"

    for cast_node in add_node_list:
        model.graph.node.append(cast_node)
    return model


def convert(args: Namespace) -> None:

    if args.format not in ("bf16", "vitisqdq", "with_cast", "simulate_bf16"):
        raise ValueError(
            f"The param {args.format} is invalid. Please set the param format as bf16 or vitisqdq or with_cast or simulate_bf16. The default value is with_cast."
        )

    fp16_model = onnx.load(args.input)
    if args.format == "bf16":
        qType = onnx_proto.TensorProto.BFLOAT16
        bf16_model = convert_to_bf16(fp16_model, qType, original_data_type=10)
        onnx.save(bf16_model, args.output, save_as_external_data=args.save_as_external_data)
        print(f"Convert the float16 model {args.input} to the bfloat16 model {args.output}.")

    else:
        fp32_model = float16.convert_float16_to_float(fp16_model)
        fp32_path = tempfile.TemporaryDirectory(prefix="vai.tools.")
        fp32_input_path = Path(fp32_path.name).joinpath("fp32.onnx").as_posix()
        onnx.save_model(fp32_model, fp32_input_path, save_as_external_data=args.save_as_external_data)
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
        bf16_path = tempfile.TemporaryDirectory(prefix="vai.tools.")
        bf16_with_fp32_input_output_path = Path(bf16_path.name).joinpath("bf16.onnx").as_posix()
        quantizer.quantize_model(fp32_input_path, bf16_with_fp32_input_output_path, None)
        bf16_with_fp32_input_output_model = onnx.load(bf16_with_fp32_input_output_path)
        bf16_model = add_fp16_input_output_cast(bf16_with_fp32_input_output_model)
        onnx.save(bf16_model, args.output, save_as_external_data=args.save_as_external_data)


if __name__ == '__main__':
    args = parse_args()
    convert(args)
