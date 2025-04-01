#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Rmove bfloat16 cast ops for an onnx model.

:param input_model_path: the path of input bfloat16 quantized model with bfloat16 cast
:param output_model_path: the path of bfloat16 quantized model with no bfloat16 cast

Use the remove_bf16_cast.py to remove bfloat16 cast for a bfloat16 quantized model:

```
python remove_bf16_cast.py --input $INPUT_MODEL_PATH --output $OUTPUT_MODEL_PATH
```

'''
import onnx
import torch
import numpy as np
from onnx import checker
from onnx import numpy_helper, TensorProto, ModelProto, NodeProto
from numpy.typing import NDArray
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def get_input_tensor_to_node_dict(model: ModelProto) -> Dict[str, NodeProto]:
    input_tensor_to_node_dict: Dict[str, NodeProto] = {}
    for node in model.graph.node:
        for input_ in node.input:
            if input_ not in input_tensor_to_node_dict:
                input_tensor_to_node_dict[input_] = []  # type: ignore
            input_tensor_to_node_dict[input_].append(node)
    return input_tensor_to_node_dict


def remove_couples_of_cast(model: ModelProto, input_tensor_to_node_dict: Dict[str, NodeProto]) -> ModelProto:

    input_list = []
    for input_ in model.graph.input:
        input_list.append(input_.name)

    output_list = []
    for output in model.graph.output:
        output_list.append(output.name)

    nodes_to_remove = []
    edges_to_reconnect: List[Tuple[NodeProto, str, NodeProto]] = []

    for node in model.graph.node:
        first_node = node
        for i in range(len(first_node.output)):
            if first_node.output[i] in input_tensor_to_node_dict and len(
                    input_tensor_to_node_dict[first_node.output[i]]) == 1:
                second_node = input_tensor_to_node_dict[first_node.output[i]][0]
                if (second_node.op_type == "Cast" and len(second_node.attribute) == 1
                        and second_node.attribute[0].name == "to" and second_node.attribute[0].i == 16
                        and len(second_node.output) == 1 and second_node.output[0] in input_tensor_to_node_dict
                        and len(input_tensor_to_node_dict[second_node.output[0]]) == 1):
                    third_node = input_tensor_to_node_dict[second_node.output[0]][0]
                    if (third_node.op_type == "Cast" and len(third_node.attribute) == 1
                            and third_node.attribute[0].name == "to" and third_node.attribute[0].i == 1
                            and third_node.output[0] not in output_list
                            and third_node.output[0] in input_tensor_to_node_dict):
                        fourth_nodes = input_tensor_to_node_dict[third_node.output[0]]
                        nodes_to_remove.extend([second_node, third_node])
                        edges_to_reconnect.extend(
                            (first_node, third_node.output[0], fourth_node) for fourth_node in fourth_nodes)
    nodes = model.graph.node
    for node in nodes_to_remove:
        nodes.remove(node)
    for first_node, third_node_output, fourth_node in edges_to_reconnect:
        for i in range(len(fourth_node.input)):
            if fourth_node.input[i] == third_node_output:
                fourth_node.input[i] = first_node.output[0]

    checker.check_model(model)
    return model


def float32_to_bfloat16(x: NDArray[np.float32]) -> NDArray[np.float32]:
    bfloat16_array = torch.tensor(x).to(torch.bfloat16)
    float32_back_array = bfloat16_array.to(torch.float32)
    new_x = float32_back_array.numpy()
    return new_x


def convert_bf16_cast_to_fp32_weights(model: ModelProto, input_tensor_to_node_dict: Dict[str, NodeProto]) -> ModelProto:
    cast_cast_node_list = []
    for node in model.graph.node:
        first_node = node
        if (first_node.op_type == "Cast" and len(first_node.attribute) == 1 and first_node.attribute[0].name == "to"
                and first_node.attribute[0].i == 16 and len(first_node.output) == 1
                and first_node.output[0] in input_tensor_to_node_dict
                and len(input_tensor_to_node_dict[first_node.output[0]]) == 1):
            second_node = input_tensor_to_node_dict[first_node.output[0]][0]
            if (second_node.op_type == "Cast" and len(second_node.attribute) == 1
                    and second_node.attribute[0].name == "to" and second_node.attribute[0].i == 1
                    and second_node.output[0] in input_tensor_to_node_dict):
                third_node = input_tensor_to_node_dict[second_node.output[0]][0]
                cast_cast_node_list.append((first_node, second_node, third_node))

    for first_node, second_node, third_node in cast_cast_node_list:
        init_name = first_node.input[0]
        for init in model.graph.initializer:
            if init.name == init_name:
                float32_init = onnx.numpy_helper.to_array(init)
                bfloat16_init = float32_to_bfloat16(float32_init)
                new_tensor = numpy_helper.from_array(bfloat16_init, name=init.name + "_bf16")
                new_tensor.data_type = TensorProto.FLOAT
                second_node_output = second_node.output[0]
                for i in range(len(third_node.input)):
                    if third_node.input[i] == second_node_output:
                        third_node.input[i] = new_tensor.name
                        model.graph.initializer.append(new_tensor)
                        model.graph.node.remove(first_node)
                        model.graph.node.remove(second_node)
                        model.graph.initializer.remove(init)

    checker.check_model(model)
    return model


def remove_output_cast(model: ModelProto, input_tensor_to_node_dict: Dict[str, NodeProto]) -> ModelProto:
    output_list = []
    for output in model.graph.output:
        output_list.append(output.name)

    for node in model.graph.node:
        first_node = node
        if (len(first_node.output) == 1 and first_node.output[0] in input_tensor_to_node_dict
                and len(input_tensor_to_node_dict[first_node.output[0]]) == 1):
            second_node = input_tensor_to_node_dict[first_node.output[0]][0]
            if (second_node.op_type == "Cast" and len(second_node.attribute) == 1
                    and second_node.attribute[0].name == "to" and second_node.attribute[0].i == 16
                    and len(second_node.output) == 1 and second_node.output[0] in input_tensor_to_node_dict
                    and len(input_tensor_to_node_dict[second_node.output[0]]) == 1):
                third_node = input_tensor_to_node_dict[second_node.output[0]][0]
                if (third_node.op_type == "Cast" and len(third_node.attribute) == 1
                        and third_node.attribute[0].name == "to" and third_node.attribute[0].i == 1
                        and third_node.output[0] in output_list):
                    model.graph.node.remove(second_node)
                    model.graph.node.remove(third_node)
                    first_node.output[0] = third_node.output[0]

    checker.check_model(model)
    return model


def parse_args() -> Namespace:
    parser = ArgumentParser("RemoveBF16Cast")
    parser.add_argument("input_model_path", type=str)
    parser.add_argument("output_model_path", type=str)
    args, _ = parser.parse_known_args()
    return args


def remove_bf16_cast(model: ModelProto) -> ModelProto:
    input_tensor_to_node_dict = get_input_tensor_to_node_dict(model)
    model = remove_couples_of_cast(model, input_tensor_to_node_dict)
    model = convert_bf16_cast_to_fp32_weights(model, input_tensor_to_node_dict)
    model = remove_output_cast(model, input_tensor_to_node_dict)
    return model


if __name__ == '__main__':
    args = parse_args()
    model = onnx.load(args.input_model_path)
    model = remove_bf16_cast(model)
    onnx.save(model, args.output_model_path)
    logger.info(
        f"Removed the bfloat16 cast from the model {args.input_model_path} to the model {args.output_model_path}.")
