#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert resize op's float scale to pof2s.

    Example : python -m quark.onnx.tools.convert_resize_fs_to_pof2s --input_model INPUT_MODEL_PATH --output_model OUTPUT_MODEL_PATH

'''

import numpy as np
import onnx
from onnx import NodeProto, TensorProto
from argparse import ArgumentParser, Namespace


def scale2pos(scale: float) -> int:
    """
    Obtain the fixed-point position corresponding to the scale.
    To avoid generating infinity during computations,
    the range of scale is limited.
    :param scale: the scale
    :return: the fixed-point position
    """
    scale = min(max(scale, float(2**-127)), float(2**127))
    return int(np.floor(-np.log2(scale)))


def pos2scale(pos: int) -> float:
    """
    Obtain the scale corresponding to the fixed-point position.
    :param scale: the fixed-point position
    :return: the scale
    """
    return float(np.power(2.0, -pos))


def fs_to_pof2s(node: NodeProto, initializer_map: dict[str, TensorProto]) -> None:
    scale_name = node.input[1]
    zero_point_name = node.input[2]

    scale_init = initializer_map[scale_name]
    zero_init = initializer_map[zero_point_name]

    scale_np = onnx.numpy_helper.to_array(scale_init).astype(float)
    zero_point_np = onnx.numpy_helper.to_array(zero_init).astype(float)

    f_min = (-128 - zero_point_np) * scale_np
    f_max = (127 - zero_point_np) * scale_np
    new_scale = np.maximum(np.abs(f_max), np.abs(f_min)) / 128
    new_zero = np.array(0, dtype=np.int8)

    pos = scale2pos(new_scale.item())
    pof2_scale = np.array(pos2scale(pos), dtype=scale_np.dtype)

    scale_init.CopyFrom(
        onnx.helper.make_tensor(name=scale_init.name,
                                data_type=onnx.TensorProto.FLOAT,
                                dims=scale_init.dims,
                                vals=pof2_scale.flatten().tolist()))

    zero_init.CopyFrom(
        onnx.helper.make_tensor(name=zero_init.name,
                                data_type=onnx.TensorProto.INT8,
                                dims=zero_init.dims,
                                vals=new_zero.flatten().tolist()))


def convert_resize_fs_to_pof2s(input_model: str, output_model: str) -> None:
    model = onnx.load(input_model)

    q_nodes = []
    dq_nodes = []
    resize_nodes = []

    for node in model.graph.node:
        if node.op_type == 'QuantizeLinear':
            q_nodes.append(node)
        elif node.op_type == 'DequantizeLinear':
            dq_nodes.append(node)
        elif node.op_type == 'Resize':
            resize_nodes.append(node)

    initializer_map = {init.name: init for init in model.graph.initializer}

    # Now, let's find the QuantizeLinear and DequantizeLinear nodes before and after Resize
    for resize_node in resize_nodes:
        resize_input = resize_node.input[0]
        resize_output = resize_node.output[0]

        # Find the preceding DequantizeLinear node
        for dq_before in dq_nodes:
            if dq_before.output[0] == resize_input:
                for q_before in q_nodes:
                    if q_before.output[0] == dq_before.input[0]:
                        if len(dq_before.input) > 1:
                            fs_to_pof2s(q_before, initializer_map)
                        if len(q_before.input) > 1:
                            fs_to_pof2s(dq_before, initializer_map)

        # Find the succeeding QuantizeLinear node
        for q_after in q_nodes:
            if q_after.input[0] == resize_output:
                for dq_after in dq_nodes:
                    if q_after.output[0] == dq_after.input[0]:
                        if len(dq_after.input) > 1:
                            fs_to_pof2s(q_after, initializer_map)
                        if len(q_after.input) > 1:
                            fs_to_pof2s(dq_after, initializer_map)

    onnx.save(model, output_model)
    print(f"Converted model saved to {output_model}")


def parse_args() -> Namespace:
    usage_str = "python -m quark.onnx.tools.convert_resize_fs_to_pof2s.py --input_model INPUT_MODEL_PATH --output_model OUTPUT_MODEL_PATH"
    parser = ArgumentParser("convert_resize_fs_to_pof2s", usage=usage_str)
    parser.add_argument("--input_model", type=str, default="", help="input onnx model file path.")
    parser.add_argument("--output_model", type=str, default="", help="output onnx model file path.")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    convert_resize_fs_to_pof2s(args.input_model, args.output_model)


if __name__ == "__main__":
    main()
