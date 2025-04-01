#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert u16u8 to u8u8.
'''

import numpy as np
import onnx
from onnx import onnx_pb as onnx_proto
from onnxruntime.quantization.onnx_model import ONNXModel
from argparse import ArgumentParser, Namespace
from typing import Union, List


def parse_args() -> Namespace:
    parser = ArgumentParser("U16U8ToU8U8Converter")
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args, _ = parser.parse_known_args()
    return args


ONNX_INT_TO_ONNX_TYPE = {
    1: onnx_proto.TensorProto.FLOAT,
    2: onnx_proto.TensorProto.UINT8,
    3: onnx_proto.TensorProto.INT8,
    4: onnx_proto.TensorProto.UINT16,
    5: onnx_proto.TensorProto.INT16,
    6: onnx_proto.TensorProto.INT32,
    7: onnx_proto.TensorProto.INT64,
    8: onnx_proto.TensorProto.STRING,
    9: onnx_proto.TensorProto.BOOL,
    10: onnx_proto.TensorProto.FLOAT16,
    11: onnx_proto.TensorProto.DOUBLE,
    12: onnx_proto.TensorProto.UINT32,
    13: onnx_proto.TensorProto.UINT64,
}

ONNX_INT_TYPE_RANGE = {
    onnx_proto.TensorProto.UINT8: (0, 255),
    onnx_proto.TensorProto.INT8: (-128, 127),
    onnx_proto.TensorProto.UINT16: (0, 65535),
    onnx_proto.TensorProto.INT16: (-32768, 32767),
    onnx_proto.TensorProto.UINT32: (0, 2**32 - 1),
    onnx_proto.TensorProto.INT32: (-2**31, 2**31 - 1),
}

ONNX_TYPE_TO_NP_TYPE = {
    onnx_proto.TensorProto.INT8: np.int8,
    onnx_proto.TensorProto.UINT8: np.uint8,
    onnx_proto.TensorProto.INT16: np.int16,
    onnx_proto.TensorProto.UINT16: np.uint16,
    onnx_proto.TensorProto.INT32: np.int32,
    onnx_proto.TensorProto.UINT32: np.uint32,
    onnx_proto.TensorProto.FLOAT16: np.float16,
    # This is mismatched conversion,
    # numpy does not support yet
    onnx_proto.TensorProto.BFLOAT16: np.float16,
}

OperationsQ = ['QuantizeLinear', 'VitisQuantizeLinear']
OperationsDQ = ['DequantizeLinear', 'VitisDequantizeLinear']
OperationsWithBias = ['Conv', 'ConvTranspose', 'Gemm']

SOURCE_TYPE = onnx.TensorProto.UINT16
TARGET_TYPE = onnx.TensorProto.UINT8


def convert_u16u8_to_u8u8(input_model_path: str, output_model_path: str) -> None:
    model = onnx.load(input_model_path)
    onnx_model = ONNXModel(model)

    output_name_to_node = onnx_model.output_name_to_node()
    input_name_to_nodes = onnx_model.input_name_to_nodes()

    model_inputs = [inp.name for inp in onnx_model.model.graph.input]
    updated_initializers: List[str] = []

    def _modify_scale_value(scale_init: onnx.TensorProto, zp_init: onnx.TensorProto,
                            quant_type: onnx.TensorProto.DataType) -> Union[onnx.TensorProto, None]:
        source_dtype: onnx.TensorProto.DataType = ONNX_INT_TO_ONNX_TYPE[zp_init.data_type]
        target_dtype: onnx.TensorProto.DataType = quant_type
        if source_dtype == target_dtype:
            return None
        elif source_dtype not in ONNX_INT_TYPE_RANGE or target_dtype not in ONNX_INT_TYPE_RANGE:
            return None

        source_qrange = ONNX_INT_TYPE_RANGE[source_dtype][1] - ONNX_INT_TYPE_RANGE[source_dtype][0]
        target_qrange = ONNX_INT_TYPE_RANGE[target_dtype][1] - ONNX_INT_TYPE_RANGE[target_dtype][0]

        scale = onnx.numpy_helper.to_array(scale_init)
        new_scale = scale * source_qrange / target_qrange
        new_scale = new_scale.astype(np.float32)

        new_init = onnx.numpy_helper.from_array(new_scale, name=scale_init.name)
        return new_init

    def _create_zp_value_and_datatype(init: onnx.TensorProto,
                                      quant_type: onnx.TensorProto.DataType) -> Union[onnx.TensorProto, None]:
        source_dtype: onnx.TensorProto.DataType = ONNX_INT_TO_ONNX_TYPE[init.data_type]
        target_dtype: onnx.TensorProto.DataType = quant_type
        if source_dtype == target_dtype:
            return None
        elif source_dtype not in ONNX_INT_TYPE_RANGE or target_dtype not in ONNX_INT_TYPE_RANGE:
            return None

        source_qrange = ONNX_INT_TYPE_RANGE[source_dtype][1] - ONNX_INT_TYPE_RANGE[source_dtype][0]
        target_qrange = ONNX_INT_TYPE_RANGE[target_dtype][1] - ONNX_INT_TYPE_RANGE[target_dtype][0]

        zp = onnx.numpy_helper.to_array(init)
        zp = zp - np.asarray(ONNX_INT_TYPE_RANGE[source_dtype][0])
        zp = zp * np.asarray(target_qrange / source_qrange, dtype=np.float32)
        zp = np.asarray(round(zp + ONNX_INT_TYPE_RANGE[target_dtype][0]))
        zp = zp.astype(ONNX_TYPE_TO_NP_TYPE[target_dtype])

        new_init = onnx.numpy_helper.from_array(zp, name=init.name)
        return new_init

    def _update_node_scale(node: onnx.NodeProto, updated_initializers: List[str]) -> None:
        scale_init = onnx_model.get_initializer(node.input[1])
        if scale_init is None:
            # print(f"ERROR: node '{node.name}' has no scale")
            pass
        else:
            zp_init = onnx_model.get_initializer(node.input[2])
            if zp_init is None:
                # print(f"ERROR: node '{node.name}' has no zero_point")
                pass
            else:
                if zp_init.data_type != SOURCE_TYPE:
                    # print(f"WARNING: unexpected zero_point '{zp_init.name}' with type {zp_init.data_type}")
                    pass
                else:
                    new_scale_init = _modify_scale_value(scale_init, zp_init, TARGET_TYPE)
                    if new_scale_init is not None and scale_init.name not in updated_initializers:
                        onnx_model.remove_initializer(scale_init)
                        onnx_model.add_initializer(new_scale_init)
                        updated_initializers.append(scale_init.name)
                        # print(f"INFO: updated scale of {scale_init.name}")

    def _update_node_zp(node: onnx.NodeProto, updated_initializers: List[str]) -> None:
        zp_init = onnx_model.get_initializer(node.input[2])
        if zp_init is not None and zp_init.name in updated_initializers:
            # print(f"WARNING: this zero_point '{zp_init.name}' has been updated")
            pass
        elif zp_init is not None and zp_init.data_type != SOURCE_TYPE:
            # print(f"WARNING: unexpected zero_point '{zp_init.name}' with type {zp_init.data_type}")
            pass
        else:
            new_zp_init = _create_zp_value_and_datatype(zp_init, TARGET_TYPE)
            if new_zp_init is not None and zp_init.name not in updated_initializers:
                onnx_model.remove_initializer(zp_init)
                onnx_model.add_initializer(new_zp_init)
                updated_initializers.append(zp_init.name)
                # print(f"INFO: updated zero_point of '{zp_init.name}'")

    # Let's update all scales first
    for node in onnx_model.model.graph.node:
        if node.op_type not in OperationsDQ:
            continue

        node_parent = onnx_model.get_parent(node, 0, output_name_to_node)
        if node_parent is None or node_parent.op_type not in OperationsQ:
            continue

        # Targeting the QDQ of activation
        if node_parent.input[0] not in output_name_to_node and node_parent.input[0] not in model_inputs:
            continue

        _update_node_scale(node, updated_initializers)
        if node.input[1] != node_parent.input[1]:
            _update_node_scale(node_parent, updated_initializers)

    # Update zero_points
    for node in onnx_model.model.graph.node:
        if node.op_type not in OperationsDQ:
            continue

        node_parent = onnx_model.get_parent(node, 0, output_name_to_node)
        if node_parent is None or node_parent.op_type not in OperationsQ:
            continue

        # Targeting the QDQ of activation
        if node_parent.input[0] not in output_name_to_node and node_parent.input[0] not in model_inputs:
            continue

        _update_node_zp(node, updated_initializers)
        if node.input[2] != node_parent.input[2]:
            _update_node_zp(node_parent, updated_initializers)

    # Deal with activation whose input is a constant
    for node in onnx_model.model.graph.node:
        if node.op_type not in OperationsDQ:
            continue

        zp_init = onnx_model.get_initializer(node.input[2])
        if zp_init.data_type != SOURCE_TYPE:
            continue

        # Targeting the constant
        weight_init = onnx_model.get_initializer(node.input[0])
        if weight_init is None:
            continue

        scale_init = onnx_model.get_initializer(node.input[1])
        old_scale = onnx.numpy_helper.to_array(scale_init)
        zp_init = onnx_model.get_initializer(node.input[2])
        old_zp = onnx.numpy_helper.to_array(zp_init)
        # Update scale and zero_point
        _update_node_scale(node, updated_initializers)
        _update_node_zp(node, updated_initializers)
        # Update the constant's initializer
        scale_init = onnx_model.get_initializer(node.input[1])
        new_scale = onnx.numpy_helper.to_array(scale_init)
        zp_init = onnx_model.get_initializer(node.input[2])
        new_zp = onnx.numpy_helper.to_array(zp_init)

        old_weight = onnx.numpy_helper.to_array(weight_init)
        old_weight = old_weight - old_zp
        weight = old_weight.astype(np.float32) * old_scale

        new_weight = weight / new_scale + new_zp
        quant_range = ONNX_INT_TYPE_RANGE[TARGET_TYPE]
        new_weight = np.clip(new_weight, quant_range[0], quant_range[1])
        new_weight = new_weight.astype(ONNX_TYPE_TO_NP_TYPE[TARGET_TYPE])

        new_weight_init = onnx.numpy_helper.from_array(new_weight, weight_init.name)

        onnx_model.remove_initializer(weight_init)
        onnx_model.add_initializer(new_weight_init)

    # Make sure scale_bias = scale_x * scale_weight
    for node in onnx_model.model.graph.node:
        if node.op_type not in OperationsWithBias:
            continue

        if len(node.input) < 3:  # It has no Bias
            continue

        x_node = onnx_model.get_parent(node, 0, output_name_to_node)
        if x_node is None or x_node.op_type not in OperationsDQ:
            continue

        x_scale_init = onnx_model.get_initializer(x_node.input[1])
        if x_scale_init is None:
            continue

        w_node = onnx_model.get_parent(node, 1, output_name_to_node)
        if w_node is None or w_node.op_type not in OperationsDQ:
            continue

        w_scale_init = onnx_model.get_initializer(w_node.input[1])
        if w_scale_init is None:
            continue

        b_node = onnx_model.get_parent(node, 2, output_name_to_node)
        if b_node is None or b_node.op_type not in OperationsDQ:
            continue

        b_scale_init = onnx_model.get_initializer(b_node.input[1])
        if b_scale_init is None:
            continue

        b_zp_init = onnx_model.get_initializer(b_node.input[2])
        if b_zp_init is None or b_zp_init.data_type != onnx_proto.TensorProto.INT32:
            continue  # Only Int32 Bias should follow the formula

        x_scale = onnx.numpy_helper.to_array(x_scale_init)
        w_scale = onnx.numpy_helper.to_array(w_scale_init)
        b_scale = x_scale * w_scale
        new_b_scale_init = onnx.numpy_helper.from_array(b_scale, b_scale_init.name)

        # print(f"INFO: updated the scale of {node.name}'s bias to meet "
        #        "scale_bias = scale_x * scale_weight")

        # Note that if didn't fold Q, the scale should be shared with Q
        onnx_model.remove_initializer(b_scale_init)
        onnx_model.add_initializer(new_b_scale_init)

    onnx_model.clean_initializers()
    onnx_model.topological_sort()

    onnx.save(onnx_model.model, output_model_path)


def convert(args: Namespace) -> None:
    input_model_path = args.input
    output_model_path = args.output

    convert_u16u8_to_u8u8(input_model_path, output_model_path)
    print(f"Convert the u16u8 model {args.input} to the u8u8 model {args.output}.")


if __name__ == '__main__':
    args = parse_args()
    convert(args)
