#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Replace BFloat16 QDQ with Cast op.
'''
import onnx
from onnx import helper, numpy_helper, ModelProto
from onnxruntime.quantization.onnx_model import ONNXModel
import argparse
import os
import numpy as np
from quark.shares.utils.log import ScreenLogger
from typing import Any, Tuple

logger = ScreenLogger(__name__)


def replace_bfloat16_qdq_cast(model: ModelProto) -> Any:
    # Load the ONNX model

    graph = model.graph

    new_nodes = []

    def check_second_third_input(graph: onnx.GraphProto, node: onnx.NodeProto) -> Tuple[bool, Any]:
        # Ensure the node has at least 3 inputs
        if len(node.input) < 3:
            return False, None

        second_input_name = node.input[1]
        third_input_name = node.input[2]

        # Get tensor values for second and third inputs
        scale_value = None
        zero_point_value = None
        third_input_type = None

        for initializer in graph.initializer:
            if initializer.name == second_input_name:
                scale_value = numpy_helper.to_array(initializer)
            if initializer.name == third_input_name:
                zero_point_value = numpy_helper.to_array(initializer)
                third_input_type = initializer.data_type

        # Check that the third input is bfloat16 and zero_point is 0
        if third_input_type != onnx.TensorProto.BFLOAT16 or not np.all(zero_point_value == 0):
            return False, None

        # Return True and the scale value
        return True, scale_value

    try:
        onnx_model = ONNXModel(model)
        for node in graph.node:
            if node.op_type in {'VitisQuantizeLinear', 'VitisDequantizeLinear'}:
                # Check if second input (scale) and third input (zero_point) meet the conditions
                is_valid, scale_value = check_second_third_input(graph, node)

                if is_valid:
                    # If scale is not 1, prepare the scale or reciprocal of scale for Mul node
                    if scale_value is not None and np.all(scale_value != 1):
                        scale_tensor_name = f"{node.name}_scale"
                        reciprocal_scale = 1.0 / scale_value if node.op_type == 'VitisQuantizeLinear' else scale_value

                        # Convert scale to ndarray and add to initializers
                        scale_initializer = helper.make_tensor(name=scale_tensor_name,
                                                               data_type=onnx.TensorProto.FLOAT,
                                                               dims=scale_value.shape,
                                                               vals=reciprocal_scale.flatten() if node.op_type
                                                               == 'VitisQuantizeLinear' else scale_value.flatten())
                        graph.initializer.append(scale_initializer)

                    # Replace node with Cast and Mul if scale != 1
                    if node.op_type == 'VitisQuantizeLinear':
                        # Add Mul before the Cast with scale's reciprocal
                        if scale_value is not None and np.all(scale_value != 1):
                            mul_before_cast = helper.make_node(
                                'Mul',
                                inputs=[node.input[0], scale_tensor_name],
                                outputs=[f"{node.name}_mul_out"]  # Intermediate output before Cast
                            )
                            new_nodes.append(mul_before_cast)
                            cast_input = f"{node.name}_mul_out"  # Mul output as input to Cast
                        else:
                            cast_input = node.input[0]  # Direct input to Cast if scale == 1

                        # Create Cast to Bfloat16
                        cast_to_bfloat16 = helper.make_node(
                            'Cast',
                            inputs=[cast_input],
                            outputs=node.output,  # Final output of the QuantizeLinear node
                            to=onnx.TensorProto.BFLOAT16)
                        new_nodes.append(cast_to_bfloat16)

                    elif node.op_type == 'VitisDequantizeLinear':
                        # Create Cast to Float
                        cast_to_float = helper.make_node(
                            'Cast',
                            inputs=[node.input[0]],  # Only keep the first input
                            outputs=[f"{node.name}_cast_out"],  # Intermediate output after Cast
                            to=onnx.TensorProto.FLOAT)
                        new_nodes.append(cast_to_float)

                        # Add Mul after the Cast with the original scale value
                        if scale_value is not None and np.all(scale_value != 1):
                            mul_after_cast = helper.make_node(
                                'Mul',
                                inputs=[f"{node.name}_cast_out", scale_tensor_name],
                                outputs=node.output  # Final output of the DequantizeLinear node
                            )
                            new_nodes.append(mul_after_cast)
                        else:
                            cast_to_float.output[0] = node.output[0]  # Directly use the cast output if scale == 1
                else:
                    # If the condition is not met, keep the original node
                    new_nodes.append(node)
            else:
                # Keep other nodes unchanged
                new_nodes.append(node)

        # Replace the graph's nodes with the new node list
        graph.ClearField('node')
        graph.node.extend(new_nodes)

        onnx_model.clean_initializers()
        onnx_model.topological_sort()

        logger.info(
            "Replaced Bfloat16 VitisQuantizeLinear and VitisDequantizeLinear to Cast with optional Mul for scale.")
    except Exception as e:
        logger.warning(f"Exception in replacing Bfloat16 VitisQuantizeLinear and VitisDequantizeLinear to Cast: {e}")

    return onnx_model.model


def main() -> None:
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=True, help="Input ONNX model file path.")
    parser.add_argument("--output_model", type=str, required=True, help="Output ONNX model file path.")
    FLAGS, _ = parser.parse_known_args()

    # Check if input file exists
    if not os.path.isfile(FLAGS.input_model):
        logger.error(f"Input model file '{FLAGS.input_model}' does not exist!")
        logger.error("Usage: python script.py --input_model INPUT_MODEL_PATH --output_model OUTPUT_MODEL_PATH.")
        exit()

    # Replace Vitis nodes with Cast
    origin_model = onnx.load(FLAGS.input_model)
    model = replace_bfloat16_qdq_cast(origin_model)
    onnx.save(model, FLAGS.output_model)

    # Save the modified model
    logger.info('Replace Finished!')
    logger.info(f'model saved in: {FLAGS.output_model}')


if __name__ == '__main__':
    main()
