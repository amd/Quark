#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Insert Clip before bfloat16 activation VitisQDQ
'''
import onnx
from onnx import helper, ModelProto, GraphProto, NodeProto, TensorProto
from onnxruntime.quantization.onnx_model import ONNXModel
import argparse
import os
from quark.shares.utils.log import ScreenLogger
from typing import Any

logger = ScreenLogger(__name__)


def insert_clip_bfloat16_qdq(model: ModelProto) -> Any:
    graph = model.graph
    onnx_model = ONNXModel(model)

    def check_bfloat16_activation_qdq(graph: GraphProto, node: NodeProto) -> bool:
        if node.op_type == 'VitisQuantizeLinear':
            input_0 = onnx_model.get_initializer(node.input[0])
            zp = onnx_model.get_initializer(node.input[2])
            if (input_0 is None) and (zp.data_type == TensorProto.BFLOAT16):
                return True
            else:
                return False
        else:
            return False

    try:
        for node in graph.node:
            if check_bfloat16_activation_qdq(graph, node):
                bf16_max = 3.38953139e38
                min_initializer = helper.make_tensor(node.input[0] + "_clip_min", TensorProto.FLOAT, [], [-bf16_max])
                max_initializer = helper.make_tensor(node.input[0] + "_clip_max", TensorProto.FLOAT, [], [bf16_max])
                onnx_model.model.graph.initializer.extend([min_initializer, max_initializer])
                clip_node = helper.make_node(
                    "Clip",
                    [node.input[0], min_initializer.name, max_initializer.name],
                    [node.input[0] + "_clip_output"],
                )
                node.input[0] = node.input[0] + "_clip_output"
                print(node.input[0])
                onnx_model.add_node(clip_node)

        onnx_model.clean_initializers()
        onnx_model.topological_sort()

        logger.info("Insert Clip before BFloat16 activition VitisQuantizeLinear")
    except Exception as e:
        logger.warning(f"Exception in inserting Clip before BFloat16 activition VitisQuantizeLinear: {e}")

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
    model = insert_clip_bfloat16_qdq(origin_model)
    onnx.save(model, FLAGS.output_model)

    # Save the modified model
    logger.info('Insertion Finished!')
    logger.info(f'model saved in: {FLAGS.output_model}')


if __name__ == '__main__':
    main()
