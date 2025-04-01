#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

try:
    import onnx
except ModuleNotFoundError as e:
    logger.error(str(e))
    raise ModuleNotFoundError("Please install onnx package if exporting onnx graph. " + str(e)) from e


def convert_model_to_uint4_int4(onnx_graph: str) -> None:
    logger.info("Converting to int4/uint4 onnx model...")
    model = onnx.load(onnx_graph)
    graph = model.graph
    for node in model.graph.node:
        if node.op_type == 'QuantizeLinear':
            node_name = node.input[2]
            for node in graph.initializer:
                if node_name == node.name:
                    found_node = node
                    break
            if found_node.data_type == 2:
                found_node.data_type = 21
            elif found_node.data_type == 3:
                found_node.data_type = 22
        elif node.op_type == 'DequantizeLinear':
            node_name = node.input[2]
            for node in graph.initializer:
                if node_name in node.name:
                    found_node = node
                    break
            if found_node.data_type == 2:
                found_node.data_type = 21
            elif found_node.data_type == 3:
                found_node.data_type = 22

    onnx_dir = os.path.dirname(onnx_graph)
    uint4_int4_dir = os.path.join(onnx_dir, "uint4_int4_onnx")
    if not os.path.exists(uint4_int4_dir):
        os.makedirs(uint4_int4_dir)

    uint4_int4_onnx = os.path.join(uint4_int4_dir, "quark_model.onnx")
    onnx.save_model(model, uint4_int4_onnx, save_as_external_data=True)
    logger.info("Converted to int4/uint4 onnx model successfully")
    logger.info("Quantized onnx model exported to {} successfully.".format(uint4_int4_onnx))
