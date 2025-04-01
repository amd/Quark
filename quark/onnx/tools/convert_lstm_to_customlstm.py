#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Convert Custom QDQ to QDQ."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import onnx
import os
import argparse

from quark.onnx.quant_utils import ONNXQuantizedModel
from typing import Any


def convert_lstm_to_customlstm(model: onnx.ModelProto) -> Any:
    """Convert Custom LSTM to LSTM.
    :param model: source model
    :return: converted model
    """

    OpMapping = {"LSTM": "VitisLSTM"}
    OpDomain = "com.vai.quantize"

    parser = ONNXQuantizedModel(model)

    nodes_to_add = []
    nodes_to_remove = []

    for node in parser.onnx_model.model.graph.node:
        if node.op_type not in OpMapping:
            continue

        inputs_dq = {}
        for tensor_index, tensor_name in enumerate(node.input):
            dq, _ = parser._find_node_input_qdq(node, tensor_name)
            if dq is None:
                print(f"Node {node.name} #{tensor_index} input has no input Q/DQ")
                continue
            inputs_dq[tensor_index] = dq

        outputs_q = {}
        for tensor_index, tensor_name in enumerate(node.output):
            q, _ = parser._find_node_output_qdq(node, tensor_name)
            if q is None:
                print(f"Node {node.name} #{tensor_index} output has no output Q/DQ")
                continue
            outputs_q[tensor_index] = q

        if len(inputs_dq) < 4 or len(outputs_q) < 1:
            print(f"Node {node.name} wasn't quantized fully,"
                  f"{len(inputs_dq)} inputs and {len(outputs_q)} outputs")

        # Get inputs and outputs scale and zero_point
        assert 0 in inputs_dq
        x_scale_init = parser.onnx_model.get_initializer(inputs_dq[0].input[1])
        x_scale = onnx.numpy_helper.to_array(x_scale_init).item()
        x_zero_point_init = parser.onnx_model.get_initializer(inputs_dq[0].input[2])
        x_zero_point = onnx.numpy_helper.to_array(x_zero_point_init).item()

        assert 1 in inputs_dq
        w_scale_init = parser.onnx_model.get_initializer(inputs_dq[1].input[1])
        w_scale = onnx.numpy_helper.to_array(w_scale_init).item()
        w_zero_point_init = parser.onnx_model.get_initializer(inputs_dq[1].input[2])
        w_zero_point = onnx.numpy_helper.to_array(w_zero_point_init).item()

        assert 2 in inputs_dq
        r_scale_init = parser.onnx_model.get_initializer(inputs_dq[2].input[1])
        r_scale = onnx.numpy_helper.to_array(r_scale_init).item()
        r_zero_point_init = parser.onnx_model.get_initializer(inputs_dq[2].input[2])
        r_zero_point = onnx.numpy_helper.to_array(r_zero_point_init).item()

        assert 3 in inputs_dq
        b_scale_init = parser.onnx_model.get_initializer(inputs_dq[3].input[1])
        b_scale = onnx.numpy_helper.to_array(b_scale_init).item()
        b_zero_point_init = parser.onnx_model.get_initializer(inputs_dq[3].input[2])
        b_zero_point = onnx.numpy_helper.to_array(b_zero_point_init).item()

        assert 0 in outputs_q
        y_scale_init = parser.onnx_model.get_initializer(outputs_q[0].input[1])
        y_scale = onnx.numpy_helper.to_array(y_scale_init).item()
        y_zero_point_init = parser.onnx_model.get_initializer(outputs_q[0].input[2])
        y_zero_point = onnx.numpy_helper.to_array(y_zero_point_init).item()

        # Get stantard attributes
        direction = next((attr.s.decode() for attr in node.attribute if attr.name == 'direction'), 'bidirectional')
        hidden_size = next((attr.i for attr in node.attribute if attr.name == 'hidden_size'), 128)
        layout = next((attr.i for attr in node.attribute if attr.name == 'layout'), 0)

        new_node = onnx.helper.make_node(
            OpMapping[node.op_type],
            inputs=node.input,
            outputs=node.output,
            name=node.name,
            domain=OpDomain,
            x_scale=x_scale,
            x_zero_point=x_zero_point,
            w_scale=w_scale,
            w_zero_point=w_zero_point,
            r_scale=r_scale,
            r_zero_point=r_zero_point,
            b_scale=b_scale,
            b_zero_point=b_zero_point,
            y_scale=y_scale,
            y_zero_point=y_zero_point,
            direction=direction,
            hidden_size=hidden_size,
            layout=layout,
        )

        nodes_to_remove.append(node)
        nodes_to_add.append(new_node)
        print(f"Node {node.name} was converted to customized version")

    parser.onnx_model.remove_nodes(nodes_to_remove)
    parser.onnx_model.add_nodes(nodes_to_add)

    return parser.onnx_model.model


def custom_ops_infer_shapes(model: onnx.ModelProto) -> Any:
    """Generate value info for output tensors of custom ops.
    :param model: source model
    :return: converted model
    """
    CustomOps = (
        "VitisQuantizeLinear",
        "VitisDequantizeLinear",
        "VitisInstanceNormalization",
        "VitisLSTM",
    )

    has_customop = False

    for node in model.graph.node:
        if node.op_type in CustomOps:
            has_customop = True
            break

    if has_customop:
        from quark.onnx.quant_utils import infer_custom_op_shape as infer_shape
        print("Infer tensor's shape to generate value info for custom ops")
        return infer_shape(model)

    return model


def run_main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, default="", help="input onnx model file path.")
    parser.add_argument("--output_model", type=str, default="", help="output onnx model file path.")
    FLAGS, uparsed = parser.parse_known_args()

    if not os.path.isfile(FLAGS.input_model):
        print("Input model file '{}' does not exist!".format(FLAGS.input_model))
        print(
            "Usage: python -m quark.onnx.tools.convert_lstm_to_customlstm --input_model INPUT_MODEL_PATH --output_model OUTPUT_MODEL_PATH."
        )
        exit()

    model = onnx.load_model(FLAGS.input_model)
    converted_model = convert_lstm_to_customlstm(model)
    onnx.save(converted_model, FLAGS.output_model)
    print('Conversion Finished!')
    print('Converted model saved in: {}'.format(FLAGS.output_model))


if __name__ == '__main__':
    run_main()
