#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
from quark.shares.utils.log import ScreenLogger
import numpy as np

logger = ScreenLogger(__name__)

try:
    import onnx
    from onnxsim import simplify
    from onnx import numpy_helper
except ModuleNotFoundError as e:
    logger.error(str(e))
    raise ModuleNotFoundError("Please install onnx package if exporting onnx graph. " + str(e)) from e

__all__ = [
    "convert_model_to_uint4_int4",
    "export_onnx_model_optimization",
]


def export_onnx_model_optimization(onnx_graph: str) -> None:
    '''
    This is the top level API, called by quark/torch/export/api.py: func:export_onnx_model
    NOTE all following function should:
        input: onnx_graph type: str
            modify the onnx and save the modified onnx to the original path.
        output: None
    '''
    # if int16/uint16 quant, change the opset so that can onnxruntime
    change_opset_version(onnx_graph)
    # if int32 quant(usually for bias), as QuantizeLinear not support int32 runtime,
    #   so we caclulate the int32 bias and saved in DequantizeLinear and delete QuantizeLinear node
    fold_quantizers_for_bias(onnx_graph)
    return


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


# ==============if int16/uint16 quant then change op_set version====
def _contain_uint16_or_int16_quant(onnx_graph: str) -> bool:
    target_data_type = ["int16", "uint16"]
    model = onnx.load(onnx_graph)
    try:
        model, check = simplify(model)
        initializers = {init.name: init for init in model.graph.initializer}

        for node in model.graph.node:
            if node.op_type == 'QuantizeLinear':
                if len(node.input) >= 3:
                    y_zero_point_name = node.input[2]
                    if y_zero_point_name in initializers:
                        y_zero_point_initializer = initializers[y_zero_point_name]
                        if onnx.helper.tensor_dtype_to_np_dtype(y_zero_point_initializer.data_type) in target_data_type:
                            return True
    except Exception as e:
        logger.warning(str(e))
        logger.warning("Check whether using int16 quant failed, skip")
    return False


def change_opset_version(onnx_path: str, opset_version: int = 21) -> None:
    '''
    If int16/Uint16 quant, the dataflow would be like:
        input_date
            |
        QuantizeLinear(uint16/int16)
            |
        DequantizeLinear(uint16/int16)
            |
        quant_data

    However
        1. Based on the official ONNX proto:
            since op_set: opset version >= 21
            QuantizeLinear support int16 & uint16 quant:
                (see: QuantizeLinear's y_zero_point type: T3)
        2. torch.onnx.export (tested on torch 2.7 & 2.6)
            The exported model with opset version less than 21.
    For better compatibility with onnxruntime:
        Once find uint16/int16 quantization, we would change the opset version to 21>=.
    '''

    # Step1: check whether contain int16/uint16 quant
    if not _contain_uint16_or_int16_quant(onnx_path):
        # not contain  uint16/int16 quant, skip optimization
        return

    # Step2: simplify the model first
    onnx_model = onnx.load(onnx_path)
    old_opset_version = onnx_model.opset_import[0].version if onnx_model.opset_import[0].domain == "" else None
    if old_opset_version is not None and old_opset_version < opset_version:
        model_simp, check = simplify(onnx_model)
        # step3: optimization, change the op_set to 21 or higher
        from onnx import version_converter
        converted_model = version_converter.convert_version(model_simp, opset_version)
        onnx.save(converted_model, onnx_path)
        logger.info("During export onnx, find int16/uint16 quant, converting opset from: {} to {}".format(
            old_opset_version, opset_version))
    return


# ==============if bias int32 quant, delete QuantizeLinear and remain DequantizeLinear node ====
def fold_quantizers_for_bias(model_path: str) -> None:
    '''
    If Conv's bias is int32 quant.
    As QuantizeLinear not support int32 quant,
       So for bias:
        1. We need calculate the INT32 quant BIAS,
        2. Delete the QuantizeLinear node,
        3. Modify the Bias and save Quant bias to DequantizeLinear.
    Before:
        QuantizeLinear (fp32_bias, zp(int32), scale(fp32))
                |
        DequantizeLinear (zp(int32), scale(fp32))
                |
    After:
        DequantizeLinear (INT32_bias, zp(int32), scale(fp32))
                |
    '''
    target_conv = ["Conv", "ConvTranspose"]  # TODO rich the target conv
    target_bias_type = ["int32"]
    # Load the model and simplify
    model = onnx.load(model_path)
    try:
        model, check = simplify(model)
    except Exception as e:
        logger.warning("During fold bias, simplify onnx model failed, skip fold_quantizers_for_bias")
        return

    # get the graph
    graph = model.graph
    name_to_initializer = {init.name: init for init in graph.initializer}
    input_0_to_dequnt_node = {
        node.input[0]: node
        for node in graph.node if node.op_type == "DequantizeLinear"
    }  # DequantizeLinear
    input_2_to_conv_node = {node.input[2]: node for node in graph.node if node.op_type in target_conv}  # Conv

    # Step1: find the demand QuantizeLinear node
    '''
    The demand patterm:
        QuantizeLinear(int32)(bias) -> DequantizeLinear(bias)(int32) -> conv
    The following not meet the demand
    '''
    target_bias_quant_node = []
    for node in model.graph.node:
        # if is a single QuantizeLinear, and input is a param
        if (not node.op_type == 'QuantizeLinear') or (not node.input[0] in name_to_initializer):
            continue
        quant_node = node
        y_zero_point_name = quant_node.input[2]
        if y_zero_point_name not in name_to_initializer:
            continue
        y_zero_point_initializer = name_to_initializer[y_zero_point_name]
        # if not int32 quant format, then skip
        if not onnx.helper.tensor_dtype_to_np_dtype(y_zero_point_initializer.data_type) in target_bias_type:
            continue

        # if followed by Dequantizer node
        output_node_name = quant_node.output[0]
        if (output_node_name not in input_0_to_dequnt_node) or (not input_0_to_dequnt_node[output_node_name].op_type
                                                                == 'DequantizeLinear'):
            continue
        dequant_outnode = input_0_to_dequnt_node[output_node_name]
        x_zero_point_name = dequant_outnode.input[2]
        if x_zero_point_name not in name_to_initializer:
            continue
        x_zero_point_initializer = name_to_initializer[x_zero_point_name]
        if not onnx.helper.tensor_dtype_to_np_dtype(x_zero_point_initializer.data_type) in target_bias_type:
            continue

        # if followed by conv node
        output_node_name = dequant_outnode.output[0]
        if (output_node_name
                not in input_2_to_conv_node) or (not input_2_to_conv_node[output_node_name].op_type in target_conv):
            continue

        target_bias_quant_node.append(quant_node)

    # Step2: merge int32 format bias to DequantLinear and delete the QuantLinear
    if len(target_bias_quant_node) == 0:
        return
    removed_nodes = []
    remove_fp32_bias_name = []
    for node in target_bias_quant_node:
        q_node = node

        # find the next DequantizeLinear node
        dq_node = None
        for next_node in graph.node:
            if next_node.op_type == "DequantizeLinear" and next_node.input[0] == q_node.output[0]:
                dq_node = next_node
                break

        assert dq_node is not None
        # get the bias and zp
        input_bias_name = q_node.input[0]
        zp_name = q_node.input[2]

        remove_fp32_bias_name.append(input_bias_name)
        # NOTE in this model all zp_name may equal, so we can not directly delect zp

        x = numpy_helper.to_array(name_to_initializer[input_bias_name])
        y_scale = numpy_helper.to_array(name_to_initializer[q_node.input[1]])
        y_zero_point = numpy_helper.to_array(name_to_initializer[zp_name])

        # get the INT32 format BIAS
        q = np.round(x / y_scale + y_zero_point).astype(np.int32)
        np.clip(q, -2**31, 2**31 - 1, out=q)

        # A new bias
        new_initer_bias_name = input_bias_name + "_int32"
        folded_bias_initializer = numpy_helper.from_array(q, name=new_initer_bias_name)
        graph.initializer.append(folded_bias_initializer)
        dq_node.input[0] = new_initer_bias_name
        # A new zp
        new_initer_zp_name = input_bias_name + "_zp_int32"
        y_zero_point = y_zero_point.astype(np.int32)
        int32_zp_initializer = numpy_helper.from_array(y_zero_point, name=new_initer_zp_name)
        graph.initializer.append(int32_zp_initializer)
        dq_node.input[2] = new_initer_zp_name

        # remove the QuantizeLinear node
        removed_nodes.extend([q_node])

    for node in removed_nodes:
        graph.node.remove(node)
    for each_init_name in remove_fp32_bias_name:
        graph.initializer.remove(name_to_initializer[each_init_name])

    bias_quant_node_num = len(target_bias_quant_node)
    model_simp, check = simplify(model)
    onnx.save_model(model_simp, model_path)
    logger.info(
        "As bias is int32 quant, fold bias QuantizeLinear to DequantizeLinear for better onnxruntime, total convert: {}"
        .format(bias_quant_node_num))
    return
