#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from quark.shares.utils.log import ScreenLogger, log_errors
import numpy as np
import onnx
import time
import pandas as pd
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import find_by_name
from quark.onnx.quant_utils import (ONNX_TYPE_TO_NP_TYPE, BFPFIX_OP_DEFAULT_ATTRS, get_tensor_type_from_qType,
                                    VitisQuantType, scale2pos, pos2scale, ONNXQuantizedModel)
from quark.onnx.finetuning.onnx_subgraph import Subgraph
from quark.onnx.finetuning.create_torch.create_model_utils import (ComputeOperations, QuantizeLinearOps,
                                                                   DequantizeLinearOps)
from quark.onnx.finetuning.onnx_evaluate import inference_model, average_L2
from quark.onnx.mprecision.mixed_bfp import mixed_bfp
from onnx import onnx_pb as onnx_proto
from tqdm import tqdm
import copy
from typing import Any, Dict

logger = ScreenLogger(__name__)

ONNX_INT_TYPE_RANGE = {
    onnx_proto.TensorProto.UINT8: (0, 255),
    onnx_proto.TensorProto.INT8: (-128, 127),
    onnx_proto.TensorProto.UINT16: (0, 65535),
    onnx_proto.TensorProto.INT16: (-32768, 32767),
    onnx_proto.TensorProto.UINT32: (0, 2**32 - 1),
    onnx_proto.TensorProto.INT32: (-2**31, 2**31 - 1),
}


@log_errors
def auto_mixprecision(float_model_path: str, quant_model_path: str, dr: Any, activation_type: Any, weight_type: Any,
                      extra_options: Any) -> Any:
    """ Automatic apply low precision quantization on Q/DQ."""

    def _update_optimized_param(qmodel: onnx.ModelProto, param_name: str, opt_param: Any) -> Any:
        for init in qmodel.graph.initializer:
            if init.name != param_name or opt_param is None:
                continue

            ori_param = onnx.numpy_helper.to_array(init)
            opt_param = opt_param.astype(ori_param.dtype)
            new_init = onnx.numpy_helper.from_array(opt_param, name=param_name)
            init.CopyFrom(new_init)
            return ori_param
        return None

    def _create_weight_value_and_datatype(init: Any, quant_type: Any) -> Any:
        source_dtype = init.data_type
        target_dtype = quant_type
        if source_dtype == target_dtype:
            return None

        source_qrange = ONNX_INT_TYPE_RANGE[source_dtype][1] - ONNX_INT_TYPE_RANGE[source_dtype][0]
        target_qrange = ONNX_INT_TYPE_RANGE[target_dtype][1] - ONNX_INT_TYPE_RANGE[target_dtype][0]

        weight = onnx.numpy_helper.to_array(init)
        weight = weight - np.asarray(ONNX_INT_TYPE_RANGE[source_dtype][0]).astype(weight.dtype)
        weight = weight * np.asarray(target_qrange / source_qrange, dtype=np.float32)
        weight = np.asarray(np.round(weight + ONNX_INT_TYPE_RANGE[target_dtype][0], decimals=0).astype(int))
        weight = weight.astype(ONNX_TYPE_TO_NP_TYPE[target_dtype])

        new_weight = onnx.numpy_helper.from_array(weight, name=init.name)
        return new_weight

    def _modify_scale_value(scale_init: Any, zp_init: Any, quant_type: Any) -> Any:
        source_dtype = zp_init.data_type
        target_dtype = quant_type
        if source_dtype == target_dtype:
            return None

        source_qrange = ONNX_INT_TYPE_RANGE[source_dtype][1] - ONNX_INT_TYPE_RANGE[source_dtype][0]
        target_qrange = ONNX_INT_TYPE_RANGE[target_dtype][1] - ONNX_INT_TYPE_RANGE[target_dtype][0]

        scale = onnx.numpy_helper.to_array(scale_init)
        new_scale = scale * source_qrange / target_qrange
        new_scale = new_scale.astype(np.float32)

        # Check if it's a power-of-two scale, the scale should be a scalar
        if pos2scale(scale2pos(float(scale))) == float(scale):
            logger.debug(f"{scale_init.name} is a power-of-two scale")
            pos = scale2pos(float(new_scale))
            new_scale = np.array(pos2scale(int(pos)))
            new_scale = np.asarray(new_scale)
            new_scale = new_scale.astype(np.float32)

        new_init = onnx.numpy_helper.from_array(new_scale, name=scale_init.name)
        return new_init

    def _create_zp_value_and_datatype(init: Any, quant_type: Any) -> Any:
        source_dtype = init.data_type
        target_dtype = quant_type
        if source_dtype == target_dtype:
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

    def _replace_quant_type(quant_model: Any, refer_model: Any, qdq: Any, quant_type: Any) -> None:
        weight_init = quant_model.get_initializer(qdq.input[0])
        scale_init = quant_model.get_initializer(qdq.input[1])
        zp_init = quant_model.get_initializer(qdq.input[2])

        if weight_init is not None:
            new_weight_init = _create_weight_value_and_datatype(weight_init, quant_type)
            if new_weight_init is not None:
                quant_model.remove_initializer(weight_init)
                quant_model.add_initializer(new_weight_init)

        new_scale_init = _modify_scale_value(scale_init, zp_init, quant_type)
        if new_scale_init is not None:
            quant_model.remove_initializer(scale_init)
            quant_model.add_initializer(new_scale_init)

        new_zp_init = _create_zp_value_and_datatype(zp_init, quant_type)
        if new_zp_init is not None:
            quant_model.remove_initializer(zp_init)
            quant_model.add_initializer(new_zp_init)

    def _restore_quant_type(quant_model: Any, refer_model: Any, qdq: Any, quant_type: Any) -> None:
        if qdq.input[0] is not None and refer_model.get_initializer(qdq.input[0]) is not None:
            weight_init = quant_model.get_initializer(qdq.input[0])
            ori_weight_init = refer_model.get_initializer(qdq.input[0])
            quant_model.remove_initializer(weight_init)
            quant_model.add_initializer(ori_weight_init)

        scale_init = quant_model.get_initializer(qdq.input[1])
        ori_scale_init = refer_model.get_initializer(qdq.input[1])
        quant_model.remove_initializer(scale_init)
        quant_model.add_initializer(ori_scale_init)

        zp_init = quant_model.get_initializer(qdq.input[2])
        ori_zp_init = refer_model.get_initializer(qdq.input[2])
        quant_model.remove_initializer(zp_init)
        quant_model.add_initializer(ori_zp_init)

    def _replace_tensor_type(quant_model: onnx.ModelProto, refer_model: onnx.ModelProto, tensor_name: str,
                             quant_type: Any) -> None:
        new_value_info_list = []
        for vi in quant_model.model.graph.value_info:
            if vi.name == tensor_name:
                new_vi = onnx.onnx_pb.ValueInfoProto()
                new_vi.CopyFrom(vi)
                new_vi.type.tensor_type.elem_type = quant_type
                new_value_info_list.append(new_vi)
            # else:
            #    new_value_info_list.append(vi)
        quant_model.model.graph.value_info.extend(new_value_info_list)

    def _restore_tensor_type(quant_model: onnx.ModelProto, refer_model: onnx.ModelProto, tensor_name: str,
                             quant_type: Any) -> None:
        origin_type = None
        for value_info in refer_model.model.graph.value_info:
            if value_info.name == tensor_name:
                if value_info.type.HasField('tensor_type'):
                    origin_type = value_info.type.tensor_type.elem_type
                break
        if origin_type:
            _replace_tensor_type(quant_model, refer_model, tensor_name, origin_type)

    def _change_tensors_qdq(quant_model: Any, refer_model: Any, target_tensors: Any, quant_type: Any) -> None:
        for tensor_name in target_tensors:
            q, dq = (None, None)
            for node in quant_model.model.graph.node:
                if "QuantizeLinear" in node.op_type and node.name == tensor_name + "_QuantizeLinear":
                    q = node
                if "DequantizeLinear" in node.op_type and node.name == tensor_name + "_DequantizeLinear":
                    dq = node
            if q is None or dq is None:
                continue

            _replace_quant_type(quant_model, refer_model, dq, quant_type)
            if q.input[2] != dq.input[2]:  # If did not share the same inputs
                _replace_quant_type(quant_model, refer_model, q, quant_type)

            logger.info(f"Have changed the quant type for {tensor_name}'s Q/DQ")

    def _insert_new_qdq(quant_model: Any, refer_model: Any, q: Any, dq: Any, quant_type: Any, tensor_name: Any) -> None:
        qdq = dq if dq is not None else q

        scale_init = quant_model.get_initializer(qdq.input[1])
        zp_init = quant_model.get_initializer(qdq.input[2])

        new_scale_init = _modify_scale_value(scale_init, zp_init, quant_type)
        new_zp_init = _create_zp_value_and_datatype(zp_init, quant_type)

        if new_scale_init is not None and new_zp_init is not None:
            new_scale_init.name = new_scale_init.name + "_Mixed"
            new_zp_init.name = new_zp_init.name + "_Mixed"

            # Create a new QuantizeLinear node
            q_new = copy.deepcopy(q)
            q_new.name = q_new.name + "_Mixed"
            q_new.input[0] = tensor_name
            q_new.input[1] = new_scale_init.name
            q_new.input[2] = new_zp_init.name
            q_new.output[0] = q_new.output[0] + "_Mixed"

            # Create a new DequantizeLinear node
            dq_new = copy.deepcopy(dq)
            dq_new.name = dq_new.name + "_Mixed"
            dq_new.input[0] = q_new.output[0]  # Connect dq_new with q_new
            dq_new.input[1] = new_scale_init.name
            dq_new.input[2] = new_zp_init.name
            dq_new.output[0] = dq_new.output[0] + "_Mixed"

            # Insert qdq to the tensor by its name
            quant_model.replace_input_of_all_nodes(tensor_name, dq_new.output[0])

            quant_model.add_initializer(new_scale_init)
            quant_model.add_initializer(new_zp_init)
            quant_model.add_node(q_new)
            quant_model.add_node(dq_new)

    def _delete_new_qdq(quant_model: Any, refer_model: Any, q: Any, dq: Any, quant_type: Any, tensor_name: str) -> None:
        q_new_name = q.name if q.name.endswith("_Mixed") else q.name + "_Mixed"
        q_new = find_by_name(q_new_name, quant_model.model.graph.node)
        dq_new_name = dq.name if dq.name.endswith("_Mixed") else dq.name + "_Mixed"
        dq_new = find_by_name(dq_new_name, quant_model.model.graph.node)

        if q_new is None or dq_new is None:
            logger.debug("Not found the new qdq to delete")
        else:
            new_scale_init = quant_model.get_initializer(q_new.input[1])
            new_zp_init = quant_model.get_initializer(q_new.input[2])

            quant_model.replace_input_of_all_nodes(dq_new.output[0], tensor_name)

            quant_model.remove_initializer(new_scale_init)
            quant_model.remove_initializer(new_zp_init)
            quant_model.remove_node(q_new)
            quant_model.remove_node(dq_new)

    def _count_child_qdqs(quant_model: Any, tensor_name: str) -> int:
        in_name_to_nodes = quant_model.input_name_to_nodes()
        child_nodes = in_name_to_nodes.get(tensor_name, None)

        count = 0
        while (child_nodes is not None):
            if child_nodes[0].op_type in QuantizeLinearOps + DequantizeLinearOps:
                child_nodes = in_name_to_nodes.get(child_nodes[0].output[0], None)
            else:
                break
            count += 1
        return count

    def _count_parent_qdqs(quant_model: Any, tensor_name: str) -> int:
        out_name_to_node = quant_model.output_name_to_node()
        parent_node = out_name_to_node.get(tensor_name, None)

        count = 0
        while (parent_node is not None):
            if parent_node.op_type in QuantizeLinearOps + DequantizeLinearOps:
                parent_node = out_name_to_node.get(parent_node.input[0], None)
            else:
                break
            count += 1
        return count

    def _update_bias_scale(quant_model: Any, node_name: str) -> None:
        onnx_model = ONNXQuantizedModel(quant_model.model)
        for node in quant_model.model.graph.node:
            if node.name == node_name and len(node.input) == 3:
                dq, q = onnx_model._find_node_input_qdq(node, node.input[2])
                if dq is None:
                    logger.warning("Not found DQ for bias of node {node_name}")
                    break

                # Only needed for Int32 quantized Bias
                zp_init = quant_model.get_initializer(dq.input[2])
                if zp_init.data_type != onnx_proto.TensorProto.INT32:
                    logger.debug("No need update bias scale for non-Int32")
                    break

                # Calculate scale for bias
                input_dq, _ = onnx_model._find_node_input_qdq(node, node.input[0])
                assert input_dq is not None
                input_scale_init = quant_model.get_initializer(input_dq.input[1])
                input_scale = onnx.numpy_helper.to_array(input_scale_init)

                weight_dq, _ = onnx_model._find_node_input_qdq(node, node.input[1])
                assert weight_dq is not None
                weight_scale_init = quant_model.get_initializer(weight_dq.input[1])
                weight_scale = onnx.numpy_helper.to_array(weight_scale_init)

                new_bias_scale = input_scale * weight_scale
                new_bias_scale = new_bias_scale.astype(np.float32)

                # Re-quantize bias if Q was foled
                if q is None:
                    bias_scale_init = quant_model.get_initializer(dq.input[1])
                    old_bias_scale = onnx.numpy_helper.to_array(bias_scale_init)

                    bias_init = quant_model.get_initializer(dq.input[0])
                    bias = onnx.numpy_helper.to_array(bias_init)

                    # Since the loss of QInt32 is ignorable, just dq and then q again
                    bias = bias.astype(np.float32)
                    bias = bias * old_bias_scale
                    bias = bias / new_bias_scale
                    bias = bias.astype(np.int32)

                    new_bias_init = onnx.numpy_helper.from_array(bias, dq.input[0])
                    bias_init.CopyFrom(new_bias_init)

                # Update scale initializer
                new_bias_scale_init = onnx.numpy_helper.from_array(new_bias_scale, dq.input[1])
                quant_model.get_initializer(dq.input[1]).CopyFrom(new_bias_scale_init)

                # If Q was not folded and has different initializer of scale
                if q is not None and q.input[1] != dq.input[1]:
                    new_bias_scale_init = onnx.numpy_helper.from_array(new_bias_scale, q.input[1])
                    quant_model.get_initializer(q.input[1]).CopyFrom(new_bias_scale_init)

                break

    @log_errors
    def _handling_target_qdqs(quant_model: Any, refer_model: Any, node_struct: Any, act_quant_type: Any,
                              weight_quant_type: Any, bias_quant_type: Any, process_type: Any) -> None:

        def __replace(quant_model: Any, refer_model: Any, dq: Any, q: Any, quant_type: Any) -> None:
            _replace_quant_type(quant_model, refer_model, dq, quant_type)
            if q is not None and q.input[2] != dq.input[2]:  # If did not share the same inputs
                _replace_quant_type(quant_model, refer_model, q, quant_type)

        def __restore(quant_model: Any, refer_model: Any, dq: Any, q: Any, quant_type: Any) -> None:
            _restore_quant_type(quant_model, refer_model, dq, quant_type)
            if q is not None and q.input[2] != dq.input[2]:  # If did not share the same inputs
                _restore_quant_type(quant_model, refer_model, q, quant_type)

        def __insert(quant_model: Any, refer_model: Any, dq: Any, q: Any, quant_type: Any, is_output: Any) -> None:
            tensor_name = q.input[0] if is_output else dq.output[0]
            qdq_num = _count_child_qdqs(quant_model, q.input[0])
            if (qdq_num == 2):
                _insert_new_qdq(quant_model, refer_model, q, dq, quant_type,
                                tensor_name)  # This tensor should link to the new qdq
            elif (qdq_num == 4):
                __replace(quant_model, refer_model, dq, q, quant_type)
            else:
                raise ValueError("Unexpected QDQ numbers in insert process")

        def __delete(quant_model: Any, refer_model: Any, dq: Any, q: Any, quant_type: Any, is_output: Any) -> None:
            tensor_name = q.input[0] if is_output else dq.output[0]
            qdq_num = _count_child_qdqs(quant_model, q.input[0])
            if (qdq_num == 4):
                _delete_new_qdq(quant_model, refer_model, q, dq, quant_type,
                                tensor_name)  # This tensor should link to the new qdq
            elif (qdq_num == 2):
                __restore(quant_model, refer_model, dq, q, quant_type)
            else:
                raise ValueError("Unexpected QDQ numbers in delete process")

        if len(node_struct['input_qdqs']):
            dq, q = node_struct['input_qdqs'][0]  # Just analyze input tensor's qdq
            if dq is not None and q is not None:
                if process_type == "replace":
                    __replace(quant_model, refer_model, dq, q, act_quant_type)
                elif process_type == "restore":
                    __restore(quant_model, refer_model, dq, q, act_quant_type)
                elif process_type == "insert":
                    __insert(quant_model, refer_model, dq, q, act_quant_type, False)
                elif process_type == "delete":
                    __delete(quant_model, refer_model, dq, q, act_quant_type, False)
                else:
                    raise ValueError("Unsupported process for auto mixprecision")

            if len(node_struct['input_qdqs']) >= 2:
                dq, q = node_struct['input_qdqs'][1]
                if dq is not None:
                    if process_type == "replace":
                        __replace(quant_model, refer_model, dq, q, weight_quant_type)
                    elif process_type == "restore":
                        __restore(quant_model, refer_model, dq, q, weight_quant_type)
                    elif process_type == "insert":
                        __insert(quant_model, refer_model, dq, q, weight_quant_type, False)
                    elif process_type == "delete":
                        __delete(quant_model, refer_model, dq, q, weight_quant_type, False)
                    else:
                        raise ValueError("Unsupported process for auto mixprecision")

            if len(node_struct['input_qdqs']) >= 3:
                dq, q = node_struct['input_qdqs'][2]
                if dq is not None:
                    if process_type == "replace":
                        __replace(quant_model, refer_model, dq, q, bias_quant_type)
                    elif process_type == "restore":
                        __restore(quant_model, refer_model, dq, q, bias_quant_type)
                    elif process_type == "insert":
                        __insert(quant_model, refer_model, dq, q, bias_quant_type, False)
                    elif process_type == "delete":
                        __delete(quant_model, refer_model, dq, q, bias_quant_type, False)
                    else:
                        raise ValueError("Unsupported process for auto mixprecision")

        # Need to update bias scale to meet bias_scale=input_scale*weight_scale
        if len(node_struct['input_qdqs']) == 3:
            # Considering simplicity, not doing this for "replace" and "restore"
            if (process_type == "replace" or process_type == "restore"):
                _update_bias_scale(quant_model, node_struct['node'].name)

        return None  # It will cause accuracy drop if we change output qdq too

        # The sub-module has no output qdqs, so we get them from refer model
        in_name_to_nodes = refer_model.input_name_to_nodes()
        child_nodes = in_name_to_nodes.get(node_struct['node'].output[0], None)
        if child_nodes is not None and child_nodes[0].op_type in QuantizeLinearOps:
            q = child_nodes[0]
            child_nodes = in_name_to_nodes.get(q.output[0], None)
            if child_nodes is not None and child_nodes[0].op_type in DequantizeLinearOps:
                dq = child_nodes[0]
                if process_type == "replace":
                    __replace(quant_model, refer_model, dq, q, quant_type)
                elif process_type == "restore":
                    __restore(quant_model, refer_model, dq, q, quant_type)
                elif process_type == "insert":
                    __insert(quant_model, refer_model, dq, q, quant_type, True)
                elif process_type == "delete":
                    __delete(quant_model, refer_model, dq, q, quant_type, True)
                else:
                    raise ValueError("Unsupported process for auto mixprecision")

    # Get the configurations of AutoMixprecision
    data_size = extra_options.get('AutoMixprecision', {}).get('DataSize', None)
    target_op_type = extra_options.get('AutoMixprecision', {}).get('TargetOpType', ComputeOperations)
    target_quant_type = extra_options.get('AutoMixprecision', {}).get('TargetQuantType', None)
    act_target_quant_type = extra_options.get('AutoMixprecision', {}).get('ActTargetQuantType', None)
    weight_target_quant_type = extra_options.get('AutoMixprecision', {}).get('WeightTargetQuantType', None)
    bias_target_quant_type = extra_options.get('AutoMixprecision', {}).get('BiasTargetQuantType', None)
    target_tensors = extra_options.get('AutoMixprecision', {}).get('TargetTensors', [])
    target_indices = extra_options.get('AutoMixprecision', {}).get('TargetIndices', [])
    exclude_indices = extra_options.get('AutoMixprecision', {}).get('ExcludeIndices', [])
    output_index = extra_options.get('AutoMixprecision', {}).get('OutputIndex', None)
    l2_target = extra_options.get('AutoMixprecision', {}).get('L2Target', None)
    top1_acc_target = extra_options.get('AutoMixprecision', {}).get('Top1AccTarget', None)
    evaluate_function = extra_options.get('AutoMixprecision', {}).get('EvaluateFunction', None)
    num_target = extra_options.get('AutoMixprecision', {}).get('NumTarget', 0)
    no_shared = extra_options.get('AutoMixprecision', {}).get('NoInputQDQShared', True)
    auto_mix_use_fast_ft = extra_options.get('AutoMixprecision', {}).get('AutoMixUseFastFT', False)
    int32_bias = extra_options.get("Int32Bias", True)

    if target_quant_type is None and act_target_quant_type is None and weight_target_quant_type is None:
        raise ValueError(
            "include_auto_mp is True, so TargetQuantType or ActTargetQuantType or WeightTargetQuantType must be given!")

    if act_target_quant_type is None and weight_target_quant_type is None:
        act_target_quant_type = target_quant_type
        weight_target_quant_type = weight_type

    if act_target_quant_type is None:
        act_target_quant_type = activation_type

    if weight_target_quant_type is None:
        weight_target_quant_type = weight_type

    if bias_target_quant_type is not None and bias_target_quant_type != weight_target_quant_type:
        raise ValueError(
            f"BiasTargetQuantType: {bias_target_quant_type} and WeightTargetQuantType: {weight_target_quant_type} should be the same."
        )

    bias_type = weight_type
    if int32_bias:
        bias_type = VitisQuantType.QInt32

    if bias_target_quant_type is None and int32_bias:
        bias_target_quant_type = VitisQuantType.QInt32
    elif bias_target_quant_type is None and not int32_bias:
        bias_target_quant_type = weight_target_quant_type

    if target_quant_type is None:
        target_quant_type = act_target_quant_type

    logger.info(
        f"Start running auto mixprecision for ops. Activation target quant type: {act_target_quant_type}, Weight target quant type: {weight_target_quant_type}, Bias target quant type: {bias_target_quant_type}..."
    )

    # If configured BFP as the target quant type, to mix BFP directly
    quantized_model = onnx.load(quant_model_path)
    if target_quant_type == VitisQuantType.QBFP:
        logger.info("Configured BFP as target quant type, start inserting BFPFixNeuron...")
        bfp_attrs = copy.deepcopy(BFPFIX_OP_DEFAULT_ATTRS)
        # To compatible with older version settings
        if 'bfp_method' in extra_options:
            bfp_attrs['bfp_method'] = extra_options['bfp_method']
        if 'bit_width' in extra_options:
            bfp_attrs['bit_width'] = extra_options['bit_width']
        if 'block_size' in extra_options:
            bfp_attrs['block_size'] = extra_options['block_size']
        if 'sub_block_shift_bits' in extra_options:
            bfp_attrs['sub_block_shift_bits'] = extra_options['sub_block_shift_bits']
        # Get attributes for BFPFixNeuron
        if "BFPAttributes" in extra_options:
            bfp_attrs.update(extra_options["BFPAttributes"])
        return mixed_bfp(quantized_model, target_op_type, bfp_attrs)

    if (l2_target is not None) and (top1_acc_target is not None):
        raise ValueError("l2_target and top1_acc_target must one of the two!")

    if top1_acc_target is not None and evaluate_function is None:
        raise ValueError("Evaluate_function must be given when top1_acc_target is given!")

    # Extract sub-graph of modules
    sg = Subgraph(float_model_path, quant_model_path, dr, extra_options)
    assert (len(sg.subgraph_qmodel_list) == len(sg.subgraph_fmodel_list) == len(sg.f_weight_list) == len(
        sg.f_input_data_list) == len(sg.f_output_data_list))

    if len(target_tensors) == 0 and len(
            target_indices) == 0 and l2_target is None and top1_acc_target is None and num_target <= 0:
        num_target = len(sg.subgraph_qmodel_list)
        logger.warning("No target was specified, all modules will be mixed.")

    refer_model = ONNXModel(sg.qmodel)  # The sg.qmodel has full shape info
    quant_model = ONNXModel(quantized_model)

    input_name_to_nodes = quant_model.input_name_to_nodes()
    output_name_to_node = quant_model.output_name_to_node()

    quant_type = get_tensor_type_from_qType(target_quant_type)
    act_quant_type = get_tensor_type_from_qType(act_target_quant_type)
    weight_quant_type = get_tensor_type_from_qType(weight_target_quant_type)
    bias_quant_type = get_tensor_type_from_qType(bias_target_quant_type)

    # If configured target tensors, just change their QDQs quant type
    if isinstance(target_tensors, list) and len(target_tensors):
        logger.info(f"Configured target tensors, start changing their QDQs to {quant_type}...")
        _change_tensors_qdq(quant_model, refer_model, target_tensors, quant_type)
        return quant_model.model

    # If configured target indices, just use it as target and skip analysis
    sorted_module = []

    module_index_name_dict = {}
    for i in range(len(sg.subgraph_qmodel_list)):
        sub_model = sg.subgraph_qmodel_list[i]
        for node in sub_model.graph.node:
            if node.op_type in target_op_type:
                module_index_name_dict[str(i)] = node.name

    distance = 0.0
    if isinstance(target_indices, list) and len(target_indices):
        logger.info("Specified target indice and will use it as target")
        num_target = len(target_indices)

        for index in target_indices:
            if index < 0 or index >= len(sg.subgraph_qmodel_list):
                logger.warning(f"The TargetIndices {target_indices} has an invalid value {index}")
                continue
            sorted_module.append((index, distance))
    elif isinstance(exclude_indices, list) and len(exclude_indices):
        logger.info("Specified exclude indice and will use it as target")

        for index in range(0, len(sg.subgraph_qmodel_list)):
            if index in exclude_indices:
                logger.warning(f"The index {index} within exclude_indices {exclude_indices} was skipped")
                continue
            sorted_module.append((index, distance))

        num_target = len(sorted_module)

    if sorted_module == []:
        float_results = inference_model(float_model_path, dr, data_size, output_index)
        quant_results = inference_model(quant_model_path, dr, data_size, output_index)
        if l2_target is not None:
            distance = average_L2(float_results, quant_results)
            logger.info(
                f"Activation: {act_target_quant_type}, Weight: {weight_target_quant_type}, Bias: {bias_target_quant_type} average L2 distance is {distance}"
            )
        if top1_acc_target is not None:
            float_top1_acc = evaluate_function(float_results)
            quant_top1_acc = evaluate_function(quant_results)
            distance = round(float_top1_acc - quant_top1_acc, 4)
            if top1_acc_target <= distance:
                logger.warning(
                    f"Quantizing all nodes to Activation: {activation_type}, Weight: {weight_type}, Bias: {bias_type}, the top1 accuracy is {quant_top1_acc}, the loss is {distance} compared with the float top1 accuracy {float_top1_acc}. The top1_acc_target {top1_acc_target} cannot be met. The following process will continue with the Activation: {act_target_quant_type}, Weight: {weight_target_quant_type}, Bias: {bias_target_quant_type} quantized model. We suggest to set a larger top1_acc_target value and try again."
                )
                quant_model.topological_sort()
                return quant_model.model
            else:
                logger.info(
                    f"Quantizing all nodes to Activation: {activation_type}, Weight: {weight_type}, Bias: {bias_type}, the top1 accuracy is {quant_top1_acc}, the loss is {distance} compared with the float top1 accuracy {float_top1_acc}. The top1_acc_target {top1_acc_target} can be met. Then start the mix-precision process."
                )

    # Auto precision analysis
    module_dict = {}
    torch_training_time: float = 0
    distance_new = 0.0
    for i, module in tqdm(enumerate(sg.subgraph_qmodel_list), total=len(sg.subgraph_qmodel_list)):
        if len(sorted_module):  # No need analysis if have sorted
            continue

        # Get the target Q/DQ of the module
        parser = ONNXQuantizedModel(module)
        node_struct = parser.find_target_op_type_qdqs(target_op_type)
        if node_struct['node'] is None or len(node_struct['input_qdqs']) == 0:
            continue

        _handling_target_qdqs(quant_model, refer_model, node_struct, act_quant_type, weight_quant_type, bias_quant_type,
                              "replace")
        if auto_mix_use_fast_ft:
            # for every subgraph to do adaround to replace
            q_input_data = np.array(sg.get_q_input_data(i))
            f_input_data = np.array(sg.f_input_data_list[i])
            f_output_data = np.array(sg.f_output_data_list[i])
            f_input_data = f_input_data.reshape((-1, *f_input_data.shape[2:]))
            f_output_data = f_output_data.reshape((-1, *f_output_data.shape[2:]))
            q_input_data = q_input_data.reshape((-1, *q_input_data.shape[2:]))
            if not q_input_data.shape == f_input_data.shape:
                logger.warning(f"Input shape for quantized module {q_input_data.shape} "
                               f"is different with the float module {f_input_data.shape}."
                               "Skip this module.")
                continue

            # Optimize weight and bias for this module
            f_weight = np.array(sg.f_weight_list[i])
            f_bias = None if sg.f_bias_list[i] is None else np.array(sg.f_bias_list[i]).reshape(-1)

            start_time = time.perf_counter()
            from quark.onnx.finetuning.torch_utils import optimize_module
            opt_weight, opt_bias = optimize_module(module, f_weight, f_bias, q_input_data, f_input_data, f_output_data,
                                                   extra_options)
            end_time = time.perf_counter()
            torch_training_time += (end_time - start_time)
            ori_weight = _update_optimized_param(quantized_model, sg.q_weight_name_list[i], opt_weight)
            ori_bias = _update_optimized_param(quantized_model, sg.q_bias_name_list[i], opt_bias)
            # Calculate the average
            quant_results = inference_model(quantized_model, dr, data_size, output_index)
            _update_optimized_param(quantized_model, sg.q_weight_name_list[i], ori_weight)
            _update_optimized_param(quantized_model, sg.q_bias_name_list[i], ori_bias)
        else:
            # Calculate the average
            quant_results = inference_model(quantized_model, dr, data_size, output_index)

        if l2_target is not None or num_target != 0:
            distance_new = average_L2(float_results, quant_results)
            logger.debug(f"The average L2 distance is from {distance} to {distance_new}.")

        if top1_acc_target is not None:
            float_top1_acc = evaluate_function(float_results)
            quant_top1_acc = evaluate_function(quant_results)
            distance_new = round(float_top1_acc - quant_top1_acc, 4)
            logger.debug(f"The top1 accuracy loss is from {distance} to {distance_new}.")

        _handling_target_qdqs(quant_model, refer_model, node_struct, act_quant_type, weight_quant_type, bias_quant_type,
                              "restore")

        module_dict[i] = distance_new

    if len(sorted_module) == 0:
        if len(module_dict) == 0:
            logger.warning("Could not apply auto mixprecision because not found available modules.")
            return sg.qmodel

        if l2_target is not None or num_target != 0:
            # Sort L2 distance in ascending order
            sorted_module = sorted(module_dict.items(), key=lambda x: x[1])
            new_sorted_module: Dict[str, Any] = {"Index": [], "Node name": [], "L2 loss": []}
            for index, loss in sorted_module:
                new_sorted_module["Index"].append(index)
                new_sorted_module["Node name"].append(module_index_name_dict[str(index)])
                new_sorted_module["L2 loss"].append(loss)
            df = pd.DataFrame(new_sorted_module)
            logger.info(f"The {len(sorted_module)} modules name, index and corresponding l2 are as follows:")

            logger.info("\n%s", df.to_string(index=False))

        if top1_acc_target is not None:
            # Sort top1 acc target in ascending order
            sorted_module = sorted(module_dict.items(), key=lambda x: x[1])
            new_sorted_module = {"Index": [], "Node name": [], "Top1 acc loss": []}
            for index, loss in sorted_module:
                new_sorted_module["Index"].append(index)
                new_sorted_module["Node name"].append(module_index_name_dict[str(index)])
                new_sorted_module["Top1 acc loss"].append(loss)
            df = pd.DataFrame(new_sorted_module)
            logger.info(f"The {len(sorted_module)} modules name, index and corresponding top1 acc are as follows:")
            logger.info("\n%s", df.to_string(index=False))

    # Execute mix precision
    mixed_num = 0  # How many modules have been mixed
    mixed_node_names = []

    for module_index, estimate in sorted_module:
        module = sg.subgraph_qmodel_list[module_index]

        # Get the target Q/DQs of the module and change its quant_type
        parser = ONNXQuantizedModel(module)
        node_struct = parser.find_target_op_type_qdqs(target_op_type)

        node = node_struct['node']  # The node will receive a new quant_type
        if node is None or len(node_struct['input_qdqs']) == 0:
            logger.warning(f"Skipped #{module_index} module that can't be mixed.")
            continue  # Skip the node that is not supported or not quantized
        elif no_shared and len(input_name_to_nodes[node.input[0]]) != 1:
            logger.info(f"Skipped {node.name} which has a shared input QDQ pairs.")
            continue  # Skip the node who shared the QDQ with other nodes
        else:
            mixed_node_names.append(node.name)
            logger.info(
                f"#{module_index} {node.op_type} named {node.name} is converted from Activation: {activation_type}, Weight: {weight_type}, Bias: {bias_type} to Activation: {act_target_quant_type}, Weight: {weight_target_quant_type}, Bias: {bias_target_quant_type}"
            )

        _handling_target_qdqs(quant_model, refer_model, node_struct, act_quant_type, weight_quant_type, bias_quant_type,
                              "replace")

        # Count the number of mixed modules
        mixed_num += 1

        # Checking whether suspension conditions are met
        if num_target > 0:
            if mixed_num >= num_target:
                logger.info(f"Have met the Num Target {num_target}, break.")
                break

        elif l2_target is not None:
            quant_results = inference_model(quant_model.model, dr, data_size, output_index)
            distance_new = average_L2(float_results, quant_results)

            if distance_new > l2_target:
                _handling_target_qdqs(quant_model, refer_model, node_struct, act_quant_type, weight_quant_type,
                                      bias_quant_type, "restore")
                if len(mixed_node_names) >= 1:
                    mixed_node_names.pop()
                logger.info(f"The average L2 distance is {distance_new}, "
                            f"which is greater than the L2 Target {l2_target}, "
                            f"so go back to its original state and break."
                            f"Mixed node names are: {mixed_node_names}.")
                break

            logger.info(f"The average L2 distance is from {distance} to {distance_new}.")
            distance = distance_new

        elif top1_acc_target is not None:
            quant_results = inference_model(quant_model.model, dr, data_size, output_index)
            float_top1_acc = evaluate_function(float_results)
            quant_top1_acc = evaluate_function(quant_results)
            distance_new = round(float_top1_acc - quant_top1_acc, 4)

            if distance_new > top1_acc_target:
                _handling_target_qdqs(quant_model, refer_model, node_struct, act_quant_type, weight_quant_type,
                                      bias_quant_type, "restore")
                if len(mixed_node_names) >= 1:
                    mixed_node_names.pop()
                logger.info(f"The Top1 accuracy is {quant_top1_acc}, "
                            f"The Top1 accuracy loss is {distance_new}, "
                            f"which is greater than the Top1 Acc Target {top1_acc_target}, "
                            f"so go back to its original state and break."
                            f"Mixed node names are: {mixed_node_names}.")
                break

            logger.info(f"The Top1 accuracy is {quant_top1_acc}, "
                        f"The Top1 accuracy loss is from {distance} to {distance_new}.")
            distance = distance_new

    if l2_target is not None:
        quant_results = inference_model(quant_model.model, dr, data_size, output_index)
        distance_new = average_L2(float_results, quant_results)
        logger.info(
            f"Activation: {activation_type}, Weight: {weight_type}, Bias: {bias_type} and Activation: {act_target_quant_type}, Weight: {weight_target_quant_type}, Bias: {bias_target_quant_type} mixed L2 distance is {distance_new}."
        )
    if top1_acc_target is not None:
        quant_results = inference_model(quant_model.model, dr, data_size, output_index)
        float_top1_acc = evaluate_function(float_results)
        quant_top1_acc = evaluate_function(quant_results)
        distance_new = round(float_top1_acc - quant_top1_acc, 4)
        logger.info(
            f"Mixed Precision Summary: Activation: {activation_type}, Weight: {weight_type}, Bias: {bias_type} and Activation: {act_target_quant_type}, Weight: {weight_target_quant_type}, Bias: {bias_target_quant_type} mixed top1 accuracy is {quant_top1_acc}, the loss is {distance_new} compared with the float top1 accuracy {float_top1_acc}. {len(mixed_node_names)}/{len(sorted_module)} are Activation: {act_target_quant_type}, Weight: {weight_target_quant_type}, Bias: {bias_target_quant_type} {len(sorted_module)-len(mixed_node_names)}/{len(sorted_module)} are Activation: {activation_type}, Weight: {weight_type}, Bias: {bias_type}."
        )
        logger.info(
            f"Activation: {act_target_quant_type}, Weight: {weight_target_quant_type}, Bias: {bias_target_quant_type} node names are {mixed_node_names}."
        )

    quant_model.topological_sort()
    return quant_model.model
