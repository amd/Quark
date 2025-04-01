#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import numpy as np
from quark.shares.utils.log import ScreenLogger
import onnxruntime
import itertools
import onnx
from onnx import TensorProto, ModelProto, numpy_helper
import tempfile
from pathlib import Path
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.onnx_model import ONNXModel
from .quant_utils import (inference_sub_model_with_data, get_output_nodes_of_node, dequantize_data, quantize_data_pof2s,
                          get_tensor_type_from_qType, PowerOfTwoMethod)
from typing import Tuple, Set, Optional, Any, Union, Dict, List
from onnxruntime.quantization.quant_utils import QuantType
from .quant_utils import VitisQuantType, CachedDataReader

logger = ScreenLogger(__name__)


class BiasCorrection():

    def __init__(self, opt_model: ModelProto, quant_model: ModelProto, use_external_data_format: bool,
                 execution_providers: List[str]) -> None:
        self.origin_model = opt_model
        self.augment_model = opt_model
        self.quant_model = quant_model
        self.quant_augmented_model_path = ""
        self.target_type = ["Conv", "Gemm"]
        self.use_external_data_format = use_external_data_format
        self.execution_providers = execution_providers
        self.origin_intermediate_outputs: List[str] = []
        self.quant_intermediate_outputs: List[str] = []

    def select_tensors_to_calibrate(self, model: onnx.ModelProto) -> Tuple[Set[str], Dict[str, onnx.ValueInfoProto]]:
        """
        select all quantization_candidates op type nodes' input/output tensors.
        returns:
            tensors (set): set of tensor name.
            value_infos (dict): tensor name to value info.
        """
        value_infos = {vi.name: vi for vi in model.graph.value_info}
        value_infos.update({ot.name: ot for ot in model.graph.output})
        value_infos.update({it.name: it for it in model.graph.input})
        initializer = {init.name for init in model.graph.initializer}

        tensors_to_calibrate = set()
        tensor_type_to_calibrate = {TensorProto.FLOAT, TensorProto.FLOAT16}

        for node in model.graph.node:
            if node.op_type in ["Conv", "Gemm"]:
                for tensor_name in itertools.chain(node.input, node.output):
                    if tensor_name in value_infos:
                        vi = value_infos[tensor_name]
                        if (vi.type.HasField("tensor_type")
                                and vi.type.tensor_type.elem_type in tensor_type_to_calibrate
                                and tensor_name not in initializer):
                            tensors_to_calibrate.add(tensor_name)

        return tensors_to_calibrate, value_infos

    def get_bias_corr_pattern_augment_graph(
            self, origin_model: ModelProto,
            augmented_model_path: str) -> Tuple[Dict[str, Tuple[str, str, Optional[str]]], Set[str]]:
        """
        make all quantization_candidates op type nodes as part of the graph output.
        :return: augmented ONNX model
        """
        node_bias_corr_output_map = {}
        onnx.save(
            origin_model,
            augmented_model_path,
            save_as_external_data=self.use_external_data_format,
        )
        model = onnx.load(augmented_model_path)

        tensors_to_calibrate, value_infos = self.select_tensors_to_calibrate(model)
        model_original_outputs = set(output.name for output in model.graph.output)
        linear_and_quant_node_type = ["Relu", "Clip", "QuantizeLinear", "DequantizeLinear"]
        all_sub_model_in_out = []
        for node in model.graph.node:
            sub_model_input = None
            sub_model_output = None
            if node.op_type in self.target_type:
                sub_model_input = node.input[0]
                sub_model_output = node.output[0]
                node_output_nodes1 = get_output_nodes_of_node(node, model.graph)
                while node_output_nodes1:
                    if node_output_nodes1[0].op_type in linear_and_quant_node_type:
                        sub_model_output = node_output_nodes1[0].output[0]
                        inter_node = node_output_nodes1[0]
                        node_output_nodes1 = get_output_nodes_of_node(inter_node, model.graph)
                    else:
                        break
                logger.debug(f"node: {node.name}, output: {sub_model_output}")
                bias_tensor_name = None
                if len(node.input) == 3:
                    bias_tensor_name = node.input[2]
                node_bias_corr_output_map[node.name] = (sub_model_input, sub_model_output, bias_tensor_name)
                if sub_model_input not in all_sub_model_in_out:
                    all_sub_model_in_out.append(sub_model_input)
                if sub_model_output not in all_sub_model_in_out:
                    all_sub_model_in_out.append(sub_model_output)
        for inter_output in all_sub_model_in_out:
            if inter_output not in model_original_outputs:
                logger.debug(f"output:{value_infos[inter_output]}")
                model.graph.output.append(value_infos[inter_output])
        onnx_model = ONNXModel(model)
        onnx_model.topological_sort()
        onnx.save(
            onnx_model.model,
            augmented_model_path,
            save_as_external_data=self.use_external_data_format,
        )
        return node_bias_corr_output_map, tensors_to_calibrate

    def augment_origin_quant_graph(self) -> Tuple[Dict[str, Tuple[str, str, Optional[str]]], Set[str]]:
        quant_bc_pattern, quant_tensors_to_bc = self.get_bias_corr_pattern_augment_graph(
            self.quant_model, self.quant_augmented_model_path)
        return quant_bc_pattern, quant_tensors_to_bc

    def create_inference_session(self, augmented_model_path: str) -> onnxruntime.InferenceSession:
        """
        create an OnnxRuntime InferenceSession.
        """
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        infer_session = onnxruntime.InferenceSession(
            augmented_model_path,
            sess_options=sess_options,
            providers=self.execution_providers,
        )
        return infer_session

    def collect_data(self, data_reader: CachedDataReader, infer_session: onnxruntime.InferenceSession,
                     tensors_to_bc: Set[str]) -> Tuple[Dict[Any, Any], List[Any]]:
        data_reader.reset_iter()
        intermediate_outputs = []
        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            intermediate_outputs.append(infer_session.run(None, inputs))

        if len(intermediate_outputs) == 0:
            logger.warning("No data is collected.")

        output_names = [infer_session.get_outputs()[i].name for i in range(len(intermediate_outputs[0]))]
        output_dicts_list = [
            dict(zip(output_names, intermediate_output)) for intermediate_output in intermediate_outputs
        ]
        merged_dict: Dict[Any, Any] = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_dict.setdefault(k, []).append(v)

        clean_merged_dict = dict((i, merged_dict[i]) for i in merged_dict)
        #                         if i in tensors_to_bc)
        return clean_merged_dict, intermediate_outputs


def bias_correction(model_input: str,
                    model_output: str,
                    use_external_data_format: bool,
                    calibration_data_reader: CachedDataReader,
                    activation_type: Union[QuantType, VitisQuantType],
                    calibrate_method: Union[PowerOfTwoMethod, CalibrationMethod],
                    extra_options: Dict[str, Any],
                    execution_providers: List[str] = ['CPUExecutionProvider']) -> Any:
    logger.info('Start BiasCorrection...')
    from onnxruntime.quantization.quant_utils import load_model_with_shape_infer

    quant_model = load_model_with_shape_infer(Path(model_output))
    topo_model = ONNXModel(onnx.load(model_input))
    topo_model.topological_sort()
    model = topo_model.model
    bias_corr = BiasCorrection(model, quant_model, use_external_data_format, execution_providers)
    bc_path = tempfile.TemporaryDirectory(prefix="vai.bc.")
    bias_corr.quant_augmented_model_path = Path(bc_path.name).joinpath("quant_augmented_model.onnx").as_posix()
    quant_bc_pattern, quant_tensors_bc = bias_corr.augment_origin_quant_graph()

    bias_corr.execution_providers = execution_providers
    quant_infer_session = bias_corr.create_inference_session(bias_corr.quant_augmented_model_path)
    quant_clean_merged_dict, _ = bias_corr.collect_data(calibration_data_reader, quant_infer_session, quant_tensors_bc)
    quant_output_tensor_node_map = {}
    activation_type = get_tensor_type_from_qType(activation_type)
    # get all output tensor name to node obj

    for node in quant_model.graph.node:
        for out in node.output:
            quant_output_tensor_node_map[out] = node
    # get all weight name to obj map
    quant_initializer_name_map = {}
    for init in quant_model.graph.initializer:
        quant_initializer_name_map[init.name] = init
    # get all node name to obj map
    float_node_obj_map = {}
    for node in model.graph.node:
        float_node_obj_map[node.name] = node
    linear_and_quant_node_type = ["Relu", "Clip"]
    for bc_node_name, bc_in_out_bias_tensor in quant_bc_pattern.items():
        if bc_node_name in float_node_obj_map:
            bc_node_start = float_node_obj_map[bc_node_name]
            bc_node_end = float_node_obj_map[bc_node_name]
            node_output_nodes1 = get_output_nodes_of_node(bc_node_start, model.graph)
            while len(node_output_nodes1) > 0:
                if node_output_nodes1[0].op_type in linear_and_quant_node_type:
                    inter_node = node_output_nodes1[0]
                    bc_node_end = inter_node
                    node_output_nodes1 = get_output_nodes_of_node(inter_node, model.graph)
                else:
                    break

            quant_in_out_bias_tensor = bc_in_out_bias_tensor
            bc_input_tensor_name = quant_in_out_bias_tensor[0]
            bc_output_tensor_name = quant_in_out_bias_tensor[1]
            quant_node_bias_name = quant_in_out_bias_tensor[2]
            quant_input = quant_clean_merged_dict[bc_input_tensor_name]
            quant_output = quant_clean_merged_dict[bc_output_tensor_name]
            quant_in_tensor = quant_input
            float_output, _ = inference_sub_model_with_data(model, {bc_node_name: quant_in_tensor}, [bc_node_end.name])

            quant_out_tensor = np.array(quant_output)
            float_out_tensor = np.array(float_output)
            if quant_out_tensor.ndim == 5:
                axis_reduce_4dim = (0, 1, 3, 4)
                float_quant_diff = np.mean((float_out_tensor - quant_out_tensor), axis=axis_reduce_4dim)
            elif quant_out_tensor.ndim == 3:
                axis_reduce_2dim = (0, 1)
                float_quant_diff = np.mean((float_out_tensor - quant_out_tensor), axis=axis_reduce_2dim)
            else:
                logger.warning("the bias correction only support ndim 2 or 4")
                continue
            float_quant_diff_mean = float_quant_diff
            if quant_node_bias_name:
                quant_bias_node = quant_output_tensor_node_map[quant_node_bias_name]
                bias_name = quant_bias_node.input[0]
                scale_name = quant_bias_node.input[1]
                zp_name = quant_bias_node.input[2]
                bias_init = quant_initializer_name_map[bias_name]
                scale_init = quant_initializer_name_map[scale_name]
                zp_init = quant_initializer_name_map[zp_name]
                bias_data = numpy_helper.to_array(bias_init)
                scale_data = numpy_helper.to_array(scale_init)
                zp_data = numpy_helper.to_array(zp_init)
                bias_data_float = dequantize_data(bias_data, scale_data, zp_data)
                max_diff = np.max(np.abs(float_quant_diff_mean))
                max_bias = np.max(np.abs(bias_data_float))
                scale = 1
                plus_bias = max_bias / 256
                if max_diff > plus_bias and max_diff > 0.1:
                    scale = scale * plus_bias / max_diff
                bias_data_bc = bias_data_float + float_quant_diff_mean * scale
                logger.debug(f'the bias_data_float max: {np.max(np.abs(bias_data_float))}')
                logger.debug(f'the diff_mean max: {np.max(np.abs(float_quant_diff_mean*scale))}')
                quantized_data = None
                if calibrate_method in [PowerOfTwoMethod.NonOverflow, PowerOfTwoMethod.MinMSE]:
                    symmetric = False if "ActivationSymmetric" not in extra_options else extra_options[
                        "ActivationSymmetric"]
                    _, _, zp, scale, quantized_data = quantize_data_pof2s(bias_data_bc,
                                                                          bias_init.data_type,
                                                                          symmetric,
                                                                          method=calibrate_method)
                elif calibrate_method in [CalibrationMethod.MinMax, CalibrationMethod.Percentile]:
                    # for cpu the bias is symmetry
                    quantized_data = (np.asarray(bias_data_bc) / scale_data).round().astype(np.int32)
                    quantized_data = np.asarray(quantized_data, dtype=np.int32).reshape(bias_init.dims)
                if quantized_data is not None:
                    bias_bc = numpy_helper.from_array(quantized_data, bias_name)
                quant_model.graph.initializer.extend([bias_bc])
                quant_model.graph.initializer.remove(bias_init)
    calibration_data_reader.reset_iter()
    logger.info('BiasCorrection Done...')
    return quant_model
