#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from quark.shares.utils.log import ScreenLogger, log_errors
import copy
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from quark.onnx.quant_utils import (register_custom_ops_library, extract_sub_model, infer_custom_op_shape,
                                    make_batch_size_dynamic, make_batch_size_fixed, get_batch_size,
                                    create_infer_session_for_onnx_model)
from onnx import numpy_helper, TensorProto, NodeProto
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import save_and_reload_model_with_shape_infer
from typing import Dict, List, Tuple, Union, Any, Optional
from numpy.typing import NDArray

logger = ScreenLogger(__name__)

TARGET_OPS = ['Conv', 'ConvTranspose', 'InstanceNormalization', 'LayerNormalization', 'Gemm', 'MatMul']
Q = ["QuantizeLinear", "ExtendedQuantizeLinear"]
DQ = ["DequantizeLinear", "ExtendedDequantizeLinear"]
FN = ["BFPQuantizeDequantize", "MXQuantizeDequantize"]
ACT_OPS = ['Relu', 'PRelu', 'LeakyRelu', 'Gelu', 'Tanh', 'Clip', 'Sigmoid', 'Softmax']


class Subgraph(object):
    """
    A class for split subgraph for adaquant or adaround.
    """

    def __init__(self, float_model: Union[str, Path, onnx.ModelProto], quant_model: Union[str, Path, onnx.ModelProto],
                 use_external_data_format: bool, data_reader: Any, extra_options: Dict[str, Any]) -> None:
        # Get the float and quantized model
        self.float_model = float_model if isinstance(float_model, onnx.ModelProto) else onnx.load(float_model)
        self.quant_model = quant_model if isinstance(quant_model, onnx.ModelProto) else onnx.load(quant_model)
        self.use_external_data_format = use_external_data_format
        self.data_reader = data_reader

        # Get parameters from extra options
        self.data_size = extra_options.get('FastFinetune', {}).get('DataSize', float('inf'))
        self.output_qdq = extra_options.get('FastFinetune', {}).get('OutputQDQ', False)
        self.target_ops = extra_options.get('FastFinetune', {}).get('TargetOpType', TARGET_OPS)
        self.quantize_bias = extra_options.get('QuantizeBias', True)
        self.dynamic_batch = extra_options.get('FastFinetune', {}).get('DynamicBatch', False)
        self.parallel = extra_options.get('FastFinetune', {}).get('Parallel', False)
        self.mem_opt_level = extra_options.get('FastFinetune', {}).get('MemOptLevel', 1)

        self.ort_infer_device = extra_options.get('FastFinetune', {}).get('InferDevice', 'cpu').lower()
        self.providers, self.provider_options = self.onnx_execution_providers()
        self.device = self.providers[0][:-len("ExecutionProvider")]

        self.qbatch_size = get_batch_size(self.quant_model)
        if self.dynamic_batch:
            self.all_input, self.dynamic_batch_size = self.get_all_input()
            self.fmodel = infer_custom_op_shape(make_batch_size_dynamic(self.float_model, self.dynamic_batch_size))
            self.qmodel = infer_custom_op_shape(make_batch_size_dynamic(self.quant_model, self.dynamic_batch_size))
        else:
            self.fmodel = infer_custom_op_shape(self.float_model)
            self.qmodel = infer_custom_op_shape(self.quant_model)

        if self.use_external_data_format:
            self.fmodel = save_and_reload_model_with_shape_infer(self.fmodel)
            self.qmodel = save_and_reload_model_with_shape_infer(self.qmodel)

        self.f_tensor_to_producer = self.get_f_tensor_to_producer()
        self.q_tensor_to_producer = self.get_q_tensor_to_producer()
        self.f_tensor_to_consumer = self.get_f_tensor_to_consumer()
        self.q_tensor_to_consumer = self.get_q_tensor_to_consumer()

        self.subgraph_qmodel, self.qsubgraph_input_tensor, self.subgraph_act = self.get_subgraph_qmodel()
        self.subgraph_fmodel, self.fsubgraph_input_output_tensors = self.get_subgraph_fmodel(
            self.subgraph_qmodel, self.subgraph_act)
        self.subgraph_qmodel_list = list(self.subgraph_qmodel.values())
        self.subgraph_fmodel_list = list(self.subgraph_fmodel.values())
        self.qsubgraph_input_tensor_list = list(self.qsubgraph_input_tensor.values())
        self.fsubgraph_input_tensor_list = [value[0] for value in self.fsubgraph_input_output_tensors.values()]
        self.fsubgraph_output_tensor_list = [value[1] for value in self.fsubgraph_input_output_tensors.values()]
        self.f_weight_list, self.q_weight_name_list, self.f_bias_list, self.q_bias_name_list = self.extract_submodel_weight(
        )

        if self.mem_opt_level == 0:
            # In this model, input and output data for each layer can be obtained
            # with just single-pass inference, which is faster but consumes more memory
            # because it has to cache all of the data
            logger.warning(
                "No memory optimization is applied, this will be faster but requires more memory for caching")
            self.f_input_data, self.f_output_data = self.get_f_input_output_data_single_pass()
            self.f_input_data_list = list(self.f_input_data.values())
            self.f_output_data_list = list(self.f_output_data.values())

        if self.parallel:
            # In this mode, input data of each laye can be obtained in parallel,
            # which is faster but hurts the accuracy because it ignores the
            # parameters that have been optimized
            logger.warning("Parallel fast finetuning, this will be faster but will have slightly impact on accuracy")
            self.q_input_data = self.get_q_input_data_in_parallel()
            self.q_input_data_list = list(self.q_input_data.values())

    def onnx_execution_providers(self) -> Tuple[List[str], Optional[List[Dict[str, str]]]]:
        providers: List[str] = ['CPUExecutionProvider']
        provider_options: Optional[List[Dict[str, str]]] = None

        # In order to be consistent with torch optimization device,
        # here it only supports 'cuda' prefix
        if self.ort_infer_device.startswith('cuda'):
            available_providers = ort.get_available_providers()
            if 'ROCMExecutionProvider' in available_providers:
                providers = ['ROCMExecutionProvider']
            elif 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider']
            else:
                logger.warning("No GPUEPs available in current ORT version, falling back to CPU")

            # If users only filled in 'cuda', then there is no need to specify the device id
            if len(self.ort_infer_device) > 5:
                decive_ids = [i for i in self.ort_infer_device[5:].split(',')]
                for ids in decive_ids:
                    if isinstance(provider_options, list):
                        provider_options.append({'device_id': ids})
                    else:
                        provider_options = [{'device_id': ids}]

            if provider_options is not None and len(provider_options) > 1:
                providers = providers * len(provider_options)

        return providers, provider_options

    def get_f_tensor_to_producer(self) -> Dict[str, NodeProto]:
        onnx_fmodel = ONNXModel(self.fmodel)
        tensor_to_producer = {}
        for node in onnx_fmodel.model.graph.node:
            for output in node.output:
                tensor_to_producer[output] = node
        for init in onnx_fmodel.model.graph.initializer:
            tensor_to_producer[init.name] = init
        return tensor_to_producer

    def get_q_tensor_to_producer(self) -> Dict[str, NodeProto]:
        onnx_qmodel = ONNXModel(self.qmodel)
        tensor_to_producer = {}
        for node in onnx_qmodel.model.graph.node:
            for output in node.output:
                tensor_to_producer[output] = node
        for init in onnx_qmodel.model.graph.initializer:
            tensor_to_producer[init.name] = init
        return tensor_to_producer

    def get_f_tensor_to_consumer(self) -> Dict[str, NodeProto]:
        onnx_fmodel = ONNXModel(self.fmodel)
        tensor_to_consumer = {}
        for node in onnx_fmodel.model.graph.node:
            for input in node.input:
                tensor_to_consumer[input] = node
        for init in onnx_fmodel.model.graph.initializer:
            tensor_to_consumer[init.name] = init
        return tensor_to_consumer

    def get_q_tensor_to_consumer(self) -> Dict[str, NodeProto]:
        onnx_qmodel = ONNXModel(self.qmodel)
        tensor_to_consumer = {}
        for node in onnx_qmodel.model.graph.node:
            for input in node.input:
                tensor_to_consumer[input] = node
        for init in onnx_qmodel.model.graph.initializer:
            tensor_to_consumer[init.name] = init
        return tensor_to_consumer

    def check_qmodel_constb_matmul(self, node: NodeProto) -> bool:
        if node.op_type != "MatMul" or len(node.input) != 2:
            return False

        inp_1 = node.input[1]  # check input b only
        if inp_1 not in self.q_tensor_to_producer:
            return False

        inp_1_node = self.q_tensor_to_producer[inp_1]
        if isinstance(inp_1_node, TensorProto):
            return True  # it's a initializer and was not quantized
        elif not (isinstance(inp_1_node, NodeProto) and inp_1_node.op_type in DQ + FN):
            return False  # it's not a quantization node

        if inp_1_node.input[0] not in self.q_tensor_to_producer:
            return False

        inp_1_node_parent = self.q_tensor_to_producer[inp_1_node.input[0]]
        if isinstance(inp_1_node_parent, TensorProto):
            return True  # it's a initializer and was quantized by DQ (Q was folded) or FN
        elif inp_1_node.op_type in DQ:
            if isinstance(inp_1_node_parent, NodeProto) and inp_1_node_parent.op_type in Q:
                if inp_1_node_parent.input[0] not in self.q_tensor_to_producer:
                    return False
                elif isinstance(self.q_tensor_to_producer[inp_1_node_parent.input[0]], TensorProto):
                    return True  # it's a initializer and was quantized by Q/DQ

        return False

    def find_start(self, node: NodeProto) -> Any:
        try:
            inp_0 = node.input[0]
            inp_0_node = self.q_tensor_to_producer[inp_0]
            inp_1 = node.input[1]
            inp_1_node = self.q_tensor_to_producer[inp_1]

            # ensure the weight was quantized
            if not (isinstance(inp_1_node, NodeProto) and inp_1_node.op_type in DQ + FN):
                return []

            # find the input tensor as a start
            if inp_0_node.op_type in DQ:
                dq_node = inp_0_node
                dq_inp_0 = dq_node.input[0]
                dq_inp_0_node = self.q_tensor_to_producer[dq_inp_0]
                if dq_inp_0_node.op_type in Q:
                    return dq_inp_0_node.input[0]
            elif inp_0_node.op_type in FN:
                return inp_0_node.input[0]
            else:
                return inp_0_node.output[0]
        except Exception as e:
            logger.debug(f"Cannot find start tensor because {e}")

        return []

    def find_end(self, node: NodeProto) -> Tuple[Any, bool]:
        try:
            out_0 = node.output[0]
            out_0_node = self.q_tensor_to_consumer[out_0]
            if out_0_node.op_type in Q:
                if not self.output_qdq:
                    return out_0, False
                else:
                    # Remove qdq after output
                    q_node = out_0_node
                    q_out_0 = q_node.output[0]
                    q_out_0_node = self.q_tensor_to_consumer[q_out_0]
                    if q_out_0_node.op_type in DQ:
                        return q_out_0_node.output[0], False
            elif out_0_node.op_type in FN:
                if not self.output_qdq:
                    return out_0, False
                else:
                    # Remove qdq after output
                    return out_0_node.output[0], False
            else:
                a_node = out_0_node
                if a_node.op_type in ACT_OPS:
                    a_out_0 = a_node.output[0]
                    if not self.output_qdq:
                        return a_out_0, True
                    else:
                        # Remove qdq after output
                        a_out_0_node = self.q_tensor_to_consumer[a_out_0]
                        if a_out_0_node.op_type in Q:
                            q_node = a_out_0_node
                            q_out_0 = q_node.output[0]
                            q_out_0_node = self.q_tensor_to_consumer[q_out_0]
                            if q_out_0_node.op_type in DQ:
                                return q_out_0_node.output[0], True
                        elif a_out_0_node.op_type in FN:
                            return a_out_0_node.output[0], True
                        else:
                            return a_out_0, True
                else:
                    return out_0, False
        except Exception as e:
            logger.debug(f"Cannot find end tensor because {e}")

        return [], False

    def get_subgraph_qmodel(self) -> Any:

        subgraph_qmodel: Dict[str, Any] = {}
        subgraph_start: Dict[str, Any] = {}
        subgraph_act: Dict[str, bool] = {}

        for node in self.qmodel.graph.node:
            if node.op_type in self.target_ops:
                if node.op_type == "MatMul" and not self.check_qmodel_constb_matmul(node):
                    continue
                start_tensor = self.find_start(node)
                end_tensor, is_act = self.find_end(node)
                if len(start_tensor) > 0 and len(end_tensor) > 0:
                    subgraph_qmodel[node.name] = extract_sub_model(self.qmodel, [start_tensor], [end_tensor])
                    subgraph_start[node.name] = start_tensor
                    subgraph_act[node.name] = is_act
        return subgraph_qmodel, subgraph_start, subgraph_act

    def get_subgraph_fmodel(
            self, subgraph_qmodel: Dict[str, onnx.ModelProto],
            subgraph_act: Dict[str, bool]) -> Tuple[Dict[str, onnx.ModelProto], Dict[str, Tuple[str, str]]]:
        subgraph_fmodel = {}
        subgraph_start_end = {}
        for k in subgraph_qmodel:
            for n in self.fmodel.graph.node:
                if n.name == k:
                    start_tensor = n.input[0]
                    # Allow using another quantized model as the reference model
                    if n.input[0] in self.f_tensor_to_producer:
                        parent = self.f_tensor_to_producer[n.input[0]]
                        if isinstance(parent, NodeProto):
                            if parent.op_type in FN:
                                start_tensor = parent.input[0]
                            elif parent.op_type in DQ:
                                if parent.input[0] in self.f_tensor_to_producer:
                                    parent = self.f_tensor_to_producer[parent.input[0]]
                                    if isinstance(parent, NodeProto) and parent.op_type in Q:
                                        start_tensor = parent.input[0]
                    if subgraph_act[k]:
                        act_node = self.f_tensor_to_consumer[n.output[0]]
                        end_tensor = act_node.output[0]
                    else:
                        end_tensor = n.output[0]
                    subgraph_fmodel[k] = extract_sub_model(self.fmodel, [start_tensor], [end_tensor])
                    subgraph_start_end[k] = (start_tensor, end_tensor)
        return subgraph_fmodel, subgraph_start_end

    def get_f_input_output_data_single_pass(
        self
    ) -> Tuple[Dict[int, Union[List[NDArray[Any]], NDArray[Any]]], Dict[int, Union[List[NDArray[Any]], NDArray[Any]]]]:
        model_original_outputs = set(output.name for output in self.fmodel.graph.output)
        for f_in in self.fsubgraph_input_tensor_list:
            if f_in not in model_original_outputs:
                model_original_outputs.add(f_in)
                self.fmodel.graph.output.extend([onnx.ValueInfoProto(name=f_in)])
        for f_out in self.fsubgraph_output_tensor_list:
            if f_out not in model_original_outputs:
                model_original_outputs.add(f_out)
                self.fmodel.graph.output.extend([onnx.ValueInfoProto(name=f_out)])

        f_input_data: Dict[int, Union[NDArray[Any], List[NDArray[Any]]]] = {}
        f_output_data: Dict[int, Union[NDArray[Any], List[NDArray[Any]]]] = {}

        f_session = create_infer_session_for_onnx_model(self.fmodel,
                                                        providers=self.providers,
                                                        provider_options=self.provider_options,
                                                        use_external_data_format=self.use_external_data_format)

        for i, _ in enumerate(self.fsubgraph_input_tensor_list):
            f_input_data[i] = []
        for i, _ in enumerate(self.fsubgraph_output_tensor_list):
            f_output_data[i] = []
        if self.dynamic_batch:
            outputs = f_session.run(self.fsubgraph_input_tensor_list + self.fsubgraph_output_tensor_list,
                                    self.all_input)
            for i, f_in in enumerate(self.fsubgraph_input_tensor_list):
                f_input_data[i] = np.expand_dims(np.array(outputs[i]), axis=0)
            offset = len(self.fsubgraph_input_tensor_list)
            for i, f_out in enumerate(self.fsubgraph_output_tensor_list):
                f_output_data[i] = np.expand_dims(np.array(outputs[i + offset]), axis=0)
        else:
            n = 1
            self.data_reader.reset_iter()
            while True:
                inputs = self.data_reader.get_next()
                if not inputs or n > self.data_size:
                    break
                outputs = f_session.run(self.fsubgraph_input_tensor_list + self.fsubgraph_output_tensor_list, inputs)
                for i, f_in in enumerate(self.fsubgraph_input_tensor_list):
                    f_input_data[i].append(np.array(outputs[i]))  # type: ignore
                offset = len(self.fsubgraph_input_tensor_list)
                for i, f_out in enumerate(self.fsubgraph_output_tensor_list):
                    f_output_data[i].append(np.array(outputs[i + offset]))  # type: ignore
                n = n + 1
        return f_input_data, f_output_data

    def get_f_input_output_data(
            self,
            index: int) -> Tuple[Union[NDArray[Any], List[NDArray[Any]]], Union[NDArray[Any], List[NDArray[Any]]]]:
        aug_model = copy.deepcopy(self.fmodel)
        model_original_inputs = [n.name for n in aug_model.graph.input]
        model_original_outputs = set(output.name for output in aug_model.graph.output)

        f_out = self.fsubgraph_output_tensor_list[index]
        assert isinstance(f_out, str), f"Invalid tensor name {f_out} for float subgraph"
        if f_out not in model_original_outputs:
            try:
                # To accelerate the inference, we only run a sub-model
                sub_aug_model = extract_sub_model(aug_model, model_original_inputs, [f_out])
                aug_model = sub_aug_model  # This line ensure the aug_model will not be updated if above line failed
            except Exception as e:
                logger.warning(f"Fail to extract sub-model from fmodel because of {e}, full model will be used.")
                aug_model.graph.output.extend([onnx.ValueInfoProto(name=f_out)])

        f_in = self.fsubgraph_input_tensor_list[index]
        assert isinstance(f_in, str), f"Invalid tensor name {f_in} for float subgraph"
        if f_in not in [output.name for output in aug_model.graph.output]:
            aug_model.graph.output.extend([onnx.ValueInfoProto(name=f_in)])

        f_session = create_infer_session_for_onnx_model(aug_model,
                                                        providers=self.providers,
                                                        provider_options=self.provider_options,
                                                        use_external_data_format=self.use_external_data_format)

        # Get input data
        f_input_data: Union[NDArray[Any], List[NDArray[Any]]] = []
        f_output_data: Union[NDArray[Any], List[NDArray[Any]]] = []

        if self.dynamic_batch:
            # Expand one dimension before batch dim for the external requirement
            outputs = f_session.run([f_in, f_out], self.all_input)
            f_input_data = np.expand_dims(np.array(outputs[0]), axis=0)
            f_output_data = np.expand_dims(np.array(outputs[1]), axis=0)
        else:
            n = 1
            self.data_reader.reset_iter()
            while True:
                inputs = self.data_reader.get_next()
                if not inputs or n > self.data_size:
                    break
                # The two variables used to output will be of type List
                outputs = f_session.run([f_in, f_out], inputs)
                f_input_data.append(np.array(outputs[0]))  # type: ignore
                f_output_data.append(np.array(outputs[1]))  # type: ignore
                n = n + 1
        return f_input_data, f_output_data

    def get_q_input_data_in_parallel(self) -> Dict[int, Union[NDArray[Any], List[NDArray[Any]]]]:
        aug_model = copy.deepcopy(self.qmodel)
        model_original_outputs = set(output.name for output in aug_model.graph.output)
        for q_in in self.qsubgraph_input_tensor_list:
            if q_in not in model_original_outputs:
                model_original_outputs.add(q_in)
                aug_model.graph.output.extend([onnx.ValueInfoProto(name=q_in)])

        q_input_data: Dict[int, Union[NDArray[Any], List[NDArray[Any]]]] = {}

        q_sess_options = ort.SessionOptions()
        q_sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        register_custom_ops_library(q_sess_options, self.device)
        q_session = create_infer_session_for_onnx_model(aug_model,
                                                        q_sess_options,
                                                        providers=self.providers,
                                                        provider_options=self.provider_options,
                                                        use_external_data_format=self.use_external_data_format)

        for i, _ in enumerate(self.qsubgraph_input_tensor_list):
            q_input_data[i] = []

        if self.dynamic_batch:
            outputs = q_session.run(self.qsubgraph_input_tensor_list, self.all_input)
            for i, _ in enumerate(self.qsubgraph_input_tensor_list):
                q_input_data[i] = np.expand_dims(np.array(outputs[i]), axis=0)

        else:
            n = 1
            self.data_reader.reset_iter()
            while True:
                inputs = self.data_reader.get_next()
                if not inputs or n > self.data_size:
                    break
                outputs = q_session.run(self.qsubgraph_input_tensor_list, inputs)
                for i, _ in enumerate(self.qsubgraph_input_tensor_list):
                    q_input_data[i].append(np.array(outputs[i]))  # type: ignore
                n = n + 1
        return q_input_data

    def get_q_input_data(self, index: int) -> Any:
        aug_model = copy.deepcopy(self.qmodel)
        q_in = self.qsubgraph_input_tensor_list[index]
        input_names = {n.name for n in aug_model.graph.input}
        output_names = {n.name for n in aug_model.graph.output}
        if q_in in input_names:
            aug_model.graph.output.extend([onnx.ValueInfoProto(name=q_in)])
        elif q_in in output_names:
            pass
        else:
            try:
                # To accelerate the inference, we only run a sub-model
                sub_aug_model = extract_sub_model(aug_model, [n.name for n in aug_model.graph.input], [q_in])
                aug_model = sub_aug_model  # This line ensure the aug_model will not be updated if above line failed
            except Exception as e:
                logger.warning(f"Fail to extract sub-model from qmodel because of {e}, full model will be used.")
                aug_model.graph.output.extend([onnx.ValueInfoProto(name=q_in)])

        q_sess_options = ort.SessionOptions()
        q_sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        register_custom_ops_library(q_sess_options, self.device)
        q_session = create_infer_session_for_onnx_model(aug_model,
                                                        q_sess_options,
                                                        providers=self.providers,
                                                        provider_options=self.provider_options,
                                                        use_external_data_format=self.use_external_data_format)

        # Get input data
        q_input_data: Union[NDArray[Any], List[NDArray[Any]]] = []

        if self.dynamic_batch:
            q_input_data = np.expand_dims(np.array(q_session.run([q_in], self.all_input)[0]), axis=0)
        else:
            n = 1
            self.data_reader.reset_iter()
            while True:
                inputs = self.data_reader.get_next()
                if not inputs or n > self.data_size:
                    break
                q_input_data.append(np.array(q_session.run([q_in], inputs)[0]))  # type: ignore
                n = n + 1
        return q_input_data

    def get_all_input(self) -> Tuple[Dict[str, List[NDArray[Any]]], int]:
        n = 1
        self.data_reader.reset_iter()
        all_inputs = self.data_reader.get_next()
        if all_inputs is not None:
            concat_dict = {key: [value] for key, value in all_inputs.items()}

        while True:
            inputs = self.data_reader.get_next()
            n = n + 1
            if not inputs or n > self.data_size:
                break

            for key in inputs.keys():
                concat_dict[key].append(inputs[key])

        for key in concat_dict.keys():
            concat_dict[key] = np.concatenate(concat_dict[key], axis=0)

        return concat_dict, n - 1

    @log_errors
    def extract_submodel_weight(
            self) -> Tuple[list[np.ndarray[Any, Any]], list[Any], list[Union[np.ndarray[Any, Any], None]], list[Any]]:
        f_weight = []
        f_bias: List[Union[NDArray[Any], None]] = []
        q_weight_name = []
        q_bias_name = []
        for i, name in enumerate(self.subgraph_fmodel):
            model = self.subgraph_fmodel[name]
            for node in model.graph.node:
                if node.name == name:
                    weight_name = node.input[1]
                    if self.quantize_bias and len(node.input) == 3:
                        bias_name = node.input[2]
                    is_weight_init = False
                    for init in model.graph.initializer:
                        if init.name == weight_name:
                            weight = numpy_helper.to_array(init)
                            f_weight.append(weight)
                            is_weight_init = True
                            break
                    if not is_weight_init:
                        raise ValueError("The weight of conv is not an initializer.")
                    is_bias_init = False
                    if self.quantize_bias and len(node.input) == 3:
                        for init in model.graph.initializer:
                            if init.name == bias_name:
                                bias = numpy_helper.to_array(init)
                                f_bias.append(bias)
                                is_bias_init = True
                                break
                        if not is_bias_init:
                            raise ValueError("The bias of conv is not an initializer.")
                    else:
                        f_bias.append(None)
        for _, name in enumerate(self.subgraph_qmodel):
            model = self.subgraph_qmodel[name]
            for node in model.graph.node:
                if node.name == name:
                    quant_node = self.q_tensor_to_producer[node.input[1]]  # DQ or FixNeuron
                    candidate = self.q_tensor_to_producer[quant_node.input[0]]  # Q or initializer
                    if isinstance(candidate, TensorProto):
                        weight_name = quant_node.input[0]
                    else:
                        weight_name = candidate.input[0]
                    if self.quantize_bias and len(node.input) == 3:
                        quant_node = self.q_tensor_to_producer[node.input[2]]  # DQ or FixNeuron
                        candidate = self.q_tensor_to_producer[quant_node.input[0]]  # Q or initializer
                        if isinstance(candidate, TensorProto):
                            bias_name = quant_node.input[0]
                        else:
                            bias_name = candidate.input[0]
                    is_weight_init = False
                    for init in model.graph.initializer:
                        if init.name == weight_name:
                            weight = numpy_helper.to_array(init)
                            q_weight_name.append(weight_name)
                            is_weight_init = True
                            break
                    if not is_weight_init:
                        raise ValueError("The weight of conv is not an initializer.")
                    if self.quantize_bias and len(node.input) == 3:
                        is_bias_init = False
                        for init in model.graph.initializer:
                            if init.name == bias_name:
                                bias = numpy_helper.to_array(init)
                                q_bias_name.append(bias_name)
                                is_bias_init = True
                                break
                        if not is_bias_init:
                            raise ValueError("The bias of conv is not an initializer.")
                    else:
                        q_bias_name.append(None)
        return f_weight, q_weight_name, f_bias, q_bias_name

    def convert_qmodel_batch_size(self) -> Any:
        return make_batch_size_fixed(self.qmodel, self.qbatch_size)
