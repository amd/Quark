#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Union, Tuple, List, Dict, Any
import numpy as np
from numpy.typing import NDArray
import onnx
from onnx import onnx_pb as onnx_proto
from onnx.onnx_ml_pb2 import AttributeProto, TensorProto, NodeProto

import torch

from quark.shares.utils.log import ScreenLogger, log_errors

logger = ScreenLogger(__name__)

ComputeOperations = ('Conv', 'ConvTranspose', 'Gemm', 'MatMul')
NormalizationOperations = ("InstanceNormalization", "VitisInstanceNormalization", "LayerNormalization")
ActivationMapping = {
    'Relu': torch.nn.ReLU(inplace=True),
    'PRelu': torch.nn.PReLU(),
    'LeakyRelu': torch.nn.LeakyReLU(),
    'Tanh': torch.nn.Tanh(),
    'Clip': torch.nn.ReLU6(inplace=True),
    'Sigmoid': torch.nn.Sigmoid(),
    'Softmax': torch.nn.Softmax(),
    'Gelu': torch.nn.GELU(),
}

QuantizeLinearOps = ("QuantizeLinear", "VitisQuantizeLinear")
DequantizeLinearOps = ("DequantizeLinear", "VitisDequantizeLinear")
FixNeuronOps = ("FixNeuron", "BFPFixNeuron", "MXFixNeuron")

AttributeType = dict(
    UNDEFINED=0,
    FLOAT=1,
    INT=2,
    STRING=3,
    TENSOR=4,
    GRAPH=5,
    SPARSE_TENSOR=11,
    FLOATS=6,
    INTS=7,
    STRINGS=8,
    TENSORS=9,
    GRAPHS=10,
    SPARSE_TENSORS=12,
)


def extract_attr_values(attr: AttributeProto) -> Any:
    """Extract onnx attribute values."""
    value: Any = None
    if attr.type == AttributeType["INT"]:
        value = attr.i
    elif attr.type == AttributeType["FLOAT"]:
        value = attr.f
    elif attr.type == AttributeType["INTS"]:
        value = tuple(attr.ints)
    elif attr.type == AttributeType["FLOATS"]:
        value = tuple(attr.floats)
    elif attr.type == AttributeType["TENSOR"]:
        value = onnx.numpy_helper.to_array(attr.t)
    elif attr.type == AttributeType["STRING"]:
        value = attr.s.decode()
    elif attr.type == AttributeType["GRAPH"]:
        value = attr.g
    else:
        raise NotImplementedError("Extraction of attribute type {} not implemented.".format(attr.type))
    return value


class ONNXModelParser(object):

    def __init__(self, onnx_model: onnx.ModelProto) -> None:

        self.model = onnx_model

        self.dtype_to_qrange = self._dtype_to_qrange()
        self.name_to_init = self._name_to_initializer()
        self.in_name_to_nodes = self._input_name_to_nodes()
        self.out_name_to_node = self._output_name_to_node()

    def _dtype_to_qrange(self) -> Dict[Any, Tuple[Union[int, float], Union[int, float]]]:
        """ Range of different integer data types quantization """
        dtype_to_qrange = {
            onnx_proto.TensorProto.UINT4: (0, 15),
            onnx_proto.TensorProto.INT4: (-8, 7),
            onnx_proto.TensorProto.UINT8: (0, 255),
            onnx_proto.TensorProto.INT8: (-128, 127),
            onnx_proto.TensorProto.UINT16: (0, 65535),
            onnx_proto.TensorProto.INT16: (-32768, 32767),
            onnx_proto.TensorProto.UINT32: (0, 2**32 - 1),
            onnx_proto.TensorProto.INT32: (-2**31, 2**31 - 1),
            onnx_proto.TensorProto.FLOAT16: (-65504.0, 65504.0),
            onnx_proto.TensorProto.BFLOAT16: (-3.38953139e38, 3.38953139e38),
        }
        return dtype_to_qrange

    def _name_to_initializer(self) -> Dict[str, TensorProto]:
        """ The initializer who provides the tensor, one to one """
        name_to_init: Dict[str, TensorProto] = {}
        for init in self.model.graph.initializer:
            name_to_init[init.name] = init
        return name_to_init

    def _input_name_to_nodes(self) -> Dict[str, List[NodeProto]]:
        """ The nodes whose inputs include the tensor, one to many """
        in_name_to_nodes: Dict[str, List[NodeProto]] = {}
        for node in self.model.graph.node:
            for in_name in node.input:
                if in_name not in in_name_to_nodes:
                    in_name_to_nodes[in_name] = [node]
                else:
                    in_name_to_nodes[in_name].append(node)
        return in_name_to_nodes

    def _output_name_to_node(self) -> Dict[str, NodeProto]:
        """ The node who outputs the tensor, one to one """
        out_name_to_node: Dict[str, NodeProto] = {}
        for node in self.model.graph.node:
            for out_name in node.output:
                out_name_to_node[out_name] = node
        return out_name_to_node

    def _find_node_input_init(self, node: NodeProto, index: int) -> Union[TensorProto, None]:
        """ Find the initializer who provides the node's input tensor """
        if index < 0 or index >= len(node.input):
            return None
        elif node.input[index] not in self.name_to_init:
            return None

        return self.name_to_init[node.input[index]]

    @log_errors
    def _find_node_input_qdq(self, node: NodeProto,
                             index: int) -> Tuple[Union[NodeProto, None], Union[NodeProto, None]]:
        """ Find node's input qdq nodes, dq always exits but q may be folded """
        if index < 0 or index >= len(node.input):
            raise ValueError(f"index {index} exceeded the number of inputs for {node.name}")
            return None, None

        tensor_name = node.input[index]
        if tensor_name not in self.out_name_to_node:
            logger.debug(f"input {tensor_name} of {node.name} came from initializer")
            return None, None

        dq_candidate = self.out_name_to_node[tensor_name]
        if dq_candidate.op_type not in DequantizeLinearOps:
            logger.debug(f"input {tensor_name} of {node.name} was not quantized")
            return None, None
        elif dq_candidate.input[0] not in self.out_name_to_node:
            logger.debug(f"input {tensor_name} of {node.name} has a folded Q")
            return dq_candidate, None

        q_candidate = self.out_name_to_node[dq_candidate.input[0]]
        if q_candidate.op_type not in QuantizeLinearOps:
            logger.warning(f"input {tensor_name} of {node.name} lost a Q")
            return dq_candidate, None

        return dq_candidate, q_candidate

    @log_errors
    def _find_node_input_fn(self, node: NodeProto, index: int) -> Union[NodeProto, None]:
        """ Find node's input fixneuron"""
        if index < 0 or index >= len(node.input):
            raise ValueError(f"index {index} exceeded the number of inputs for {node.name}")
            return None

        tensor_name = node.input[index]
        if tensor_name not in self.out_name_to_node:
            logger.debug(f"input {tensor_name} of {node.name} came from initializer")
            return None

        fn_candidate = self.out_name_to_node[tensor_name]
        if fn_candidate.op_type not in FixNeuronOps:
            logger.debug(f"input {tensor_name} of {node.name} was not quantized by FixNeuron")
            return None

        return fn_candidate

    @log_errors
    def _parse_qdq_quant_info(
        self, dq: Union[NodeProto, None], q: Union[NodeProto, None]
    ) -> Union[Tuple[NDArray[np.float32], NDArray[Any], NDArray[Any], NDArray[Any], int, bool, Any], None]:
        """ Parse quantization info from the QantizeLinear or DeqantizeLinear.
        The quantization info contains scale, zero_point, max value of the quantized data type,
        min value of the quantized data type, the flag of whether QantizeLinear was folded or not.
        """
        qnode = q if dq is None else dq

        if qnode is None:
            logger.warning("not found qdq for parsing quantizaion information")
            return None

        scale_init = self.name_to_init[qnode.input[1]]
        scale = onnx.numpy_helper.to_array(scale_init)
        scale = scale.copy()  # The numpy array from init may be not writable

        zero_point_init = self.name_to_init[qnode.input[2]]
        zero_point = onnx.numpy_helper.to_array(zero_point_init)
        zero_point = zero_point.copy()  # Make it writable to avoid the warning

        if zero_point_init.data_type not in self.dtype_to_qrange:
            # This data type is not supported
            raise ValueError(f"unsupport this qdq that has data type {zero_point_init.data_type}")
        elif zero_point_init.data_type in [TensorProto.FLOAT16, TensorProto.BFLOAT16]:
            # For the float16 and bfloat16 quantization, we just unify the zp as float32 data type,
            # but in fact it's float32 already at the conversion of onnx.numpy_helper.to_array()
            zero_point = zero_point.astype(np.float32)
        else:
            # The torch.from_numpy does not support uint16 and uint32,
            # therefore unified it as int64 for the integer quantizer
            zero_point = zero_point.astype(np.int64)

        min_q_init, max_q_init = self.dtype_to_qrange[zero_point_init.data_type]
        min_q = np.array(min_q_init)
        max_q = np.array(max_q_init)

        ch_axis = 0  # The axis to apply per-channel, which is always for weight
        for attr in qnode.attribute:
            if attr.name == "axis":
                ch_axis = attr.i  # It's a INT attribute of Q/DQ

        q_folded = q is None

        quant_type = zero_point_init.data_type  # To distinguish integer or floating point quantization

        return (scale, zero_point, min_q, max_q, ch_axis, q_folded, quant_type)

    def _parse_fn_quant_info(self, fn: NodeProto) -> Union[Dict[str, Any], None]:
        """ Parse quantization info from the FixNeuron and return its attributes.
        """
        qnode = fn

        if qnode is None:
            logger.warning("no fixneuron for parsing quantizaion information")
            return None

        quant_info: Dict[str, Any] = {}

        attrs = {}
        for attr in qnode.attribute:
            attrs[attr.name] = extract_attr_values(attr)
        quant_info['op_attrs'] = attrs

        quant_info['op_type'] = qnode.op_type  # To distinguish different fix neurons

        return quant_info

    def get_inputs_qinfo(
        self, node: NodeProto
    ) -> List[Union[Tuple[NDArray[np.float32], NDArray[Any], NDArray[Any], NDArray[Any], int, bool, Any], Dict[
            str, Any], None]]:
        """ Get the quantization info of each input for the node """
        qinfos: List[Union[Tuple[NDArray[np.float32], NDArray[Any], NDArray[Any], NDArray[Any], int, bool, Any],
                           Dict[str, Any], None]] = [None] * len(node.input)

        for index, name in enumerate(node.input):
            dq, q = self._find_node_input_qdq(node, index)
            if dq is not None or q is not None:
                qinfos[index] = self._parse_qdq_quant_info(dq, q)
            else:
                fn = self._find_node_input_fn(node, index)
                if fn is not None:
                    qinfos[index] = self._parse_fn_quant_info(fn)

        return qinfos

    @log_errors
    def _find_node_output_qdq(self, node: NodeProto,
                              index: int) -> Tuple[Union[NodeProto, None], Union[NodeProto, None]]:
        """ Find node's output qdq nodes, dq and q may not have either """
        if index < 0 or index >= len(node.output):
            raise ValueError(f"index {index} exceeded the number of outputs for {node.name}")

        tensor_name = node.output[index]
        if tensor_name not in self.in_name_to_nodes:
            logger.debug(f"output {tensor_name} of {node.name} may be model's output")
            return None, None

        q_candidate = self.in_name_to_nodes[tensor_name][0]
        if q_candidate.op_type not in QuantizeLinearOps:
            logger.debug(f"output {tensor_name} of {node.name} was not quantized")
            return None, None
        elif q_candidate.output[0] not in self.in_name_to_nodes:
            raise ValueError(f"output {tensor_name} of {node.name} has an isolate Q")

        dq_candidate = self.in_name_to_nodes[q_candidate.output[0]][0]
        if dq_candidate.op_type not in DequantizeLinearOps:
            logger.warning(f"output {tensor_name} of {node.name} lost a DQ")
            return q_candidate, None

        return q_candidate, dq_candidate

    def _find_node_output_fn(self, node: NodeProto, index: int) -> Union[NodeProto, None]:
        """ Find node's output fixneuron """
        if index < 0 or index >= len(node.output):
            raise ValueError(f"index {index} exceeded the number of outputs for {node.name}")

        tensor_name = node.output[index]
        if tensor_name not in self.in_name_to_nodes:
            logger.debug(f"output {tensor_name} of {node.name} may be model's output")
            return None

        fn_candidate = self.in_name_to_nodes[tensor_name][0]
        if fn_candidate.op_type not in FixNeuronOps:
            logger.debug(f"output {tensor_name} of {node.name} was not quantized by FixNeuron")
            return None

        return fn_candidate

    def get_outputs_qinfo(
        self, node: NodeProto
    ) -> List[Union[Tuple[NDArray[np.float32], NDArray[Any], NDArray[Any], NDArray[Any], int, bool, Any], Dict[
            str, Any], None]]:
        """ Get the quantization info of each output for the node """
        qinfos: List[Union[Tuple[NDArray[np.float32], NDArray[Any], NDArray[Any], NDArray[Any], int, bool, Any],
                           Dict[str, Any], None]] = [None] * len(node.output)

        for index, name in enumerate(node.output):
            q, dq = self._find_node_output_qdq(node, index)
            if q is not None and dq is not None:
                qinfos[index] = self._parse_qdq_quant_info(dq, q)
            else:
                fn = self._find_node_output_fn(node, index)
                if fn is not None:
                    qinfos[index] = self._parse_fn_quant_info(fn)

        return qinfos

    def get_inputs_param(self, node: NodeProto) -> List[NDArray[Any]]:
        """ Get the weight and bias of the node in numpy array format """
        params = []

        for index, name in enumerate(node.input):
            if index == 0:  # It's always activation not weights
                continue

            param_init = self._find_node_input_init(node, index)

            if param_init is None:
                dq, q = self._find_node_input_qdq(node, index)
                # Has QDQ pair
                if q is not None:
                    param_init = self.name_to_init[q.input[0]]
                # Q was folded
                elif dq is not None:
                    # Note this is quantized parameter not original
                    param_init = self.name_to_init[dq.input[0]]
                # May be quantized by FixNeuron
                else:
                    fn = self._find_node_input_fn(node, index)
                    if fn is not None:
                        param_init = self.name_to_init[fn.input[0]]
                    else:
                        continue  # Not quantized

            params.append(onnx.numpy_helper.to_array(param_init))

        return params

    def get_output_node(self, node: NodeProto) -> Union[NodeProto, None]:
        """ Get the target whose input is from the node's output """
        if node.output[0] not in self.in_name_to_nodes:
            return None

        output_nodes = self.in_name_to_nodes[node.output[0]]
        return output_nodes[0]
