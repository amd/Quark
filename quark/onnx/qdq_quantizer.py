#
# Modifications copyright(c) 2023 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy
import numpy as np

from quark.shares.utils.log import ScreenLogger, log_errors

import onnx
import onnx.numpy_helper
from onnx import TensorProto, ModelProto, NodeProto
from onnx import onnx_pb as onnx_proto
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer as OrtQDQQuantizer
from onnxruntime.quantization.qdq_quantizer import (QDQQuantTensorType, QDQTensorQuantInfo)
from onnxruntime.quantization.quant_utils import (
    QuantType,
    QuantizationMode,
    DEQUANT_OP_NAME,
    QUANT_OP_NAME,
    QuantizedValue,
    QuantizedValueType,
    add_dequant_output_suffix,
    add_dequant_suffix,
    add_quant_input_suffix,
    add_quant_output_suffix,
    add_quant_suffix,
    find_by_name,
)

from .quant_utils import (
    VitisQuantFormat,
    __producer__,
    __version__,
    VitisQuantType,
    FIX_OP_NAME,
    FIX_OP_DEFAULT_ATTRS,
    BFPFIX_OP_NAME,
    BFPFIX_OP_DEFAULT_ATTRS,
    MXFIX_OP_NAME,
    MXFIX_OP_DEFAULT_ATTRS,
    VAI_DOMAIN,
    COP_DOMAIN,
    COP_QUANT_OP_NAME,
    COP_DEQUANT_OP_NAME,
    ONNX_WBIT_QTYPES_LIST,
    ONNX_FP_QTYPES_LIST,
    ONNX_BFP_QTYPES_LIST,
    get_annotate_tensors,
    get_qdq_to_remove,
    remove_nodes,
    modified_annotate_input,
    get_tensor_type_from_qType,
)
from .registry import (CreateQDQQuantizer, CreateNPUCnnQDQQuantizer, CreateNPUTransformerQDQQuantizer)
from .refine import adjust_quantize_info, align_quantize_info
from .simulate_dpu import simulate_transforms
from .onnx_quantizer import VitisONNXQuantizer

from typing import Any, List, Dict, Optional

logger = ScreenLogger(__name__)


class QDQQuantizer(OrtQDQQuantizer):  # type: ignore
    """
    A class to perform quantization on an ONNX model using Quantize-Dequantize (QDQ) nodes.

    Args:
        model (ModelProto): The ONNX model to be quantized.
        per_channel (bool): Whether to perform per-channel quantization.
        reduce_range (bool): Whether to reduce the quantization range.
        mode (QuantizationMode.QLinearOps): The quantization mode to be used.
        static (bool): Whether to use static quantization.
        weight_qType (Any): The quantization type for weights.
        activation_qType (Any): The quantization type for activations.
        tensors_range (Any): Dictionary specifying the min and max values for tensors.
        nodes_to_quantize (List[str]): List of node names to be quantized.
        nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
        op_types_to_quantize (List[str]): List of operation types to be quantized.
        extra_options (Any, optional): Additional options for quantization.

    Inherits from:
        OrtQDQQuantizer: Base class for ONNX QDQ quantization.
    """

    def __init__(
        self,
        model: ModelProto,
        per_channel: bool,
        reduce_range: bool,
        mode: QuantizationMode.QLinearOps,
        static: bool,
        weight_qType: Any,
        activation_qType: Any,
        tensors_range: Any,
        nodes_to_quantize: List[str],
        nodes_to_exclude: List[str],
        op_types_to_quantize: List[str],
        extra_options: Any = None,
    ):
        super().__init__(
            model,
            per_channel=per_channel,
            reduce_range=reduce_range,
            mode=mode,
            static=static,
            weight_qType=weight_qType,
            activation_qType=activation_qType,
            tensors_range=tensors_range,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            op_types_to_quantize=op_types_to_quantize,
            extra_options=extra_options,
        )
        self.int32_bias = True if extra_options is None or "Int32Bias" not in extra_options else extra_options[
            "Int32Bias"]

        # weights-only quantization switch
        self.weights_only = False if "WeightsOnly" not in extra_options else extra_options["WeightsOnly"]

        # include-gptq quantization switch
        self.use_gptq = False if extra_options is None or "UseGPTQ" not in extra_options else extra_options["UseGPTQ"]
        # If GPTQ is turned on, the quantizer will only quantize weights and leave the activations in floating-point for GPTQ.
        if self.use_gptq is True:
            self.weights_only = True

    def _is_tensor_quantizable(self, tensor_name: str) -> bool:
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            if weight.data_type in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16):
                return True
        elif self.weights_only is True:
            return False
        elif tensor_name in self.value_infos:
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type in (
                    TensorProto.FLOAT,
                    TensorProto.FLOAT16,
            ):
                return True
        else:
            logger.warning(
                f"failed to infer the type of tensor: {tensor_name}. Skip to quantize it. Please check if it is expected."
            )

        return False

    def quantize_bias_tensor(self, bias_name: str, input_name: str, weight_name: str, beta: float = 1.0) -> None:
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                if self.quantize_bias:
                    if self.int32_bias:
                        self.bias_to_quantize.append((bias_name, input_name, weight_name, beta))
                    else:
                        if self.per_channel:
                            self.quantize_weight_tensor_per_channel(bias_name, 0)
                        else:
                            self.quantize_weight_tensor(bias_name)
        else:
            logger.warning(f"Expected {bias_name} to be a weight")

    def quantize_model(self) -> Any:
        annotate_tensors = get_annotate_tensors(self.model.model)

        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)

        self._quantize_normal_tensors()
        self._quantize_sharing_param_tensors()
        if self.quantize_bias and self.int32_bias and not self.weights_only:
            self._quantize_bias_tensors()
        self.remove_nodes()

        dq_nodes_to_remove, q_nodes_to_remove, input_node_mapping = get_qdq_to_remove(
            self.model.model, annotate_tensors)
        pruned_model = copy.deepcopy(self.model)
        modified_annotate_input(pruned_model.model, input_node_mapping)
        pruned_model.model = remove_nodes(pruned_model.model, dq_nodes_to_remove)
        pruned_model.model = remove_nodes(pruned_model.model, q_nodes_to_remove)
        try:
            pruned_model.topological_sort()
            logger.info("Remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu).")
            self.model.model = pruned_model.model
        except Exception as e:
            logger.warning(
                f"Unable to remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu). Exception: {e}"
            )

        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def _add_qdq_pair_for_initializer(self, weight_proto: TensorProto, tensor_type: Any, axis: Any = None) -> None:
        weight_name = weight_proto.name
        if axis is not None:
            if self.opset_version < 13:
                raise ValueError("Per-Channel support with QDQ format requires onnx opset version 13 or above.")
            q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel(
                weight_name,
                # Quantization type is forced to be TensorProto.INT8.
                # when the expected value would be (see below)
                # self.weight_qType if tensor_type is QDQQuantTensorType.WEIGHT else self.activation_qType.
                # QLinearConv expects to have a unique value for all channels.
                # This code does not enforce that but it is necessarily the case when the
                # quantization is symmetric (as for INT8).
                onnx_proto.TensorProto.INT8,
                axis,
                keep_float_weight=self.add_qdq_pair_to_weight,
            )
        else:
            q_weight_name, zp_name, scale_name = self.quantize_initializer(
                weight_proto,
                self.weight_qType,
                keep_float_weight=self.add_qdq_pair_to_weight,
            )

        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name, weight_dequant_output)
        if self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)

            self._create_qdq_nodes(
                weight_name,
                weight_quant_output,
                add_quant_suffix(weight_name),
                weight_quant_output,
                weight_dequant_output,
                add_dequant_suffix(weight_name),
                scale_name,
                zp_name,
                axis,
            )
        else:
            dequant_node = onnx.helper.make_node(
                DEQUANT_OP_NAME,
                [q_weight_name, scale_name, zp_name],
                [weight_dequant_output],
                add_dequant_suffix(weight_name),
                axis=axis,
            )
            self.model.add_node(dequant_node)


class QDQNPUTransformerQuantizer(QDQQuantizer):
    """
    A class to perform quantization on an ONNX model using Quantize-Dequantize (QDQ) nodes
    optimized for NPU (Neural Processing Unit) Transformers.

    Args:
        model (ModelProto): The ONNX model to be quantized.
        per_channel (bool): Whether to perform per-channel quantization.
        reduce_range (bool): Whether to reduce the quantization range.
        mode (QuantizationMode.QLinearOps): The quantization mode to be used.
        static (bool): Whether to use static quantization.
        weight_qType (Any): The quantization type for weights.
        activation_qType (Any): The quantization type for activations.
        tensors_range (Any): Dictionary specifying the min and max values for tensors.
        nodes_to_quantize (List[str]): List of node names to be quantized.
        nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
        op_types_to_quantize (List[str]): List of operation types to be quantized.
        extra_options (Optional[Dict[str, Any]], optional): Additional options for quantization.

    Inherits from:
        QDQQuantizer: Base class for ONNX QDQ quantization.
    """

    def __init__(
        self,
        model: ModelProto,
        per_channel: bool,
        reduce_range: bool,
        mode: QuantizationMode.QLinearOps,
        static: bool,
        weight_qType: Any,
        activation_qType: Any,
        tensors_range: Any,
        nodes_to_quantize: List[str],
        nodes_to_exclude: List[str],
        op_types_to_quantize: List[str],
        extra_options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model,
            per_channel=per_channel,
            reduce_range=reduce_range,
            mode=mode,
            static=static,
            weight_qType=weight_qType,
            activation_qType=activation_qType,
            tensors_range=tensors_range,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            op_types_to_quantize=op_types_to_quantize,
            extra_options=extra_options,
        )
        self.int32_bias = True if extra_options is None or "Int32Bias" not in extra_options else extra_options[
            "Int32Bias"]

    def quantize_bias_tensor(self, bias_name: str, input_name: str, weight_name: str, beta: float = 1.0) -> None:
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                if self.quantize_bias:
                    if self.int32_bias:
                        self.bias_to_quantize.append((bias_name, input_name, weight_name, beta))
                    else:
                        if self.per_channel:
                            self.quantize_weight_tensor_per_channel(bias_name, 0)
                        else:
                            self.quantize_weight_tensor(bias_name)
        else:
            logger.warning(f"Expected {bias_name} to be a weight")

    def quantize_model(self) -> Any:

        annotate_tensors = get_annotate_tensors(self.model.model)

        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateNPUTransformerQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)

        self.remove_nodes()
        self._quantize_normal_tensors()
        self._quantize_sharing_param_tensors()
        if self.quantize_bias and self.int32_bias and not self.weights_only:
            self._quantize_bias_tensors()

        dq_nodes_to_remove, q_nodes_to_remove, input_node_mapping = get_qdq_to_remove(
            self.model.model, annotate_tensors)
        pruned_model = copy.deepcopy(self.model)
        modified_annotate_input(pruned_model.model, input_node_mapping)
        pruned_model.model = remove_nodes(pruned_model.model, dq_nodes_to_remove)
        pruned_model.model = remove_nodes(pruned_model.model, q_nodes_to_remove)
        try:
            pruned_model.topological_sort()
            logger.info("Remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu).")
            self.model.model = pruned_model.model
        except Exception as e:
            logger.warning(
                f"Unable to remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu). Exception: {e}"
            )

        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model


class VitisQDQQuantizer(VitisONNXQuantizer):
    """
    A class to perform Vitis-specific Quantize-Dequantize (QDQ) quantization on an ONNX model.

    Args:
        model (ModelProto): The ONNX model to be quantized.
        per_channel (bool): Whether to perform per-channel quantization.
        reduce_range (bool): Whether to reduce the quantization range.
        mode (QuantizationMode.QLinearOps): The quantization mode to be used.
        static (bool): Whether to use static quantization.
        weight_qType (Any): The quantization type for weights.
        activation_qType (Any): The quantization type for activations.
        tensors_range (Any): Dictionary specifying the min and max values for tensors.
        nodes_to_quantize (List[str]): List of node names to be quantized.
        nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
        op_types_to_quantize (List[str]): List of operation types to be quantized.
        calibrate_method (Any): The method used for calibration.
        quantized_tensor_type (Dict[Any, Any], optional): Dictionary specifying quantized tensor types.
        extra_options (Any, optional): Additional options for quantization.

    Inherits from:
        VitisONNXQuantizer: Base class for Vitis-specific ONNX quantization.

    Attributes:
        tensors_to_quantize (Dict[Any, Any]): Dictionary of tensors to be quantized.
        bias_to_quantize (List[Any]): List of bias tensors to be quantized.
        nodes_to_remove (List[Any]): List of nodes to be removed during quantization.
        op_types_to_exclude_output_quantization (List[str]): List of op types to exclude from output quantization.
        quantize_bias (bool): Whether to quantize bias tensors.
        add_qdq_pair_to_weight (bool): Whether to add QDQ pairs to weights.
        dedicated_qdq_pair (bool): Whether to create dedicated QDQ pairs for each node.
        tensor_to_its_receiving_nodes (Dict[Any, Any]): Dictionary mapping tensors to their receiving nodes.
        qdq_op_type_per_channel_support_to_axis (Dict[str, int]): Dictionary mapping op types to channel axis for per-channel quantization.
        int32_bias (bool): Whether to quantize bias using int32.
        weights_only (bool): Whether to perform weights-only quantization.

    """

    def __init__(
        self,
        model: ModelProto,
        per_channel: bool,
        reduce_range: bool,
        mode: QuantizationMode.QLinearOps,
        static: bool,
        weight_qType: Any,
        activation_qType: Any,
        tensors_range: Any,
        nodes_to_quantize: List[str],
        nodes_to_exclude: List[str],
        op_types_to_quantize: List[str],
        calibrate_method: Any,
        quantized_tensor_type: Dict[Any, Any] = {},
        extra_options: Any = None,
    ):
        """
        Initializes the VitisQDQQuantizer with the provided configuration.

        Args:
            model (ModelProto): The ONNX model to be quantized.
            per_channel (bool): Whether to perform per-channel quantization.
            reduce_range (bool): Whether to reduce the quantization range.
            mode (QuantizationMode.QLinearOps): The quantization mode to be used.
            static (bool): Whether to use static quantization.
            weight_qType (Any): The quantization type for weights.
            activation_qType (Any): The quantization type for activations.
            tensors_range (Any): Dictionary specifying the min and max values for tensors.
            nodes_to_quantize (List[str]): List of node names to be quantized.
            nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
            op_types_to_quantize (List[str]): List of operation types to be quantized.
            calibrate_method (Any): The method used for calibration.
            quantized_tensor_type (Dict[Any, Any], optional): Dictionary specifying quantized tensor types.
            extra_options (Any, optional): Additional options for quantization.
        """
        self.calibrate_method = calibrate_method
        VitisONNXQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            quantized_tensor_type,
            extra_options,
        )
        self.tensors_to_quantize: Dict[Any, Any] = {}
        self.bias_to_quantize: List[Any] = []

        self.nodes_to_remove: List[Any] = []

        # Specific op types to exclude qdq quantization for their outputs.
        # In TRT, it's not recommended to quantize outputs for weighted ops such as Conv, MatMul, Gemm
        # because those ops may be followed by nodes that require high resolution inputs.
        # Adding QDQ for those ops' output may end up with worse accuracy.
        # So, we don't recommend to add QDQ to node's output under such condition.
        self.op_types_to_exclude_output_quantization = ([] if extra_options is None
                                                        or "OpTypesToExcludeOutputQuantization" not in extra_options
                                                        else extra_options["OpTypesToExcludeOutputQuantization"])

        # Some scenarios do not need the bias quantized. For example, in the case of Quantization Aware Training,
        # quantizing the bias is not needed. This is because in QAT, all model parameters are expected to be in
        # floating point format. To that end, we can use the FakeQuant operator for weights and activations that
        # can always have QDQ pairs (by using AddQDQPairToWeight). But for biases in a quantized model, we can't use
        # FakeQuant because it only ever appears before a DQ (since it is quantized as int32).
        self.quantize_bias = True if extra_options is None or "QuantizeBias" not in extra_options else extra_options[
            "QuantizeBias"]

        # We do quantization on Dequantizelinear's input to remove Quantizelinear for weight as an optimization.
        # In some cases, for example QDQ BERT model for TensorRT, QDQ should always appear as a pair.
        # Therefore, we need to disable this optimization and add qdq pair to weight.
        self.add_qdq_pair_to_weight = (False if extra_options is None or "AddQDQPairToWeight" not in extra_options else
                                       extra_options["AddQDQPairToWeight"])

        # The default behavior is that multiple nodes can share a QDQ pair as their inputs.
        # In TRT, QDQ pair can't be shared between nodes, so it will create dedicated QDQ pairs for each node.
        self.dedicated_qdq_pair = (False if extra_options is None or "DedicatedQDQPair" not in extra_options else
                                   extra_options["DedicatedQDQPair"])
        if self.dedicated_qdq_pair:
            self.tensor_to_its_receiving_nodes: Dict[Any, Any] = {}

        # Let user set channel axis for specific op type and it's effective only when per channel quantization is supported and per_channel is True.
        self.qdq_op_type_per_channel_support_to_axis = ({} if extra_options is None
                                                        or "QDQOpTypePerChannelSupportToAxis" not in extra_options else
                                                        extra_options["QDQOpTypePerChannelSupportToAxis"])

        # We quantize Bias using Int32 by default except floating point type quantization
        if self.weight_qType in ONNX_FP_QTYPES_LIST + ONNX_BFP_QTYPES_LIST:
            self.int32_bias = False
        else:
            self.int32_bias = True
        if extra_options is not None and "Int32Bias" in extra_options:
            self.int32_bias = extra_options["Int32Bias"]
        if self.int32_bias and (self.weight_qType in ONNX_BFP_QTYPES_LIST
                                or self.activation_qType in ONNX_BFP_QTYPES_LIST):
            self.int32_bias = False  # Cannot meet the requirement of bias_scale = input_scale * weight_scale
            logger.warning("Disabled Int32 Bias, because the quant type of activaion is BFP or MX")

        # weights-only quantization switch
        self.weights_only = False if "WeightsOnly" not in extra_options else extra_options["WeightsOnly"]
        # include-gptq quantization switch
        self.use_gptq = False if extra_options is None or "UseGPTQ" not in extra_options else extra_options["UseGPTQ"]
        # If GPTQ is turned on, the quantizer will only quantize weights and leave the activations in floating-point for GPTQ.
        if self.use_gptq is True:
            self.weights_only = True

    def _get_tensor_type(self, tensor_name: str) -> Any:
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            return weight.data_type
        elif tensor_name in self.value_infos:
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type"):
                return vi.type.tensor_type.elem_type
        return None

    def _is_tensor_quantizable(self, tensor_name: str) -> bool:
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                return True
        elif self.weights_only is True:
            return False
        elif tensor_name in self.value_infos.keys():
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == TensorProto.FLOAT:
                return True
        else:
            logger.warning(
                "failed to infer the type of tensor: {}. Skip to quantize it. Please check if it is expected.".format(
                    tensor_name))

        return False

    def __quantize_tensor(self,
                          tensor_name: str,
                          quant_sharing_param: Any = None,
                          tensor_type: Any = QDQQuantTensorType.ACTIVATION) -> None:
        if self._is_tensor_quantizable(tensor_name):
            if quant_sharing_param:
                data_type = self._get_tensor_type(tensor_name)
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(tensor_type=tensor_type,
                                                                           quant_para_provider=quant_sharing_param,
                                                                           data_type=data_type)
            elif tensor_name not in self.tensors_to_quantize:
                data_type = self._get_tensor_type(tensor_name)
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(tensor_type=tensor_type, data_type=data_type)

    def quantize_activation_tensor(self, tensor_name: str, quant_sharing_param: Any = None) -> Any:
        return self.__quantize_tensor(tensor_name, quant_sharing_param, QDQQuantTensorType.ACTIVATION)

    def quantize_weight_tensor(self, tensor_name: str, quant_sharing_param: Any = None) -> Any:
        return self.__quantize_tensor(tensor_name, quant_sharing_param, QDQQuantTensorType.WEIGHT)

    def quantize_weight_tensor_per_channel(self, tensor_name: str, axis: Any) -> None:

        weight = find_by_name(tensor_name, self.model.initializer())
        if weight:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(tensor_type=QDQQuantTensorType.WEIGHT,
                                                                           axis=axis,
                                                                           data_type=weight.data_type)
        else:
            logger.warning(
                "only support per-channel quantization on weight. Tensor: {} is not quantized.".format(tensor_name))

    def quantize_bias_tensor(self, bias_name: str, input_name: str, weight_name: str, beta: float = 1.0) -> None:
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                if self.quantize_bias:
                    if self.int32_bias:
                        self.bias_to_quantize.append((bias_name, input_name, weight_name, beta))
                    else:
                        if self.per_channel:
                            self.quantize_weight_tensor_per_channel(bias_name, 0)
                        else:
                            self.quantize_weight_tensor(bias_name)
        else:
            logger.warning(f"Expected {bias_name} to be a weight")

    def remove_node(self, node: NodeProto) -> None:
        self.nodes_to_remove.append(node)

    def remove_nodes(self) -> None:
        self.model.remove_nodes(self.nodes_to_remove)

    def quantize_model(self) -> Any:
        annotate_tensors = get_annotate_tensors(self.model.model)

        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)

        self._quantize_normal_tensors()
        self._quantize_sharing_param_tensors()
        if self.quantize_bias and self.int32_bias and not self.weights_only:
            self._quantize_bias_tensors()

        self.remove_nodes()
        dq_nodes_to_remove, q_nodes_to_remove, input_node_mapping = get_qdq_to_remove(
            self.model.model, annotate_tensors)
        pruned_model = copy.deepcopy(self.model)
        modified_annotate_input(pruned_model.model, input_node_mapping)
        pruned_model.model = remove_nodes(pruned_model.model, dq_nodes_to_remove)
        pruned_model.model = remove_nodes(pruned_model.model, q_nodes_to_remove)
        try:
            pruned_model.topological_sort()
            logger.info("Remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu).")
            self.model.model = pruned_model.model
        except Exception as e:
            logger.warning(
                f"Unable to remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu). Exception: {e}"
            )

        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def try_replacing_upstream_output(self, upstream_output_name: str, output_name: str) -> bool:
        if (output_name in self.quantization_params.keys()
                and len(self.model.input_name_to_nodes()[upstream_output_name]) == 1
                and not self.model.is_graph_output(upstream_output_name)
                and not self.model.is_graph_input(upstream_output_name)):
            self.model.replace_output_of_all_nodes(upstream_output_name, output_name)
            if upstream_output_name in self.tensors_to_quantize:
                del self.tensors_to_quantize[upstream_output_name]
            return True
        return False

    def _create_qdq_nodes(self,
                          q_input: Any,
                          q_output: Any,
                          quant_node_name: str,
                          dq_input: Any,
                          dq_output: Any,
                          dequant_node_name: str,
                          scale_name: str,
                          zp_name: str,
                          axis: Any = None) -> None:
        qlinear_node = onnx.helper.make_node(
            QUANT_OP_NAME,
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
            axis=axis,
        )
        dequant_node = onnx.helper.make_node(
            DEQUANT_OP_NAME,
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
        )
        self.model.add_nodes([qlinear_node, dequant_node])

    def _add_qdq_pair_for_initializer(self, weight_proto: TensorProto, tensor_type: Any, axis: Any = None) -> None:
        weight_name = weight_proto.name
        if axis is not None:
            if self.opset_version < 13:
                raise ValueError("Per-Channel support with QDQ format requires onnx opset version 13 or above.")
            q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel(
                weight_name,
                onnx_proto.TensorProto.INT8,
                axis,
                self.calibrate_method,
                keep_float_weight=self.add_qdq_pair_to_weight)
        else:
            q_weight_name, zp_name, scale_name = self.quantize_initializer(
                weight_proto,
                self.weight_qType,
                self.calibrate_method,
                keep_float_weight=self.add_qdq_pair_to_weight,
            )

        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name, weight_dequant_output)
        if self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)

            self._create_qdq_nodes(
                weight_name,
                weight_quant_output,
                add_quant_suffix(weight_name),
                weight_quant_output,
                weight_dequant_output,
                add_dequant_suffix(weight_name),
                scale_name,
                zp_name,
                axis,
            )
        else:
            dequant_node = onnx.helper.make_node(
                DEQUANT_OP_NAME,
                [q_weight_name, scale_name, zp_name],
                [weight_dequant_output],
                add_dequant_suffix(weight_name),
                axis=axis,
            )
            self.model.add_node(dequant_node)

    def _add_qdq_pair_for_activation(self, tensor_name: str, scale_name: str, zp_name: str) -> None:
        if (self.dedicated_qdq_pair and tensor_name in self.tensor_to_its_receiving_nodes
                and len(self.tensor_to_its_receiving_nodes[tensor_name]) > 1):
            num_dedicated_qdq_pair = len(self.tensor_to_its_receiving_nodes[tensor_name])
            for i in range(num_dedicated_qdq_pair):
                postfix = f"_{i + 1}"
                tensor_name_quant_output_postfix = add_quant_output_suffix(tensor_name) + postfix
                tensor_name_dequant_output_postfix = add_dequant_output_suffix(tensor_name) + postfix
                quant_node_name_postfix = add_quant_suffix(tensor_name) + postfix
                dequant_node_name_postfix = add_dequant_suffix(tensor_name) + postfix
                self._create_qdq_nodes(
                    tensor_name,
                    tensor_name_quant_output_postfix,
                    quant_node_name_postfix,
                    tensor_name_quant_output_postfix,
                    tensor_name_dequant_output_postfix,
                    dequant_node_name_postfix,
                    scale_name,
                    zp_name,
                )

                node = self.tensor_to_its_receiving_nodes[tensor_name][i]
                self.model.replace_node_input(node, tensor_name, tensor_name_dequant_output_postfix)
                if i == 0:
                    quantized_value = QuantizedValue(
                        tensor_name,
                        tensor_name_dequant_output_postfix,
                        scale_name,
                        zp_name,
                        QuantizedValueType.Input,
                    )
                    self.quantized_value_map[tensor_name] = quantized_value
        else:
            q_input = tensor_name
            dq_output = add_dequant_output_suffix(tensor_name)
            if self.model.is_graph_output(tensor_name):
                q_input = add_quant_input_suffix(tensor_name)
                dq_output = tensor_name
                self.model.replace_output_of_all_nodes(tensor_name, q_input)
            else:
                self.model.replace_input_of_all_nodes(tensor_name, dq_output)

            self._create_qdq_nodes(
                q_input,
                add_quant_output_suffix(tensor_name),
                add_quant_suffix(tensor_name),
                add_quant_output_suffix(tensor_name),
                dq_output,
                add_dequant_suffix(tensor_name),
                scale_name,
                zp_name,
            )

            quantized_value = QuantizedValue(
                tensor_name,
                dq_output,
                scale_name,
                zp_name,
                QuantizedValueType.Input,
            )
            self.quantized_value_map[tensor_name] = quantized_value

    def _quantize_normal_tensors(self) -> None:
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
            if tensor_name in self.quantized_value_map.keys():
                continue

            if not tensor_info.is_shared:
                # Quantize the input
                initializer = find_by_name(tensor_name, self.model.initializer())
                if initializer:
                    self._add_qdq_pair_for_initializer(initializer, tensor_info.tensor_type, tensor_info.axis)
                else:
                    used_scale, used_zp = self.find_quant_scale_zp(tensor_name)
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(
                        tensor_name, used_scale, used_zp)

                    if not data_found:
                        raise ValueError(
                            f"Quantization parameters are not specified for param {tensor_name}. "
                            "In static mode quantization params for inputs and outputs of nodes to be quantized are required."
                        )

                    self._add_qdq_pair_for_activation(tensor_name, scale_name, zp_name)

                del self.tensors_to_quantize[tensor_name]

    def _quantize_sharing_param_tensors(self) -> None:
        while self.tensors_to_quantize:
            for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
                tensor_provider_name = tensor_info.quant_para_provider
                if tensor_provider_name in self.quantized_value_map:
                    del self.tensors_to_quantize[tensor_name]

                    quantized_value = self.quantized_value_map[tensor_provider_name]
                    # Quantize the input
                    initializer = find_by_name(tensor_name, self.model.initializer())
                    if initializer is not None:
                        raise ValueError("Quantization parameter shared mode is not supported for weight yet")
                    self._add_qdq_pair_for_activation(tensor_name, quantized_value.scale_name, quantized_value.zp_name)

    def _quantize_bias_tensors(self) -> None:
        for bias_name, input_name, weight_name, beta in self.bias_to_quantize:
            if bias_name in self.quantized_value_map.keys():
                continue
            # Quantize the input
            self.quantize_bias_static(bias_name, input_name, weight_name, beta)
            self.model.remove_initializer(find_by_name(bias_name, self.model.initializer()))
            quant_value = self.quantized_value_map[bias_name]
            inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
            node_name = add_dequant_suffix(bias_name)
            if quant_value.axis is not None:
                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs,
                    [bias_name],
                    node_name,
                    axis=quant_value.axis,
                )
            else:
                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs,
                    [bias_name],
                    node_name,
                )
            self.model.add_node(dequant_node)

    def is_tensor_quantized(self, tensor_name: str) -> bool:
        return tensor_name in self.tensors_to_quantize or tensor_name in self.bias_to_quantize


class VitisQDQNPUCNNQuantizer(VitisQDQQuantizer):
    """
    A class to perform Vitis-specific Quantize-Dequantize (QDQ) quantization for NPU (Neural Processing Unit) on CNN models.

    Args:
        model (ModelProto): The ONNX model to be quantized.
        per_channel (bool): Whether to perform per-channel quantization (must be False for NPU).
        reduce_range (bool): Whether to reduce the quantization range (must be False for NPU).
        mode (QuantizationMode.QLinearOps): The quantization mode to be used.
        static (bool): Whether to use static quantization.
        weight_qType (Any): The quantization type for weights (must be QuantType.QInt8 for NPU).
        activation_qType (Any): The quantization type for activations.
        tensors_range (Any): Dictionary specifying the min and max values for tensors.
        nodes_to_quantize (List[str]): List of node names to be quantized.
        nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
        op_types_to_quantize (List[str]): List of operation types to be quantized.
        calibrate_method (Any): The method used for calibration.
        quantized_tensor_type (Dict[Any, Any], optional): Dictionary specifying quantized tensor types.
        extra_options (Optional[Dict[str, Any]], optional): Additional options for quantization.

    Inherits from:
        VitisQDQQuantizer: Base class for Vitis-specific QDQ quantization.

    Attributes:
        tensors_to_quantize (Dict[Any, Any]): Dictionary of tensors to be quantized.
        is_weight_symmetric (bool): Whether to enforce symmetric quantization for weights.
        is_activation_symmetric (bool): Whether to enforce symmetric quantization for activations.

    """

    @log_errors
    def __init__(
        self,
        model: ModelProto,
        per_channel: bool,
        reduce_range: bool,
        mode: QuantizationMode.QLinearOps,
        static: bool,
        weight_qType: Any,
        activation_qType: Any,
        tensors_range: Any,
        nodes_to_quantize: List[str],
        nodes_to_exclude: List[str],
        op_types_to_quantize: List[str],
        calibrate_method: Any,
        quantized_tensor_type: Dict[Any, Any] = {},
        extra_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the VitisQDQNPUCNNQuantizer with the provided configuration.

        Args:
            model (ModelProto): The ONNX model to be quantized.
            per_channel (bool): Whether to perform per-channel quantization (must be False for NPU).
            reduce_range (bool): Whether to reduce the quantization range (must be False for NPU).
            mode (QuantizationMode.QLinearOps): The quantization mode to be used.
            static (bool): Whether to use static quantization.
            weight_qType (Any): The quantization type for weights (must be QuantType.QInt8 for NPU).
            activation_qType (Any): The quantization type for activations.
            tensors_range (Any): Dictionary specifying the min and max values for tensors.
            nodes_to_quantize (List[str]): List of node names to be quantized.
            nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
            op_types_to_quantize (List[str]): List of operation types to be quantized.
            calibrate_method (Any): The method used for calibration.
            quantized_tensor_type (Dict[Any, Any], optional): Dictionary specifying quantized tensor types.
            extra_options (Optional[Dict[str, Any]], optional): Additional options for quantization.
        """
        self.calibrate_method = calibrate_method
        VitisQDQQuantizer.__init__(
            self,
            model,
            False,
            False,
            mode,
            static,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            quantized_tensor_type,
            extra_options,
        )
        self.tensors_to_quantize = {}

        if per_channel:
            raise ValueError(
                "Only per-tensor quantization is supported when enable_dpu=True, `per_channel` must be set to False.")

        if reduce_range:
            raise ValueError("reduce_range is not supported when enable_dpu=True, `reduce_range` must be set to False.")

        if weight_qType != QuantType.QInt8:
            raise ValueError("Only QuantType.QInt8 weight_type is supported when enable_dpu=True.")

        # If using nable_dpu, QDQ should always set WeightSymmetric as True.
        if "WeightSymmetric" in self.extra_options and not self.extra_options["WeightSymmetric"]:
            raise ValueError("When enable_dpu=True, WeightSymmetric must be set to true.")
        self.is_weight_symmetric = True

        # If using enable_dpu, QDQ should always always set ActivationSymmetric as True.
        if "ActivationSymmetric" in self.extra_options and not self.extra_options["ActivationSymmetric"]:
            raise ValueError("When enable_dpu=True, ActivationSymmetric must be set to true.")
        self.is_activation_symmetric = True

    def quantize_model(self) -> Any:
        annotate_tensors = get_annotate_tensors(self.model.model)

        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateNPUCnnQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)

        self._quantize_normal_tensors()

        self._quantize_sharing_param_tensors()
        self.remove_nodes()
        dq_nodes_to_remove, q_nodes_to_remove, input_node_mapping = get_qdq_to_remove(
            self.model.model, annotate_tensors)
        pruned_model = copy.deepcopy(self.model)
        modified_annotate_input(pruned_model.model, input_node_mapping)
        pruned_model.model = remove_nodes(pruned_model.model, dq_nodes_to_remove)
        pruned_model.model = remove_nodes(pruned_model.model, q_nodes_to_remove)
        try:
            pruned_model.topological_sort()
            logger.info("Remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu).")
            self.model.model = pruned_model.model
        except Exception as e:
            logger.warning(
                f"Unable to remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu). Exception: {e}"
            )
        if "SimulateDPU" not in self.extra_options or self.extra_options["SimulateDPU"] is True:
            self._simulate_transforms()

        if "NPULimitationCheck" not in self.extra_options or self.extra_options["NPULimitationCheck"] is True:
            self._quantize_refine()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def _add_qdq_pair_for_initializer(self, weight_proto: TensorProto, tensor_type: Any, axis: Any = None) -> None:
        weight_name = weight_proto.name
        q_weight_name, zp_name, scale_name = self.quantize_initializer(
            weight_proto,
            self.weight_qType,
            self.calibrate_method,
            keep_float_weight=self.add_qdq_pair_to_weight,
        )
        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name, weight_dequant_output)
        if self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)

            self._create_qdq_nodes(
                weight_name,
                weight_quant_output,
                add_quant_suffix(weight_name),
                weight_quant_output,
                weight_dequant_output,
                add_dequant_suffix(weight_name),
                scale_name,
                zp_name,
                axis,
            )
        else:
            dequant_node = onnx.helper.make_node(
                DEQUANT_OP_NAME,
                [q_weight_name, scale_name, zp_name],
                [weight_dequant_output],
                add_dequant_suffix(weight_name),
                axis=axis,
            )

            self.model.add_node(dequant_node)

    def quantize_bias_tensor(self, bias_name: str, input_name: str, weight_name: str, beta: float = 1.0) -> None:
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                # Use int8 quantization for bias as well as weights.
                self.quantize_weight_tensor(bias_name)
        else:
            logger.warning("Expected {} to be a weight".format(bias_name))

    def _quantize_refine(self) -> None:
        max_loop_num = 5
        if "MaxLoopNum" in self.extra_options:
            max_loop_num = self.extra_options["MaxLoopNum"]

        adjust_shift_cut = True
        if "AdjustShiftCut" in self.extra_options:
            adjust_shift_cut = self.extra_options["AdjustShiftCut"]
        adjust_shift_bias = True
        if "AdjustShiftBias" in self.extra_options:
            adjust_shift_bias = self.extra_options["AdjustShiftBias"]
        adjust_shift_read = True
        if "AdjustShiftRead" in self.extra_options:
            adjust_shift_read = self.extra_options["AdjustShiftRead"]
        adjust_shift_write = True
        if "AdjustShiftWrite" in self.extra_options:
            adjust_shift_write = self.extra_options["AdjustShiftWrite"]
        adjust_hard_sigmoid = True
        if "AdjustHardSigmoid" in self.extra_options:
            adjust_hard_sigmoid = self.extra_options["AdjustHardSigmoid"]
        adjust_shift_swish = True
        if "AdjustShiftSwish" in self.extra_options:
            adjust_shift_swish = self.extra_options["AdjustShiftSwish"]
        align_concat = True
        if "AlignConcat" in self.extra_options:
            align_concat = self.extra_options["AlignConcat"]
        align_pool = True
        if "AlignPool" in self.extra_options:
            align_pool = self.extra_options["AlignPool"]
        align_pad = True
        if "AlignPad" in self.extra_options:
            align_pad = self.extra_options["AlignPad"]
        align_slice = True
        if "AlignSlice" in self.extra_options:
            align_slice = self.extra_options["AlignSlice"]

        self.model = adjust_quantize_info(
            self.model,
            max_loop_num=max_loop_num,
            adjust_shift_cut=adjust_shift_cut,
            adjust_shift_bias=adjust_shift_bias,
            adjust_shift_read=adjust_shift_read,
            adjust_shift_write=adjust_shift_write,
            adjust_hard_sigmoid=adjust_hard_sigmoid,
            adjust_shift_swish=adjust_shift_swish,
            align_concat=align_concat,
            align_pool=align_pool,
            align_pad=align_pad,
            align_slice=align_slice,
        )

    def _simulate_transforms(self) -> None:
        convert_leaky_relu_to_dpu_version = True
        if "ConvertLeakyReluToDPUVersion" in self.extra_options:
            convert_leaky_relu_to_dpu_version = self.extra_options["ConvertLeakyReluToDPUVersion"]
        convert_sigmoid_to_hard_sigmoid = True
        if "ConvertSigmoidToHardSigmoid" in self.extra_options:
            convert_sigmoid_to_hard_sigmoid = self.extra_options["ConvertSigmoidToHardSigmoid"]
        convert_hard_sigmoid_to_dpu_version = True
        if "ConvertHardSigmoidToDPUVersion" in self.extra_options:
            convert_hard_sigmoid_to_dpu_version = self.extra_options["ConvertHardSigmoidToDPUVersion"]
        convert_avg_pool_to_dpu_version = True
        if "ConvertAvgPoolToDPUVersion" in self.extra_options:
            convert_avg_pool_to_dpu_version = self.extra_options["ConvertAvgPoolToDPUVersion"]
        convert_reduce_mean_to_dpu_version = True
        if "ConvertReduceMeanToDPUVersion" in self.extra_options:
            convert_reduce_mean_to_dpu_version = self.extra_options["ConvertReduceMeanToDPUVersion"]
        convert_softmax_to_dpu_version = False
        if "ConvertSoftmaxToDPUVersion" in self.extra_options:
            convert_softmax_to_dpu_version = self.extra_options["ConvertSoftmaxToDPUVersion"]
        convert_instance_norm_to_dpu_version = False
        if "ConvertInstanceNormToDPUVersion" in self.extra_options:
            convert_instance_norm_to_dpu_version = self.extra_options["ConvertInstanceNormToDPUVersion"]

        self.model.model, self.nodes_to_exclude = simulate_transforms(
            self.model.model,
            self.should_quantize_node,
            self.nodes_to_quantize,
            self.nodes_to_exclude,
            convert_leaky_relu_to_dpu_version=convert_leaky_relu_to_dpu_version,
            convert_sigmoid_to_hard_sigmoid=convert_sigmoid_to_hard_sigmoid,
            convert_hard_sigmoid_to_dpu_version=convert_hard_sigmoid_to_dpu_version,
            convert_avg_pool_to_dpu_version=convert_avg_pool_to_dpu_version,
            convert_reduce_mean_to_dpu_version=convert_reduce_mean_to_dpu_version,
            convert_softmax_to_dpu_version=convert_softmax_to_dpu_version,
            convert_instance_norm_to_dpu_version=convert_instance_norm_to_dpu_version,
        )


class VitisExtendedQuantizer(VitisQDQQuantizer):
    """
    A class to perform extended Vitis-specific Quantize-Dequantize (QDQ) quantization.

    Args:
        model (ModelProto): The ONNX model to be quantized.
        per_channel (bool): Whether to perform per-channel quantization.
        reduce_range (bool): Whether to reduce the quantization range.
        mode (QuantizationMode.QLinearOps): The quantization mode to be used.
        quant_format (Any): The format for quantization.
        static (bool): Whether to use static quantization.
        weight_qType (Any): The quantization type for weights.
        activation_qType (Any): The quantization type for activations.
        tensors_range (Any): Dictionary specifying the min and max values for tensors.
        nodes_to_quantize (List[str]): List of node names to be quantized.
        nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
        op_types_to_quantize (List[str]): List of operation types to be quantized.
        calibrate_method (Any): The method used for calibration.
        quantized_tensor_type (Dict[Any, Any]): Dictionary specifying quantized tensor types.
        extra_options (Optional[Dict[str, Any]], optional): Additional options for quantization.

    Inherits from:
        VitisQDQQuantizer: Base class for Vitis-specific QDQ quantization.

    Attributes:
        tensors_to_quantize (Dict[Any, Any]): Dictionary of tensors to be quantized.
        quant_format (Any): The format for quantization.
        add_qdq_pair_to_weight (bool): Whether to add QDQ pair to weight (and bias).
        fold_relu (bool): Whether to fold ReLU layers.

    """

    def __init__(
        self,
        model: ModelProto,
        per_channel: bool,
        reduce_range: bool,
        mode: QuantizationMode.QLinearOps,
        quant_format: Any,
        static: bool,
        weight_qType: Any,
        activation_qType: Any,
        tensors_range: Any,
        nodes_to_quantize: List[str],
        nodes_to_exclude: List[str],
        op_types_to_quantize: List[str],
        calibrate_method: Any,
        quantized_tensor_type: Dict[Any, Any],
        extra_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the VitisExtendedQuantizer with the provided configuration.

        Args:
            model (ModelProto): The ONNX model to be quantized.
            per_channel (bool): Whether to perform per-channel quantization.
            reduce_range (bool): Whether to reduce the quantization range.
            mode (QuantizationMode.QLinearOps): The quantization mode to be used.
            quant_format (Any): The format for quantization.
            static (bool): Whether to use static quantization.
            weight_qType (Any): The quantization type for weights.
            activation_qType (Any): The quantization type for activations.
            tensors_range (Any): Dictionary specifying the min and max values for tensors.
            nodes_to_quantize (List[str]): List of node names to be quantized.
            nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
            op_types_to_quantize (List[str]): List of operation types to be quantized.
            calibrate_method (Any): The method used for calibration.
            quantized_tensor_type (Dict[Any, Any]): Dictionary specifying quantized tensor types.
            extra_options (Optional[Dict[str, Any]], optional): Additional options for quantization.
        """
        self.calibrate_method = calibrate_method
        VitisQDQQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            quantized_tensor_type,
            extra_options,
        )
        self.tensors_to_quantize = {}

        self.quant_format = quant_format
        assert self.quant_format == VitisQuantFormat.QDQ

        # We add Q/DQ pair to weight (and bias) for float16 and bfloat16 by default,
        # which is aimed to avoid failure of data persistence check.
        # For Interger quantization type, we fold Q to support fast finetune.
        if self.weight_qType in ONNX_FP_QTYPES_LIST:
            self.add_qdq_pair_to_weight = True
        else:
            self.add_qdq_pair_to_weight = False
        if extra_options is not None and "AddQDQPairToWeight" in extra_options:
            self.add_qdq_pair_to_weight = extra_options["AddQDQPairToWeight"]
        self.quantized_tensor_type = quantized_tensor_type
        self.fold_relu = extra_options.get("FoldRelu", False) if extra_options is not None else False

        self.fn_name_w, self.fn_attrs_w = self._fn_name_and_attrs(weight_qType)
        self.fn_name_a, self.fn_attrs_a = self._fn_name_and_attrs(activation_qType)

    def quantize_model(self) -> Any:
        annotate_tensors = get_annotate_tensors(self.model.model)

        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateNPUCnnQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)

        self._quantize_normal_tensors()
        self._quantize_sharing_param_tensors()
        if self.quantize_bias and self.int32_bias and not self.weights_only:
            self._quantize_bias_tensors()

        self.remove_nodes()
        dq_nodes_to_remove, q_nodes_to_remove, input_node_mapping = get_qdq_to_remove(
            self.model.model, annotate_tensors)
        pruned_model = copy.deepcopy(self.model)
        modified_annotate_input(pruned_model.model, input_node_mapping)
        pruned_model.model = remove_nodes(pruned_model.model, dq_nodes_to_remove)
        pruned_model.model = remove_nodes(pruned_model.model, q_nodes_to_remove)
        try:
            pruned_model.topological_sort()
            logger.info("Remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu).")
            self.model.model = pruned_model.model
        except Exception as e:
            logger.warning(
                f"Unable to remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu). Exception: {e}"
            )
        if "SimulateDPU" not in self.extra_options or self.extra_options["SimulateDPU"] is True:
            self._simulate_transforms()

        if "NPULimitationCheck" not in self.extra_options or self.extra_options["NPULimitationCheck"] is True:
            self._quantize_refine()

        self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def try_replacing_upstream_output(self, upstream_output_name: str, output_name: str) -> bool:
        # TODO : Understand the principle here and fix the issue caused by QDQRemovableActivation.
        # As showed at onnxruntime/quantization/operators/activation.py, if activation uses asymmetric,
        # the QDQRemovableActivation remove nodes, which caused the graph broken.
        if (self.fold_relu and output_name in self.quantization_params
                and len(self.model.input_name_to_nodes()[upstream_output_name]) == 1
                and not self.model.is_graph_output(upstream_output_name)
                and not self.model.is_graph_input(upstream_output_name)):
            self.model.replace_output_of_all_nodes(upstream_output_name, output_name)
            if upstream_output_name in self.tensors_to_quantize:
                del self.tensors_to_quantize[upstream_output_name]
            return True
        return False

    '''
    def _create_fn_nodes(self,
                         q_input: Any,
                         dq_output: Any,
                         dequant_node_name: str,
                         scale_name: str,
                         zp_name: str,
                         axis: Any = None) -> None:
        """
        create fix_neuron node
        """
        fix_neuron_node = onnx.helper.make_node(
            FIX_OP_NAME,
            [q_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
            domain=VAI_DOMAIN,
        )
        bit_width = onnx.helper.make_attribute("bit_width", "8")
        fix_neuron_node.attribute.append(bit_width)

        scale = find_by_name(scale_name, self.model.initializer())
        scale = scale.float_data[0]
        pos = int(np.rint(-np.log2(scale)))
        pos_attr = onnx.helper.make_attribute("pos", str(pos))
        fix_neuron_node.attribute.append(pos_attr)

        self.model.add_nodes([fix_neuron_node])
    '''

    def _fn_name_and_attrs(self, qType: Any) -> tuple[str, Dict[str, Any]]:
        if qType == VitisQuantType.QBFP:
            fn_name = BFPFIX_OP_NAME
            fn_attrs = copy.deepcopy(BFPFIX_OP_DEFAULT_ATTRS)
            # Get attributes for BFPFixNeuron
            if self.extra_options is not None and "BFPAttributes" in self.extra_options:
                fn_attrs.update(self.extra_options["BFPAttributes"])
        elif qType == VitisQuantType.QMX:
            fn_name = MXFIX_OP_NAME
            fn_attrs = copy.deepcopy(MXFIX_OP_DEFAULT_ATTRS)
            # Get attributes for MXFixNeuron
            if self.extra_options is not None and "MXAttributes" in self.extra_options:
                fn_attrs.update(self.extra_options["MXAttributes"])
        else:
            fn_name = FIX_OP_NAME
            fn_attrs = {
                **FIX_OP_DEFAULT_ATTRS,
            }
        return fn_name, fn_attrs

    def _create_fn_nodes(self, q_input: Any, dq_output: Any, dequant_node_name: str, scale_name: str, zp_name: str,
                         fn_name: str, fn_attrs: Any) -> None:
        """
        create fix_neuron node
        """
        fix_neuron_node = onnx.helper.make_node(
            fn_name,
            [q_input],
            [dq_output],
            dequant_node_name,
            domain=COP_DOMAIN,
        )

        for k, v in fn_attrs.items():
            fix_neuron_node.attribute.append(onnx.helper.make_attribute(k, v))

        self.model.add_nodes([fix_neuron_node])

    def _create_pof2qdq_nodes(self,
                              q_input: Any,
                              q_output: Any,
                              quant_node_name: str,
                              dq_input: Any,
                              dq_output: Any,
                              dequant_node_name: str,
                              scale_name: str,
                              zp_name: str,
                              axis: Any = None) -> None:
        qlinear_node = onnx.helper.make_node(
            QUANT_OP_NAME,
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
            axis=axis,
            domain=VAI_DOMAIN,
        )
        dequant_node = onnx.helper.make_node(
            DEQUANT_OP_NAME,
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
            domain=VAI_DOMAIN,
        )
        bit_width = onnx.helper.make_attribute("bit_width", "8")

        scale = find_by_name(scale_name, self.model.initializer())
        scale = scale.float_data[0]
        pos = int(np.rint(-np.log2(scale)))
        pos_attr = onnx.helper.make_attribute("pos", str(pos))

        qlinear_node.attribute.append(bit_width)
        qlinear_node.attribute.append(pos_attr)
        dequant_node.attribute.append(bit_width)
        dequant_node.attribute.append(pos_attr)
        self.model.add_nodes([qlinear_node, dequant_node])

    def _create_customqdq_nodes(self,
                                q_input: Any,
                                q_output: Any,
                                quant_node_name: str,
                                dq_input: Any,
                                dq_output: Any,
                                dequant_node_name: str,
                                scale_name: str,
                                zp_name: str,
                                axis: Any = None) -> None:
        qlinear_node = onnx.helper.make_node(
            COP_QUANT_OP_NAME,
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
            axis=axis,
            domain=COP_DOMAIN,
        )
        dequant_node = onnx.helper.make_node(
            COP_DEQUANT_OP_NAME,
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
            domain=COP_DOMAIN,
        )
        self.model.add_nodes([qlinear_node, dequant_node])

    def _add_fn_pair_for_weight(self, weight_proto: TensorProto, axis: Any = None, zp_type: Any = None) -> None:
        weight_name = weight_proto.name

        if zp_type is not None:
            fn_name, fn_attrs = self._fn_name_and_attrs(zp_type)
            zp_type = get_tensor_type_from_qType(zp_type)
        else:
            fn_name, fn_attrs = self.fn_name_w, self.fn_attrs_w
            zp_type = self.weight_qType

        for key in fn_attrs.keys():
            if key == "axis" and len(weight_proto.dims) == 1:
                fn_attrs[key] = 0  # For scalar, the axis should always be 0
            if key == "convert_to_bfloat_before_bfp":
                fn_attrs[key] = 0  # Initializer is a constant, no conversion required

        if axis is not None:
            if self.opset_version < 13:
                raise ValueError("Per-Channel support with QDQ format requires onnx opset version 13 or above.")
            q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel(
                weight_name, zp_type, axis, self.calibrate_method, keep_float_weight=self.add_qdq_pair_to_weight)
        else:
            q_weight_name, zp_name, scale_name = self.quantize_initializer(
                weight_proto,
                zp_type,
                self.calibrate_method,
                keep_float_weight=self.add_qdq_pair_to_weight,
            )

        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name, weight_dequant_output)
        if zp_type in ONNX_BFP_QTYPES_LIST:
            self._create_fn_nodes(
                weight_name,
                weight_dequant_output,
                add_dequant_suffix(weight_name),
                scale_name,
                zp_name,
                fn_name,
                fn_attrs,
            )
        elif self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)
            if zp_type in ONNX_WBIT_QTYPES_LIST or self.use_qdq_vitis_custom_ops:
                self._create_customqdq_nodes(
                    weight_name,
                    weight_quant_output,
                    add_quant_suffix(weight_name),
                    weight_quant_output,
                    weight_dequant_output,
                    add_dequant_suffix(weight_name),
                    scale_name,
                    zp_name,
                    axis,
                )
            else:
                self._create_pof2qdq_nodes(
                    weight_name,
                    weight_quant_output,
                    add_quant_suffix(weight_name),
                    weight_quant_output,
                    weight_dequant_output,
                    add_dequant_suffix(weight_name),
                    scale_name,
                    zp_name,
                    axis,
                )
        else:
            if zp_type in ONNX_WBIT_QTYPES_LIST or self.use_qdq_vitis_custom_ops:
                dequant_node = onnx.helper.make_node(
                    COP_DEQUANT_OP_NAME,
                    [q_weight_name, scale_name, zp_name],
                    [weight_dequant_output],
                    add_dequant_suffix(weight_name),
                    axis=axis,
                    domain=COP_DOMAIN,
                )
                self.model.add_node(dequant_node)
            else:
                dequant_node = onnx.helper.make_node(
                    DEQUANT_OP_NAME,
                    [q_weight_name, scale_name, zp_name],
                    [weight_dequant_output],
                    add_dequant_suffix(weight_name),
                    axis=axis,
                    domain=VAI_DOMAIN,
                )
                bit_width = onnx.helper.make_attribute("bit_width", "8")
                dequant_node.attribute.append(bit_width)
                self.model.add_node(dequant_node)

    def _add_fn_pair_for_activation(self, tensor_name: str, scale_name: str, zp_name: str, zp_type: Any = None) -> Any:
        if zp_type is not None:
            fn_name, fn_attrs = self._fn_name_and_attrs(zp_type)
            zp_type = get_tensor_type_from_qType(zp_type)
        else:
            fn_name, fn_attrs = self.fn_name_a, self.fn_attrs_a
            zp_type = self.activation_qType
        if (self.dedicated_qdq_pair and tensor_name in self.tensor_to_its_receiving_nodes
                and len(self.tensor_to_its_receiving_nodes[tensor_name]) > 1):
            num_dedicated_qdq_pair = len(self.tensor_to_its_receiving_nodes[tensor_name])
            for i in range(num_dedicated_qdq_pair):
                postfix = f"_{i + 1}"
                tensor_name_quant_output_postfix = add_quant_output_suffix(tensor_name) + postfix
                tensor_name_dequant_output_postfix = add_dequant_output_suffix(tensor_name) + postfix
                quant_node_name_postfix = add_quant_suffix(tensor_name) + postfix
                dequant_node_name_postfix = add_dequant_suffix(tensor_name) + postfix

                if zp_type in ONNX_BFP_QTYPES_LIST:
                    self._create_fn_nodes(tensor_name, tensor_name_dequant_output_postfix, dequant_node_name_postfix,
                                          scale_name, zp_name, fn_name, fn_attrs)
                else:
                    if zp_type in ONNX_WBIT_QTYPES_LIST or self.use_qdq_vitis_custom_ops:
                        self._create_customqdq_nodes(
                            tensor_name,
                            tensor_name_quant_output_postfix,
                            quant_node_name_postfix,
                            tensor_name_quant_output_postfix,
                            tensor_name_dequant_output_postfix,
                            dequant_node_name_postfix,
                            scale_name,
                            zp_name,
                        )
                    else:
                        self._create_pof2qdq_nodes(
                            tensor_name,
                            tensor_name_quant_output_postfix,
                            quant_node_name_postfix,
                            tensor_name_quant_output_postfix,
                            tensor_name_dequant_output_postfix,
                            dequant_node_name_postfix,
                            scale_name,
                            zp_name,
                        )

                node = self.tensor_to_its_receiving_nodes[tensor_name][i]
                self.model.replace_node_input(node, tensor_name, tensor_name_dequant_output_postfix)
                if i == 0:
                    quantized_value = QuantizedValue(
                        tensor_name,
                        tensor_name_dequant_output_postfix,
                        scale_name,
                        zp_name,
                        QuantizedValueType.Input,
                    )
                    self.quantized_value_map[tensor_name] = quantized_value
        else:
            q_input = tensor_name
            dq_output = add_dequant_output_suffix(tensor_name)
            if self.model.is_graph_output(tensor_name):
                q_input = add_quant_input_suffix(tensor_name)
                dq_output = tensor_name
                self.model.replace_output_of_all_nodes(tensor_name, q_input)
            else:
                self.model.replace_input_of_all_nodes(tensor_name, dq_output)

            if zp_type in ONNX_BFP_QTYPES_LIST:
                self._create_fn_nodes(q_input, dq_output, add_dequant_suffix(tensor_name), scale_name, zp_name, fn_name,
                                      fn_attrs)
            else:
                if zp_type in ONNX_WBIT_QTYPES_LIST or self.use_qdq_vitis_custom_ops:
                    self._create_customqdq_nodes(
                        q_input,
                        add_quant_output_suffix(tensor_name),
                        add_quant_suffix(tensor_name),
                        add_quant_output_suffix(tensor_name),
                        dq_output,
                        add_dequant_suffix(tensor_name),
                        scale_name,
                        zp_name,
                    )
                else:
                    self._create_pof2qdq_nodes(
                        q_input,
                        add_quant_output_suffix(tensor_name),
                        add_quant_suffix(tensor_name),
                        add_quant_output_suffix(tensor_name),
                        dq_output,
                        add_dequant_suffix(tensor_name),
                        scale_name,
                        zp_name,
                    )

            quantized_value = QuantizedValue(
                tensor_name,
                dq_output,
                scale_name,
                zp_name,
                QuantizedValueType.Input,
            )
            self.quantized_value_map[tensor_name] = quantized_value

    def _quantize_normal_tensors(self) -> None:
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():

            if tensor_name in self.quantized_value_map.keys():
                continue

            if not tensor_info.is_shared:
                # This is for tensor-wise mixed precision
                zp_type = None
                if tensor_name in self.quantized_tensor_type:
                    zp_type = self.quantized_tensor_type[tensor_name]
                # Quantize the input
                initializer = find_by_name(tensor_name, self.model.initializer())
                if initializer:
                    if self.weight_qType == TensorProto.BFLOAT16:
                        weight = onnx.numpy_helper.to_array(initializer)
                        # clip weight to the range of BFLOAT16 [1.17549435e-38, 3.38953139e38]
                        if np.max(np.abs(weight)) > 3.38953139e38 or np.min(np.abs(weight)) < 1.17549435e-38:
                            original_weight = weight
                            weight = (np.sign(original_weight) *
                                      np.clip(np.abs(original_weight), 1.17549435e-38, 3.38953139e38)).astype(
                                          original_weight.dtype)
                            logger.info(
                                f"The original weight of {tensor_name}: {original_weight} has been clipped to new weight: {weight} because it is out of BFLOAT16 boundary."
                            )
                        initializer_new = onnx.numpy_helper.from_array(weight, name=initializer.name)
                        initializer.CopyFrom(initializer_new)
                    self._add_fn_pair_for_weight(initializer, tensor_info.axis, zp_type)
                else:
                    if (zp_type is None and self.activation_qType in ONNX_BFP_QTYPES_LIST) or (
                            zp_type is not None and zp_type in [VitisQuantType.QBFP, VitisQuantType.QMX]):
                        self._add_fn_pair_for_activation(tensor_name, '', '',
                                                         zp_type)  # BFP doesn't need scale and zero point
                        del self.tensors_to_quantize[tensor_name]
                        continue
                    used_scale, used_zp = self.find_quant_scale_zp(tensor_name)
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(
                        tensor_name, used_scale, used_zp, zp_type)

                    if not data_found:
                        raise ValueError(
                            f"Quantization parameters are not specified for param {tensor_name}. "
                            "In static mode quantization params for inputs and outputs of nodes to be quantized are required."
                        )

                    self._add_fn_pair_for_activation(tensor_name, scale_name, zp_name, zp_type)

                del self.tensors_to_quantize[tensor_name]

    def _quantize_sharing_param_tensors(self) -> None:
        while self.tensors_to_quantize:
            for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
                tensor_provider_name = tensor_info.quant_para_provider
                if tensor_provider_name in self.quantized_value_map:
                    del self.tensors_to_quantize[tensor_name]

                    quantized_value = self.quantized_value_map[tensor_provider_name]
                    # Quantize the input
                    initializer = find_by_name(tensor_name, self.model.initializer())
                    if initializer is not None:
                        raise ValueError("Quantization parameter shared mode is not supported for weight yet")
                    self._add_fn_pair_for_activation(tensor_name, quantized_value.scale_name, quantized_value.zp_name)

    def _quantize_bias_tensors(self) -> None:
        for bias_name, input_name, weight_name, beta in self.bias_to_quantize:
            if bias_name in self.quantized_value_map.keys():
                continue
            # Quantize the input
            self.quantize_bias_static(bias_name, input_name, weight_name, beta)
            self.model.remove_initializer(find_by_name(bias_name, self.model.initializer()))
            quant_value = self.quantized_value_map[bias_name]
            inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
            node_name = add_dequant_suffix(bias_name)

            # Keep the QDQ type of bias consistent with the weights
            if self.weight_qType in ONNX_WBIT_QTYPES_LIST or self.use_qdq_vitis_custom_ops:
                if quant_value.axis is not None:
                    dequant_node = onnx.helper.make_node(
                        COP_DEQUANT_OP_NAME,
                        inputs,
                        [bias_name],
                        node_name,
                        axis=quant_value.axis,
                        domain=COP_DOMAIN,
                    )
                else:
                    dequant_node = onnx.helper.make_node(
                        COP_DEQUANT_OP_NAME,
                        inputs,
                        [bias_name],
                        node_name,
                        domain=COP_DOMAIN,
                    )
                self.model.add_node(dequant_node)
                continue

            if quant_value.axis is not None:
                dequant_node = onnx.helper.make_node(
                    DEQUANT_OP_NAME,
                    inputs,
                    [bias_name],
                    node_name,
                    axis=quant_value.axis,
                    domain=VAI_DOMAIN,
                )
            else:
                dequant_node = onnx.helper.make_node(
                    DEQUANT_OP_NAME,
                    inputs,
                    [bias_name],
                    node_name,
                    domain=VAI_DOMAIN,
                )
            bit_width = onnx.helper.make_attribute("bit_width", "8")
            dequant_node.attribute.append(bit_width)
            self.model.add_node(dequant_node)

    def quantize_bias_tensor(self, bias_name: str, input_name: str, weight_name: str, beta: float = 1.0) -> None:
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                if self.quantize_bias:
                    if self.int32_bias:
                        self.bias_to_quantize.append((bias_name, input_name, weight_name, beta))
                    else:
                        if self.per_channel:
                            self.quantize_weight_tensor_per_channel(bias_name, 0)
                        else:
                            self.quantize_weight_tensor(bias_name)
        else:
            logger.warning(f"Expected {bias_name} to be a weight")

    def _adjust_model_scale(self) -> None:
        for node in self.model.model.graph.node:
            if node.op_type == "DequantizeLinear" or node.op_type == "QuantizeLinear":
                pos = None
                for attr in node.attribute:
                    if attr.name == "pos":
                        pos = int(attr.s)
                if pos is None:
                    continue
                new_scale = float(np.power(2., -pos))
                for i in self.model.model.graph.initializer:
                    if i.name == node.input[1]:
                        if i.float_data[0] != new_scale:
                            i.float_data[0] = new_scale

    def _quantize_refine(self) -> None:
        max_loop_num = 5
        if "MaxLoopNum" in self.extra_options:
            max_loop_num = self.extra_options["MaxLoopNum"]

        align_concat = False
        if "AlignConcat" in self.extra_options:
            align_concat = self.extra_options["AlignConcat"]
        align_pool = False
        if "AlignPool" in self.extra_options:
            align_pool = self.extra_options["AlignPool"]
        align_pad = False
        if "AlignPad" in self.extra_options:
            align_pad = self.extra_options["AlignPad"]
        align_slice = False
        if "AlignSlice" in self.extra_options:
            align_slice = self.extra_options["AlignSlice"]
        align_transpose = False
        if "AlignTranspose" in self.extra_options:
            align_transpose = self.extra_options["AlignTranspose"]
        align_reshape = False
        if "AlignReshape" in self.extra_options:
            align_reshape = self.extra_options["AlignReshape"]
        adjust_bias_scale = True
        if "AdjustBiasScale" in self.extra_options:
            adjust_bias_scale = self.extra_options["AdjustBiasScale"]

        self.model = align_quantize_info(
            self.model,
            max_loop_num=max_loop_num,
            align_concat=align_concat,
            align_pool=align_pool,
            align_pad=align_pad,
            align_slice=align_slice,
            align_transpose=align_transpose,
            align_reshape=align_reshape,
            adjust_bias_scale=adjust_bias_scale,
        )

        if self.weight_qType in [TensorProto.INT8, TensorProto.UINT8] and self.activation_qType in [
                TensorProto.INT8, TensorProto.UINT8
        ] and self.use_qdq_vitis_custom_ops:
            self._adjust_model_scale()

    def _simulate_transforms(self) -> None:
        convert_leaky_relu_to_dpu_version = False
        if "ConvertLeakyReluToDPUVersion" in self.extra_options:
            convert_leaky_relu_to_dpu_version = self.extra_options["ConvertLeakyReluToDPUVersion"]
        convert_sigmoid_to_hard_sigmoid = False
        if "ConvertSigmoidToHardSigmoid" in self.extra_options:
            convert_sigmoid_to_hard_sigmoid = self.extra_options["ConvertSigmoidToHardSigmoid"]
        convert_hard_sigmoid_to_dpu_version = False
        if "ConvertHardSigmoidToDPUVersion" in self.extra_options:
            convert_hard_sigmoid_to_dpu_version = self.extra_options["ConvertHardSigmoidToDPUVersion"]
        convert_avg_pool_to_dpu_version = False
        if "ConvertAvgPoolToDPUVersion" in self.extra_options:
            convert_avg_pool_to_dpu_version = self.extra_options["ConvertAvgPoolToDPUVersion"]
        convert_reduce_mean_to_dpu_version = False
        if "ConvertReduceMeanToDPUVersion" in self.extra_options:
            convert_reduce_mean_to_dpu_version = self.extra_options["ConvertReduceMeanToDPUVersion"]
        convert_softmax_to_dpu_version = False
        if "ConvertSoftmaxToDPUVersion" in self.extra_options:
            convert_softmax_to_dpu_version = self.extra_options["ConvertSoftmaxToDPUVersion"]
        convert_instance_norm_to_dpu_version = False
        if "ConvertInstanceNormToDPUVersion" in self.extra_options:
            convert_instance_norm_to_dpu_version = self.extra_options["ConvertInstanceNormToDPUVersion"]

        self.model.model, self.nodes_to_exclude = simulate_transforms(
            self.model.model,
            self.should_quantize_node,
            self.nodes_to_quantize,
            self.nodes_to_exclude,
            convert_leaky_relu_to_dpu_version=convert_leaky_relu_to_dpu_version,
            convert_sigmoid_to_hard_sigmoid=convert_sigmoid_to_hard_sigmoid,
            convert_hard_sigmoid_to_dpu_version=convert_hard_sigmoid_to_dpu_version,
            convert_avg_pool_to_dpu_version=convert_avg_pool_to_dpu_version,
            convert_reduce_mean_to_dpu_version=convert_reduce_mean_to_dpu_version,
            convert_softmax_to_dpu_version=convert_softmax_to_dpu_version,
            convert_instance_norm_to_dpu_version=convert_instance_norm_to_dpu_version,
        )


class VitisBFPQuantizer(VitisQDQQuantizer):
    """
    A class to perform Vitis-specific Block Floating Point (BFP) Quantization-Dequantization (QDQ) quantization.

    Args:
        model (ModelProto): The ONNX model to be quantized.
        per_channel (bool): Whether to perform per-channel quantization.
        reduce_range (bool): Whether to reduce the quantization range.
        mode (QuantizationMode.QLinearOps): The quantization mode to be used.
        quant_format (Any): The format for quantization.
        static (bool): Whether to use static quantization.
        weight_qType (Any): The quantization type for weights.
        activation_qType (Any): The quantization type for activations.
        tensors_range (Any): Dictionary specifying the min and max values for tensors.
        nodes_to_quantize (List[str]): List of node names to be quantized.
        nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
        op_types_to_quantize (List[str]): List of operation types to be quantized.
        calibrate_method (Any): The method used for calibration.
        quantized_tensor_type (Dict[Any, Any], optional): Dictionary specifying quantized tensor types.
        extra_options (Optional[Dict[str, Any]], optional): Additional options for quantization.

    Inherits from:
        VitisQDQQuantizer: Base class for Vitis-specific QDQ quantization.

    Attributes:
        int32_bias (bool): Whether to quantize bias as int32.
        is_activation_symmetric (bool): Whether to use symmetric quantization for activations.
        quant_format (Any): The format for quantization.
        fn_type: (string): The op type of the fix neuron.
        fn_attrs (Dict[str, Any]): Attributes for BFP/MX fix neuron.
    """

    def __init__(self,
                 model: ModelProto,
                 per_channel: bool,
                 reduce_range: bool,
                 mode: QuantizationMode.QLinearOps,
                 quant_format: Any,
                 static: bool,
                 weight_qType: Any,
                 activation_qType: Any,
                 tensors_range: Any,
                 nodes_to_quantize: List[str],
                 nodes_to_exclude: List[str],
                 op_types_to_quantize: List[str],
                 calibrate_method: Any,
                 quantized_tensor_type: Dict[Any, Any] = {},
                 extra_options: Optional[Dict[str, Any]] = None):
        """
        Initializes the VitisBFPQuantizer with the provided configuration.

        Args:
            model (ModelProto): The ONNX model to be quantized.
            per_channel (bool): Whether to perform per-channel quantization.
            reduce_range (bool): Whether to reduce the quantization range.
            mode (QuantizationMode.QLinearOps): The quantization mode to be used.
            static (bool): Whether to use static quantization.
            weight_qType (Any): The quantization type for weights.
            activation_qType (Any): The quantization type for activations.
            tensors_range (Any): Dictionary specifying the min and max values for tensors.
            nodes_to_quantize (List[str]): List of node names to be quantized.
            nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
            op_types_to_quantize (List[str]): List of operation types to be quantized.
            calibrate_method (Any): The method used for calibration.
            quantized_tensor_type (Dict[Any, Any], optional): Dictionary specifying quantized tensor types.
            extra_options (Optional[Dict[str, Any]], optional): Additional options for quantization.
        """
        super().__init__(model, per_channel, reduce_range, mode, static, weight_qType, activation_qType, tensors_range,
                         nodes_to_quantize, nodes_to_exclude, op_types_to_quantize, calibrate_method,
                         quantized_tensor_type, extra_options)

        self.int32_bias = False
        if extra_options is not None and "Int32Bias" in extra_options and extra_options["Int32Bias"]:
            self.int32_bias = extra_options["Int32Bias"]
            logger.warning("Will not quantize Bias since do not support Int32Bias in BFP/MX mode")

        self.is_activation_symmetric = True
        if self.extra_options is not None and "ActivationSymmetric" in self.extra_options and not self.extra_options[
                "ActivationSymmetric"]:
            self.is_activation_symmetric = self.extra_options["ActivationSymmetric"]
            logger.warning("Setting ActivationSymmetric to False has no effect on BFP/MX mode")

        self.quant_format = quant_format
        assert self.quant_format in [VitisQuantFormat.BFPFixNeuron, VitisQuantFormat.MXFixNeuron]

        self.fn_name = ""
        self.fn_attrs = {}
        if self.quant_format == VitisQuantFormat.BFPFixNeuron:
            self.fn_name = BFPFIX_OP_NAME
            self.fn_attrs = copy.deepcopy(BFPFIX_OP_DEFAULT_ATTRS)
            # Get attributes for BFPFixNeuron
            if extra_options is not None and "BFPAttributes" in extra_options:
                self.fn_attrs.update(extra_options["BFPAttributes"])
        else:
            self.fn_name = MXFIX_OP_NAME
            self.fn_attrs = copy.deepcopy(MXFIX_OP_DEFAULT_ATTRS)
            # Get attributes for MXFixNeuron
            if extra_options is not None and "MXAttributes" in extra_options:
                self.fn_attrs.update(extra_options["MXAttributes"])

    def _create_fn_nodes(self,
                         q_input: Any,
                         dq_output: Any,
                         dequant_node_name: str,
                         axis: Any = None,
                         convert_to: Any = None) -> None:
        """
        create fix_neuron node
        """
        fix_neuron_node = onnx.helper.make_node(
            self.fn_name,
            [q_input],
            [dq_output],
            dequant_node_name,
            domain=COP_DOMAIN,
        )

        for k, v in self.fn_attrs.items():
            if k == "axis" and axis is not None:
                v = axis
            elif k == "convert_to_bfloat_before_bfp" and convert_to is not None:
                v = convert_to
            fix_neuron_node.attribute.append(onnx.helper.make_attribute(k, v))

        self.model.add_nodes([fix_neuron_node])

    def _add_fn_pair_for_weight(self, weight_proto: TensorProto) -> None:
        weight_name = weight_proto.name
        dq_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name, dq_output)
        axis = 0 if len(weight_proto.dims) == 1 else None  # For scalar, the axis should be 0
        convert_to = 0  # Initializer is a constant, no conversion required
        self._create_fn_nodes(weight_name, dq_output, add_dequant_suffix(weight_name), axis, convert_to)

    def _add_fn_pair_for_activation(self, tensor_name: str) -> None:
        q_input = tensor_name
        dq_output = add_dequant_output_suffix(tensor_name)
        if self.model.is_graph_output(tensor_name):
            q_input = add_quant_input_suffix(tensor_name)
            dq_output = tensor_name
            self.model.replace_output_of_all_nodes(tensor_name, q_input)
        else:
            self.model.replace_input_of_all_nodes(tensor_name, dq_output)

        self._create_fn_nodes(q_input, dq_output, add_dequant_suffix(tensor_name))

    def _quantize_normal_tensors(self) -> None:
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():

            if tensor_name in self.quantized_value_map.keys():
                continue

            if not tensor_info.is_shared:
                # Quantize the input
                initializer = find_by_name(tensor_name, self.model.initializer())
                if initializer:
                    self._add_fn_pair_for_weight(initializer)
                else:
                    self._add_fn_pair_for_activation(tensor_name)

                del self.tensors_to_quantize[tensor_name]

    def quantize_model(self) -> Any:
        annotate_tensors = get_annotate_tensors(self.model.model)

        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

        self._quantize_normal_tensors()
        # Do not support Int32 Bias in BFP mode
        # if self.quantize_bias and self.int32_bias:
        # self._quantize_bias_tensors()

        self.remove_nodes()
        dq_nodes_to_remove, q_nodes_to_remove, input_node_mapping = get_qdq_to_remove(
            self.model.model, annotate_tensors)
        pruned_model = copy.deepcopy(self.model)
        modified_annotate_input(pruned_model.model, input_node_mapping)
        pruned_model.model = remove_nodes(pruned_model.model, dq_nodes_to_remove)
        pruned_model.model = remove_nodes(pruned_model.model, q_nodes_to_remove)
        try:
            pruned_model.topological_sort()
            logger.info("Remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu).")
            self.model.model = pruned_model.model
        except Exception as e:
            logger.warning(
                f"Unable to remove QuantizeLinear & DequantizeLinear on certain operations(such as conv-relu). Exception: {e}"
            )

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model
