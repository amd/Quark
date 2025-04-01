#
# Modifications copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy
import numpy as np
from quark.shares.utils.log import ScreenLogger
from typing import Any, List, Dict, Optional

import onnx
import onnx.numpy_helper
from onnx import TensorProto, ModelProto
from onnx import onnx_pb as onnx_proto
from onnxruntime.quantization.base_quantizer import (QuantizationParams, to_array_extended)
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer as OrtQDQQuantizer
from onnxruntime.quantization.qdq_quantizer import (QDQBiasQuantInfo, QDQTensorQuantParams, QDQTensorQuantizedValue,
                                                    QDQScaleZpInitializers)

from onnxruntime.quantization.quant_utils import (
    QuantizedValue,
    QuantizedValueType,
    QuantType,
    QuantizationMode,
    TENSOR_NAME_QUANT_SUFFIX,
    DEQUANT_OP_NAME,
    add_dequant_output_suffix,
    add_dequant_suffix,
    add_quant_output_suffix,
    add_quant_suffix,
    find_by_name,
    ms_domain,
    ONNX_TYPE_TO_NP_TYPE,
    normalize_axis,
    quantize_nparray,
)

from quark.onnx.quant_utils import is_ort_version_below
if not is_ort_version_below("1.19.0"):
    from onnxruntime.quantization.quant_utils import pack_bytes_to_4bit

from onnxruntime.quantization.onnx_quantizer import tensor_proto_to_array
from onnxruntime.quantization.calibrate import TensorData
from ..quant_utils import (
    __producer__,
    __version__,
    VitisQuantType,
    get_annotate_tensors,
    get_qdq_to_remove,
    remove_nodes,
    modified_annotate_input,
    compute_scale_zp,
    ONNX_FP_QTYPES_LIST,
    ONNX_BFP_QTYPES_LIST,
    compute_scale_zp_fp,
    quantize_data_pof2s,
    get_qmin_qmax_for_qType,
    get_tensor_type_from_qType,
)
from ..registry import CreateQDQQuantizer

logger = ScreenLogger(__name__)


class QDQQuantizer(OrtQDQQuantizer):  # type: ignore

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
        self.weights_only = False if extra_options is None or "WeightsOnly" not in extra_options else extra_options[
            "WeightsOnly"]

        # include-gptq quantization switch
        self.use_gptq = False if extra_options is None or "UseGPTQ" not in extra_options else extra_options["UseGPTQ"]
        # If GPTQ is turned on, the quantizer will only quantize weights and leave the activations in floating-point for GPTQ.
        if self.use_gptq is True:
            self.weights_only = True

    def _is_tensor_quantizable(self, tensor_name: str) -> bool:
        """
        Check if tensor can be quantized
        """
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

    def quantize_bias_tensor(self,
                             node_name: str,
                             bias_name: str,
                             input_name: str,
                             weight_name: str,
                             beta: float = 1.0) -> None:
        """
        Adds a bias tensor to the list of bias tensors to quantize. Called by op quantizers that
        want to quantize a bias with bias_zero_point = 0 and bias_scale = input_scale * weight_scale * beta.
        TODO: Explain the reasoning for using this formula.

        Args:
            node_name: name of the node that consumes the bias, input, and weight tensors.
            bias_name: name of the bias tensor to quantize.
            input_name: name of the input tensor whose scale is used to compute the bias's scale.
            weight_name: name of the weight tensor whose scale is used to compute the bias's scale.
            beta: Multiplier used to compute the bias's scale.
        """
        # If the user provided quantization overrides for this tensor, treat it as a regular weight.
        if self.tensor_quant_overrides.get(bias_name):
            logger.info(
                f"Quantizing bias tensor '{bias_name}' as a weight due to the presence of user-specified overrides")
            is_per_channel, axis = self.is_tensor_per_channel(bias_name, default_axis=0)
            if is_per_channel:
                self.quantize_weight_tensor_per_channel(bias_name, axis)
            else:
                self.quantize_weight_tensor(bias_name)
            return

        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16):
                if self.quantize_bias:
                    if bias_name not in self.bias_to_quantize:
                        if self.int32_bias:
                            self.bias_to_quantize[bias_name] = QDQBiasQuantInfo(node_name, input_name, weight_name,
                                                                                beta)
                        else:
                            if self.per_channel:
                                self.quantize_weight_tensor_per_channel(bias_name, 0)
                            else:
                                self.quantize_weight_tensor(bias_name)
                    else:
                        logger.warning(f"Bias {bias_name} has already been marked for quantization")
        else:
            logger.warning(f"Expected {bias_name} to be a weight")

    def _quantize_bias_tensors(self) -> None:
        """
        Adds DQ ops (or Cast) for bias tensors that have been marked for quantization by op quantizers.
        """
        for bias_name, bias_info in self.bias_to_quantize.items():
            if bias_name in self.quantized_value_map:
                continue
            # Quantize the input
            self.quantize_bias_static(bias_name, bias_info)
            # TODO: Figure out why the program exits on Windows if we don't add a print statement here
            logger.debug('Adds DQ ops (or Cast) for bias')
            init = find_by_name(bias_name, self.model.initializer())
            self.model.remove_initializer(init)
            quant_value = self.quantized_value_map[bias_name].original
            if quant_value.node_type == "Cast":
                # simple cast to float 16 and not DequantizeLinear
                # cublasLtMatmul only supports (b)float16, float bias.
                if not isinstance(init.data_type, int):
                    raise TypeError(f"Unexpected type {type(init.data_type)} for input={bias_info.input_name!r}")
                node_name = add_dequant_suffix(bias_name)
                dequant_node = onnx.helper.make_node(
                    "Cast",
                    [quant_value.q_name],
                    [bias_name],
                    name=node_name,
                    to=init.data_type,
                )
            elif quant_value.node_type in (None, "DequantizeLinear"):
                if quant_value.node_qtype in {
                        onnx.TensorProto.FLOAT16,
                        onnx.TensorProto.BFLOAT16,
                        onnx.TensorProto.FLOAT,
                }:
                    raise RuntimeError(f"Unexpected quantize type {quant_value.node_qtype} for DequantizeLinear.")
                inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
                node_name = add_dequant_suffix(bias_name)
                if quant_value.axis is not None:
                    dequant_node = onnx.helper.make_node(
                        "DequantizeLinear",
                        inputs,
                        [bias_name],
                        node_name,
                        axis=quant_value.axis,
                        domain=self.qdq_op_domain,
                    )
                else:
                    dequant_node = onnx.helper.make_node(
                        "DequantizeLinear",
                        inputs,
                        [bias_name],
                        node_name,
                        domain=self.qdq_op_domain,
                    )
            else:
                raise RuntimeError(f"Unexpected operator type {quant_value.node_type!r}.")
            self.model.add_node(dequant_node)

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
        if self.qdq_op_domain == ms_domain:
            self.model.set_opset_import(ms_domain, 1)

        return self.model.model

    def _add_qdq_pair_for_initializer(self, weight_proto: TensorProto, tensor_type: Any, axis: Any = None) -> None:
        weight_name = weight_proto.name
        if axis is not None:
            if self.opset_version < 13:
                raise ValueError("Per-Channel support with QDQ format requires onnx opset version 13 or above.")
            qtype = self.weight_qType
            if self.activation_qType == onnx.onnx_pb.TensorProto.UINT8:
                qtype = onnx_proto.TensorProto.INT8
            q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel(
                weight_name,
                # Quantization type is forced to be TensorProto.INT8.
                # when the expected value would be (see below)
                # self.weight_qType if tensor_type is QDQQuantTensorType.WEIGHT else self.activation_qType.
                # QLinearConv expects to have a unique value for all channels.
                # This code does not enforce that but it is necessarily the case when the
                # quantization is symmetric (as for INT8).
                qtype,
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
                domain=self.qdq_op_domain,
            )
            self.model.add_node(dequant_node)

    def _quantize_normal_tensors(self) -> None:
        """
        Adds Q/DQ ops to tensors (activations and weights) that have been marked for quantization by op quantizers.
        """
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
            if tensor_name in self.quantized_value_map:
                continue

            if not tensor_info.is_shared:
                # Quantize the input
                initializer = find_by_name(tensor_name, self.model.initializer())
                if initializer:
                    self._add_qdq_pair_for_initializer(initializer, tensor_info.tensor_type, tensor_info.axis)
                else:
                    tensor_qparam_initializers = self._make_tensor_scale_zp_initializers(tensor_name)
                    if not tensor_qparam_initializers:
                        raise ValueError(
                            f"Quantization parameters are not specified for param {tensor_name}. "
                            "In static mode quantization params for inputs and outputs of nodes to be quantized are required."
                        )

                    if tensor_qparam_initializers.converted is None:
                        # Normal case: <producer> --> Q --> DQ --> <consumers>
                        self._add_qdq_pair_for_activation(
                            tensor_name,
                            tensor_qparam_initializers.original.scale.name,
                            tensor_qparam_initializers.original.zero_point.name,
                            data_type=tensor_info.data_type,
                        )
                    else:
                        # Conversion case: <producer> ---> Q1 -+-> DQ1 --> <consumers of original type>
                        #                                      |
                        #                                      +-> DQ1' --> Q2 --> DQ2 --> <consumers of converted type>
                        assert tensor_info.data_type == tensor_qparam_initializers.original.scale.data_type
                        self._add_qdq_ops_for_converted_activation(
                            tensor_name,
                            tensor_qparam_initializers.original.scale.name,
                            tensor_qparam_initializers.original.zero_point.name,
                            tensor_info.data_type,
                            tensor_qparam_initializers.converted.scale.name,
                            tensor_qparam_initializers.converted.zero_point.name,
                            tensor_qparam_initializers.converted_recv_nodes,
                        )

                del self.tensors_to_quantize[tensor_name]

    def quantize_initializer(
        self,
        weight: onnx.TensorProto,
        qType: onnx.TensorProto.DataType,
        reduce_range: bool = False,
        keep_float_weight: bool = False,
    ) -> tuple[str, str, str]:
        """
        :param weight: TensorProto initializer
        :param qType: type to quantize to
        :param keep_float_weight: Whether to quantize the weight. In some cases, we only want to qunatize scale and zero point.
                                  If keep_float_weight is False, quantize the weight, or don't quantize the weight.
        :return: quantized weight name, zero point name, scale name
        """
        # Find if this input is already quantized
        if weight.name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight.name].original
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        q_weight_name, zp_name, scale_name = self.quantize_initializer_impl(weight, qType, reduce_range,
                                                                            keep_float_weight)

        # Log entry for this quantized weight
        quantized_value = QuantizedValue(
            weight.name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight.name] = QDQTensorQuantizedValue(quantized_value, None, None)
        return q_weight_name, zp_name, scale_name


class VitisQDQQuantizer(OrtQDQQuantizer):  # type: ignore

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
        self.calibrate_method = calibrate_method
        self.quantized_tensor_type = quantized_tensor_type

        self.use_power_of_2_scale = True
        if extra_options is not None and "UsePowerOf2Scale" in extra_options:
            self.use_power_of_2_scale = extra_options["UsePowerOf2Scale"]

        # weights-only quantization switch
        self.weights_only = False if extra_options is None or "WeightsOnly" not in extra_options else extra_options[
            "WeightsOnly"]

        # include-gptq quantization switch
        self.use_gptq = False if extra_options is None or "UseGPTQ" not in extra_options else extra_options["UseGPTQ"]
        # If GPTQ is turned on, the quantizer will only quantize weights and leave the activations in floating-point for GPTQ.
        if self.use_gptq is True:
            self.weights_only = True

        # Scale weight and activation for floating point data types' quantization
        self.is_weight_scaled = True
        if weight_qType in (VitisQuantType.QFloat16, VitisQuantType.QBFloat16, VitisQuantType.QBFP, VitisQuantType.QMX):
            self.is_weight_scaled = False if (extra_options is None
                                              or "WeightScaled" not in extra_options) else extra_options["WeightScaled"]

        self.is_activation_scaled = True
        if activation_qType in (VitisQuantType.QFloat16, VitisQuantType.QBFloat16, VitisQuantType.QBFP,
                                VitisQuantType.QMX):
            self.is_activation_scaled = False if (
                extra_options is None or "ActivationScaled" not in extra_options) else extra_options["ActivationScaled"]

        OrtQDQQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
        self.tensors_to_quantize: Dict[str, Any] = {}
        self.bias_to_quantize: Dict[str, Any] = {}

        self.nodes_to_remove: List[str] = []

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
        self.tensor_to_its_receiving_nodes: Dict[str, Any] = {}

        # Let user set channel axis for specific op type and it's effective only when per channel quantization is supported and per_channel is True.
        self.qdq_op_type_per_channel_support_to_axis = ({} if extra_options is None
                                                        or "QDQOpTypePerChannelSupportToAxis" not in extra_options else
                                                        extra_options["QDQOpTypePerChannelSupportToAxis"])
        self.qdq_op_domain = ms_domain if extra_options is None or extra_options.get("UseQDQContribOps",
                                                                                     False) else None

        # The ONNX spec did not support 16-bit Q/DQ ops before opset 21.
        # So, may have to override the Q/DQ op domain to 'com.microsoft' if the activation or weight types
        # are 16-bit or 4-bit integers.
        if self.opset_version < 21:
            opset21_types = (TensorProto.UINT16, TensorProto.INT16, TensorProto.UINT4, TensorProto.INT4)
            overrides_have_opset21_types = any(t.tensor_type in opset21_types
                                               for t in self.tensor_quant_override_qtypes)
            if not self.qdq_op_domain and (self.activation_qType in opset21_types or self.weight_qType in opset21_types
                                           or overrides_have_opset21_types):
                logger.warning(
                    "ONNX QuantizeLinear and DequantizeLinear operators do not support "
                    "16-bit/4-bit integer quantization types prior to opset 21. "
                    f"The domain of QuantizeLinear and DequantizeLinear operators will be set to '{ms_domain}' to "
                    "enable support.")
                self.qdq_op_domain = ms_domain

        self.is_weight_symmetric = (weight_qType in (QuantType.QInt8, QuantType.QInt16, VitisQuantType.QInt16,
                                                     VitisQuantType.QInt32, VitisQuantType.QFloat16,
                                                     VitisQuantType.QBFloat16, VitisQuantType.QBFP,
                                                     VitisQuantType.QMX) if "WeightSymmetric" not in self.extra_options
                                    else self.extra_options["WeightSymmetric"])
        self.is_activation_symmetric = (activation_qType in (VitisQuantType.QFloat16, VitisQuantType.QBFloat16,
                                                             VitisQuantType.QBFP,
                                                             VitisQuantType.QMX) if "ActivationSymmetric"
                                        not in self.extra_options else self.extra_options["ActivationSymmetric"])

        self.quantization_params = self.calc_graph_quant_params()

        # Map of all original value names to quantized value names
        self.quantized_value_map: Dict[str, Any] = {}

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

    def _is_tensor_quantizable(self, tensor_name: str) -> bool:
        """
        Check if tensor can be quantized
        """
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
                "failed to infer the type of tensor: {}. Skip to quantize it. Please check if it is expected.".format(
                    tensor_name))

        return False

    def quantize_bias_tensor(self,
                             node_name: str,
                             bias_name: str,
                             input_name: str,
                             weight_name: str,
                             beta: float = 1.0) -> None:
        """
        Adds a bias tensor to the list of bias tensors to quantize. Called by op quantizers that
        want to quantize a bias with bias_zero_point = 0 and bias_scale = input_scale * weight_scale * beta.
        TODO: Explain the reasoning for using this formula.

        Args:
            node_name: name of the node that consumes the bias, input, and weight tensors.
            bias_name: name of the bias tensor to quantize.
            input_name: name of the input tensor whose scale is used to compute the bias's scale.
            weight_name: name of the weight tensor whose scale is used to compute the bias's scale.
            beta: Multiplier used to compute the bias's scale.
        """
        # If the user provided quantization overrides for this tensor, treat it as a regular weight.
        if self.tensor_quant_overrides.get(bias_name):
            logger.info(
                f"Quantizing bias tensor '{bias_name}' as a weight due to the presence of user-specified overrides")
            is_per_channel, axis = self.is_tensor_per_channel(bias_name, default_axis=0)
            if is_per_channel:
                self.quantize_weight_tensor_per_channel(bias_name, axis)
            else:
                self.quantize_weight_tensor(bias_name)
            return

        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16):
                if self.quantize_bias:
                    if bias_name not in self.bias_to_quantize:
                        if self.int32_bias:
                            self.bias_to_quantize[bias_name] = QDQBiasQuantInfo(node_name, input_name, weight_name,
                                                                                beta)
                        else:
                            if self.per_channel:
                                self.quantize_weight_tensor_per_channel(bias_name, 0)
                            else:
                                self.quantize_weight_tensor(bias_name)
                    else:
                        logger.warning(f"Bias {bias_name} has already been marked for quantization")
        else:
            logger.warning(f"Expected {bias_name} to be a weight")

    def _quantize_bias_tensors(self) -> None:
        """
        Adds DQ ops (or Cast) for bias tensors that have been marked for quantization by op quantizers.
        """
        for bias_name, bias_info in self.bias_to_quantize.items():
            if bias_name in self.quantized_value_map:
                continue
            # Quantize the input
            self.quantize_bias_static(bias_name, bias_info)
            # TODO: Figure out why the program exits on Windows if we don't add a print statement here
            logger.debug('Adds DQ ops (or Cast) for bias')
            init = find_by_name(bias_name, self.model.initializer())
            self.model.remove_initializer(init)
            quant_value = self.quantized_value_map[bias_name].original
            if quant_value.node_type == "Cast":
                # simple cast to float 16 and not DequantizeLinear
                # cublasLtMatmul only supports (b)float16, float bias.
                if not isinstance(init.data_type, int):
                    raise TypeError(f"Unexpected type {type(init.data_type)} for input={bias_info.input_name!r}")
                node_name = add_dequant_suffix(bias_name)
                dequant_node = onnx.helper.make_node(
                    "Cast",
                    [quant_value.q_name],
                    [bias_name],
                    name=node_name,
                    to=init.data_type,
                )
            elif quant_value.node_type in (None, "DequantizeLinear"):
                if quant_value.node_qtype in {
                        onnx.TensorProto.FLOAT16,
                        onnx.TensorProto.BFLOAT16,
                        onnx.TensorProto.FLOAT,
                }:
                    raise RuntimeError(f"Unexpected quantize type {quant_value.node_qtype} for DequantizeLinear.")
                inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
                node_name = add_dequant_suffix(bias_name)
                if quant_value.axis is not None:
                    dequant_node = onnx.helper.make_node(
                        "DequantizeLinear",
                        inputs,
                        [bias_name],
                        node_name,
                        axis=quant_value.axis,
                        domain=self.qdq_op_domain,
                    )
                else:
                    dequant_node = onnx.helper.make_node(
                        "DequantizeLinear",
                        inputs,
                        [bias_name],
                        node_name,
                        domain=self.qdq_op_domain,
                    )
            else:
                raise RuntimeError(f"Unexpected operator type {quant_value.node_type!r}.")
            self.model.add_node(dequant_node)

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
        if self.qdq_op_domain == ms_domain:
            self.model.set_opset_import(ms_domain, 1)

        return self.model.model

    def _add_qdq_pair_for_initializer(self, weight_proto: TensorProto, tensor_type: Any, axis: Any = None) -> None:
        weight_name = weight_proto.name
        if axis is not None:
            if self.opset_version < 13:
                raise ValueError("Per-Channel support with QDQ format requires onnx opset version 13 or above.")
            qtype = self.weight_qType
            if self.activation_qType == onnx.onnx_pb.TensorProto.UINT8:
                qtype = onnx_proto.TensorProto.INT8
            q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel(
                weight_name,
                # Quantization type is forced to be TensorProto.INT8.
                # when the expected value would be (see below)
                # self.weight_qType if tensor_type is QDQQuantTensorType.WEIGHT else self.activation_qType.
                # QLinearConv expects to have a unique value for all channels.
                # This code does not enforce that but it is necessarily the case when the
                # quantization is symmetric (as for INT8).
                qtype,
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
                domain=self.qdq_op_domain,
            )
            self.model.add_node(dequant_node)

    def quantize_initializer_impl(self,
                                  weight: TensorProto,
                                  qType: Any,
                                  reduce_range: bool = False,
                                  keep_float_weight: bool = False) -> Any:
        """
        :param weight: TensorProto initializer
        :param qType: type to quantize to. Note that it may be different with weight_qType because of mixed precision
        :param keep_float_weight: Whether to quantize the weight. In some cases, we only want to qunatize scale and zero point.
                                If keep_float_weight is False, quantize the weight, or don't quantize the weight.
        :return: quantized weight name, zero point name, scale name
        """
        q_weight_name = weight.name + TENSOR_NAME_QUANT_SUFFIX
        zp_name = weight.name + "_zero_point"
        scale_name = weight.name + "_scale"

        # Quantize weight data. Use quantization overrides if provided by the user.
        weight_data = tensor_proto_to_array(weight)
        quant_overrides = self.tensor_quant_overrides.get_per_tensor_overrides(weight.name, default_val={})
        if "quant_type" in quant_overrides:
            qType = quant_overrides["quant_type"].tensor_type  # noqa: N806

        if "scale" in quant_overrides and "zero_point" in quant_overrides:
            zero_point = np.array(quant_overrides["zero_point"], dtype=ONNX_TYPE_TO_NP_TYPE[qType])
            scale = np.array(quant_overrides["scale"])
            q_weight_data = quantize_nparray(qType, weight_data.flatten(), scale, zero_point)
            assert isinstance(zero_point, np.ndarray), f"Unexpected type {type(zero_point)}"
            assert isinstance(scale, np.ndarray), f"Unexpected type {type(scale)}"

        else:
            _, _, zero_point, scale, q_weight_data = quantize_data_pof2s(
                weight_data.flatten(),
                qType,
                self.is_weight_symmetric,
                reduce_range=self.reduce_range and reduce_range,
                method=self.calibrate_method,
                use_pof2s=self.use_power_of_2_scale,
                use_scaling=self.is_weight_scaled,
            )

            assert isinstance(zero_point, np.ndarray), f"Unexpected type {type(zero_point)}"
            assert isinstance(scale, np.ndarray), f"Unexpected type {type(scale)}"

        scale_dtype = weight.data_type
        scale_initializer = onnx.helper.make_tensor(scale_name, scale_dtype, [], scale.reshape((-1, )).tolist())
        if qType in ONNX_BFP_QTYPES_LIST:
            # BFP data types do not need zero point, but we need to consider the case of reusing zero point of
            # weight for activation, such as Gather aligns its output with input.
            if self.activation_qType in ONNX_BFP_QTYPES_LIST:
                zero_initializer = onnx.helper.make_tensor(zp_name, onnx_proto.TensorProto.FLOAT, [],
                                                           zero_point.reshape((-1, )).tolist())
            else:
                if self.activation_qType not in ONNX_FP_QTYPES_LIST:
                    zero_point = zero_point.astype(ONNX_TYPE_TO_NP_TYPE[self.activation_qType])
                zero_initializer = onnx.helper.make_tensor(zp_name, self.activation_qType, [],
                                                           zero_point.reshape((-1, )).tolist())
        else:
            zero_initializer = onnx.helper.make_tensor(zp_name, qType, [], zero_point.reshape((-1, )).tolist())
        self.model.initializer_extend([scale_initializer, zero_initializer])
        if not keep_float_weight:
            if qType == onnx.TensorProto.FLOAT8E4M3FN:
                q_weight_initializer = onnx.TensorProto()
                q_weight_initializer.data_type = qType
                q_weight_initializer.dims.extend(weight.dims)
                q_weight_initializer.name = q_weight_name
                # Do not remove .flatten().copy() numpy is not clear about data persistence.
                q_weight_initializer.raw_data = q_weight_data.flatten().copy().tobytes()
                if to_array_extended is not None:
                    # This test should not be needed but it helped catch some issues
                    # with data persistence and tobytes.
                    check = to_array_extended(q_weight_initializer)
                    if check.shape != weight_data.shape or check.tobytes() != q_weight_data.tobytes():
                        raise RuntimeError(
                            f"The initializer of shape {weight_data.shape} could not be created, expecting "
                            f"{q_weight_data.tobytes()[:10]}, got {check.tobytes()[:10]} and shape={weight.shape}"
                            f"\nraw={str(q_weight_initializer)[:200]}.")
            elif qType in (onnx.TensorProto.INT4, onnx.TensorProto.UINT4):
                if is_ort_version_below("1.19.0"):
                    raise RuntimeError(f"onnxruntime version >= 1.19 is required to support {qType} quantization.")

                if q_weight_data.dtype not in (np.int8, np.uint8):
                    raise RuntimeError(
                        f"Quantized weights for {q_weight_name} must be 8-bit before packing as 4-bit values.")

                # We do not use onnx.helper.pack_float32_to_4bit() due to performance.
                # This can be the difference between a large model taking 30 minutes to quantize vs 5 minutes.
                packed_data = bytes(pack_bytes_to_4bit(q_weight_data.tobytes()))

                # We only use onnx.helper.make_tensor with raw data due to bug: https://github.com/onnx/onnx/pull/6161
                q_weight_initializer = onnx.helper.make_tensor(q_weight_name, qType, weight.dims, packed_data, raw=True)
            elif qType in ONNX_FP_QTYPES_LIST:
                q_weight_initializer = onnx.TensorProto()
                q_weight_initializer.data_type = qType
                q_weight_initializer.dims.extend(weight.dims)
                q_weight_initializer.name = q_weight_name
                # Do not remove .flatten().copy() numpy is not clear about data persistence.
                q_weight_initializer.raw_data = q_weight_data.flatten().copy().tobytes()
            elif qType in ONNX_BFP_QTYPES_LIST:
                # We just use original values for BFP data types, because the quantized weight is not actually used
                q_weight_initializer = onnx.TensorProto()
                q_weight_initializer.CopyFrom(weight)
                q_weight_initializer.name = q_weight_name
            else:
                q_weight_data = np.asarray(q_weight_data,
                                           dtype=onnx.helper.tensor_dtype_to_np_dtype(qType)).reshape(weight.dims)
                q_weight_initializer = onnx.numpy_helper.from_array(q_weight_data, q_weight_name)
            self.model.initializer_extend([q_weight_initializer])

        return q_weight_name, zp_name, scale_name

    def quantize_weight_per_channel_impl(
        self,
        weight_name: str,
        weight_qType: Any,
        channel_axis: Any,
        reduce_range: bool = True,
        keep_float_weight: bool = False,
    ) -> Any:
        initializer = find_by_name(weight_name, self.model.initializer())
        if initializer is None:
            raise ValueError("{} is not an initializer", weight_name)

        weights = tensor_proto_to_array(initializer)
        weights_rank = len(weights.shape)
        is_axis_valid, axis_norm = normalize_axis(channel_axis, weights_rank)
        if not is_axis_valid:
            raise ValueError(f"Weight {weight_name} has a per-channel axis with value {channel_axis} that is "
                             f"out-of-bounds for rank {weights_rank}")

        channel_axis = axis_norm
        channel_count = weights.shape[channel_axis]
        quant_overrides_for_channels = self.tensor_quant_overrides.get_per_channel_overrides(weight_name,
                                                                                             default_val=[{
                                                                                                 "axis":
                                                                                                 channel_axis
                                                                                             }])

        num_channel_overrides = len(quant_overrides_for_channels)
        if num_channel_overrides != 1 and num_channel_overrides != channel_count:
            raise ValueError(f"Per-channel tensor quantization overrides for {weight_name} must have "
                             f"either 1 or {channel_count} elements in the list of dictionaries.")

        is_axis_override_valid, axis_override = normalize_axis(quant_overrides_for_channels[0]["axis"], weights_rank)
        if not is_axis_override_valid or axis_override != channel_axis:
            raise ValueError(f"Tensor quantization overrides for {weight_name} specify an unexpected axis. "
                             f"Expected {channel_axis}, but got {quant_overrides_for_channels[0]['axis']}.")

        # If user provides per-channel quantization overrides, all channels must use the same quant_type,
        # axis, symmetric, and reduce_range values. So, just use the first channel's values.
        if "quant_type" in quant_overrides_for_channels[0]:
            weight_qType = quant_overrides_for_channels[0]["quant_type"].tensor_type  # noqa: N806
        reduce_range = quant_overrides_for_channels[0].get("reduce_range", self.reduce_range and reduce_range)
        zero_point_list = []
        scale_list = []
        quantized_per_channel_data_list = []
        for i in range(channel_count):
            per_channel_data = weights.take(i, channel_axis)
            channel_override_index = i if i < num_channel_overrides else 0
            channel_quant_overrides = quant_overrides_for_channels[channel_override_index]

            if "scale" in channel_quant_overrides and "zero_point" in channel_quant_overrides:
                zero_point = np.array(channel_quant_overrides["zero_point"], dtype=ONNX_TYPE_TO_NP_TYPE[weight_qType])
                scale = np.array(channel_quant_overrides["scale"])
                quantized_per_channel_data = quantize_nparray(weight_qType, per_channel_data.flatten(), scale,
                                                              zero_point)
                assert isinstance(zero_point, np.ndarray), f"Unexpected type {type(zero_point)}"
                assert (zero_point.dtype != np.float32
                        and zero_point.dtype != np.float16), f"Unexpected dtype {zero_point.dtype}"
                assert isinstance(scale, np.ndarray), f"Unexpected type {type(scale)}"
                assert isinstance(quantized_per_channel_data,
                                  np.ndarray), f"Unexpected type {type(quantized_per_channel_data)}"

            else:
                _, _, zero_point, scale, quantized_per_channel_data = quantize_data_pof2s(
                    per_channel_data.flatten(),
                    weight_qType,
                    self.is_weight_symmetric,
                    self.reduce_range and reduce_range,
                    method=self.calibrate_method,
                    use_pof2s=self.use_power_of_2_scale,
                )

                assert isinstance(zero_point, np.ndarray), f"Unexpected type {type(zero_point)}"
                assert (zero_point.dtype != np.float32
                        and zero_point.dtype != np.float16), f"Unexpected dtype {zero_point.dtype}"
                assert isinstance(scale, np.ndarray), f"Unexpected type {type(scale)}"
                assert isinstance(quantized_per_channel_data,
                                  np.ndarray), f"Unexpected type {type(quantized_per_channel_data)}"

            zero_point_list.append(zero_point)
            scale_list.append(scale)
            quantized_per_channel_data_list.append(quantized_per_channel_data)

        # combine per_channel_data into one
        reshape_dims = list(weights.shape)  # deep copy
        reshape_dims[channel_axis] = 1  # only one per channel for reshape
        quantized_weights = np.asarray(quantized_per_channel_data_list[0]).reshape(reshape_dims)
        for i in range(1, len(quantized_per_channel_data_list)):
            channel_weights = np.asarray(quantized_per_channel_data_list[i]).reshape(reshape_dims)
            quantized_weights = np.concatenate((quantized_weights, channel_weights), channel_axis)

        q_weight_name = weight_name + TENSOR_NAME_QUANT_SUFFIX
        zp_name = weight_name + "_zero_point"
        scale_name = weight_name + "_scale"

        # Update packed weight, zero point, and scale initializers
        zero_scale_shape = [initializer.dims[channel_axis]]
        scale_initializer = onnx.helper.make_tensor(scale_name, initializer.data_type, zero_scale_shape,
                                                    np.hstack(scale_list).tolist())
        zero_initializer = onnx.helper.make_tensor(zp_name, weight_qType, zero_scale_shape,
                                                   np.hstack(zero_point_list).tolist())

        self.model.initializer_extend([scale_initializer, zero_initializer])

        if not keep_float_weight:
            quantized_weights = np.asarray(
                quantized_weights,
                dtype=onnx.helper.tensor_dtype_to_np_dtype(weight_qType),
            ).reshape(initializer.dims)
            q_weight_initializer = onnx.numpy_helper.from_array(quantized_weights, q_weight_name)
            self.model.initializer_extend([q_weight_initializer])

        return q_weight_name, zp_name, scale_name

    def calc_graph_quant_params(self) -> dict[str, QDQTensorQuantParams]:
        """
        Calculates quantization parameters (scale/zero-point) for all tensors in the graph using each tensor's min/max range
        and optional user-provided overrides.
        """
        if self.tensors_range is None:
            return {}

        self.adjust_tensor_ranges()

        quantization_params = {}
        for tensor_name in self.tensors_range:
            td = self.tensors_range[tensor_name]
            if not isinstance(td, TensorData):
                raise TypeError(f"Unexpected type {type(td)} for {tensor_name!r}.")

            quant_overrides = self.tensor_quant_overrides.get_per_tensor_overrides(tensor_name, default_val={})
            original = self.calc_quant_params(tensor_name, td, quant_overrides)
            converted = None
            converted_recv_nodes = None

            if "convert" in quant_overrides:
                converted = self.calc_quant_params(tensor_name, td, quant_overrides["convert"])
                converted_recv_nodes = quant_overrides["convert"].get("recv_nodes")

            quantization_params[tensor_name] = QDQTensorQuantParams(original, converted, converted_recv_nodes)

        return quantization_params

    def calc_quant_params(self, tensor_name: str, tensor_data: TensorData,
                          quant_overrides: Dict[str, Any]) -> QuantizationParams:
        """
        Calculates quantization parameters (scale/zero-point) given a tensor's min/max range and optional
        user-provided overrides.
        """
        quant_type = self.activation_qType
        if tensor_name in self.quantized_tensor_type:
            quant_type = get_tensor_type_from_qType(self.quantized_tensor_type[tensor_name])
            logger.info(
                f"The type of tensor {tensor_name} is {self.quantized_tensor_type[tensor_name]}: using specific tensor precision"
            )

        rmin = quant_overrides.get("rmin", tensor_data.range_value[0])
        rmax = quant_overrides.get("rmax", tensor_data.range_value[1])

        if quant_type in ONNX_FP_QTYPES_LIST:
            reduce_range = self.is_activation_scaled  # If scale the activation, it will use a reduced range
            qmin, qmax = get_qmin_qmax_for_qType(quant_type, reduce_range=reduce_range)
            zero, scale = compute_scale_zp_fp(rmin, rmax, qmin, qmax, quant_type, self.calibrate_method,
                                              self.is_activation_symmetric, self.is_activation_scaled)
        else:
            if "quant_type" in quant_overrides:
                quant_type = quant_overrides["quant_type"].tensor_type

            if "scale" in quant_overrides and "zero_point" in quant_overrides:
                zero, scale = quant_overrides["zero_point"], quant_overrides["scale"]
            else:
                symmetric = quant_overrides.get("symmetric", self.is_activation_symmetric)
                reduce_range = quant_overrides.get("reduce_range", False)
                qmin, qmax = get_qmin_qmax_for_qType(quant_type, reduce_range=reduce_range, symmetric=symmetric)
                zero, scale = compute_scale_zp(rmin, rmax, qmin, qmax, quant_type, self.calibrate_method,
                                               self.is_activation_symmetric, self.use_power_of_2_scale)

        return QuantizationParams(zero_point=zero, scale=scale, quant_type=quant_type)

    def _quantize_normal_tensors(self) -> None:
        """
        Adds Q/DQ ops to tensors (activations and weights) that have been marked for quantization by op quantizers.
        """
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
            if tensor_name in self.quantized_value_map:
                continue

            if not tensor_info.is_shared:
                # Quantize the input
                initializer = find_by_name(tensor_name, self.model.initializer())
                if initializer:
                    self._add_qdq_pair_for_initializer(initializer, tensor_info.tensor_type, tensor_info.axis)
                else:
                    tensor_qparam_initializers = self._make_tensor_scale_zp_initializers(tensor_name)
                    if not tensor_qparam_initializers:
                        raise ValueError(
                            f"Quantization parameters are not specified for param {tensor_name}. "
                            "In static mode quantization params for inputs and outputs of nodes to be quantized are required."
                        )

                    if tensor_qparam_initializers.converted is None:
                        # Normal case: <producer> --> Q --> DQ --> <consumers>
                        self._add_qdq_pair_for_activation(
                            tensor_name,
                            tensor_qparam_initializers.original.scale.name,
                            tensor_qparam_initializers.original.zero_point.name,
                            data_type=tensor_info.data_type,
                        )
                    else:
                        # Conversion case: <producer> ---> Q1 -+-> DQ1 --> <consumers of original type>
                        #                                      |
                        #                                      +-> DQ1' --> Q2 --> DQ2 --> <consumers of converted type>
                        assert tensor_info.data_type == tensor_qparam_initializers.original.scale.data_type
                        self._add_qdq_ops_for_converted_activation(
                            tensor_name,
                            tensor_qparam_initializers.original.scale.name,
                            tensor_qparam_initializers.original.zero_point.name,
                            tensor_info.data_type,
                            tensor_qparam_initializers.converted.scale.name,
                            tensor_qparam_initializers.converted.zero_point.name,
                            tensor_qparam_initializers.converted_recv_nodes,
                        )

                del self.tensors_to_quantize[tensor_name]

    def quantize_initializer(
        self,
        weight: onnx.TensorProto,
        qType: onnx.TensorProto.DataType,
        reduce_range: bool = False,
        keep_float_weight: bool = False,
    ) -> tuple[str, str, str]:
        """
        :param weight: TensorProto initializer
        :param qType: type to quantize to
        :param keep_float_weight: Whether to quantize the weight. In some cases, we only want to qunatize scale and zero point.
                                  If keep_float_weight is False, quantize the weight, or don't quantize the weight.
        :return: quantized weight name, zero point name, scale name
        """
        # Find if this input is already quantized
        if weight.name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight.name].original
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        q_weight_name, zp_name, scale_name = self.quantize_initializer_impl(weight, qType, reduce_range,
                                                                            keep_float_weight)

        # Log entry for this quantized weight
        quantized_value = QuantizedValue(
            weight.name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight.name] = QDQTensorQuantizedValue(quantized_value, None, None)
        return q_weight_name, zp_name, scale_name

    def quantize_weight_per_channel(
        self,
        weight_name: str,
        weight_qType: onnx.TensorProto.DataType,
        channel_axis: int,
        reduce_range: bool = True,
        keep_float_weight: bool = False,
    ) -> tuple[str, str, str]:
        # Find if this input is already quantized
        if weight_name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight_name].original
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel_impl(weight_name, weight_qType,
                                                                                   channel_axis, reduce_range,
                                                                                   keep_float_weight)
        quantized_value = QuantizedValue(
            weight_name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight_name] = QDQTensorQuantizedValue(quantized_value, None, None)

        return q_weight_name, zp_name, scale_name

    def _make_scale_zp_initializers(self,
                                    param_name: str,
                                    params: QuantizationParams,
                                    init_name_suffix: str = "") -> QDQScaleZpInitializers:
        """
        Creates and returns scale and zero-point initializers for the given quantization params. The initializers are
        named:
            - {param_name}_zero_point{init_name_suffix}
            - {param_name}_scale{init_name_suffix}
        """
        zero_point_values = np.array([params["zero_point"]])
        if not hasattr(params["scale"], "dtype") or params["scale"].dtype not in (np.float32, np.float16):
            raise ValueError(f"Unexpected type {type(params['scale'])} and param_name={param_name!r}")
        scale_values = np.array([params["scale"]])
        assert scale_values.dtype != np.float64
        zero_point_type = params.data.get("quant_type", self.activation_qType)

        zero_point_shape: List[Any] = []
        zero_point_name = param_name + "_zero_point" + init_name_suffix
        scale_shape: List[Any] = []
        scale_name = param_name + "_scale" + init_name_suffix

        # Add initializers to model
        init_zp = onnx.helper.make_tensor(zero_point_name, zero_point_type, zero_point_shape,
                                          zero_point_values.ravel().tolist())
        self.model.add_initializer(init_zp)

        if scale_values.dtype == np.float32:
            scale_type = onnx_proto.TensorProto.FLOAT
        elif scale_values.dtype == np.float16:
            scale_type = onnx_proto.TensorProto.FLOAT16
        else:
            raise ValueError(f"Unexpected dtype={scale_values.dtype} for param_name={param_name!r}")
        init_scale = onnx.helper.make_tensor(scale_name, scale_type, scale_shape, scale_values.reshape((-1, )).tolist())
        self.model.add_initializer(init_scale)

        return QDQScaleZpInitializers(init_scale, init_zp)
