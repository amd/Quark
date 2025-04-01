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
from quark.shares.utils.log import ScreenLogger
from typing import List, Any, Dict, Optional

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import ModelProto, TensorProto
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.qdq_quantizer import QDQTensorQuantizedValue
from onnxruntime.quantization.quant_utils import (
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
    ms_domain,
)

from .qdq_quantizer import VitisQDQQuantizer
from ..quant_utils import (
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
from ..registry import CreateNPUCnnQDQQuantizer
from ..refine import align_quantize_info
from ..simulate_dpu import simulate_transforms

logger = ScreenLogger(__name__)


class VitisExtendedQuantizer(VitisQDQQuantizer):

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
        self.model = ONNXModel(model)
        self.nodes_to_exclude = nodes_to_exclude
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
        self.use_qdq_vitis_custom_ops = True
        if "UseQDQVitisCustomOps" in self.extra_options:
            self.use_qdq_vitis_custom_ops = self.extra_options["UseQDQVitisCustomOps"]

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
        if self.qdq_op_domain == ms_domain:
            self.model.set_opset_import(ms_domain, 1)

        return self.model.model

    def try_replacing_upstream_output(self, upstream_output_name: str, output_name: str) -> bool:
        '''
        if (output_name in self.quantization_params.keys() and len(
                self.model.input_name_to_nodes()[upstream_output_name]) == 1 and
                not self.model.is_graph_output(upstream_output_name) and
                not self.model.is_graph_input(upstream_output_name)):
            if upstream_output_name in self.tensors_to_quantize:
                del self.tensors_to_quantize[upstream_output_name]
            return True
        '''
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
            if zp_type in ONNX_BFP_QTYPES_LIST:
                raise ValueError("Per-Channel does not support BFP data types and its variants.")
            if self.opset_version < 13:
                raise ValueError("Per-Channel support with QDQ format requires onnx opset version 13 or above.")
            q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel(
                weight_name, zp_type, axis, keep_float_weight=self.add_qdq_pair_to_weight)
        else:
            q_weight_name, zp_name, scale_name = self.quantize_initializer(
                weight_proto,
                zp_type,
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
                    self.quantized_value_map[tensor_name] = QDQTensorQuantizedValue(quantized_value, None, None)
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
            self.quantized_value_map[tensor_name] = QDQTensorQuantizedValue(quantized_value, None, None)

    def _quantize_normal_tensors(self) -> None:
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():

            if tensor_name in self.quantized_value_map:
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
                    tensor_qparam_initializers = self._make_tensor_scale_zp_initializers(tensor_name)
                    if not tensor_qparam_initializers:
                        raise ValueError(
                            f"Quantization parameters are not specified for param {tensor_name}. "
                            "In static mode quantization params for inputs and outputs of nodes to be quantized are required."
                        )
                    if tensor_qparam_initializers.converted is None:
                        # Normal case: <producer> --> Q --> DQ --> <consumers>
                        self._add_fn_pair_for_activation(
                            tensor_name,
                            tensor_qparam_initializers.original.scale.name,
                            tensor_qparam_initializers.original.zero_point.name,
                            zp_type,
                        )
                    else:
                        raise ValueError("Do not support conversion case.")

                del self.tensors_to_quantize[tensor_name]

    def _quantize_sharing_param_tensors(self) -> None:
        """
        Adds Q/DQ ops to tensors that have been marked for quantization by op quantizers.
        Only operates on tensors that want to use the quantization parameter initializers from an upstream tensor.
        For example, a Transpose node's output tensor will typically want to use the same quantization parameter
        initializers as the Transpose node's input.
        """
        while self.tensors_to_quantize:
            for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
                quant_provider = tensor_info.quant_para_provider
                if quant_provider and quant_provider.input_name in self.quantized_value_map:
                    del self.tensors_to_quantize[tensor_name]

                    quantized_value = self.quantized_value_map[quant_provider.input_name].get_for_consumer(
                        quant_provider.node_name)
                    if self.is_input_a_initializer(tensor_name):
                        raise ValueError("Quantization parameter shared mode is not supported for weight yet")

                    # Need to check if this tensor's quant_type is converted for some consumers.
                    # If so, create new scale/zp initializers for these consumers.
                    converted_qparam_inits = None

                    if converted_qparam_inits is None:
                        # Normal case: <producer> --> Q_shared --> DQ_shared --> <consumers>
                        self._add_fn_pair_for_activation(tensor_name, quantized_value.scale_name,
                                                         quantized_value.zp_name)
                    else:
                        # Conversion case: <producer> ---> Q_shared -+-> DQ_shared --> <consumers of original type>
                        #                                            |
                        #                                            +-> DQ_shared' --> Q2 --> DQ2 --> <consumers of converted type>
                        raise ValueError("Do not support conversion case.")

    def _quantize_bias_tensors(self) -> None:
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
