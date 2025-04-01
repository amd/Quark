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
from typing import Any, List, Dict, Optional

import onnx
from onnx import TensorProto, ModelProto
import onnx.numpy_helper
from onnxruntime.quantization.quant_utils import (
    QuantizationMode,
    add_dequant_output_suffix,
    add_dequant_suffix,
    add_quant_input_suffix,
    find_by_name,
)

from ..quant_utils import (
    __producer__,
    __version__,
    VitisQuantFormat,
    BFPFIX_OP_NAME,
    BFPFIX_OP_DEFAULT_ATTRS,
    MXFIX_OP_NAME,
    MXFIX_OP_DEFAULT_ATTRS,
    COP_DOMAIN,
    get_annotate_tensors,
    get_qdq_to_remove,
    remove_nodes,
    modified_annotate_input,
)
from .qdq_quantizer import VitisQDQQuantizer
from ..registry import CreateQDQQuantizer

logger = ScreenLogger(__name__)


class VitisBFPQuantizer(VitisQDQQuantizer):

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
        super().__init__(model, per_channel, reduce_range, mode, static, weight_qType, activation_qType, tensors_range,
                         nodes_to_quantize, nodes_to_exclude, op_types_to_quantize, calibrate_method,
                         quantized_tensor_type, extra_options)

        self.int32_bias = False
        if extra_options is not None and "Int32Bias" in extra_options and extra_options["Int32Bias"]:
            self.int32_bias = extra_options["Int32Bias"]
            logger.warning("Will not quantize Bias since do not support Int32Bias in BFP/MX mode")

        self.is_activation_symmetric = True
        if "ActivationSymmetric" in self.extra_options and not self.extra_options["ActivationSymmetric"]:
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

            if tensor_name in self.quantized_value_map:
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
        #     self._quantize_bias_tensors()

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
