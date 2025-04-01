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

from onnx import ModelProto
from onnxruntime.quantization.quant_utils import QuantizationMode, ms_domain

from .qdq_quantizer import QDQQuantizer
from ..quant_utils import (__producer__, __version__, get_annotate_tensors, get_qdq_to_remove, remove_nodes,
                           modified_annotate_input)
from ..registry import CreateNPUTransformerQDQQuantizer

logger = ScreenLogger(__name__)


class QDQNPUTransformerQuantizer(QDQQuantizer):

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
        if self.qdq_op_domain == ms_domain:
            self.model.set_opset_import(ms_domain, 1)

        return self.model.model
