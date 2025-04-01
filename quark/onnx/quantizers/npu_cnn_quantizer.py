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
from quark.shares.utils.log import ScreenLogger, log_errors
from typing import List, Any, Dict, Optional

import onnx
import onnx.numpy_helper
from onnx import TensorProto, ModelProto
from onnx import onnx_pb as onnx_proto
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.qdq_quantizer import QDQBiasQuantInfo
from onnxruntime.quantization.quant_utils import (QuantType, QuantizationMode, DEQUANT_OP_NAME,
                                                  add_dequant_output_suffix, add_dequant_suffix,
                                                  add_quant_output_suffix, add_quant_suffix, find_by_name)

from ..quant_utils import (__producer__, __version__, get_annotate_tensors, get_qdq_to_remove, remove_nodes,
                           modified_annotate_input)
from .qdq_quantizer import VitisQDQQuantizer
from ..registry import CreateNPUCnnQDQQuantizer
from ..refine import adjust_quantize_info
from ..simulate_dpu import simulate_transforms

logger = ScreenLogger(__name__)


class VitisQDQNPUCNNQuantizer(VitisQDQQuantizer):

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
        self.calibrate_method = calibrate_method
        self.model = ONNXModel(model)
        self.nodes_to_exclude = nodes_to_exclude
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
        self.int32_bias = False if extra_options is None or "Int32Bias" not in extra_options else extra_options[
            "Int32Bias"]

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

    def quantize_bias_tensor(self,
                             node_name: str,
                             bias_name: str,
                             input_name: str,
                             weight_name: str,
                             beta: float = 1.0) -> None:
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16):
                if self.quantize_bias:
                    if bias_name not in self.bias_to_quantize:
                        if self.int32_bias:
                            self.bias_to_quantize[bias_name] = QDQBiasQuantInfo(node_name, input_name, weight_name,
                                                                                beta)
                        else:
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
