#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Align scale and zero-point of Q/DQ nodes for Concat, MaxPool, AveragePool, GlobalAveragePool, Pad, Slice, Transpose, Reshape.

This pass aligns quantization parameters (scale and zero_point) between the inputs and
outputs of selected op types by copying Q/DQ info (e.g. output Q/DQ to inputs for Concat,
or input Q/DQ to outputs for MaxPool/AveragePool/GlobalAveragePool/Slice) so that downstream
compiler constraints are met. Supports both power-of-two and float scales.
"""

from typing import Any

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import ModelProto, NodeProto, TensorProto
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.common.utils.log import ScreenLogger
from quark.onnx.quantization.quant_utils import (
    DEQUANT_OP_TYPES,
    QUANT_OP_TYPES,
)
from quark.onnx.utils.model_utils import ONNXQuantizedModel
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

# Q/DQ node types (QuantizeLinear and DequantizeLinear, including custom variants).
QDQ_OP_TYPES = QUANT_OP_TYPES + DEQUANT_OP_TYPES

logger = ScreenLogger(__name__)


@register_pass
class ONNXAlignScalePass(ONNXPass):
    """
    Adapter pass that aligns scale and zero-point of QuantizeLinear/DequantizeLinear
    nodes for Concat, MaxPool, AveragePool, GlobalAveragePool, Pad, Slice, Transpose,
    and Reshape to meet compiler constraints for float-scale quantized models.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Return the default configuration for this pass.

        Returns:
            dict[str, PassConfigParam]: Config dict containing ``align_scale``, which may be
            a bool, a single op type name (e.g. ``"Concat"``, ``"MaxPool"``), or a list of names.
        """
        config = {
            "align_scale": PassConfigParam(
                type_=str | list[str],
                default_value="",
                required=True,
                description="Enable align scale; can be bool, op type name (e.g. Concat, MaxPool, AveragePool, GlobalAveragePool, Slice), or list of op type names.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_align_scale(
        self,
        model: ModelProto,
        max_loop_num: int = 5,
        align_concat: bool = False,
        align_max_pool: bool = False,
        align_average_pool: bool = False,
        align_global_average_pool: bool = False,
        align_pad: bool = False,
        align_slice: bool = False,
        align_transpose: bool = False,
        align_reshape: bool = False,
    ) -> Any:
        """
        Align scale and zero-point for selected op types so that each op's inputs and
        outputs share the same Q/DQ parameters. Supports power-of-two and float scales.
        Iterates until no further change or ``max_loop_num`` rounds.
        """
        manager = QuantInfoManager(model)

        while manager.alignment_changed and (manager.num_alignment_rounds < max_loop_num):
            manager.num_alignment_rounds += 1
            if manager.num_alignment_rounds == max_loop_num:
                logger.warning(
                    "Align-scale reached max rounds (%s). Check model for inconsistent Q/DQ.",
                    max_loop_num,
                )
            manager.alignment_changed = False
            logger.info("Adjust the quantize info to meet the compiler constraints")

            if align_concat:
                manager._align_concat()

            if align_max_pool:
                manager._align_max_pool()

            if align_average_pool:
                manager._align_average_pool()

            if align_global_average_pool:
                manager._align_global_average_pool()

            if align_pad:
                manager._align_pad()

            if align_slice:
                manager._align_slice()

            if align_transpose:
                manager._align_transpose()

            if align_reshape:
                manager._align_reshape()

        return manager.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Run the align-scale pass on the given model according to ``config``.

        Args:
            model: The ONNX model to modify.
            config: Runtime config; must contain ``align_scale`` (bool, str, or list of str).

        Returns:
            The model after aligning Q/DQ scale/zero-point for the selected op types.
        """
        if "align_scale" in config and config["align_scale"]:
            model = ONNXModel(model)

            align_concat = False
            align_max_pool = False
            align_average_pool = False
            align_global_average_pool = False
            align_pad = False
            align_slice = False
            align_transpose = False
            align_reshape = False

            if isinstance(config["align_scale"], str):
                if config["align_scale"] == "Concat":
                    align_concat = True
                elif config["align_scale"] == "MaxPool":
                    align_max_pool = True
                elif config["align_scale"] == "AveragePool":
                    align_average_pool = True
                elif config["align_scale"] == "GlobalAveragePool":
                    align_global_average_pool = True
                elif config["align_scale"] == "Pad":
                    align_pad = True
                elif config["align_scale"] == "Slice":
                    align_slice = True
                elif config["align_scale"] == "Transpose":
                    align_transpose = True
                elif config["align_scale"] == "Reshape":
                    align_reshape = True

            elif isinstance(config["align_scale"], list):
                for node_type in config["align_scale"]:
                    if node_type == "Concat":
                        align_concat = True
                    elif node_type == "MaxPool":
                        align_max_pool = True
                    elif node_type == "AveragePool":
                        align_average_pool = True
                    elif node_type == "GlobalAveragePool":
                        align_global_average_pool = True
                    elif node_type == "Pad":
                        align_pad = True
                    elif node_type == "Slice":
                        align_slice = True
                    elif node_type == "Transpose":
                        align_transpose = True
                    elif node_type == "Reshape":
                        align_reshape = True

            model = self._onnx_align_scale(
                model,
                max_loop_num=5,
                align_concat=align_concat,
                align_max_pool=align_max_pool,
                align_average_pool=align_average_pool,
                align_global_average_pool=align_global_average_pool,
                align_pad=align_pad,
                align_slice=align_slice,
                align_transpose=align_transpose,
                align_reshape=align_reshape,
            )
            model = model.model

        else:
            logger.warning("align_scale is missing or falsy; enable align_scale to run this pass.")
        return model


class QuantInfoManager:
    """
    Manages scale and zero-point of Q/DQ nodes and aligns them for selected op types
    (Concat, MaxPool, AveragePool, GlobalAveragePool, Pad, Slice, Transpose, Reshape)
    by copying one side's Q/DQ params to the other.
    """

    def __init__(self, model: ModelProto) -> None:
        # Expects model to be an ONNXModel-like wrapper with .model and get_initializer.
        self.model = model
        self.alignment_changed = True
        self.num_alignment_rounds = 0
        self._qdq_parser = ONNXQuantizedModel(self.model.model)

    def _get_quant_info(self, node: NodeProto) -> list[TensorProto]:
        """Return [scale_tensor, zero_point_tensor] for a Q/DQ node (inputs at index 1 and 2)."""
        assert node.op_type in QDQ_OP_TYPES
        return [
            self.model.get_initializer(node.input[1]),
            self.model.get_initializer(node.input[2]),
        ]

    def _set_quant_info(self, node: NodeProto, qdq_info: list[TensorProto]) -> None:
        """Overwrite this node's scale and zero_point initializers with the given Q/DQ info."""
        assert node.op_type in QDQ_OP_TYPES
        scale_init = self.model.get_initializer(node.input[1])
        scale_init.CopyFrom(qdq_info[0])
        scale_init.name = node.input[1]
        zp_init = self.model.get_initializer(node.input[2])
        zp_init.CopyFrom(qdq_info[1])
        zp_init.name = node.input[2]

    def _quant_info_equal(self, qdq_info_left: list[TensorProto], qdq_info_right: list[TensorProto]) -> bool:
        """Return True if the two Q/DQ infos (scale and zero_point) are equal."""
        scale_left = onnx.numpy_helper.to_array(qdq_info_left[0])
        scale_right = onnx.numpy_helper.to_array(qdq_info_right[0])
        zp_left = onnx.numpy_helper.to_array(qdq_info_left[1])
        zp_right = onnx.numpy_helper.to_array(qdq_info_right[1])
        return np.array_equal(scale_left, scale_right) and np.array_equal(zp_left, zp_right)

    def _copy_output_qinfo_to_inputs(self, op_types: list[str]) -> None:
        """
        For each node of the given op types, copy the output Q/DQ scale and zero_point
        to all input Q/DQ nodes so that inputs match the output quant params.
        """
        for node in self.model.model.graph.node:
            if node.op_type not in op_types:
                continue

            qdq_struct = self._qdq_parser.find_target_node_qdqs(node)
            if not (qdq_struct["input_qdqs"] and qdq_struct["output_qdqs"]):
                continue

            out_q_node, out_dq_node = qdq_struct["output_qdqs"][0]
            if out_q_node is None or out_dq_node is None:
                continue

            ref_qdq_info = self._get_quant_info(out_q_node)
            round_changed = False

            for in_dq_node, in_q_node in qdq_struct["input_qdqs"]:
                if in_dq_node is None or in_q_node is None:
                    continue
                if self._quant_info_equal(ref_qdq_info, self._get_quant_info(in_dq_node)):
                    continue

                self._set_quant_info(in_dq_node, ref_qdq_info)
                if (in_q_node.input[1] != in_dq_node.input[1]) or (in_q_node.input[2] != in_dq_node.input[2]):
                    self._set_quant_info(in_q_node, ref_qdq_info)
                round_changed = True

            if round_changed:
                self.alignment_changed = True
                logger.info(f"Have aligned {node.op_type} node {node.name} inputs")

    def _copy_input_qinfo_to_outputs(self, op_types: list[str]) -> None:
        """
        For each node of the given op types, copy the input Q/DQ scale and zero_point
        to all output Q/DQ nodes so that outputs match the input quant params.
        """
        for node in self.model.model.graph.node:
            if node.op_type not in op_types:
                continue

            qdq_struct = self._qdq_parser.find_target_node_qdqs(node)
            if not (qdq_struct["input_qdqs"] and qdq_struct["output_qdqs"]):
                continue

            in_dq_node, in_q_node = qdq_struct["input_qdqs"][0]
            if in_dq_node is None or in_q_node is None:
                continue

            ref_qdq_info = self._get_quant_info(in_dq_node)
            round_changed = False

            for out_q_node, out_dq_node in qdq_struct["output_qdqs"]:
                if out_q_node is None or out_dq_node is None:
                    continue
                if self._quant_info_equal(ref_qdq_info, self._get_quant_info(out_q_node)):
                    continue

                self._set_quant_info(out_q_node, ref_qdq_info)
                if (out_q_node.input[1] != out_dq_node.input[1]) or (out_q_node.input[2] != out_dq_node.input[2]):
                    self._set_quant_info(out_dq_node, ref_qdq_info)
                round_changed = True

            if round_changed:
                self.alignment_changed = True
                logger.info(f"Have aligned {node.op_type} node {node.name} outputs")

    def _align_concat(self) -> None:
        """Align Concat: copy output Q/DQ scale and zero_point to all input Q/DQ nodes."""
        self._copy_output_qinfo_to_inputs(["Concat"])

    def _align_max_pool(self) -> None:
        """Align MaxPool: copy input Q/DQ info to output Q/DQ."""
        self._copy_input_qinfo_to_outputs(["MaxPool"])

    def _align_average_pool(self) -> None:
        """Align AveragePool: copy input Q/DQ info to output Q/DQ."""
        self._copy_input_qinfo_to_outputs(["AveragePool"])

    def _align_global_average_pool(self) -> None:
        """Align GlobalAveragePool: copy input Q/DQ info to output Q/DQ."""
        self._copy_input_qinfo_to_outputs(["GlobalAveragePool"])

    def _align_pad(self) -> None:
        """Align Pad: copy output Q/DQ info to input Q/DQ."""
        self._copy_output_qinfo_to_inputs(["Pad"])

    def _align_slice(self) -> None:
        """Align Slice: copy input Q/DQ info to output Q/DQ (handles multiple outputs)."""
        self._copy_input_qinfo_to_outputs(["Slice"])

    def _align_transpose(self) -> None:
        """Align Transpose: copy output Q/DQ info to input Q/DQ."""
        self._copy_output_qinfo_to_inputs(["Transpose"])

    def _align_reshape(self) -> None:
        """Align Reshape: copy output Q/DQ info to input Q/DQ."""
        self._copy_output_qinfo_to_inputs(["Reshape"])
