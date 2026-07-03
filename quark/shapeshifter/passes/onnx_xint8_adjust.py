#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""XINT8 quantize-position adjustment pass for DPU/NPU compiler constraints.

This pass adjusts fixed-point positions (derived from Q/DQ scales via scale2pos/pos2scale)
so that shift_cut, shift_bias, shift_read, shift_write, hard_sigmoid, and shift_swish
stay within DPU/NPU limits. It operates on quantized ONNX models that use scale (and thus
position) representation. Each adjustment clamps the corresponding expression and updates
the relevant Q/DQ scale initializers.
"""

from typing import Any

import numpy as np
import onnx
from onnx import ModelProto, NodeProto
from onnxruntime.quantization.onnx_quantizer import tensor_proto_to_array

from quark.common.utils.log import ScreenLogger
from quark.onnx.quantization.quant_utils import (
    DEQUANT_OP_TYPES,
    QUANT_OP_TYPES,
    annotate_op_type,
    avg_pool_op_type,
    check_hard_sigmoid_condition,
    is_node_needs_annotated,
    pos2scale,
    scale2pos,
)
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

# Q/DQ node types (QuantizeLinear and DequantizeLinear, including custom variants).
REFINE_OP_TYPES = QUANT_OP_TYPES + DEQUANT_OP_TYPES

# Suffix used in graph to match Q output name to DQ input name.
postfix = "_Output"

logger = ScreenLogger(__name__)


@register_pass
class ONNXXInt8AdjustPass(ONNXPass):
    """
    Adapter pass that adjusts XINT8 quantize positions so that shift_cut, shift_bias,
    concat, pool, pad, slice, shift_read, shift_write, hard_sigmoid, and shift_swish satisfy DPU/NPU constraints.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Return the default configuration for this pass.

        Returns:
            dict[str, PassConfigParam]: Config dict containing ``xint8_adjust`` (bool),
            which enables or disables the full set of adjustments.
        """
        config = {
            "xint8_adjust": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Enable XINT8 adjustment; accepts bool value.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_xint8_adjust(
        self,
        model: ModelProto,
        max_loop_num: int = 5,
        align_concat: bool = True,
        align_pool: bool = True,
        align_pad: bool = True,
        align_slice: bool = True,
        adjust_shift_cut: bool = True,
        adjust_shift_bias: bool = True,
        adjust_shift_read: bool = True,
        adjust_shift_write: bool = True,
        adjust_hard_sigmoid: bool = True,
        adjust_shift_swish: bool = True,
    ) -> ModelProto:
        """
        Run position adjustments in a loop until no change or ``max_loop_num`` rounds.
        Order: concat, pool, pad, slice, shift_read, shift_write, shift_cut, shift_bias, hard_sigmoid, shift_swish.
        """
        manager = QuantPosManager(model)

        while manager.has_change and (manager.adjust_loop_count < max_loop_num):
            manager.adjust_loop_count += 1
            if manager.adjust_loop_count == max_loop_num:
                logger.warning(
                    "XINT8 adjust reached max rounds (%s). Check model for repeated adjustments.",
                    max_loop_num,
                )
            manager.has_change = False
            logger.info("Adjust the quantize info to meet the compiler constraints")

            if align_concat:
                manager._align_concat()

            if align_pool:
                manager._align_pool()

            if align_pad:
                manager._align_pad()

            if align_slice:
                manager._align_slice()

            if adjust_shift_read:
                manager._adjust_shift_read()

            if adjust_shift_write:
                manager._adjust_shift_write()

            if adjust_shift_cut:
                manager._adjust_shift_cut()

            if adjust_shift_bias:
                manager._adjust_shift_bias()

            if adjust_hard_sigmoid:
                manager._adjust_hard_sigmoid()

            if adjust_shift_swish:
                manager._adjust_shift_swish()

        return manager.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Run the XINT8 adjust pass on the given model according to ``config``.

        Args:
            model: The ONNX model to modify (XINT8 quantized with Q/DQ).
            config: Runtime config; must contain ``xint8_adjust`` (bool).

        Returns:
            The model after adjusting quantize positions to meet DPU/NPU constraints.
        """
        if "xint8_adjust" in config and config["xint8_adjust"]:
            model = self._onnx_xint8_adjust(
                model,
                max_loop_num=5,
                align_concat=True,
                align_pool=True,
                align_pad=True,
                align_slice=True,
                adjust_shift_cut=True,
                adjust_shift_bias=True,
                adjust_shift_read=True,
                adjust_shift_write=True,
                adjust_hard_sigmoid=True,
                adjust_shift_swish=True,
            )
        else:
            logger.warning("xint8_adjust is missing or falsy; enable xint8_adjust to run this pass.")
        return model


class QuantPosManager:
    """
    Manages Q/DQ scale and derived position (pos) for the graph, and applies
    shift_cut, shift_bias, shift_read, shift_write, hard_sigmoid, and shift_swish
    adjustments so that all expressions stay within DPU/NPU bounds.
    """

    def __init__(self, model: ModelProto) -> None:
        self.model: ModelProto = model
        self.has_change = True
        self.adjust_loop_count = 0

    def _get_scale(self, node: NodeProto) -> Any:
        """Return the scale value (float) for a Q/DQ node from its scale initializer (input[1])."""
        for i in self.model.graph.initializer:
            if i.name == node.input[1]:
                if i.float_data:
                    return i.float_data[0]
                elif i.raw_data:
                    return np.frombuffer(i.raw_data, dtype=np.float32).tolist()[0]
                else:
                    # Handle float16 scale
                    val = tensor_proto_to_array(i).tolist()
                    return val[0] if isinstance(val, list) else val
        raise ValueError("DequantizeLinear and QuantizeLinear do not have scale.")

    def _set_scale(self, node: NodeProto, new_scale: float) -> None:
        """Overwrite the scale initializer (node.input[1]) with new_scale."""
        for i in self.model.graph.initializer:
            if i.name == node.input[1]:
                if i.float_data:
                    if i.float_data[0] != new_scale:
                        i.float_data[0] = new_scale
                elif i.raw_data:
                    if np.frombuffer(i.raw_data, dtype=np.float32).tolist()[0] != new_scale:
                        np.frombuffer(i.raw_data, dtype=np.float32).tolist()[0] = new_scale
                else:
                    # Handle float16 scale
                    ort_val = tensor_proto_to_array(i).dtype
                    new_val = np.array(new_scale).astype(ort_val.dtype)
                    new_init = onnx.numpy_helper.from_array(new_val, name=i.name)
                    i.CopyFrom(new_init)

    def _get_pos(self, node: NodeProto) -> Any:
        """Return the fixed-point position (int) for a Q/DQ node from its scale."""
        if node.op_type in REFINE_OP_TYPES:
            return scale2pos(self._get_scale(node))
        return None

    def _set_pos(self, node: NodeProto, new_pos: int) -> None:
        """Set the Q/DQ node (and its paired Q/DQ) to the given position by updating scale to pos2scale(new_pos)."""
        new_scale = pos2scale(new_pos)
        if node.op_type in QUANT_OP_TYPES:
            self._set_scale(node, new_scale)
            # Keep paired DQ (name matches Q.output minus postfix) in sync
            if node.output:
                for n in self.model.graph.node:
                    if n.name == node.output[0].strip(postfix) and n.op_type in DEQUANT_OP_TYPES:
                        self._set_scale(n, new_scale)
        elif node.op_type in DEQUANT_OP_TYPES:
            self._set_scale(node, new_scale)
            # Keep paired Q (name matches DQ.input[0] minus postfix) in sync
            for n in self.model.graph.node:
                if n.name == node.input[0].strip(postfix) and n.op_type in QUANT_OP_TYPES:
                    self._set_scale(n, new_scale)

    def _find_node_name(self, name: str) -> Any:
        """Return the name of the Q/DQ node whose output tensor is the given name."""
        for node in self.model.graph.node:
            if len(node.output) > 0 and node.output[0] == name and node.op_type in REFINE_OP_TYPES:
                return node.name
        return None

    def _get_ipos_name(self, node: NodeProto) -> Any:
        """Return the name of the Q/DQ node that provides the main input (input[0]) to this node; for Pool, may skip one op."""
        if len(node.input) > 0:
            i_name = node.input[0]
            ipos_name = self._find_node_name(i_name)
            if ipos_name:
                return ipos_name
            op_type = node.op_type
            for n in self.model.graph.node:
                if len(n.output) >= 1 and n.output[0] == i_name and op_type in avg_pool_op_type:
                    i_name = n.input[0]
                    ipos_name = self._find_node_name(i_name)
                    if ipos_name:
                        return ipos_name
        else:
            return None

    def _get_ipos_name_by_id(self, node: NodeProto, input_id: int = 0) -> Any:
        """Return the name of the Q/DQ node that provides node.input[input_id]."""
        if len(node.input) > input_id:
            i_name = node.input[input_id]
            return self._find_node_name(i_name)
        return None

    def _get_node_by_name(self, node_name: str) -> Any:
        """Return the graph node with the given name, or None."""
        for node in self.model.graph.node:
            if node.name == node_name:
                return node
        return None

    def _get_pos_by_name(self, name: str) -> Any:
        """Return (position, node) for the Q/DQ node with the given name; (None, None) if not found."""
        for node in self.model.graph.node:
            if node.op_type in REFINE_OP_TYPES and node.name == name:
                return self._get_pos(node), node
        return None, None

    def _find_o_name(self, o_name: str) -> Any:
        """Return the name of the Q/DQ node that consumes the given tensor as input[0]."""
        for node in self.model.graph.node:
            if len(node.input) >= 1 and node.input[0] == o_name and node.op_type in REFINE_OP_TYPES:
                return node.name
        return None

    def _get_opos_name(self, node: NodeProto) -> Any:
        """Return the name of the Q/DQ node that consumes this node's output; may follow Pool/HardSigmoid then Mul for annotate."""

        def _is_node_connected(pre_node_type: str, node: NodeProto) -> bool:
            if (
                pre_node_type in avg_pool_op_type + ["HardSigmoid"]
                and node.op_type == "Mul"
                or pre_node_type in annotate_op_type
                and is_node_needs_annotated(self.model, node)
            ):
                return True
            return False

        o_name = node.output[0]
        opos_name = self._find_o_name(o_name)
        if opos_name:
            return opos_name
        pre_node_type = node.op_type
        for n in self.model.graph.node:
            if (len(n.input) >= 1 and n.input[0] == o_name) and _is_node_connected(pre_node_type, n):
                o_name = n.output[0]
                opos_name = self._find_o_name(o_name)
                if opos_name:
                    return opos_name
        return None

    def _get_wpos_name(self, node: NodeProto) -> Any:
        """Return the name of the Q/DQ node that provides the weight (input[1]) for Conv/Gemm."""
        if len(node.input) > 1:
            return self._find_node_name(node.input[1])
        return None

    def _get_bpos_name(self, node: NodeProto) -> Any:
        """Return the name of the Q/DQ node that provides the bias (input[2]) for Conv/Gemm, or None if no bias."""
        if len(node.input) > 2:
            return self._find_node_name(node.input[2])
        return None

    def _adjust_shift_cut(self) -> None:
        """Adjust shift_cut for Conv/Gemm: shift_cut = wpos + ipos - opos; clamp to [0, 16]; update wpos if needed."""
        for i, node in enumerate(self.model.graph.node):
            if node.op_type not in ["Conv", "Gemm"]:
                continue
            ipos_name = self._get_ipos_name(node)
            ipos, _ = self._get_pos_by_name(ipos_name)

            opos_name = self._get_opos_name(node)
            opos, _ = self._get_pos_by_name(opos_name)
            wpos_name = self._get_wpos_name(node)
            wpos, wpos_node = self._get_pos_by_name(wpos_name)

            # Adjust shift_cut
            min_sc = 0
            max_sc = 16
            if wpos is None or ipos is None or opos is None:
                logger.debug(f"Found a pos that is None. Shift cut of layer {node.name} has not taken effect.")
                continue
            sc = wpos + ipos - opos
            new_sc = None
            if sc < min_sc:
                new_sc = min_sc
            elif sc > max_sc:
                new_sc = max_sc

            if new_sc is not None:
                self.has_change = True
                new_wpos = new_sc + opos - ipos
                self._set_pos(wpos_node, new_wpos)
                logger.info(
                    f"Shift cut of layer {node.input[1]} is {int(sc)}. It exceeds range [{int(min_sc)}, {int(max_sc)}]. "
                    f"Modify wpos from {int(wpos)} to {int(new_wpos)}."
                )

    def _adjust_shift_bias(self) -> None:
        """Adjust shift_bias for Conv/Gemm with bias: shift_bias = wpos + ipos - bpos; clamp to [min_sb, 15] per DPU constraints."""
        for i, node in enumerate(self.model.graph.node):
            if node.op_type not in ["Conv", "Gemm"]:
                continue

            ipos_name = self._get_ipos_name(node)
            ipos, _ = self._get_pos_by_name(ipos_name)

            opos_name = self._get_opos_name(node)
            opos, _ = self._get_pos_by_name(opos_name)
            wpos_name = self._get_wpos_name(node)
            wpos, wpos_node = self._get_pos_by_name(wpos_name)
            bpos_name = self._get_bpos_name(node)
            if bpos_name:
                bpos, bpos_node = self._get_pos_by_name(bpos_name)
                # Adjust shift_bias
                if wpos is None or ipos is None or opos is None or bpos is None:
                    logger.debug(f"Found a pos that is None. Shift bias of layer {node.name} has not taken effect.")
                    continue
                shift_cut = wpos + ipos - opos
                # min_sb from NPU/DPU formula; LeakyRelu after this Conv may relax to 0
                min_sb = min(0, -(24 - (8 + shift_cut)))
                for n in self.model.graph.node:
                    if n.op_type == "LeakyRelu" and n.input[0] == node.output[0]:
                        min_sb = 0
                max_sb = 15
                shift_bias = wpos + ipos - bpos

                new_sb = None
                if shift_bias < min_sb:
                    new_sb = min_sb
                elif shift_bias > max_sb:
                    new_sb = max_sb

                if new_sb is not None:
                    self.has_change = True
                    new_bpos = wpos + ipos - new_sb
                    self._set_pos(self._get_node_by_name(bpos_name), new_bpos)
                    logger.info(
                        f"Shift bias of layer {node.input[2]} is {int(shift_bias)}. It exceeds range [{int(min_sb)}, {int(max_sb)}]. "
                        f"Modify bpos from {int(bpos)} to {int(new_bpos)}."
                    )

    def _adjust_shift_swish(self) -> None:
        """Adjust shift for Swish Mul: shift_swish = ipos0 + ipos1 - opos; clamp to [0, 15] per DPU constraints."""

        def _is_sigmoid_layer(node_input: str) -> bool:
            """Return True if the tensor is produced by a Swish's HardSigmoid (sigmoid) branch."""
            for node in self.model.graph.node:
                if check_hard_sigmoid_condition(node) and node.input[0] == node_input:
                    return True
            return False

        def _belong_to_swish(node0: NodeProto, node1: NodeProto) -> bool:
            """Return True if the two inputs form a Swish: Mul(x, sigmoid(x))."""
            if _is_sigmoid_layer(node0) or _is_sigmoid_layer(node1):
                return True
            return False

        for i, node in enumerate(self.model.graph.node):
            if node.op_type not in ["Mul"]:
                continue

            if len(node.input) != 2:
                continue

            # Confirm it is a Swish Mul (one input from HardSigmoid).
            if not _belong_to_swish(node.input[0], node.input[1]):
                continue

            opos_name = self._get_opos_name(node)
            opos, _ = self._get_pos_by_name(opos_name)

            if opos is not None:
                ipos0_name = self._get_ipos_name_by_id(node, 0)
                ipos0, _ = self._get_pos_by_name(ipos0_name)

                ipos1_name = self._get_ipos_name_by_id(node, 1)
                ipos1, _ = self._get_pos_by_name(ipos1_name)

                if ipos1 is None or ipos0 is None:
                    logger.warning(
                        f"Fail to get quantized position for layer {node.name} input, skip adjust_shift_swish for it."
                    )
                    continue

                min_sh, max_sh = 0, 15

                shift_swish = ipos0 + ipos1 - opos

                new_opos = opos
                if shift_swish < min_sh:
                    new_opos = ipos0 + ipos1 - min_sh
                elif shift_swish > max_sh:
                    new_opos = ipos0 + ipos1 - max_sh

                if new_opos != opos:
                    self.has_change = True
                    self._set_pos(self._get_node_by_name(opos_name), new_opos)
                    logger.info(
                        f"Shift Swish of layer {node.name} is {int(shift_swish)}({int(ipos0)}+{int(ipos1)}-{int(opos)}). It exceeds range [{int(min_sh)}, {int(max_sh)}]. "
                        f"Modify opos from {int(opos)} to {int(new_opos)}."
                    )
            else:
                logger.debug(
                    f"Fail to get quantized position for layer {node.name}(output:0), skip adjust shift swish for it."
                )

    def _adjust_hard_sigmoid(self) -> None:
        """Adjust input/output pos of HardSigmoid to satisfy DPU: input pos in [0,15], output pos >= 7, shift_sigmoid in [0,31]."""
        for i, node in enumerate(self.model.graph.node):
            if node.op_type not in ["HardSigmoid"]:
                continue
            if not check_hard_sigmoid_condition(node):
                continue
            ipos_name = self._get_ipos_name(node)
            ipos, _ = self._get_pos_by_name(ipos_name)

            opos_name = self._get_opos_name(node)
            opos, _ = self._get_pos_by_name(opos_name)

            if ipos is None or opos is None:
                logger.debug(
                    "Found a pos that is None. Adjust quantize info of HardSigmoid "
                    f"nodes of layer {node.name} has not taken effect."
                )
                continue

            new_ipos = ipos if ipos > 0 else 0
            new_ipos = new_ipos if new_ipos <= 15 else 15

            new_opos = opos if opos > 7 else 7
            shift_sigmoid = 14 + new_ipos - new_opos  # DPU constraint: shift_sigmoid in [0, 31]
            new_opos = new_opos if shift_sigmoid > 0 else 14 + new_ipos

            if new_ipos != ipos:
                self.has_change = True
                self._set_pos(self._get_node_by_name(ipos_name), new_ipos)
                logger.info(
                    f"Input quantize pos of HardSigmoid layer {node.input[0]} is {int(ipos)}, modify it to {int(new_ipos)} "
                    "to meet the DPU constraints."
                )

            if new_opos != opos:
                self.has_change = True
                self._set_pos(self._get_node_by_name(opos_name), new_opos)
                logger.info(
                    f"Output quantize pos of HardSigmoid layer {node.output[0]} is {int(opos)}, modify it to {int(new_opos)} "
                    "to meet the DPU constraints."
                )

    def _adjust_shift_read(self) -> None:
        """Adjust shift_read for Add/Sub: shift_read = max(ipos)-min(ipos); clamp to [0, 7]; update max-ipos node if needed."""
        for i, node in enumerate(self.model.graph.node):
            if node.op_type not in ["Add", "Sub"]:
                continue
            ipos_layers = []
            iposes = []
            skip = False

            for i in range(len(node.input)):
                ipos_name = self._get_ipos_name_by_id(node, i)
                if ipos_name is None:
                    logger.debug(f"Fail to get input quantized position for layer {node.name}, please check it.")
                    skip = True
                    break
                ipos_layers.append(ipos_name)

            for name in ipos_layers:
                ipos, _ = self._get_pos_by_name(name)
                if ipos is None:
                    logger.debug(f"Fail to get quantized position for layer {name}, skip adjust_shift_read for it.")
                    skip = True
                    break
                iposes.append(ipos)
            if skip:
                continue
            id_max = np.argmax(iposes)
            id_min = np.argmin(iposes)
            sr = iposes[id_max] - iposes[id_min]
            min_sr, max_sr = 0, 7

            new_sr = None
            if sr > max_sr:
                new_sr = max_sr

            if new_sr is not None:
                self.has_change = True
                new_ipos_max = iposes[id_min] + new_sr
                self._set_pos(self._get_node_by_name(ipos_layers[id_max]), new_ipos_max)
                logger.info(
                    f"Shift read of layer {node.name} is {int(sr)}({int(iposes[id_max])}-{int(iposes[id_min])}). It exceeds range [{int(min_sr)}, {int(max_sr)}]. "
                    f"Modify ipos from {int(iposes[id_max])} to {int(new_ipos_max)}."
                )

    def _adjust_shift_write(self) -> None:
        """Adjust shift_write: Add uses min(ipos)-opos (clamp [-7,25]); Mul uses sum(ipos)-opos (clamp [0,32])."""
        for i, node in enumerate(self.model.graph.node):
            if node.op_type not in ["Add", "Mul"]:
                continue
            if node.op_type == "Add":
                ipos_layers = []
                iposes = []
                skip = False

                for i in range(len(node.input)):
                    ipos_name = self._get_ipos_name_by_id(node, i)
                    if ipos_name is None:
                        logger.debug(f"Fail to get input quantized position for layer {node.name}, please check it.")
                        skip = True
                        break
                    ipos_layers.append(ipos_name)

                for name in ipos_layers:
                    ipos, _ = self._get_pos_by_name(name)
                    if ipos is None:
                        logger.debug(f"Fail to get quantized position for layer {name}, skip adjust_shift_read for it.")
                        skip = True
                        break
                    iposes.append(ipos)
                if skip:
                    continue

                opos_name = self._get_opos_name(node)
                opos, _ = self._get_pos_by_name(opos_name)
                if opos is None:
                    logger.debug(
                        f"Fail to get quantized position for layer {node.name}(output:0), "
                        "skip adjust_shift_write for it."
                    )
                    continue

                id_min = np.argmin(iposes)
                sw = iposes[id_min] - opos
                min_sw, max_sw = -7, 25

                new_sw = None
                if sw > max_sw:
                    new_sw = max_sw
                elif sw < min_sw:
                    new_sw = min_sw

                if new_sw is not None:
                    self.has_change = True
                    new_opos = iposes[id_min] - new_sw
                    self._set_pos(self._get_node_by_name(opos_name), new_opos)
                    logger.info(
                        f"Shift write of layer {node.name} is {int(sw)}({int(iposes[id_min])}-{int(opos)}). It exceeds range [{int(min_sw)}, {int(max_sw)}]. "
                        f"Modify opos from {int(opos)} to {int(new_opos)}."
                    )
            elif node.op_type == "Mul":
                ipos_layers = []
                iposes = []
                skip = False

                for i in range(len(node.input)):
                    ipos_name = self._get_ipos_name_by_id(node, i)
                    if ipos_name is None:
                        logger.debug(f"Fail to get input quantized position for layer {node.name}, please check it.")
                        skip = True
                        break
                    ipos_layers.append(ipos_name)
                for name in ipos_layers:
                    ipos, _ = self._get_pos_by_name(name)
                    if ipos is None:
                        logger.debug(f"Fail to get quantized position for layer {name}, skip adjust_shift_read for it.")
                        skip = True
                        break
                    iposes.append(ipos)
                if skip:
                    continue
                opos_name = self._get_opos_name(node)
                opos, _ = self._get_pos_by_name(opos_name)
                if opos is None:
                    logger.debug(
                        f"Fail to get quantized position for layer {node.name}(output:0), "
                        "skip adjust_shift_write for it."
                    )
                    continue

                sw = sum(iposes) - opos
                min_sw, max_sw = 0, 32

                new_sw = None
                if sw > max_sw:
                    new_sw = max_sw
                elif sw < min_sw:
                    new_sw = min_sw

                if new_sw is not None:
                    new_opos = sum(iposes) - new_sw
                    self._set_pos(self._get_node_by_name(opos_name), new_opos)
                    logger.info(
                        f"Shift write of layer {node.name} is {int(sw)}({int(sum(iposes))}-{int(opos)}). It exceeds range [{int(min_sw)}, {int(max_sw)}]. "
                        f"Modify opos from {int(opos)} to {int(new_opos)}."
                    )

    def _align_concat(self) -> None:
        """Align concat op's inputs and output pos."""
        for i, node in enumerate(self.model.graph.node):
            if node.op_type not in ["Concat"]:
                continue
            input_node_num = len(node.input)
            opos_name = self._get_opos_name(node)
            opos, _ = self._get_pos_by_name(opos_name)
            if opos is not None:
                min_pos = opos
                ipos_layers = []

                for i in range(input_node_num):
                    ipos_name = self._get_ipos_name_by_id(node, i)
                    ipos_layers.append(ipos_name)
                for name in ipos_layers:
                    ipos, _ = self._get_pos_by_name(name)
                    if ipos is not None:
                        min_pos = min(ipos, min_pos)
                if opos != min_pos:
                    self.has_change = True
                    self._set_pos(self._get_node_by_name(opos_name), min_pos)
                    logger.info(
                        f"Output pos of concat node {node.name} is {int(opos)}, min_pos is {int(min_pos)}. "
                        f"Modify opos from {int(opos)} to {int(min_pos)}."
                    )
                for name in ipos_layers:
                    ipos, ipos_node = self._get_pos_by_name(name)
                    if ipos is not None and ipos != min_pos:
                        self.has_change = True
                        self._set_pos(ipos_node, min_pos)
                        logger.info(
                            f"Input pos of concat node {node.name} is {int(ipos)}, min_pos is {int(min_pos)}. "
                            f"Modify ipos from {int(ipos)} to {int(min_pos)}."
                        )
            else:
                logger.debug(
                    f"Fail to get quantized position for layer {node.name}(output:0), skip align concat for it."
                )

    def _align_pool(self) -> None:
        """Align max/avg pooling input and output pos."""
        for i, node in enumerate(self.model.graph.node):
            if node.op_type not in ["MaxPool", "AveragePool", "GlobalAveragePool"]:
                continue
            ipos_name = self._get_ipos_name(node)
            ipos, ipos_layer = self._get_pos_by_name(ipos_name)

            opos_name = self._get_opos_name(node)
            opos, opos_layer = self._get_pos_by_name(opos_name)
            if ipos is None or opos is None:
                logger.debug(f"Found a pos that is None. Align pool of layer {node.name} has not taken effect.")
                continue
            if ipos is not None and opos is not None and opos > ipos:
                self.has_change = True
                self._set_pos(opos_layer, ipos)
                logger.info(
                    f"Input pos of pooling layer {node.name} is {int(ipos)}. Output pos of pooling layer {node.name} is {int(opos)}."
                    f"Modify opos from {int(opos)} to {int(ipos)}."
                )
            elif ipos is not None and opos is not None and opos < ipos:
                self.has_change = True
                self._set_pos(ipos_layer, opos)
                logger.info(
                    f"Input pos of pooling layer {node.name} is {int(ipos)}. Output pos of pooling layer {node.name} is {int(opos)}."
                    f"Modify ipos from {int(ipos)} to {int(opos)}."
                )

    def _align_pad(self) -> None:
        """Align pad input and output pos."""
        for i, node in enumerate(self.model.graph.node):
            if node.op_type != "Pad":
                continue
            ipos_name = self._get_ipos_name(node)
            ipos, ipos_layer = self._get_pos_by_name(ipos_name)

            opos_name = self._get_opos_name(node)
            opos, opos_layer = self._get_pos_by_name(opos_name)
            if ipos is None or opos is None:
                logger.debug(f"Found a pos that is None. Align pad of layer {node.name} has not taken effect.")
                continue
            if ipos is not None and opos is not None and opos > ipos:
                self.has_change = True
                self._set_pos(opos_layer, ipos)
                logger.info(
                    f"Input pos of pad layer {node.name} is {int(ipos)}. Output pos of pad layer {node.name} is {int(opos)}."
                    f"Modify opos from {int(opos)} to {int(ipos)}."
                )
            elif ipos is not None and opos is not None and opos < ipos:
                self.has_change = True
                self._set_pos(ipos_layer, opos)
                logger.info(
                    f"Input pos of pad layer {node.name} is {int(ipos)}. Output pos of pooling layer {node.name} is {int(opos)}."
                    f"Modify ipos from {int(ipos)} to {int(opos)}."
                )

    def _align_slice(self) -> None:
        """Align slice input and output pos."""
        for i, node in enumerate(self.model.graph.node):
            if node.op_type not in ["Slice"]:
                continue
            ipos_name = self._get_ipos_name(node)
            ipos, ipos_layer = self._get_pos_by_name(ipos_name)

            opos_name = self._get_opos_name(node)
            opos, opos_layer = self._get_pos_by_name(opos_name)
            if ipos is None or opos is None:
                logger.debug(f"Found a pos that is None. Align Slice of layer {node.name} has not taken effect.")
                continue
            if ipos is not None and opos is not None and opos > ipos:
                self.has_change = True
                self._set_pos(opos_layer, ipos)
                logger.info(
                    f"Input pos of Slice layer {node.name} is {int(ipos)}. Output pos of Slice layer {node.name} is {int(opos)}."
                    f"Modify opos from {int(opos)} to {int(ipos)}."
                )
            elif ipos is not None and opos is not None and opos < ipos:
                self.has_change = True
                self._set_pos(ipos_layer, opos)
                logger.info(
                    f"Input pos of Slice layer {node.name} is {int(ipos)}. Output pos of Slice layer {node.name} is {int(opos)}."
                    f"Modify ipos from {int(ipos)} to {int(opos)}."
                )
