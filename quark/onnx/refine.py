#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import numpy as np
from quark.shares.utils.log import ScreenLogger
import onnx
import onnx.numpy_helper

from onnx import NodeProto, ModelProto, TensorProto
from typing import Any, List
from .quant_utils import (pos2scale, scale2pos, is_node_needs_annotated, annotate_op_type, avg_pool_op_type,
                          check_hard_sigmoid_condition, ONNXQuantizedModel, FIX_OP_NAME, QUANT_OP_TYPES,
                          DEQUANT_OP_TYPES)

refine_pos_op_type = [FIX_OP_NAME]
refine_scale_op_type = QUANT_OP_TYPES + DEQUANT_OP_TYPES

postfix = "_Output"

logger = ScreenLogger(__name__)


class QuantPosManager(object):

    def __init__(self, model: ONNXQuantizedModel) -> None:
        self.model = model
        self.has_change = True
        self.adjust_loop_count = 0

    def get_scale(self, node: NodeProto) -> Any:
        for i in self.model.model.graph.initializer:
            if i.name == node.input[1]:
                if i.float_data:
                    return i.float_data[0]
                elif i.raw_data:
                    return np.frombuffer(i.raw_data, dtype=np.float32).tolist()[0]
        raise ValueError("DequantizeLinear and QuantizeLinear do not have scale.")

    def set_scale(self, node: NodeProto, new_scale: float) -> None:
        for i in self.model.model.graph.initializer:
            if i.name == node.input[1]:
                if i.float_data:
                    if i.float_data[0] != new_scale:
                        i.float_data[0] = new_scale
                elif i.raw_data:
                    if np.frombuffer(i.raw_data, dtype=np.float32).tolist()[0] != new_scale:
                        np.frombuffer(i.raw_data, dtype=np.float32).tolist()[0] = new_scale

    def get_pos(self, node: NodeProto) -> Any:
        if node.op_type in refine_pos_op_type:
            for attr in node.attribute:
                if attr.name == "pos":
                    return attr.s
        if node.op_type in refine_scale_op_type:
            return scale2pos(self.get_scale(node))
        return None

    def set_pos(self, node: NodeProto, new_pos: int) -> None:
        if node.op_type in refine_pos_op_type:
            for attr in node.attribute:
                if attr.name == "pos":
                    attr.s = str(new_pos).encode()

        elif node.op_type in QUANT_OP_TYPES:
            new_scale = pos2scale(new_pos)
            self.set_scale(node, new_scale)
            if node.output:
                for n in self.model.model.graph.node:
                    if n.name == node.output[0].strip(postfix) and n.op_type in DEQUANT_OP_TYPES:
                        self.set_scale(node, new_scale)

        elif node.op_type in DEQUANT_OP_TYPES:
            new_scale = pos2scale(new_pos)
            self.set_scale(node, new_scale)
            for n in self.model.model.graph.node:
                if n.name == node.input[0].strip(postfix) and n.op_type in QUANT_OP_TYPES:
                    self.set_scale(node, new_scale)

    def find_node_name(self, name: str) -> Any:
        for node in self.model.model.graph.node:
            if len(node.output
                   ) > 0 and node.output[0] == name and node.op_type in refine_pos_op_type + refine_scale_op_type:
                return node.name
        return None

    def get_ipos_name(self, node: NodeProto) -> Any:
        if len(node.input) > 0:
            i_name = node.input[0]
            ipos_name = self.find_node_name(i_name)
            if ipos_name:
                return ipos_name
            op_type = node.op_type
            for n in self.model.model.graph.node:
                if len(n.output) >= 1 and n.output[0] == i_name and op_type in avg_pool_op_type:
                    i_name = n.input[0]
                    ipos_name = self.find_node_name(i_name)
                    if ipos_name:
                        return ipos_name
        else:
            return None

    def get_ipos_name_by_id(self, node: NodeProto, input_id: int = 0) -> Any:
        if len(node.input) > input_id:
            i_name = node.input[input_id]
            return self.find_node_name(i_name)
        else:
            return None

    def get_node_by_name(self, node_name: str) -> Any:
        for node in self.model.model.graph.node:
            if node.name == node_name:
                return node
        return None

    def get_pos_by_name(self, name: str) -> Any:
        for node in self.model.model.graph.node:
            if node.op_type in refine_pos_op_type and node.name == name:
                for attr in node.attribute:
                    if attr.name == "pos":
                        return int(attr.s), node
            elif node.op_type in refine_scale_op_type and node.name == name:
                return self.get_pos(node), node

        return None, None

    def find_o_name(self, o_name: str) -> Any:
        for node in self.model.model.graph.node:
            if (len(node.input) >= 1 and node.input[0] == o_name
                    and node.op_type in refine_pos_op_type + refine_scale_op_type):
                return node.name
        return None

    def get_opos_name(self, node: NodeProto) -> Any:

        def is_node_connected(pre_node_type: str, node: NodeProto) -> bool:
            if pre_node_type in avg_pool_op_type + ["HardSigmoid"] and node.op_type == "Mul":
                return True
            elif pre_node_type in annotate_op_type and is_node_needs_annotated(self.model.model, node):
                return True
            return False

        o_name = node.output[0]
        opos_name = self.find_o_name(o_name)
        if opos_name:
            return opos_name
        pre_node_type = node.op_type
        for n in self.model.model.graph.node:
            if (len(n.input) >= 1 and n.input[0] == o_name) and is_node_connected(pre_node_type, n):
                o_name = n.output[0]
                opos_name = self.find_o_name(o_name)
                if opos_name:
                    return opos_name
        return None

    def get_wpos_name(self, node: NodeProto) -> Any:
        if len(node.input) > 1:
            w_name = node.input[1]
            return self.find_node_name(w_name)
        else:
            return None

    def get_bpos_name(self, node: NodeProto) -> Any:
        if len(node.input) > 2:
            b_name = node.input[2]
            return self.find_node_name(b_name)
        else:
            return None

    def adjust_shift_cut(self) -> None:
        """Adjust the shift cut of nodes.

        shift_cut = wpos + ipos - opos

        DPU compiler constraints of shift_cut:
        1. 0 <= shift_cut <= 16
        """
        for i, node in enumerate(self.model.model.graph.node):
            if node.op_type not in ["Conv", "Gemm"]:
                continue
            ipos_name = self.get_ipos_name(node)
            ipos, _ = self.get_pos_by_name(ipos_name)

            opos_name = self.get_opos_name(node)
            opos, _ = self.get_pos_by_name(opos_name)
            wpos_name = self.get_wpos_name(node)
            wpos, wpos_node = self.get_pos_by_name(wpos_name)

            # Adjust shift_cut
            min_sc = 0
            max_sc = 16
            if wpos is None or ipos is None or opos is None:
                logger.debug("Found a pos that is None. Shift cut of layer {} has not taken effect.".format(node.name))
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
                self.set_pos(wpos_node, new_wpos)
                logger.info("Shift cut of layer {} is {}. It exceeds range [{}, {}]. "
                            "Modify wpos from {} to {}.".format(node.input[1], int(sc), int(min_sc), int(max_sc),
                                                                int(wpos), int(new_wpos)))

    def adjust_shift_bias(self) -> None:
        """Adjust the shift bias of node.

        shift_bias = wpos + ipos - bpos

        DPU compiler constraints of shift_bias:
        1. min(0, -(24 - (8 + shift_cut))) <= shift_bias <= 15
        """
        for i, node in enumerate(self.model.model.graph.node):
            if node.op_type not in ["Conv", "Gemm"]:
                continue

            ipos_name = self.get_ipos_name(node)
            ipos, _ = self.get_pos_by_name(ipos_name)

            opos_name = self.get_opos_name(node)
            opos, _ = self.get_pos_by_name(opos_name)
            wpos_name = self.get_wpos_name(node)
            wpos, wpos_node = self.get_pos_by_name(wpos_name)
            bpos_name = self.get_bpos_name(node)
            if bpos_name:
                bpos, bpos_node = self.get_pos_by_name(bpos_name)
                # Adjust shift_bias
                if wpos is None or ipos is None or opos is None or bpos is None:
                    logger.debug("Found a pos that is None. Shift bias of layer {} has not taken effect.".format(
                        node.name))
                    continue
                shift_cut = wpos + ipos - opos

                min_sb = min(0, -(24 - (8 + shift_cut)))
                # TODO: Optimize code structure
                for n in self.model.model.graph.node:
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
                    self.set_pos(self.get_node_by_name(bpos_name), new_bpos)
                    logger.info("Shift bias of layer {} is {}. It exceeds range [{}, {}]. "
                                "Modify bpos from {} to {}.".format(node.input[2], int(shift_bias), int(min_sb),
                                                                    int(max_sb), int(bpos), int(new_bpos)))

    def adjust_shift_swish(self) -> None:
        """Adjust the shift of Swish layer's Multiply op.
        shift_swish = 'input 0 pos' + 'input 1 pos' - 'output pos'
        DPU compiler constraints of shift_swish:
          1. 0 <= shift_swish <= 15
        """

        def _is_sigmoid_layer(node_input: str) -> bool:
            '''
            it's a swish's sigmoid layer or not
            '''
            for node in self.model.model.graph.node:
                if check_hard_sigmoid_condition(node) and node.input[0] == node_input:
                    return True
            return False

        def _belong_to_swish(node0: NodeProto, node1: NodeProto) -> bool:
            '''
            swish = mul(x, sigmoid(x))
            so one is sigmoid and another is x
            '''
            if _is_sigmoid_layer(node0) or _is_sigmoid_layer(node1):
                return True
            return False

        for i, node in enumerate(self.model.model.graph.node):
            if node.op_type not in ["Mul"]:
                continue

            if len(node.input) != 2:
                continue

            # Comfirm it's a swish's mul layer or not
            if not _belong_to_swish(node.input[0], node.input[1]):
                continue

            opos_name = self.get_opos_name(node)
            opos, _ = self.get_pos_by_name(opos_name)

            if opos is not None:

                ipos0_name = self.get_ipos_name_by_id(node, 0)
                ipos0, _ = self.get_pos_by_name(ipos0_name)

                ipos1_name = self.get_ipos_name_by_id(node, 1)
                ipos1, _ = self.get_pos_by_name(ipos1_name)

                if (ipos1 is None or ipos0 is None):
                    logger.warning('Fail to get quantized position for layer {} input, '
                                   'skip adjust_shift_swish for it.'.format(node.name))
                    continue

                min_sh, max_sh = 0, 15

                shift_swish = ipos0 + ipos1 - opos

                new_opos = opos
                if (shift_swish < min_sh):
                    new_opos = ipos0 + ipos1 - min_sh
                elif (shift_swish > max_sh):
                    new_opos = ipos0 + ipos1 - max_sh

                if new_opos != opos:
                    self.has_change = True
                    self.set_pos(self.get_node_by_name(opos_name), new_opos)
                    logger.info('Shift Swish of layer {} is {}({}+{}-{}). It exceeds range [{}, {}]. '
                                'Modify opos from {} to {}.'.format(node.name, int(shift_swish), int(ipos0), int(ipos1),
                                                                    int(opos), int(min_sh), int(max_sh), int(opos),
                                                                    int(new_opos)))
            else:
                logger.debug("Fail to get quantized position for layer {}(output:0), "
                             "skip adjust shift swish for it.".format(node.name))

    def adjust_hard_sigmoid(self) -> None:
        """Adjust quantize info of HardSigmoid nodes.

        DPU compiler constraints for HardSigmoid:
        1. input pos of HardSigmoid >= 0 && <= 15
        2. output pos of HardSigmoid >= 7
        3. shift_sigmoid >= 0 && shift_sigmoid <= 31 where
            shift_sigmoid = 14 + 'input pos' - ' output pos'
        """
        for i, node in enumerate(self.model.model.graph.node):

            if node.op_type not in ["HardSigmoid"]:
                continue
            if not check_hard_sigmoid_condition(node):
                continue
            ipos_name = self.get_ipos_name(node)
            ipos, _ = self.get_pos_by_name(ipos_name)

            opos_name = self.get_opos_name(node)
            opos, _ = self.get_pos_by_name(opos_name)

            if ipos is None or opos is None:
                logger.debug("Found a pos that is None. Adjust quantize info of HardSigmoid "
                             "nodes of layer {} has not taken effect.".format(node.name))
                continue

            new_ipos = ipos if ipos > 0 else 0
            new_ipos = new_ipos if new_ipos <= 15 else 15

            new_opos = opos if opos > 7 else 7
            shift_sigmoid = 14 + new_ipos - new_opos  # will not bigger than 31 now
            new_opos = new_opos if shift_sigmoid > 0 else 14 + new_ipos

            if new_ipos != ipos:
                self.has_change = True
                self.set_pos(self.get_node_by_name(ipos_name), new_ipos)
                logger.info("Input quantize pos of HardSigmoid layer {} is {}, modify it to {} "
                            "to meet the DPU constraints.".format(node.input[0], int(ipos), int(new_ipos)))

            if new_opos != opos:
                self.has_change = True
                self.set_pos(self.get_node_by_name(opos_name), new_opos)
                logger.info("Output quantize pos of HardSigmoid layer {} is {}, modify it to {} "
                            "to meet the DPU constraints.".format(node.output[0], int(opos), int(new_opos)))

    def adjust_shift_read(self) -> None:
        """Adjust the shift read of node.

        shift_read = max(ipos) - min(ipos)

        NPU compiler constraints of shift_read:
        1. 0 <= shift_read <= 7
        """
        for i, node in enumerate(self.model.model.graph.node):

            if node.op_type not in ["Add", "Sub"]:
                continue
            ipos_layers = []
            iposes = []
            skip = False

            for i in range(len(node.input)):
                ipos_name = self.get_ipos_name_by_id(node, i)
                if ipos_name is None:
                    logger.debug("Fail to get input quantized position for layer {}, "
                                 "please check it.".format(node.name))
                    skip = True
                    break
                ipos_layers.append(ipos_name)

            for name in ipos_layers:
                ipos, _ = self.get_pos_by_name(name)
                if ipos is None:
                    logger.debug("Fail to get quantized position for layer {}, "
                                 "skip adjust_shift_read for it.".format(name))
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
                self.set_pos(self.get_node_by_name(ipos_layers[id_max]), new_ipos_max)
                logger.info("Shift read of layer {} is {}({}-{}). It exceeds range [{}, {}]. "
                            "Modify ipos from {} to {}.".format(
                                node.name,
                                int(sr),
                                int(iposes[id_max]),
                                int(iposes[id_min]),
                                int(min_sr),
                                int(max_sr),
                                int(iposes[id_max]),
                                int(new_ipos_max),
                            ))

    def adjust_shift_write(self) -> None:
        """Adjust the shift write of node.

        For Add:
        shift_write = min(ipos) - opos

        NPU compiler constraints of shift_write:
        1. -7 <= shift_write <= 25

        For Mul:
        shift_write = sum(ipos) - opos

        NPU compiler constraints of shift_write:
        1. 0 <= shift_write <= 32
        """
        for i, node in enumerate(self.model.model.graph.node):

            if node.op_type not in ["Add", "Mul"]:
                continue
            if node.op_type == "Add":
                ipos_layers = []
                iposes = []
                skip = False

                for i in range(len(node.input)):
                    ipos_name = self.get_ipos_name_by_id(node, i)
                    if ipos_name is None:
                        logger.debug("Fail to get input quantized position for layer {}, "
                                     "please check it.".format(node.name))
                        skip = True
                        break
                    ipos_layers.append(ipos_name)

                for name in ipos_layers:
                    ipos, _ = self.get_pos_by_name(name)
                    if ipos is None:
                        logger.debug("Fail to get quantized position for layer {}, "
                                     "skip adjust_shift_read for it.".format(name))
                        skip = True
                        break
                    iposes.append(ipos)
                if skip:
                    continue

                opos_name = self.get_opos_name(node)
                opos, _ = self.get_pos_by_name(opos_name)
                if opos is None:
                    logger.debug("Fail to get quantized position for layer {}(output:0), "
                                 "skip adjust_shift_write for it.".format(node.name))
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
                    self.set_pos(self.get_node_by_name(opos_name), new_opos)
                    logger.info("Shift write of layer {} is {}({}-{}). It exceeds range [{}, {}]. "
                                "Modify opos from {} to {}.".format(
                                    node.name,
                                    int(sw),
                                    int(iposes[id_min]),
                                    int(opos),
                                    int(min_sw),
                                    int(max_sw),
                                    int(opos),
                                    int(new_opos),
                                ))
            elif node.op_type == "Mul":
                ipos_layers = []
                iposes = []
                skip = False

                for i in range(len(node.input)):
                    ipos_name = self.get_ipos_name_by_id(node, i)
                    if ipos_name is None:
                        logger.debug("Fail to get input quantized position for layer {}, "
                                     "please check it.".format(node.name))
                        skip = True
                        break
                    ipos_layers.append(ipos_name)
                for name in ipos_layers:
                    ipos, _ = self.get_pos_by_name(name)
                    if ipos is None:
                        logger.debug("Fail to get quantized position for layer {}, "
                                     "skip adjust_shift_read for it.".format(name))
                        skip = True
                        break
                    iposes.append(ipos)
                if skip:
                    continue
                opos_name = self.get_opos_name(node)
                opos, _ = self.get_pos_by_name(opos_name)
                if opos is None:
                    logger.debug("Fail to get quantized position for layer {}(output:0), "
                                 "skip adjust_shift_write for it.".format(node.name))
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
                    self.set_pos(self.get_node_by_name(opos_name), new_opos)
                    logger.info("Shift write of layer {} is {}({}-{}). It exceeds range [{}, {}]. "
                                "Modify opos from {} to {}.".format(
                                    node.name,
                                    int(sw),
                                    int(sum(iposes)),
                                    int(opos),
                                    int(min_sw),
                                    int(max_sw),
                                    int(opos),
                                    int(new_opos),
                                ))

    def align_concat(self) -> None:
        """Align concat op's inputs and output pos."""
        for i, node in enumerate(self.model.model.graph.node):
            if node.op_type not in ["Concat"]:
                continue
            input_node_num = len(node.input)
            opos_name = self.get_opos_name(node)
            opos, _ = self.get_pos_by_name(opos_name)
            if opos is not None:
                min_pos = opos
                ipos_layers = []

                for i in range(input_node_num):
                    ipos_name = self.get_ipos_name_by_id(node, i)
                    ipos_layers.append(ipos_name)
                for name in ipos_layers:
                    ipos, _ = self.get_pos_by_name(name)
                    if ipos is not None:
                        min_pos = min(ipos, min_pos)
                if opos != min_pos:
                    self.has_change = True
                    self.set_pos(self.get_node_by_name(opos_name), min_pos)
                    logger.info(("Output pos of concat node {} is {}, min_pos is {}. "
                                 "Modify opos from {} to {}.".format(node.name, int(opos), int(min_pos), int(opos),
                                                                     int(min_pos))))
                for name in ipos_layers:
                    ipos, ipos_node = self.get_pos_by_name(name)
                    if ipos is not None and ipos != min_pos:
                        self.has_change = True
                        self.set_pos(ipos_node, min_pos)
                        logger.info("Input pos of concat node {} is {}, min_pos is {}. "
                                    "Modify ipos from {} to {}.".format(node.name, int(ipos), int(min_pos), int(ipos),
                                                                        int(min_pos)))
            else:
                logger.debug("Fail to get quantized position for layer {}(output:0), "
                             "skip align concat for it.".format(node.name))

    def align_pool(self) -> None:
        """Align max/avg pooling input and output pos."""
        for i, node in enumerate(self.model.model.graph.node):
            if node.op_type not in ["MaxPool", "AveragePool", "GlobalAveragePool"]:
                continue
            ipos_name = self.get_ipos_name(node)
            ipos, ipos_layer = self.get_pos_by_name(ipos_name)

            opos_name = self.get_opos_name(node)
            opos, opos_layer = self.get_pos_by_name(opos_name)
            if ipos is None or opos is None:
                logger.debug("Found a pos that is None. Align pool of layer {} has not taken effect.".format(node.name))
                continue
            if ipos is not None and opos is not None and opos > ipos:
                self.has_change = True
                self.set_pos(opos_layer, ipos)
                logger.info("Input pos of pooling layer {} is {}. Output pos of pooling layer {} is {}."
                            "Modify opos from {} to {}.".format(node.name, int(ipos), node.name, int(opos), int(opos),
                                                                int(ipos)))
            elif ipos is not None and opos is not None and opos < ipos:
                self.has_change = True
                self.set_pos(ipos_layer, opos)
                logger.info("Input pos of pooling layer {} is {}. Output pos of pooling layer {} is {}."
                            "Modify ipos from {} to {}.".format(node.name, int(ipos), node.name, int(opos), int(ipos),
                                                                int(opos)))

    def align_pad(self) -> None:
        """Align pad input and output pos."""
        for i, node in enumerate(self.model.model.graph.node):
            if node.op_type != "Pad":
                continue
            ipos_name = self.get_ipos_name(node)
            ipos, ipos_layer = self.get_pos_by_name(ipos_name)

            opos_name = self.get_opos_name(node)
            opos, opos_layer = self.get_pos_by_name(opos_name)
            if ipos is None or opos is None:
                logger.debug("Found a pos that is None. Align pad of layer {} has not taken effect.".format(node.name))
                continue
            if ipos is not None and opos is not None and opos > ipos:
                self.has_change = True
                self.set_pos(opos_layer, ipos)
                logger.info("Input pos of pad layer {} is {}. Output pos of pad layer {} is {}."
                            "Modify opos from {} to {}.".format(node.name, int(ipos), node.name, int(opos), int(opos),
                                                                int(ipos)))
            elif ipos is not None and opos is not None and opos < ipos:
                self.has_change = True
                self.set_pos(ipos_layer, opos)
                logger.info("Input pos of pad layer {} is {}. Output pos of pooling layer {} is {}."
                            "Modify ipos from {} to {}.".format(node.name, int(ipos), node.name, int(opos), int(ipos),
                                                                int(opos)))

    def align_slice(self) -> None:
        """Align slice input and output pos."""
        for i, node in enumerate(self.model.model.graph.node):
            if node.op_type not in ["Slice"]:
                continue
            ipos_name = self.get_ipos_name(node)
            ipos, ipos_layer = self.get_pos_by_name(ipos_name)

            opos_name = self.get_opos_name(node)
            opos, opos_layer = self.get_pos_by_name(opos_name)
            if ipos is None or opos is None:
                logger.debug("Found a pos that is None. Align Slice of layer {} has not taken effect.".format(
                    node.name))
                continue
            if ipos is not None and opos is not None and opos > ipos:
                self.has_change = True
                self.set_pos(opos_layer, ipos)
                logger.info("Input pos of Slice layer {} is {}. Output pos of Slice layer {} is {}."
                            "Modify opos from {} to {}.".format(node.name, int(ipos), node.name, int(opos), int(opos),
                                                                int(ipos)))
            elif ipos is not None and opos is not None and opos < ipos:
                self.has_change = True
                self.set_pos(ipos_layer, opos)
                logger.info("Input pos of Slice layer {} is {}. Output pos of Slice layer {} is {}."
                            "Modify ipos from {} to {}.".format(node.name, int(ipos), node.name, int(opos), int(ipos),
                                                                int(opos)))


def adjust_quantize_info(model: ModelProto,
                         max_loop_num: int = 5,
                         adjust_shift_cut: bool = True,
                         adjust_shift_bias: bool = True,
                         adjust_shift_read: bool = True,
                         adjust_shift_write: bool = True,
                         adjust_hard_sigmoid: bool = True,
                         adjust_shift_swish: bool = True,
                         align_concat: bool = True,
                         align_pool: bool = True,
                         align_pad: bool = True,
                         align_slice: bool = True) -> ONNXQuantizedModel:
    """Adjust the quantize info to meet the compiler constraints."""

    manager = QuantPosManager(model)

    while manager.has_change and (manager.adjust_loop_count < max_loop_num):
        manager.adjust_loop_count += 1
        if manager.adjust_loop_count == max_loop_num:
            logger.warning('The number of adjustments has reached the limit. '
                           'Please check the model')
        manager.has_change = False
        logger.info('Adjust the quantize info to meet the compiler constraints')

        # First do the alignment, then make adjustments to ensure all adjustments are effective
        if align_concat:
            manager.align_concat()

        if align_pool:
            manager.align_pool()

        if align_pad:
            manager.align_pad()

        if align_slice:
            manager.align_slice()

        if adjust_shift_read:
            manager.adjust_shift_read()

        if adjust_shift_write:
            manager.adjust_shift_write()

        if adjust_shift_cut:
            manager.adjust_shift_cut()

        if adjust_shift_bias:
            manager.adjust_shift_bias()

        if adjust_hard_sigmoid:
            manager.adjust_hard_sigmoid()

        if adjust_shift_swish:
            manager.adjust_shift_swish()

    return manager.model


class QuantInfoManager(object):

    def __init__(self, model: ModelProto) -> None:
        self.model = model
        self.has_change = True
        self.adjust_loop_count = 0

        # Note that the model is a ONNXModel instance
        self.parser = ONNXQuantizedModel(self.model.model)

    def get_quant_info(self, node: NodeProto) -> List[TensorProto]:
        assert node.op_type in refine_scale_op_type

        quant_info = [
            self.model.get_initializer(node.input[1]),
            self.model.get_initializer(node.input[2]),
        ]

        return quant_info

    def set_quant_info(self, node: NodeProto, quant_info: List[TensorProto]) -> None:
        assert node.op_type in refine_scale_op_type

        scale = self.model.get_initializer(node.input[1])
        scale.CopyFrom(quant_info[0])
        scale.name = node.input[1]

        zero_point = self.model.get_initializer(node.input[2])
        zero_point.CopyFrom(quant_info[1])
        zero_point.name = node.input[2]

    def quant_info_equal(self, quant_info_a: List[TensorProto], quant_info_b: List[TensorProto]) -> Any:
        scale_a = onnx.numpy_helper.to_array(quant_info_a[0])
        scale_b = onnx.numpy_helper.to_array(quant_info_b[0])
        zero_point_a = onnx.numpy_helper.to_array(quant_info_a[1])
        zero_point_b = onnx.numpy_helper.to_array(quant_info_b[1])
        return np.array_equal(scale_a, scale_b) and np.array_equal(zero_point_a, zero_point_b)

    def copy_output_qinfo_to_inputs(self, op_types: List[str]) -> None:
        """Copy output tensor's quant info to input tensor."""
        for node in self.model.model.graph.node:
            if node.op_type not in op_types:
                continue

            node_struct = self.parser.find_target_node_qdqs(node)
            if not (len(node_struct['input_qdqs']) and len(node_struct['output_qdqs'])):
                continue

            output_q, output_dq = node_struct['output_qdqs'][0]
            if output_q is None or output_dq is None:
                continue

            quant_info = self.get_quant_info(output_q)

            has_change = False

            for input_dq, input_q in node_struct['input_qdqs']:
                if input_dq is None or input_q is None:
                    continue

                if self.quant_info_equal(quant_info, self.get_quant_info(input_dq)):
                    continue

                self.set_quant_info(input_dq, quant_info)

                # Q always shares the same scale and zp with DQ
                if (input_q.input[1] != input_dq.input[1]) or (input_q.input[2] != input_dq.input[2]):
                    self.set_quant_info(input_q, quant_info)

                has_change = True

            if has_change:
                self.has_change = True
                logger.info(f"Have aligned {node.op_type} node {node.name} inputs")

    def copy_input_qinfo_to_outputs(self, op_types: List[str]) -> None:
        """Copy input tensor's quant info to output tensor."""
        for node in self.model.model.graph.node:
            if node.op_type not in op_types:
                continue

            node_struct = self.parser.find_target_node_qdqs(node)
            if not (len(node_struct['input_qdqs']) and len(node_struct['output_qdqs'])):
                continue

            input_dq, input_q = node_struct['input_qdqs'][0]
            if input_dq is None or input_q is None:
                continue

            quant_info = self.get_quant_info(input_dq)

            has_change = False

            for output_q, output_dq in node_struct['output_qdqs']:
                if output_q is None or output_dq is None:
                    continue

                if self.quant_info_equal(quant_info, self.get_quant_info(output_q)):
                    continue

                self.set_quant_info(output_q, quant_info)

                # DQ always shares the same scale and zp with Q
                if (output_q.input[1] != output_dq.input[1]) or (output_q.input[2] != output_dq.input[2]):
                    self.set_quant_info(output_dq, quant_info)

                has_change = True

            if has_change:
                self.has_change = True
                logger.info(f"Have aligned {node.op_type} node {node.name} outputs")

    def align_concat(self) -> None:
        """Align concat input and output quant info."""
        self.copy_output_qinfo_to_inputs(["Concat"])

    def align_pool(self) -> None:
        """Align pool input and output quant info."""
        self.copy_input_qinfo_to_outputs(["MaxPool", "AveragePool", "GlobalAveragePool"])

    def align_pad(self) -> None:
        """Align pad input and output quant info."""
        self.copy_output_qinfo_to_inputs(["Pad"])

    def align_slice(self) -> None:
        """Align slice input and output quant info.
           Note that Slice may have multiple outputs
        """
        self.copy_input_qinfo_to_outputs(["Slice"])

    def align_transpose(self) -> None:
        """Align transpose input and output quant info."""
        self.copy_output_qinfo_to_inputs(["Transpose"])

    def align_reshape(self) -> None:
        """Align reshape input and output quant info."""
        self.copy_output_qinfo_to_inputs(["Reshape"])

    def adjust_bias_scale(self) -> None:
        """ Make sure that bias scale = activation scale * weights scale. """
        scale_values = {}
        output2node = {}

        for node in self.model.model.graph.node:
            if node.op_type in ['DequantizeLinear', 'VitisDequantizeLinear']:
                scale_input_name = node.input[1]
                scale_initializer = next(
                    (init for init in self.model.model.graph.initializer if init.name == scale_input_name), None)
                if scale_initializer:
                    scale_value = onnx.numpy_helper.to_array(scale_initializer)
                    scale_values[node.output[0]] = scale_value
                    output2node[node.output[0]] = node

        for node in self.model.model.graph.node:
            if node.op_type in ['Conv', 'Gemm', 'ConvTranspose']:
                if len(node.input) == 3 and node.input[0] in scale_values and node.input[
                        1] in scale_values and node.input[2] in scale_values:
                    act_scale = scale_values[node.input[0]]
                    weights_scale = scale_values[node.input[1]]
                    bias_scale = scale_values[node.input[2]]
                    bias_node = output2node[node.input[2]]

                    if act_scale * weights_scale != bias_scale:
                        new_bias_scale = act_scale * weights_scale
                        for initializer in self.model.model.graph.initializer:
                            if initializer.name == bias_node.input[2]:
                                data_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]
                                if data_type != np.int32:
                                    logger.warning(
                                        "The bias scale != activation scale * weights scale in QDQ of {} because the bias qdq is not int32. Please check it."
                                        .format(node.name))
                                    continue
                                else:
                                    for initializer in self.model.model.graph.initializer:
                                        if initializer.name == bias_node.input[0]:
                                            array = onnx.numpy_helper.to_array(initializer)
                                            new_array = array / (act_scale * weights_scale / bias_scale)
                                            new_array = new_array.astype(np.int32)
                                            new_initializer = onnx.numpy_helper.from_array(new_array,
                                                                                           name=bias_node.input[0])
                                            initializer.CopyFrom(new_initializer)
                                        if initializer.name == bias_node.input[1]:
                                            array = onnx.numpy_helper.to_array(initializer)
                                            new_array = act_scale * weights_scale
                                            new_initializer = onnx.numpy_helper.from_array(new_array,
                                                                                           name=bias_node.input[1])
                                            initializer.CopyFrom(new_initializer)
                                    logger.info(
                                        "Have adjusted bias scale == activation scale * weights scale in QDQ of {}.".
                                        format(node.name))


def align_quantize_info(
    model: ModelProto,
    max_loop_num: int = 5,
    align_concat: bool = True,
    align_pool: bool = True,
    align_pad: bool = True,
    align_slice: bool = True,
    align_transpose: bool = True,
    align_reshape: bool = True,
    adjust_bias_scale: bool = True,
) -> Any:
    """Align the quantize info to meet the compiler constraints.
       This function supports pof2 scale and float scale both
    """

    manager = QuantInfoManager(model)

    while manager.has_change and (manager.adjust_loop_count < max_loop_num):
        manager.adjust_loop_count += 1
        if manager.adjust_loop_count == max_loop_num:
            logger.warning('The number of adjustments has reached the limit. '
                           'Please check the model')
        manager.has_change = False
        logger.info('Adjust the quantize info to meet the compiler constraints')

        if align_concat:
            manager.align_concat()

        if align_pool:
            manager.align_pool()

        if align_pad:
            manager.align_pad()

        if align_slice:
            manager.align_slice()

        if align_transpose:
            manager.align_transpose()

        if align_reshape:
            manager.align_reshape()

        if adjust_bias_scale:
            manager.adjust_bias_scale()

    return manager.model
