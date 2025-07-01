#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from torch.fx import GraphModule, Node
from abc import abstractmethod
from quark.torch.quantization.graph.optimization.utils import _copy_node_meta_info
from quark.torch.quantization.graph.torch_utils import is_clip_node, is_cat_node, is_call_module_node, \
    is_hardsigmoid_node, is_mul_node, ADD_OPS, SUB_OPS, MUL_OPS, LINEAR_OPS, CONV2D_OPS, _STRATEGY_SHIFT_CUT_MODULE
from quark.torch.quantization.graph.optimization.opt_pass_manager import OptPassBase
from quark.torch.quantization.graph.optimization.utils import is_quantizer_node, is_quantizer, get_quantizer_scale_pos
from typing import List, Tuple, Any, Optional
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)
__all__ = [
    "ConvertClip2ReLUQOPass", "ApplyConstrain2ConcatQOPass", "AlignSingleInOutOpScaleQOPass", "AdjustShiftReadQOPass",
    "ConvertHardSigmoidDpuVersionQOPass", "AdjustShiftWriteQOPass", "AdjustShiftCutQOPass", "AdjustShiftBiasQOPass",
    "AdjustHardSigmoidQOPass", "AdjustShiftSwishQOPass"
]


def _is_has_one_user_and_followed_quantizer(m: GraphModule, n: Node) -> bool:
    if len(n.users) != 1 or (not is_quantizer_node(m, next(iter(n.users)))):
        logger.warning("node: {} user num != 1 or not followed by quantizer node, \
                        Please check whether meet the quant demand.".format(n.name))
        return False
    return True


def _get_out_quantizer_node(graph_module: GraphModule, node: Node, max_depth: int) -> Optional[Node]:
    if max_depth > 2:
        return None
    if max_depth <= 2 and is_quantizer_node(graph_module, node):
        return node
    max_depth += 1
    if len(node.users) > 1:
        logger.warning("During finding out quantizer, Node: {} have more than one user".format(node.name))
        return None
    if len(node.users) == 0:
        return None
    next_user = next(iter(node.users))
    return _get_out_quantizer_node(graph_module, next_user, max_depth)


class ConvertClip2ReLUQOPass(OptPassBase):
    '''
    This is a post-quantization optimization.
    after quantization, we get a model as follows:
        ...
        x = torch.clip(x, clip_min=num1, clip_max=num2)
        x = fake_quantizer(x)
        ...
        x = torch.clip(x, clip_min=num3, clip_max=num4)
        x = other_type_layer(x) # not fake_quantizer
    post quant optimization:
        the clip that satisfies some condition can be transferred to ReLU layer:
            1. following a fake quantizer layer
            2. clip_min =  0
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        '''
        convert a clip layer to relu layer
        (only activate under condition that relu can be act as clip)
        '''
        to_delete_node = []
        for n in m.graph.nodes:
            if (not is_clip_node(n)) or (not _is_has_one_user_and_followed_quantizer(m, n)):
                continue
            clip_node = n
            clip_min, clip_max = clip_node.args[1], clip_node.args[2]
            if isinstance(clip_min, Node) or isinstance(clip_max, Node):
                raise NotImplementedError("clip node's min/max are not float, found Node")
            # condition check
            if clip_min < 0:
                continue  # could not be replaced with Relu
            input_activation_node = clip_node.args[0]
            to_delete_node.append(clip_node)
            with m.graph.inserting_after(input_activation_node):
                relu_node = m.graph.create_node(
                    'call_function',
                    torch.ops.aten.relu_.default,  # type: ignore[attr-defined]
                    (input_activation_node, ),
                    {})
                # NOTE modify the node's meta info
                _copy_node_meta_info(org_node=clip_node, target_node=relu_node)
                clip_node.replace_all_uses_with(relu_node)
        if len(to_delete_node):
            logger.info("Totally replace ops.aten.clip to ops.aten.relu count:\t{}.".format(len(to_delete_node)))
            [m.graph.erase_node(node) for node in to_delete_node]
            m.graph.eliminate_dead_code()
            m.recompile()
        return m


class ApplyConstrain2ConcatQOPass(OptPassBase):
    '''
    This is a post-quantization optimization.
    after quantization, we get a model as follows:
        ...
        x1 = self.fake_quantizer_1(x1)   # with scale_1
        x2 = self.fake_quantizer_1(x2)   # with scale_2
        cat = torch.ops.aten.cat.default([x1, x2], 1)
        cat_quant = self.fake_quantizer_2(cat)   # with scale_3
        ...
    Align concat op's inputs and output pos:
        inputs' quantizer scales should be same with concat's output scale
        if inputs's quantizer scales diff with concat's output scale,
        we select and modify them to the max scale
            e.g. scale1: 0.5 scale_2: 0.25 scale_3: 0.5 -> all scale will be 0.5
    NOTE: This strategy is align with quark/onnx/refine.py: manager.align_concat()
        different with onnx tool strategy (only consider powof2),
            this strategy select max scale regardless of scale type
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        '''
        1. find the concat node
        2. check the input's quant scale and output quant scale
        '''
        for n in m.graph.nodes:
            if (not is_cat_node(n)) or (not _is_has_one_user_and_followed_quantizer(m, n)):
                continue
            need_change = False
            cat_node = n
            # Session 1: get the max scale from inputs & outputs quantizer
            output_node = next(iter(cat_node.users))  # output_node should be quantizer
            output_quantizer = getattr(m, output_node.target)
            if output_quantizer.scale.numel() != 1:
                logger.warning("In quantizer Node {} scale is not a scale num, Skip apply this strategy".format(
                    output_node.name))
                continue
            scale_list = [output_quantizer.scale.item()]

            skip_optimize_this_concat = False
            for input_node in cat_node.args[0]:
                if not is_quantizer_node(m, input_node):
                    skip_optimize_this_concat = True
                    continue
                input_quantizer = getattr(m, input_node.target)
                if input_quantizer.scale.numel() != 1:
                    skip_optimize_this_concat = True
                    continue
                scale_list.append(input_quantizer.scale.item())
            if skip_optimize_this_concat:
                logger.warning("Skip apply this strategy for cat node: {}".format(cat_node.name))
                continue
            if len(set(scale_list)) != 1:
                need_change = True
                max_scale = max(scale_list)
            if not need_change:
                continue

            # Session2: if need change, change quantizer's scale
            if output_quantizer.scale.item() != max_scale:
                logger.info("Node: {} output quantizer: {} scale change from {} to {}".format(
                    cat_node.name, output_node.name, output_quantizer.scale.item(), max_scale))
                output_quantizer.scale.fill_(max_scale)
            for input_node in cat_node.args[0]:
                input_quantizer = getattr(m, input_node.target)
                if input_quantizer.scale.item() != max_scale:
                    logger.info("Node: {} input quantizer: {} scale change from {} to {}".format(
                        cat_node.name, input_node.name, input_quantizer.scale.item(), max_scale))
                    input_quantizer.scale.fill_(max_scale)
        return m


class AlignSingleInOutScaleBase(OptPassBase):
    '''
    This is a post-quantization optimization.
    after quantization, we get a model as follows:
        ...
        x1 = self.fake_quantizer_1(x1)   # with scale_1
        max_pool = torch.ops.aten.max_pool.default(x1)
        x2 = self.fake_quantizer_1(max_pool)   # with scale_1
        ...
    Align Single Input & Output quantizer's scale:
        scale_1 should be same as scale_1e, and value should be  max(scale_1, scale_2)

    NOTE: This strategy is similar to quark/onnx/refine.py: align_slice, align_pool, align_pad
        different with onnx tool strategy (only consider powof2),
            this strategy select max scale regardless of scale type.
    '''

    @abstractmethod
    def get_target_node(self, g: GraphModule, n: Node) -> bool:
        pass

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        '''
        1. fine the target op node
        2. check the input's quant scale and output quant scale
        '''
        for n in m.graph.nodes:
            if not self.get_target_node(m, n):
                continue
            need_change = False
            target_node = n
            # Session 1: get the max scale from inputs & outputs
            output_node = next(iter(target_node.users))
            input_node = target_node.args[0]
            if (not is_quantizer_node(m, input_node)):
                logger.warning("Please check Node: {} is not quantizer Node".format(input_node.name))
                continue
            output_quantizer = getattr(m, output_node.target)
            input_quantizer = getattr(m, input_node.target)
            if output_quantizer.scale.numel() != 1 or input_quantizer.scale.numel() != 1:
                logger.warning(
                    "At present, for AMD NPU, only support Per-Tensor quantization, Please close hw optimization")
                continue
            scale_list = [output_quantizer.scale.item(), input_quantizer.scale.item()]

            if len(set(scale_list)) != 1:
                need_change = True
                max_scale = max(scale_list)
            if not need_change:
                continue

            # Session 2: if need change modify the scale
            if output_quantizer.scale.item() != max_scale:
                logger.info("Node: {} output quantizer: {} scale change from {} to {}".format(
                    target_node.name, output_node.name, output_quantizer.scale.item(), max_scale))
                output_quantizer.scale.fill_(max_scale)
            if input_quantizer.scale.item() != max_scale:
                logger.info("Node: {} input quantizer: {} scale change from {} to {}".format(
                    target_node.name, input_node.name, input_quantizer.scale.item(), max_scale))
                input_quantizer.scale.fill_(max_scale)
        return m


# TODO may meed run multi times
class AlignSingleInOutOpScaleQOPass(AlignSingleInOutScaleBase):

    def __init__(self, op: List[torch._ops.OpOverload]) -> None:
        self.target_op = op

    def get_target_node(self, g: GraphModule, n: Node) -> bool:
        if n.op == "call_function" and (n.target in self.target_op) and _is_has_one_user_and_followed_quantizer(g, n):
            return True
        else:
            return False


# TODO may meed run multi times
class AlignSingleInOutModuleScaleQOPass(AlignSingleInOutScaleBase):

    def __init__(self, modules: Tuple[Any]) -> None:
        self.target_module = modules

    def get_target_node(self, g: GraphModule, n: Node) -> bool:
        if is_call_module_node(n) and isinstance(
                getattr(g, n.target),  # type: ignore[arg-type]
                self.target_module) and _is_has_one_user_and_followed_quantizer(g, n):
            return True
        else:
            return False


# TODO may meed run multi times
class AdjustShiftReadQOPass(OptPassBase):
    '''
    This is a post-quantization optimization.
    shift_read = max(input_quant_pos) - min(ipos)
    This strategy is applied for Add and Sub opeartion
    NPU compiler constraints of shift_read:
    1. 0 <= shift_read <= 7
    NOTE: align with quark/onnx/refine.py: adjust_shift_read
    '''

    def __init__(self) -> None:
        self.min_sr = 0
        self.max_sr = 7

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        '''
        1. fine the target op node
        2. check the input's quant scale and output quant scale
        '''
        for n in m.graph.nodes:
            if not (n.op == "call_function" and n.target in ADD_OPS + SUB_OPS):
                continue
            add_or_sub_node = n
            input_node_1, input_node_2 = add_or_sub_node.args[0], add_or_sub_node.args[1]
            if (not is_quantizer_node(m, input_node_1)) or (not is_quantizer_node(m, input_node_2)):
                logger.warning("Node: {} input is not all quantizer".format(add_or_sub_node.name))
                continue

            # Session 1: get the max scale from inputs & outputs
            input_quantizer_1 = getattr(m, input_node_1.target)
            input_quantizer_2 = getattr(m, input_node_2.target)
            if input_quantizer_1.scale.numel() != 1 or input_quantizer_2.scale.numel() != 1:
                logger.warning(
                    "At present, for AMD NPU, only support Per-Tensor quantization, Please close hw optimization")
                continue
            input_pos1 = get_quantizer_scale_pos(input_quantizer_1)
            input_pos2 = get_quantizer_scale_pos(input_quantizer_2)
            iposes = [input_pos1, input_pos2]

            sr = max(iposes) - min(iposes)
            new_sr = None
            if sr > self.max_sr:
                new_sr = self.max_sr
            if new_sr is not None:
                new_ipos_max = min(iposes) + new_sr
                modify_quantizer = input_quantizer_1 if input_pos1 == max(iposes) else input_quantizer_2
                modify_quantizer_name = input_node_1.name if input_pos1 == max(iposes) else input_node_2.name
                logger.info("Node: {} input quantizer: {} scale change from {} to {}".format(
                    add_or_sub_node.name, modify_quantizer_name, modify_quantizer.scale.item(), 1 / (2**new_ipos_max)))
                modify_quantizer.scale.fill_(1 / (2**new_ipos_max))
        return m


# TODO may meed run multi times
class AdjustShiftWriteQOPass(OptPassBase):
    '''
    This is a post-quantization optimization.
    Adjust the shift write of node.
    For Add:
    shift_write = min(ipos) - opos
    NPU compiler constraints of shift_write:
    1. -7 <= shift_write <= 25

    For Mul:
    shift_write = sum(ipos) - opos
    NPU compiler constraints of shift_write:
    1. 0 <= shift_write <= 32

    NOTE: align with quark/onnx/refine.py: adjust_shift_write
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        for n in m.graph.nodes:
            if not (n.op == "call_function" and n.target in ADD_OPS + MUL_OPS
                    and _is_has_one_user_and_followed_quantizer(m, n)):
                continue
            target_node = n
            input_node_1 = target_node.args[0]
            input_node_2 = target_node.args[1]
            output_node = next(iter(target_node.users))
            if (not is_quantizer_node(m, input_node_1)) or (not is_quantizer_node(m, input_node_2)):
                logger.warning("AdjustShiftWrite Node: {} inputs not all quantizer".format(target_node.name))
                continue
            input_quantizer_1 = getattr(m, input_node_1.target)
            input_quantizer_2 = getattr(m, input_node_2.target)
            output_quantizer = getattr(m, output_node.target)
            if input_quantizer_1.scale.numel() != 1 or input_quantizer_2.scale.numel() != 1\
                    or output_quantizer.scale.numel() != 1:
                logger.warning("For DPU version, only support Per-Tensor quantization, Please close hw optimization")
                continue
            input_pos1 = get_quantizer_scale_pos(input_quantizer_1)
            input_pos2 = get_quantizer_scale_pos(input_quantizer_2)
            output_pos = get_quantizer_scale_pos(output_quantizer)
            inputposes = [input_pos1, input_pos2]
            new_sw = None
            if n.target in ADD_OPS:
                '''
                shift_write = min(ipos) - opos
                NPU compiler constraints of shift_write:
                1. -7 <= shift_write <= 25
                '''
                sw = min(inputposes) - output_pos
                min_sw, max_sw = -7, 25

                if sw > max_sw:
                    new_sw = max_sw
                elif sw < min_sw:
                    new_sw = min_sw
                if new_sw is not None:
                    new_opos = min(inputposes) - new_sw
            elif n.target in MUL_OPS:
                '''
                For Mul:
                shift_write = sum(ipos) - opos
                NPU compiler constraints of shift_write:
                1. 0 <= shift_write <= 32
                '''
                sw = sum(inputposes) - output_pos
                min_sw, max_sw = 0, 32
                if sw > max_sw:
                    new_sw = max_sw
                elif sw < min_sw:
                    new_sw = min_sw

                if new_sw is not None:
                    new_opos = sum(inputposes) - new_sw
            if new_sw is not None:
                logger.info("Node: {} output quantizer: {} scale change from {} to {}".format(
                    target_node.name, output_node.name, output_quantizer.scale.item(), 1 / (2**new_opos)))
                output_quantizer.scale.fill_(1 / (2**new_opos))
        return m


class AdjustShiftBase(OptPassBase):
    '''
    Baseclass for adjust_shift_cut & adjust_shift_bias
    Func: get the input, weight, output quantizer
    align with quark/onnx/refine.py: adjust_shift_cut
    '''

    def __init__(self) -> None:
        super().__init__()
        self._target_op = LINEAR_OPS + CONV2D_OPS
        self._target_md = _STRATEGY_SHIFT_CUT_MODULE

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def _is_conv_node(self, m: GraphModule, n: Node) -> bool:
        cond_1 = n.op == "call_function" and n.target in self._target_op
        cond_2 = is_call_module_node(n) and isinstance(getattr(m, n.target), self._target_md)  # type:ignore[arg-type]
        return cond_1 or cond_2

    def get_input_weight_output_quantizer(self, m: GraphModule, n: Node) -> Optional[List[Any]]:
        if not self._is_conv_node(m, n):
            return None
        conv_like_node = n
        # 1. get input quantizer
        i_or_i_qt_node = conv_like_node.args[0]
        if (not isinstance(i_or_i_qt_node, Node)) or (not is_quantizer_node(m, i_or_i_qt_node)):
            logger.warning("AdjustShiftBase Skip as Node: {} args[0] is not Node/quantized.".format(
                conv_like_node.name))
            return None
        input_quantizer = getattr(m, i_or_i_qt_node.target)

        # 2. get weight quantizer
        if conv_like_node.op == "call_function":
            w_or_w_qt_node = conv_like_node.args[1]
            if (not isinstance(w_or_w_qt_node, Node)) or (not is_quantizer_node(m, w_or_w_qt_node)):
                logger.warning("AdjustShiftBase Skip as Node: {} args[1] is not Node/quantized".format(
                    conv_like_node.name))
                return None
            weight_quantizer = getattr(m, w_or_w_qt_node.target)
        else:
            conv_like_module = getattr(m, conv_like_node.target)  # type:ignore
            weight_quantizer = conv_like_module._weight_quantizer
            if (weight_quantizer is None) or (not is_quantizer(weight_quantizer)):
                logger.warning("AdjustShiftBase Skip as Node: {} weight are not quantized".format(conv_like_node.name))
                return None
        # 3. get output quantizer
        o_qt_node = _get_out_quantizer_node(m, conv_like_node, 0)
        if o_qt_node is None:
            logger.warning("AdjustShiftBase Skip as Node: {} output are not quantized".format(conv_like_node.name))
            return None
        output_quantizer = getattr(m, o_qt_node.target)  # type:ignore[arg-type]
        return [input_quantizer, weight_quantizer, output_quantizer]


# TODO align with Quark ONNX may only apply for conv/linear
# NOTE For e.g QuantConv, input quantizer is outof the Module, diff with eager mode
class AdjustShiftCutQOPass(AdjustShiftBase):
    '''
        Adjust the shift cut of nodes.

        shift_cut = wpos + ipos - opos

        DPU compiler constraints of shift_cut:
        1. 0 <= shift_cut <= 16

    NOTE: align with quark/onnx/refine.py: adjust_shift_cut
    '''

    def call(self, m: GraphModule) -> GraphModule:
        for n in m.graph.nodes:
            may_quantizers = self.get_input_weight_output_quantizer(m, n)
            if may_quantizers is None:
                continue
            conv_like_node = n
            input_quantizer, weight_quantizer, output_quantizer = may_quantizers
            if input_quantizer.scale.numel() != 1 or weight_quantizer.scale.numel(
            ) != 1 or output_quantizer.scale.numel() != 1:
                logger.warning("AdjustShiftCut as input/weight/output quantizer scale is not single scale, \
                                not appliable for NPU deploy, skip")
                continue
            # get pos
            ipos = get_quantizer_scale_pos(input_quantizer)
            wpos = get_quantizer_scale_pos(weight_quantizer)
            opos = get_quantizer_scale_pos(output_quantizer)
            min_sc = 0
            max_sc = 16
            sc = wpos + ipos - opos
            new_sc = None
            if sc < min_sc:
                new_sc = min_sc
            elif sc > max_sc:
                new_sc = max_sc
            if new_sc is not None:
                new_wpos = new_sc + opos - ipos
                logger.info("Node: {} weight quantizer scale change from {} to {}".format(
                    conv_like_node.name, weight_quantizer.scale.item(), 1 / (2**new_wpos)))
                weight_quantizer.scale.fill_(1 / (2**new_wpos))
        return m


# TODO align with Quark ONNX may only apply for conv/linear not transpose
# NOTE as even QuantConv, input quantizer is outof the Module, diff with eager mode
class AdjustShiftBiasQOPass(AdjustShiftBase):
    '''
        Adjust the shift bias of node.

        shift_bias = wpos + ipos - bpos

        DPU compiler constraints of shift_bias:
        1. min(0, -(24 - (8 + shift_cut))) <= shift_bias <= 15
    NOTE: align with quark/onnx/refine.py: adjust_shift_bias
    '''

    def call(self, m: GraphModule) -> GraphModule:
        for n in m.graph.nodes:
            may_quantizers = self.get_input_weight_output_quantizer(m, n)
            if may_quantizers is None:
                continue
            conv_like_node = n
            input_quantizer, weight_quantizer, output_quantizer = may_quantizers

            # get bias quantizer
            if conv_like_node.op == "call_function":
                b_or_b_qt_node = conv_like_node.args[2] if len(conv_like_node.args) >= 3 else None
                if (b_or_b_qt_node is None) or (not is_quantizer_node(m, b_or_b_qt_node)):
                    logger.warning("AdjustShiftBiasQOPass Skip as Node: {} bias is not quantized".format(
                        conv_like_node.name))
                    continue
                bias_quantizer = getattr(m, b_or_b_qt_node.target)
            else:
                conv_like_module = getattr(m, conv_like_node.target)
                bias_quantizer = conv_like_module._bias_quantizer
                if (bias_quantizer is None) or (not is_quantizer(bias_quantizer)):
                    logger.warning("AdjustShiftBias Skip as Node: {} bias is not quantized".format(conv_like_node.name))
                    continue
            if input_quantizer.scale.numel() != 1 or weight_quantizer.scale.numel(
            ) != 1 or output_quantizer.scale.numel() != 1 or bias_quantizer.scale.numel() != 1:
                logger.warning("AdjustShiftBiasQOPass as input/weight/output/bias quantizer scale is not single scale, \
                                not appliable for NPU deploy, skip")
                continue

            ipos = get_quantizer_scale_pos(input_quantizer)
            wpos = get_quantizer_scale_pos(weight_quantizer)
            opos = get_quantizer_scale_pos(output_quantizer)
            bpos = get_quantizer_scale_pos(bias_quantizer)

            shift_cut = wpos + ipos - opos
            min_sb = min(0, -(24 - (8 + shift_cut)))
            # TODO align with onnx code
            # for n in self.model.model.graph.node:
            #         if n.op_type == "LeakyRelu" and n.input[0] == node.output[0]:
            #             min_sb = 0
            max_sb = 15
            shift_bias = wpos + ipos - bpos
            new_sb = None
            if shift_bias < min_sb:
                new_sb = min_sb
            elif shift_bias > max_sb:
                new_sb = max_sb

            if new_sb is not None:
                new_bpos = wpos + ipos - new_sb
                logger.info("Node: {} bias quantizer scale change from {} to {}".format(
                    conv_like_node.name, bias_quantizer.scale.item(), 1 / (2**new_bpos)))
                bias_quantizer.scale.fill_(1 / (2**new_bpos))
        return m


class AdjustHardSigmoidQOPass(OptPassBase):
    '''
        Adjust quantize info of HardSigmoid nodes.
        DPU compiler constraints for HardSigmoid:
        1. input pos of HardSigmoid >= 0 && <= 15
        2. output pos of HardSigmoid >= 7
        3. shift_sigmoid >= 0 && shift_sigmoid <= 31 where
            shift_sigmoid = 14 + 'input pos' - ' output pos'

    NOTE: align with quark/onnx/refine.py: adjust_hard_sigmoid
    TODO: align with check_hard_sigmoid_condition (seems ok)
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        for n in m.graph.nodes:
            if not is_hardsigmoid_node(n):
                continue
            hardsigmoid_node = n

            i_qt_node = hardsigmoid_node.args[0]
            o_qt_node = _get_out_quantizer_node(m, hardsigmoid_node, 0)

            if (not isinstance(i_qt_node, Node)) or (not isinstance(o_qt_node, Node)) or (not is_quantizer_node(m, i_qt_node)) or\
                    (not is_quantizer_node(m, o_qt_node)):
                logger.warning(
                    "AdjustHardSigmoid for node: {}, scale is not single scale, skip optimize for NPU".format(
                        hardsigmoid_node.name))
                continue
            input_quantizer = getattr(m, i_qt_node.target)
            output_quantizer = getattr(m, o_qt_node.target)

            if input_quantizer.scale.numel() != 1 or output_quantizer.scale.numel() != 1:
                logger.warning(
                    "AdjustHardSigmoid for node: {}, scale is not single scale, skip optimize for NPU".format(
                        hardsigmoid_node.name))
                continue
            ipos = get_quantizer_scale_pos(input_quantizer)
            opos = get_quantizer_scale_pos(output_quantizer)

            new_ipos = ipos if ipos > 0 else 0
            new_ipos = new_ipos if new_ipos <= 15 else 15

            new_opos = opos if opos > 7 else 7
            shift_sigmoid = 14 + new_ipos - new_opos  # will not bigger than 31 now
            new_opos = new_opos if shift_sigmoid > 0 else 14 + new_ipos

            if new_ipos != ipos:
                logger.info("Node: {} input quantizer scale change from {} to {}".format(
                    hardsigmoid_node.name, input_quantizer.scale.item(), 1 / (2**new_ipos)))
                input_quantizer.scale.fill_(1 / (2**new_ipos))
            if new_opos != opos:
                logger.info("Node: {} outputbias quantizer scale change from {} to {}".format(
                    hardsigmoid_node.name, output_quantizer.scale.item(), 1 / (2**new_opos)))
                output_quantizer.scale.fill_(1 / (2**new_opos))
        return m


class AdjustShiftSwishQOPass(OptPassBase):
    '''
        Adjust the shift of Swish layer's Multiply op.
        shift_swish = 'input 0 pos' + 'input 1 pos' - 'output pos'
        DPU compiler constraints of shift_swish:
          1. 0 <= shift_swish <= 15
    NOTE: align with quark/onnx/refine.py: adjust_shift_swish
    NOTE & TODO : x = x * sigmoid(x) is equal to nn.SiLU()
          x = relu(x) * sigmoid(x) is not considered.
    NOTE: this strategy should prioritize ConvertHardSigmoidDpuVersionQOPass to execute
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def _followed_by_sigmoid(self, n: Node) -> bool:
        if len(n.users) > 2:
            return False
        for each_user in n.users.keys():
            if is_hardsigmoid_node(each_user):
                return True
        return False

    def _is_silu_block(self, g: GraphModule, n1: Node, n2: Node, mul_node: Node) -> bool:
        '''
             quantizer(n1)
                |
              -   -
            /       |
(aten.op)sigmoid    |
            |       |
    quantizer(n2)   |
            |       |
            |       |
               Mul
                |
            quantizer
        '''
        node_1, node_2 = None, None
        if self._followed_by_sigmoid(n1) and (not self._followed_by_sigmoid(n2)):
            node_1, node_2 = n1, n2
        elif (not self._followed_by_sigmoid(n1)) and (self._followed_by_sigmoid(n2)):
            node_1, node_2 = n2, n1
        else:
            return False
        if len(node_1.users) != 2:
            return False
        sigmoid_node = list(node_1.users.keys())[0] if is_hardsigmoid_node(list(node_1.users.keys())[0]) else list(
            node_1.users.keys())[1]
        if not _is_has_one_user_and_followed_quantizer(g, sigmoid_node):
            return False
        sigmoid_quantizer = next(iter(sigmoid_node.users))
        if sigmoid_quantizer != node_2:
            return False
        return True

    def call(self, m: GraphModule) -> GraphModule:
        for n in m.graph.nodes:
            if (not is_mul_node(n)) or (not _is_has_one_user_and_followed_quantizer(m, n)):
                continue
            mul_node = n
            input_1, input_2 = mul_node.args[0], mul_node.args[1]
            if (not is_quantizer_node(m, input_1)) or (not is_quantizer_node(m, input_2)):
                logger.warning("Node: {} input is not all quantizer".format(mul_node.name))
                continue

            if not self._is_silu_block(m, input_1, input_2, mul_node):
                continue
            output_node = next(iter(mul_node.users))
            input_1_quantizer = getattr(m, input_1.target)
            input_2_quantizer = getattr(m, input_2.target)
            output_quantizer = getattr(m, output_node.target)

            if output_quantizer.scale.numel() != 1 or input_1_quantizer.scale.numel(
            ) != 1 or input_2_quantizer.scale.numel() != 1:
                logger.warning(
                    "AdjustShiftSwish, for AMD NPU, only support Per-Tensor quantization, Please close hw optimization")
                continue

            ipos0 = get_quantizer_scale_pos(input_1_quantizer)
            ipos1 = get_quantizer_scale_pos(input_2_quantizer)
            opos = get_quantizer_scale_pos(output_quantizer)
            min_sh, max_sh = 0, 15
            shift_swish = ipos0 + ipos1 - opos
            new_opos = opos
            if (shift_swish < min_sh):
                new_opos = ipos0 + ipos1 - min_sh
            elif (shift_swish > max_sh):
                new_opos = ipos0 + ipos1 - max_sh

            if new_opos != opos:
                logger.info("AdjustShiftSwish: Node: {} outputbias quantizer scale change from {} to {}".format(
                    mul_node.name, output_quantizer.scale.item(), 1 / (2**new_opos)))
                output_quantizer.scale.fill_(1 / (2**new_opos))
        return m


class ConvertHardSigmoidDpuVersionQOPass(OptPassBase):
    '''
        Convert HardSigmoid to DPU version.

    NOTE: quark/onnx/simulate_dpu.py convert_hard_sigmoid_to_dpu_version
    '''

    def __init__(self) -> None:
        self.hard_sigmoid_scale = (2731. / 16384.) / (1. / 6.)

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        for n in m.graph.nodes:
            if (not is_hardsigmoid_node(n)) or (not _is_has_one_user_and_followed_quantizer(m, n)):
                continue
            hardsigmoid_node = n
            input_node = hardsigmoid_node.args[0]
            output_node = next(iter(hardsigmoid_node.users))
            if (not is_quantizer_node(m, input_node)):
                logger.warning("Node: {} input is not all quantizer".format(hardsigmoid_node.name))
                continue
            input_quantizer = getattr(m, input_node.target)
            output_quantizer = getattr(m, output_node.target)
            if input_quantizer.scale.numel() != 1 or output_quantizer.scale.numel() != 1:
                logger.warning(
                    "At present, for AMD NPU, only support Per-Tensor quantization, Please close hw optimization")
                continue

            # insert the Mul node after the hardsigmoid node
            with m.graph.inserting_after(hardsigmoid_node):
                mul_node = m.graph.create_node(
                    'call_function',
                    torch.ops.aten.mul.Tensor,  # type: ignore[attr-defined]
                    (None, self.hard_sigmoid_scale),
                    {})
                # NOTE modify the node's meta info
                _copy_node_meta_info(org_node=hardsigmoid_node, target_node=mul_node)
                mul_node.meta["skip_quant"] = True  # NOTE this is post quant strategy, so set to True
                hardsigmoid_node.replace_all_uses_with(mul_node)
                mul_node.update_arg(0, hardsigmoid_node)

                logger.info("Found HardSigmoid node: {}Convert to DPU version. Mul with scale: {}".format(
                    hardsigmoid_node.name, self.hard_sigmoid_scale))

        m.graph.eliminate_dead_code()
        m.recompile()
        return m
