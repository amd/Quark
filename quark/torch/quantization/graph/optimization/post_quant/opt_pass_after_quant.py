#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from torch.fx import GraphModule, Node
from quark.torch.quantization.graph.torch_utils import is_clip_node
from quark.torch.quantization.graph.optimization.opt_pass_manager import OptPassBase
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, FreezedScaledFakeQuantize
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ["ConvertClip2ReLUQOPass"]


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

    # def _condition_check()
    def call(self, m: GraphModule) -> GraphModule:
        '''
        convert a clip layer to relu layer
        (only activate under condition that relu can be act as clip)
        '''
        to_delete_node = []
        for n in m.graph.nodes:
            if not is_clip_node(n):
                continue
            clip_node = n
            clip_min, clip_max = clip_node.args[1], clip_node.args[2]

            if isinstance(clip_min, Node) or isinstance(clip_max, Node):
                raise NotImplementedError("clip node's min/max are not float, found Node")

            # condition check
            if clip_min < 0:
                continue  # could not be replaced with Relu
            if len(clip_node.users) != 1:
                continue
            user_node = list(clip_node.users.keys())[0]
            # NOTE TODO may change futhre
            if not (user_node.op == 'call_module' and isinstance(getattr(m, user_node.target),
                                                                 (FreezedScaledFakeQuantize, FakeQuantizeBase))):
                continue

            # logger.info("After quant, found clip node: {} convert to Relu Node".format(clip_node.name))
            input_activation_node = clip_node.args[0]
            to_delete_node.append(clip_node)
            with m.graph.inserting_after(input_activation_node):
                relu_node = m.graph.create_node(
                    'call_function',
                    torch.ops.aten.relu_.default,  # type: ignore[attr-defined]
                    (input_activation_node, ),
                    {})
                relu_node.meta = clip_node.meta  # TODO use deepcopy future
                clip_node.replace_all_uses_with(relu_node)
        if len(to_delete_node):
            logger.info("Totally replace ops.aten.clip to ops.aten.relu count:\t{}.".format(len(to_delete_node)))
            [m.graph.erase_node(node) for node in to_delete_node]
            m.graph.eliminate_dead_code()
            m.recompile()
        return m
