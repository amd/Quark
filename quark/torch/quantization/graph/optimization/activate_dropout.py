#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from torch.fx import GraphModule
from quark.torch.quantization.graph.fx.base import GraphTransform
from quark.torch.quantization.graph.torch_utils import is_dropout_node
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ActivateDropoutNode(GraphTransform):

    def __init__(self) -> None:
        super(ActivateDropoutNode, self).__init__()
        # NOTE: dropout may different under different device(cpu, cuda, rocm)

    def apply(self, graph_model: GraphModule, activate: bool = True) -> GraphModule:
        find_dropout = False
        for node in graph_model.graph.nodes:
            if not is_dropout_node(node):
                continue
            find_dropout = True
            if activate is True:
                node.args = (node.args[0], node.args[1], True)
            else:
                node.args = (node.args[0], node.args[1], False)

        logger.info("Whether find Droutout: {}, change mode to: {}".format(find_dropout, activate))
        graph_model.graph.eliminate_dead_code()
        graph_model.recompile()
        return graph_model
