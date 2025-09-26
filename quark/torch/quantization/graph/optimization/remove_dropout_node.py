#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from typing import List

from torch.fx import GraphModule, Node

from quark.torch.quantization.graph.fx.base import GraphTransform
from quark.torch.quantization.graph.torch_utils import is_dropout_node


class RemoveDropoutNode(GraphTransform):
    def __init__(self) -> None:
        super(RemoveDropoutNode, self).__init__()

    def apply(self, graph_model: GraphModule) -> GraphModule:
        need_to_delete_node: list[Node] = []
        # func: dropout(Tensor input, float p, bool train) -> Tensor
        for node in graph_model.graph.nodes:
            if not is_dropout_node(node):
                continue
            dropout_node = node
            need_to_delete_node.append(dropout_node)
            input_node = dropout_node.args[0]
            dropout_node.replace_all_uses_with(input_node)

        [graph_model.graph.erase_node(node) for node in need_to_delete_node]
        graph_model.graph.eliminate_dead_code()
        graph_model.recompile()
        return graph_model
