#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from torch import ops  # type: ignore[attr-defined]
from torch.fx import GraphModule, Node
from quark.torch.quantization.graph.fx.base import GraphTransform


class RemoveDropoutNode(GraphTransform):

    def __init__(self) -> None:
        super(RemoveDropoutNode, self).__init__()

    def apply(self, graph_model: GraphModule) -> GraphModule:
        for node in graph_model.graph.nodes:
            if node.op != "call_function" or node.target != ops.aten.clone.default:
                continue
            if node.meta["original_aten"]._name == ops.aten.dropout.default._name:
                if self.node_has_single_io(node, graph_model):
                    self.remove_single_io_nodes(graph_model, node)

        graph_model.graph.eliminate_dead_code()
        graph_model.recompile()
        return graph_model

    def remove_single_io_nodes(self, graph_model: GraphModule, node: Node) -> None:
        for user in list(node.users):
            user.replace_input_with(node, node.args[0])
        graph_model.graph.erase_node(node)

    def node_has_single_io(
        self,
        node: Node,
        graph_model: GraphModule,
    ) -> bool:
        has_single_input = len(node.args) == 1
        outputs_count = sum(1 for n in graph_model.graph.nodes if node in n.args)
        has_single_output = outputs_count == 1
        return has_single_input and has_single_output
