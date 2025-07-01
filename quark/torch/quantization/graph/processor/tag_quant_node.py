#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from typing import List
from torch.fx import GraphModule, Node
from quark.torch.quantization.graph.ops import DeQuantStub, QuantStub
from quark.torch.quantization.graph.export.onnx_export import register_custom_ops
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ['tag_quant_nodes', 'mask_op_with_no_grad_no_quant']


def _mark_node_skip_quant(node: Node, skip_quant: bool = True) -> None:
    assert hasattr(node, "meta")
    node.meta["skip_quant"] = skip_quant


def tag_quant_nodes(m: GraphModule) -> None:
    """
    if the user specifies the quant scope (start from QuantStub and end with DeQuantStub),
    Only the operations that are within the scope will be quantized.
    e.g. (users' network)
        network_block_1 -> QuantStub -> network_block_2 -> DeQuantStub -> network_block_3
    meaning only the network_block_2 will be quantized, other network parts will keep the original format(e.g FP32)
    """

    # NOTE case 1
    # if user not specify the quant start and end point, all node set quantiable by dafault.
    if not any([node.target in [DeQuantStub.default, QuantStub.default] for node in m.graph.nodes]):
        logger.debug("As user not specify Quant scpoe, all layers/operations will be quantize by default, \n" +
                     "if want to partly quant the model, please use QuantStub & QuantStub to specify the quant scope.")
        for node in m.graph.nodes:
            _mark_node_skip_quant(node, False)
        return

    # only contain DeQuantStub and QuantStub then we need register the quark operation
    register_custom_ops()
    # NOTE case 2
    if not any([node.target == QuantStub.default for node in m.graph.nodes]):
        raise ValueError("Found DeQuantStub in model, but not found QuantStub, Please modify the input model")
    # tag the node that among the quantable scope

    # ------session 1, dfs from input to output, depth first
    def depth_first_search(node: Node, visited: List[Node]) -> None:
        if node.target == DeQuantStub.default:
            _mark_node_skip_quant(node)
            return

        visited.append(node)
        _mark_node_skip_quant(node, False)
        for user_node in node.users.keys():
            if not isinstance(user_node, Node):
                continue
            if user_node not in visited:
                depth_first_search(user_node, visited)

    source: List[Node] = []
    for node in m.graph.nodes:
        if node.target == QuantStub.default:
            source.append(node)

    visited: List[Node] = []
    for source_node in source:
        depth_first_search(source_node, visited)

    # ------session2,expand scope from node, width first
    def width_first_search(node: Node, visited: List[Node]) -> None:
        if node.target == DeQuantStub.default:
            _mark_node_skip_quant(node)
            return

        visited.append(node)
        _mark_node_skip_quant(node, False)
        for user_node in node.args:
            if not isinstance(user_node, Node):
                continue
            if user_node not in visited:
                width_first_search(user_node, visited)

    expand_source = []
    for node in visited:
        for input_node_or_args in node.args:
            if (isinstance(input_node_or_args, Node)
                    and input_node_or_args.target not in [QuantStub.default, DeQuantStub.default]
                    and input_node_or_args not in visited
                    and node.target not in [QuantStub.default, DeQuantStub.default]):
                expand_source.append(input_node_or_args)

    width_visited: List[Node] = []
    for source_node in expand_source:
        width_first_search(source_node, width_visited)

    # tag other nodes that out of the quantable scope
    for node in m.graph.nodes:
        # type 'get_attr' no need to tag unquantable
        if node.meta.get("skip_quant", None) is None:
            if node.op in ["placeholder", "output"]:  # "get_attr",
                node.meta["skip_quant"] = False
                continue
            else:
                node.meta["skip_quant"] = True
    return


def mask_op_with_no_grad_no_quant(model: torch.fx.GraphModule) -> List[str]:
    # TODO haoliang this is a temponary func, hope to use QuantStub and DeQuantStub
    # NOTE this is tempory func and may be changed in the future.
    '''
    For assuming that the operations that no need grad will not be quantized
    e.g:
        op0 = **
        _set_grad_enabled_1 = torch._C._set_grad_enabled(False)
        op1 = **
        op2 = **
        _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
        op3 = **
    Tha above eample we will not intend to quant op1 & op2, so we mark op1 & op2 not to quant.
    '''
    skip_quant = False
    skip_quant_node_name = []
    for node in model.graph.nodes:
        if node.op == 'call_function' and node.target == torch._C._set_grad_enabled:
            skip_quant = True if node.args[0] is False else False
        node.meta['skip_quant'] = skip_quant if skip_quant else node.meta['skip_quant']
        if skip_quant:
            skip_quant_node_name.append(node.name)
    return skip_quant_node_name
