#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from typing import List

import torch
from torch.fx import GraphModule, Node

from quark.shares.utils.log import ScreenLogger
from quark.torch.quantization.graph.optimization.utils import _copy_node_meta_info
from quark.torch.quantization.graph.torch_utils import is_silu_node

logger = ScreenLogger(__name__)


def replace_silu_node(m: GraphModule) -> GraphModule:
    """
    Silu:
        silu(x)= mul(x, sigmoid(x))

    torch.ops.aten.silu.default(input)
    """
    count_num = 0
    need_to_delete_node: list[Node] = []
    for n in m.graph.nodes:
        if not is_silu_node(n):
            continue
        count_num += 1
        silu_node = n
        input_node = silu_node.args[0]
        with m.graph.inserting_after(input_node):
            sigmoid_node = m.graph.create_node(
                "call_function",
                torch.ops.aten.sigmoid.default,  # type: ignore[attr-defined]
                (input_node,),
            )
            _copy_node_meta_info(org_node=silu_node, target_node=sigmoid_node)
        with m.graph.inserting_after(silu_node):
            mul_node = m.graph.create_node(
                "call_function",
                torch.ops.aten.mul.Tensor,  # type: ignore[attr-defined]
                (sigmoid_node, input_node),
            )
            _copy_node_meta_info(org_node=silu_node, target_node=mul_node)
            silu_node.replace_all_uses_with(mul_node)
        need_to_delete_node.append(silu_node)

    if count_num != 0:
        [m.graph.erase_node(node) for node in need_to_delete_node]
        logger.info(f"Replace sliu to x * sigmoid(x), count: {count_num}")
        m.graph.eliminate_dead_code()
        m.recompile()
    return m
