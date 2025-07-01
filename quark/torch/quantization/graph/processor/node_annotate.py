#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch.fx
from typing import List
from quark.torch.quantization.graph.processor.processor import mark_exclude_nodes
# Graph

__all__ = ["mark_exclude_quant_node"]


def mark_exclude_quant_node(model: torch.fx.GraphModule) -> List[str]:
    '''
    # TODO move code here
    '''
    exclude_node_name_list = mark_exclude_nodes(model)
    return exclude_node_name_list
