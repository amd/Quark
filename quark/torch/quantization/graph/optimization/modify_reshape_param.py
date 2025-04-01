#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from torch.fx import GraphModule
from quark.torch.quantization.graph.torch_utils import is_reshape_node
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def modify_reshape_param(m: GraphModule) -> GraphModule:
    '''
    In some case, reshape param in fx graph is traced by input datashape.
    For example: reshape.default(conv2d_85, [25, 80, 6400]) # where 25 is batch size
    we can change: [25, 80, 6400] to [-1, 80, 6400]
    '''
    for n in m.graph.nodes:
        if not is_reshape_node(n):
            continue
        reshape_node = n
        shape_param = reshape_node.args[1]
        if -1 in shape_param:
            continue
        new_shape_param = [-1] + shape_param[1:]
        reshape_node.update_arg(1, new_shape_param)
        logger.warning("For reshape node: {}, change from {} to {}".format(reshape_node.name, shape_param,
                                                                           new_shape_param))

    m.graph.eliminate_dead_code()
    m.recompile()
    return m
