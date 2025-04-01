#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from abc import ABC
from abc import abstractmethod
from torch.fx import GraphModule


class Transform(ABC):
    pass


class GraphTransform(Transform):

    @abstractmethod
    def apply(self, graph_model: GraphModule) -> GraphModule:
        pass
