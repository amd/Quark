#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from abc import ABC, abstractmethod

from torch.fx import GraphModule


class Transform(ABC):  # noqa
    pass


class GraphTransform(Transform):
    @abstractmethod
    def apply(self, graph_model: GraphModule) -> GraphModule:
        pass
