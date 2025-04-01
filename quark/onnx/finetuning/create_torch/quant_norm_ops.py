#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch

from .quant_base_ops import QuantizeWrapper
from torch.nn import InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, LayerNorm
from typing import Any


class QInstanceNorm1d(QuantizeWrapper, InstanceNorm1d):

    def __init__(self, **kwargs: Any) -> None:
        QuantizeWrapper.__init__(self, **kwargs)
        InstanceNorm1d.__init__(self, **kwargs)

    def forward_impl(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.instance_norm(
            input,
            running_mean=None,
            running_var=None,
            weight=weight,
            bias=None,
            use_input_stats=True,
            momentum=self.momentum,  # type: ignore
            eps=self.eps)


class QInstanceNorm2d(QuantizeWrapper, InstanceNorm2d):

    def __init__(self, **kwargs: Any) -> None:
        QuantizeWrapper.__init__(self, **kwargs)
        InstanceNorm2d.__init__(self, **kwargs)

    def forward_impl(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.instance_norm(
            input,
            running_mean=None,
            running_var=None,
            weight=weight,
            bias=None,
            use_input_stats=True,
            momentum=self.momentum,  # type: ignore
            eps=self.eps)


class QInstanceNorm3d(QuantizeWrapper, InstanceNorm3d):

    def __init__(self, **kwargs: Any) -> None:
        QuantizeWrapper.__init__(self, **kwargs)
        InstanceNorm3d.__init__(self, **kwargs)

    def forward_impl(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.instance_norm(
            input,
            running_mean=None,
            running_var=None,
            weight=weight,
            bias=None,
            use_input_stats=True,
            momentum=self.momentum,  # type: ignore
            eps=self.eps)


class QLayerNorm(QuantizeWrapper, LayerNorm):

    def __init__(self, **kwargs: Any) -> None:
        QuantizeWrapper.__init__(self, **kwargs)
        LayerNorm.__init__(self, **kwargs)

    def forward_impl(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(input, self.normalized_shape, weight=weight, bias=None, eps=self.eps)
