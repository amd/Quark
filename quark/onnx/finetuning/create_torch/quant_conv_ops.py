#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch

from .quant_base_ops import QuantizeWrapper
from torch.nn import (Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d)
from typing import Any


class QConv1d(QuantizeWrapper, Conv1d):

    def __init__(self, **kwargs: Any) -> None:
        QuantizeWrapper.__init__(self, **kwargs)
        Conv1d.__init__(self, **kwargs)

    def forward_impl(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv1d(input,
                                          weight,
                                          bias=None,
                                          stride=self.stride,
                                          padding=self.padding,
                                          dilation=self.dilation,
                                          groups=self.groups)


class QConv2d(QuantizeWrapper, Conv2d):

    def __init__(self, **kwargs: Any) -> None:
        QuantizeWrapper.__init__(self, **kwargs)
        Conv2d.__init__(self, **kwargs)

    def forward_impl(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(input,
                                          weight,
                                          bias=None,
                                          stride=self.stride,
                                          padding=self.padding,
                                          dilation=self.dilation,
                                          groups=self.groups)


class QConv3d(QuantizeWrapper, Conv3d):

    def __init__(self, **kwargs: Any) -> None:
        QuantizeWrapper.__init__(self, **kwargs)
        Conv3d.__init__(self, **kwargs)

    def forward_impl(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv3d(input,
                                          weight,
                                          bias=None,
                                          stride=self.stride,
                                          padding=self.padding,
                                          dilation=self.dilation,
                                          groups=self.groups)


class QConvTranspose1d(QuantizeWrapper, ConvTranspose1d):

    def __init__(self, **kwargs: Any) -> None:
        QuantizeWrapper.__init__(self, **kwargs)
        ConvTranspose1d.__init__(self, **kwargs)

    def forward_impl(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.padding, tuple)
        return torch.nn.functional.conv_transpose1d(input,
                                                    weight,
                                                    bias=None,
                                                    stride=self.stride,
                                                    padding=self.padding,
                                                    output_padding=self.output_padding,
                                                    dilation=self.dilation,
                                                    groups=self.groups)


class QConvTranspose2d(QuantizeWrapper, ConvTranspose2d):

    def __init__(self, **kwargs: Any) -> None:
        QuantizeWrapper.__init__(self, **kwargs)
        ConvTranspose2d.__init__(self, **kwargs)

    def forward_impl(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.padding, tuple)
        return torch.nn.functional.conv_transpose2d(input,
                                                    weight,
                                                    bias=None,
                                                    stride=self.stride,
                                                    padding=self.padding,
                                                    output_padding=self.output_padding,
                                                    dilation=self.dilation,
                                                    groups=self.groups)


class QConvTranspose3d(QuantizeWrapper, ConvTranspose3d):

    def __init__(self, **kwargs: Any) -> None:
        QuantizeWrapper.__init__(self, **kwargs)
        ConvTranspose3d.__init__(self, **kwargs)

    def forward_impl(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.padding, tuple)
        return torch.nn.functional.conv_transpose3d(input,
                                                    weight,
                                                    bias=None,
                                                    stride=self.stride,
                                                    padding=self.padding,
                                                    output_padding=self.output_padding,
                                                    dilation=self.dilation,
                                                    groups=self.groups)
