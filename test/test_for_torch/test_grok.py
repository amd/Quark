#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
from quark.torch.algorithm.awq.scale import scale_ln_fcs
from quark.torch.algorithm.utils.prepare import get_layers_for_scaling
from quark.shares.utils.testing_utils import torch_device


def test_grok():
    class RMSNorm(nn.Module):
        def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-5,
            create_scale: bool = True,
        ) -> None:
            super().__init__()
            self.variance_epsilon = eps
            self.w = nn.Linear(in_features=512, out_features=512).to(device=torch_device)
            if create_scale:
                self.scale = nn.Parameter(torch.ones(hidden_size))
            else:
                self.scale = 1.0

    fake_layer = RMSNorm(hidden_size=512).to(device=torch_device)
    linear_layer = nn.Linear(in_features=512, out_features=512).to(device=torch_device)
    scale_ln_fcs(ln=fake_layer, fcs=[linear_layer], scales=torch.tensor(1.0).to(device=torch_device))

    scaling_layers = [
        {
            "prev_op": "w",
            "layers": ["w"],
            "inp": "w",
            "module2inspect": "",
        },
        {
            "prev_op": "w",
            "layers": ["w"],
            "inp": "w",
            "module2inspect": "",
        }]

    get_layers_for_scaling(fake_layer, input_feat={"w": torch.tensor(1.0).to(device=torch_device)}, module_kwargs=None, scaling_layers=scaling_layers)

if __name__ == "__main__":
    test_grok()
