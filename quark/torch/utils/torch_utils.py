#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch.nn as nn


def get_op_name(root_module: nn.Module, op: nn.Module) -> str:
    # get the name of the op relative to the module
    for name, submodule in root_module.named_modules():
        if submodule is op:
            return name  # type: ignore
    raise ValueError(f"Cannot find op {op} in module {root_module}")
