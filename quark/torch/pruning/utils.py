#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy

import torch.nn as nn

from quark.common.utils.log import ScreenLogger
from quark.torch.algorithm.utils.utils import clear_memory, get_device_map, set_device_map
from quark.torch.pruning.config import PConfig

logger = ScreenLogger(__name__)


def pre_process_tuning(model: nn.Module, config: PConfig, is_accelerate: bool | None) -> nn.Module:
    if config.blockwise_tuning_config is not None:
        device_map = get_device_map(model, is_accelerate)

        model.cpu()
        clear_memory()

        fp_model = copy.deepcopy(model)

        model = set_device_map(model, device_map)

        return fp_model
    else:
        return model
