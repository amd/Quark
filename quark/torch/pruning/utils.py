#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy
import torch.nn as nn
from typing import Optional
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.pruning.config import Config
from quark.torch.algorithm.utils.utils import get_device_map, set_device_map
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def pre_process_tuning(model: nn.Module, config: Config, is_accelerate: Optional[bool]) -> nn.Module:

    if config.blockwise_tuning_config is not None:

        device_map = get_device_map(model, is_accelerate)

        model.cpu()
        clear_memory()

        fp_model = copy.deepcopy(model)

        model = set_device_map(model, device_map)

        return fp_model
    else:
        return model
