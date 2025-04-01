#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
from typing import Dict, Any


def split_model_info(info: Dict[str, Any], params_dict: Dict[str, torch.Tensor]) -> None:
    """Util function to split the weights or any torch.Tensor in nested config to weights."""
    if isinstance(info, dict):
        info_name = info.get("name", None)
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                key_name = f"{info_name}.{k}" if info_name is not None else f"{k}"
                if info_name is not None:
                    if "quant" in info_name.split(".")[-1]:
                        key_name = f"{info_name}_{k}"
                    else:
                        key_name = f"{info_name}.{k}"
                else:
                    key_name = f"{k}"
                params_dict[key_name] = v
                info[k] = f"{key_name}"
            else:
                split_model_info(v, params_dict)
    elif isinstance(info, list):
        for v in info:
            split_model_info(v, params_dict)
