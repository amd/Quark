#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from tqdm import tqdm
from typing import Any, TYPE_CHECKING, List, Dict, Sequence
import torch
import torch.nn as nn
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.algorithm.utils.prepare import get_model_layers
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.rotation.rotation_utils import transform_norm_and_linear, get_rotation_matrix, rotate_in_channels, rotate_out_channels
from quark.torch.algorithm.utils.module import get_nested_attr_from_module

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import RotationConfig

__all__ = ["RotationProcessor"]


class RotationProcessor(BaseAlgoProcessor):

    def __init__(self, model: nn.Module, pre_quant_opt_config: RotationConfig, _data_loader: Any) -> None:
        self.pre_quant_opt_config = pre_quant_opt_config
        self.scaling_modules = self.pre_quant_opt_config.scaling_layers
        self.modules = get_model_layers(model, self.pre_quant_opt_config.model_decoder_layers)
        self.scaling_layers = self.get_scaling_layers()
        assert self.scaling_layers is not None
        self.model = model

    def apply(self) -> None:
        rotation = get_rotation_matrix(self.model.config.hidden_size, random=self.pre_quant_opt_config.random)
        self.rotate(rotation)
        clear_memory()

    def rotate(self, rotation: torch.Tensor) -> None:

        rotated_list = []

        for layers_pattern in tqdm(self.scaling_layers, desc="Rotation:"):
            prev_modules = [
                get_nested_attr_from_module(self.model, layer_name) for layer_name in layers_pattern["prev_modules"]
            ]
            norm_module = get_nested_attr_from_module(self.model, layers_pattern["norm_module"])
            next_modules = [
                get_nested_attr_from_module(self.model, layer_name) for layer_name in layers_pattern["next_modules"]
            ]

            prev_out_channels_dims = self.get_prev_out_channels_dims(prev_modules)

            transform_norm_and_linear(prev_modules=prev_modules,
                                      norm_module=norm_module,
                                      next_modules=next_modules,
                                      prev_out_channels_dims=prev_out_channels_dims)

            for index in range(len(prev_out_channels_dims)):
                if prev_out_channels_dims[index] == 0 and prev_modules[index] not in rotated_list:
                    rotate_out_channels(prev_modules[index], rotation=rotation)
                    rotated_list.append(prev_modules[index])
                else:
                    if prev_modules[index] not in rotated_list:
                        rotate_in_channels(prev_modules[index], rotation=rotation)
                        rotated_list.append(prev_modules[index])

            for fc in next_modules:
                if fc not in rotated_list:
                    rotate_in_channels(fc, rotation=rotation)
                    rotated_list.append(fc)

    def get_prev_out_channels_dims(self, prev_modules: List[nn.Module]) -> List[int]:
        prev_out_channels_dims = []
        for module in prev_modules:
            if isinstance(module, nn.Embedding):
                prev_out_channels_dims.append(1)
            elif isinstance(module, nn.Linear):
                prev_out_channels_dims.append(0)
            else:
                raise ValueError("prev_modules is wrong")
        return prev_out_channels_dims

    def get_scaling_layers(self) -> List[Dict[str, Sequence[str]]]:
        scaling_layers = []
        for i in range(len(self.modules)):
            scaling_layers_cur = []
            if i == 0:
                for layers_pattern in self.scaling_modules["first_layer"]:
                    scaling_layers_cur.append({
                        "prev_modules":
                        [layer_name.replace("layer_id", str(i)) for layer_name in layers_pattern["prev_modules"]],
                        "norm_module":
                        layers_pattern["norm_module"].replace("layer_id", str(i)),
                        "next_modules":
                        [layer_name.replace("layer_id", str(i)) for layer_name in layers_pattern["next_modules"]]
                    })
            else:
                for layers_pattern in self.scaling_modules["middle_layers"]:
                    scaling_layers_cur.append({
                        "prev_modules": [
                            layer_name.replace("pre_layer_id", str(i - 1)).replace("layer_id", str(i))
                            for layer_name in layers_pattern["prev_modules"]
                        ],
                        "norm_module":
                        layers_pattern["norm_module"].replace("layer_id", str(i)),
                        "next_modules":
                        [layer_name.replace("layer_id", str(i)) for layer_name in layers_pattern["next_modules"]]
                    })
                if i == len(self.modules) - 1:
                    for layers_pattern in self.scaling_modules["last_layer"]:
                        scaling_layers_cur.append({
                            "prev_modules":
                            [layer_name.replace("layer_id", str(i)) for layer_name in layers_pattern["prev_modules"]],
                            "norm_module":
                            layers_pattern["norm_module"].replace("layer_id", str(i)),
                            "next_modules":
                            [layer_name.replace("layer_id", str(i)) for layer_name in layers_pattern["next_modules"]]
                        })
            scaling_layers.extend(scaling_layers_cur)
        return scaling_layers
