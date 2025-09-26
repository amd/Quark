#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Pruning Config API for PyTorch"""

from __future__ import annotations

from abc import ABC
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

T = TypeVar("T", bound="ConfigBase")


@dataclass(eq=True)
class ConfigBase(ABC):
    name = ""

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        return cls(**data)

    def update_from_dict(self, data: dict[str, Any]) -> None:
        for field_name in data:
            setattr(self, field_name, data[field_name])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(eq=True)
class Config(ConfigBase):
    """
    A class that encapsulates comprehensive pruning configurations for a machine learning model, allowing for detailed and hierarchical control over pruning parameters across different model components.

    :param Optional[AlgoConfig] algo_config: Optional configuration for the pruning algorithm, such as OSSCAR. After this process, the params will be reduced. Default is None.
    :param Optional[int] log_severity_level: 0:DEBUG, 1:INFO, 2:WARNING. 3:ERROR, 4:CRITICAL/FATAL. Default is 1.
    """

    # Optional configuration for the pruning algorithm, such as OSSCAR
    # After this process, the datatype/fake_datatype of weights will be changed with pruning scales.
    algo_config: AlgoConfig | None = None

    blockwise_tuning_config: AlgoConfig | None = None

    # Log level for printing on screen
    log_severity_level: int | None = 1


@dataclass
class AlgoConfigBase(ConfigBase):
    pass


@dataclass
class AlgoConfig(AlgoConfigBase):
    pass


@dataclass
class OSSCARConfig(AlgoConfig):
    name: str = "osscar"
    damp_percent: float = 0.01
    true_sequential: bool = True
    inside_layer_modules: list[str] = field(default_factory=list)
    mlp_pruning_modules: list[str] = field(default_factory=list)
    mlp_scaling_layers: dict[str, str | None] = field(default_factory=dict)
    mlp_pruning_ratio: float = 0.1
    mlp_intermediate_size_name: str = field(default_factory=str)
    model_decoder_layers: str = field(default_factory=str)


@dataclass
class LayerImportancePruneConfig(AlgoConfig):
    """
    Configuration for layer importance depth wise prune algorithm (for LLM model).

    :param int delete_layer_num: Number of layers to delete (at least 1).
    :param List[int] delete_layers_index: Specific indexes of layers to delete.
    :param bool save_gpu_memory: Whether to save GPU memory (tradeoff speed vs. memory).
    :param str layer_num_field: Field name for number of layers.
    :param str model_decoder_layers: Field name for decoder layers.
    :param str layer_norm_field: Field name for normalization layer.
    """

    name: str = "layer_importance_depth_pruning"
    delete_layer_num: int = 1  # at least one layer # NOTE used for search
    delete_layers_index: list[int] = field(default_factory=list)
    save_gpu_memory: bool = False  # balance between speed and
    layer_num_field: str = field(default_factory=str)
    model_decoder_layers: str = field(default_factory=str)
    layer_norm_field: str = field(default_factory=str)


@dataclass
class BlockwiseTuningConfig(AlgoConfig):
    name: str = "blockwise_tuning"
    epochs: int = 5
    weight_lr: float = 0.0001
    weight_decay: float = 0.0
    min_lr_factor: float = 20.0
    max_grad_norm: float = 0.3
    model_decoder_layers: str = field(default_factory=str)
    trainable_modules: list[str] = field(default_factory=list)
