#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Pruning Config API for PyTorch"""

from __future__ import annotations

from dataclasses import dataclass, field

from quark.common.config import BaseAlgoConfig, BaseConfigImpl


@dataclass(eq=True)
class PConfig(BaseConfigImpl):
    """
    A class that encapsulates comprehensive pruning configurations for a machine learning model, allowing for detailed and hierarchical control over pruning parameters across different model components.

    :param Optional[BaseAlgoConfig] algo_config: Optional configuration for the pruning algorithm, such as OSSCAR. After this process, the params will be reduced. Default is None.
    """

    # Optional configuration for the pruning algorithm, such as OSSCAR
    # After this process, the datatype/fake_datatype of weights will be changed with pruning scales.
    algo_config: BaseAlgoConfig | None = None

    blockwise_tuning_config: BaseAlgoConfig | None = None


@dataclass
class OSSCARConfig(BaseAlgoConfig):
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
class LayerImportancePruneConfig(BaseAlgoConfig):
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
