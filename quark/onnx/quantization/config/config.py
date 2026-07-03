#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization Config API for ONNX"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .algorithm import AlgoConfig
from .algorithm import from_dict as dict_to_algo
from .data_type import DataType
from .legacy import QuantizationConfig
from .spec import Int8Spec, QLayerConfig
from .utils import config_to_dict


@dataclass(eq=True)
class Config:
    """
    A class that encapsulates comprehensive quantization configurations for a machine learning model, allowing for detailed and hierarchical control over quantization parameters across different model components.

    :param QuantizationConfig global_quant_config: Global quantization configuration applied to the entire model unless overridden at the layer level.
    """

    # Global quantization configuration applied to the entire model unless overridden at the layer level.
    global_quant_config: QuantizationConfig


# TODO: Move QConfig into quark/common
@dataclass(eq=True, init=False)
class QConfig:
    """
    A class that defines quantization configuration at multiple levels (global, specific layers, specific operation types),
    and provides flexibility for specifying algorithm settings.

    :param QLayerConfig global_config: Global quantization configuration applied to all layers unless overridden.
    :param Dict[DataType, List[str]] specific_layer_config: Dictionary mapping specific layer names to their quantization
        configuration. Overrides ``global_config`` for those layers. Default is ``None``.
    :param Dict[Optional[DataType], List[str]] layer_type_config: Dictionary mapping layer types (e.g., Conv, Gemm) to
        quantization configurations. Overrides ``global_config`` for those operation types. Default is ``None``.
    :param List[Union[str, List[Tuple[List[str]]]]] exclude: List of nodes or subgraphs excluded from quantization. Default is ``None``.
    :param List[AlgoConfig] algo_config: Algorithm configuration(s), such as CLE, SmoothQuant,
        or AdaRound. Can be a list of algorithm configurations. Default is ``None``.
    :param bool use_external_data_format: Whether to use ONNX external data format when saving the quantized model.
        Default is ``False``.
        advanced customization and extension.
    :param Dict[str, Any] extra_options: Dictionary for additional options. Default is ``None``.
    """

    global_config: QLayerConfig = QLayerConfig(activation=Int8Spec(), weight=Int8Spec())
    specific_layer_config: dict[DataType, list[str]] | None
    layer_type_config: dict[DataType | None, list[str]] | None
    exclude: list[str | list[tuple[list[str]]]] | None
    algo_config: list[AlgoConfig] | None
    use_external_data_format: bool

    def __init__(
        self,
        global_config: QLayerConfig,
        specific_layer_config: dict[DataType, list[str]] | None = None,
        layer_type_config: dict[DataType | None, list[str]] | None = None,
        exclude: list[str | list[tuple[list[str]]]] | None = None,
        algo_config: list[AlgoConfig] | None = None,
        use_external_data_format: bool = False,
        **kwargs: dict[str, Any],
    ):
        self.global_config = global_config
        self.specific_layer_config = specific_layer_config or {}
        self.layer_type_config = layer_type_config or {}
        self.exclude = exclude or []
        self.algo_config = algo_config or []  # type: ignore
        self.use_external_data_format = use_external_data_format
        self.extra_options = kwargs

    def to_dict(self) -> dict[str, Any]:
        return config_to_dict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> QConfig:
        """
        Convert a dictionary into a QConfig object.

        This includes global configuration, per-layer configuration,
        algorithm configurations, and extra options.

        :param dict[str, Any] d: Dictionary representation of a quantization config.
        :return: Constructed QConfig instance.
        """
        if "global_config" in d:
            global_cfg = QLayerConfig.from_dict(d["global_config"])
        else:
            raise ValueError("QConfig dict must contain field 'global_config'.")

        specific_layer_cfg = {
            QLayerConfig.from_dict(batch[0]) if batch[0] is not None else None: batch[1]
            for batch in d.get("specific_layer_config", [])
        }

        layer_type_cfg = {
            QLayerConfig.from_dict(batch[0]) if batch[0] is not None else None: batch[1]
            for batch in d.get("layer_type_config", [])
        }

        algo_cfgs = [dict_to_algo(x) for x in d.get("algo_config", [])]

        extra = d.get("extra_options", {})

        return QConfig(
            global_config=global_cfg,
            specific_layer_config=specific_layer_cfg,
            layer_type_config=layer_type_cfg,
            exclude=d.get("exclude", []),
            algo_config=algo_cfgs,
            use_external_data_format=d.get("use_external_data_format", False),
            **extra,
        )

    @staticmethod
    def get_default_config(config_name: str) -> Config:
        """
        Retrieve the default quantization configuration by name.

        This function looks up the provided `config_name` in the
        `DefaultConfigMapping`. If a match is found, it returns a
        `Config` object with the corresponding global quantization
        configuration. Otherwise, it raises a ValueError.

        Args:
            config_name (str): The name of the default configuration
            to look up like XINT8.

        Returns:
            Config: A configuration object containing the default
            quantization settings.

        Raises:
            ValueError: If the provided `config_name` is not found
            in `DefaultConfigMapping`.
        """
        from . import DefaultConfigMapping

        if config_name in DefaultConfigMapping:
            return Config(global_quant_config=DefaultConfigMapping[config_name])
        else:
            raise ValueError("The quantization config is invalid.")
