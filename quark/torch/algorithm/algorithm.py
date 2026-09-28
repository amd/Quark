#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""The unit of extension for Quark's PyTorch quantization algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from quark.common.config import BaseAlgoConfig
from quark.torch.algorithm.processor import BaseAlgoProcessor

__all__ = ["QuarkAlgorithm"]


@dataclass(frozen=True)
class QuarkAlgorithm:
    """One quantization algorithm, packaged so core never has to name it.

    :param str name: The algorithm name as it appears in ``{"name": ...}`` in a config file.
        Case-insensitive; normalized to lower case.
    :param type[BaseAlgoConfig] algo_config: The config dataclass the algorithm's JSON is
        deserialized into. Replaces a branch of ``_load_quant_algo_config_from_dict``.
    :param type[BaseAlgoProcessor] algo_processor: The processor that runs the algorithm.
        Replaces a ``PROCESSOR_MAP`` entry.
    :param Mapping[str, BaseAlgoConfig] algo_config_map: Default settings per model architecture
        (``"llama"``, ``"qwen2"``, ...). Read in place of an ``ALGORITHM_CONFIG_MAPS`` entry, so
        like core's maps its values are shared singletons — build a fresh config per model type
        rather than reusing one instance.

    Example:

    .. code-block:: python

        MY_ALGORITHM = QuarkAlgorithm(
            name="myalgo",
            algo_config=MyAlgoConfig,
            algo_processor=MyAlgoProcessor,
            algo_config_map={"llama": MyAlgoConfig(bits=4)},
        )
    """

    name: str
    algo_config: type[BaseAlgoConfig]
    algo_processor: type[BaseAlgoProcessor]
    # `repr=False`: a per-model default map is a page of nested layer names per architecture, and
    # this object shows up in error messages and test failure output.
    algo_config_map: Mapping[str, BaseAlgoConfig] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("QuarkAlgorithm.name must be a non-empty string.")

        # Names are matched case-insensitively everywhere, so normalize once here rather than
        # lower-casing at every lookup site. `frozen=True` means going through object.__setattr__.
        object.__setattr__(self, "name", self.name.strip().lower())

        if not (isinstance(self.algo_config, type) and issubclass(self.algo_config, BaseAlgoConfig)):
            raise TypeError(
                f"QuarkAlgorithm(name={self.name!r}).algo_config must be a subclass of "
                f"BaseAlgoConfig, got {self.algo_config!r}."
            )

        if not (isinstance(self.algo_processor, type) and issubclass(self.algo_processor, BaseAlgoProcessor)):
            raise TypeError(
                f"QuarkAlgorithm(name={self.name!r}).algo_processor must be a subclass of "
                f"BaseAlgoProcessor, got {self.algo_processor!r}."
            )

    def build_config(self, config_dict: Mapping[str, Any]) -> BaseAlgoConfig:
        """Deserialize a raw config dict into this algorithm's config object.

        Override this when the plain field-for-field mapping is not enough.

        :param Mapping[str, Any] config_dict: The parsed JSON for this algorithm.
        :return: An instance of :py:attr:`algo_config`.
        :rtype: BaseAlgoConfig
        """
        return self.algo_config.from_dict(dict(config_dict))
