#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PretrainedConfig


def get_quantization_config(config: Mapping[str, Any] | PretrainedConfig) -> dict[str, Any] | None:
    """
    Resolve ``quantization_config`` from a model config or raw config dict,
    searching sub-configs if not found at the top level.

    Accepts either a ``PretrainedConfig`` object or a ``Mapping`` (e.g.
    a ``dict`` or ``MappingProxyType`` parsed from ``config.json``).

    Raises ``NotImplementedError`` if multiple distinct entries are found.
    """
    if not isinstance(config, Mapping):
        config = config.to_dict()

    if "quantization_config" in config:
        return config["quantization_config"]

    found = {}
    for key, value in config.items():
        if isinstance(value, dict) and "quantization_config" in value:
            found[key] = value["quantization_config"]

    if len(found) == 0:
        return None
    if len(found) > 1:
        raise NotImplementedError(
            f"Found multiple quantization_config entries across sub-configs "
            f"({list(found.keys())}). This is not supported yet."
        )
    return next(iter(found.values()))
