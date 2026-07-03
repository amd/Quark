#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quantization config bridge between diffusers and Quark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from diffusers.quantizers.quantization_config import QuantizationConfigMixin


class _QuarkQuantMethod(str):
    """String subclass that exposes a ``.value`` attribute.

    Diffusers' ``from_pretrained`` calls ``quant_method.value`` assuming
    every ``quant_method`` is an ``Enum``.  Quark is not (yet) in the
    upstream ``QuantizationMethod`` enum, so we use a thin shim that
    satisfies the ``str`` + ``.value`` contract.
    """

    @property
    def value(self) -> str:
        return str(self)


_QUARK_METHOD = _QuarkQuantMethod("quark")


@dataclass
class QuarkQuantizationConfig(QuantizationConfigMixin):
    """Wraps a serialised Quark ``QConfig`` dict for the diffusers quantizer API.

    The raw dict is persisted in ``config.json`` under the key
    ``quantization_config`` by :class:`DiffusersSafetensorsExporter` and
    reconstructed here so that :class:`QuarkDiffusersQuantizer` can call
    ``QConfig.from_dict()`` to rebuild the full config at load time.
    """

    quant_method: str = _QUARK_METHOD  # type: ignore[assignment]
    quant_config_dict: dict[str, Any] = field(default_factory=dict)

    def __init__(self, quant_config_dict: dict[str, Any]) -> None:
        self.quant_method = _QUARK_METHOD  # type: ignore[assignment]
        self.quant_config_dict = quant_config_dict

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any], return_unused_kwargs: bool = False, **kwargs: Any
    ) -> QuarkQuantizationConfig | tuple[QuarkQuantizationConfig, dict[str, Any]]:
        """Reconstruct from the ``quantization_config`` section of ``config.json``."""
        config = cls(quant_config_dict=config_dict)
        if return_unused_kwargs:
            return config, kwargs
        return config

    def to_dict(self) -> dict[str, Any]:
        return {"quant_method": "quark", **self.quant_config_dict}

    def to_diff_dict(self) -> dict[str, Any]:
        return self.to_dict()
