#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from quark.torch.export.utils import apply_export_state_dict_mappings, fix_loaded_state_dict_mismatch


class QuarkLinearBase(nn.Module, ABC):
    """Shared base for Quark linear modules used in evaluation/export flows.

    Inherits from ``nn.Module`` so that ``state_dict`` / ``_load_from_state_dict``
    overrides sit before ``nn.Module`` in the MRO for any concrete subclass.
    """

    _native_inference_enabled: bool = False
    _custom_mode: str
    weight_quantizer: Any
    input_quantizer: Any
    output_quantizer: Any
    bias_quantizer: Any

    @classmethod
    def from_module(
        cls,
        linear: nn.Linear,
        custom_mode: str = "quark",
        pack_method: str | None = "reorder",
        quant_config: Any = None,
        algo_config: Any = None,
    ) -> QuarkLinearBase:
        """
        Build a QuarkLinearBase from a QuantLinear or nn.Linear.
        Initialize the shape and data type of weight and bias in importing.
        Initialize weight and bias in exporting.
        """
        quarklinearbase = cls(
            linear=linear,
            custom_mode=custom_mode,
            pack_method=pack_method,
            quant_config=quant_config,
            algo_config=algo_config,
        )
        return quarklinearbase

    def preprocess_weight(self) -> None:
        """Transform weight from storage format to kernel-ready format."""
        return

    def postprocess_weight(self) -> None:
        """Transform weight from kernel format back to export/storage format."""
        return

    def state_dict(self, *args: Any, destination: Any = None, prefix: str = "", keep_vars: bool = False) -> Any:
        """
        Wraps ``nn.Module.state_dict`` with ``postprocess_weight`` / ``preprocess_weight``
        and applies export key mappings via ``apply_export_state_dict_mappings``.

        External / serialized keys use ``weight_scale``, ``weight_zero_point``, etc.
        instead of the internal ``weight_quantizer.scale`` layout.  The mapping is
        handled by ``apply_export_state_dict_mappings``.
        """
        try:
            self.postprocess_weight()
            destination_local = super().state_dict(*args, prefix=prefix, keep_vars=keep_vars)
            apply_export_state_dict_mappings(self, destination_local, prefix)

            if destination is not None:
                destination.update(destination_local)
            else:
                destination = destination_local

            return destination
        finally:
            self.preprocess_weight()

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        state_dict = fix_loaded_state_dict_mismatch(self, state_dict, prefix)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass contract for all Quark linear modules."""
        ...
