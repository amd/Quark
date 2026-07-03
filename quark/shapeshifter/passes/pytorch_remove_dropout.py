#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Pass to remove dropout layers from PyTorch models.

This pass recursively replaces all dropout layers (nn.Dropout, nn.Dropout2d, nn.Dropout3d)
with nn.Identity layers. This is useful for pre-export preparation since dropout is
typically disabled during inference anyway.
"""

from typing import Any

import torch.nn as nn

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_base import PytorchPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


@register_pass
class PytorchRemoveDropoutPass(PytorchPass):
    """Remove dropout layers from PyTorch model.

    This pass recursively traverses the model and replaces all dropout layers
    with Identity layers. Useful for pre-export preparation since dropout is
    typically disabled during inference anyway.

    Supported dropout types:
    - nn.Dropout
    - nn.Dropout2d
    - nn.Dropout3d

    Configuration:
        remove_dropout (bool): Whether to remove dropout layers. Default: True.

    Example:
        >>> config = {"remove_dropout": True}
        >>> pass_instance = PytorchRemoveDropoutPass(config)
        >>> optimized_model = pass_instance.run(model)
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        config = {
            "remove_dropout": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to remove dropout layers.",
            ),
        }
        config.update(self.config)
        return config

    def _remove_dropout_recursive(self, module: nn.Module) -> None:
        """Recursively replace dropout layers with Identity.

        Args:
            module: The module to process recursively.
        """
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Dropout | nn.Dropout2d | nn.Dropout3d):
                logger.info(f"Removing dropout layer: {name} (type: {type(child).__name__})")
                setattr(module, name, nn.Identity())
            else:
                self._remove_dropout_recursive(child)

    def _run_for_config(self, model: Any, config: dict[str, Any]) -> Any:
        """Execute the pass to remove dropout layers.

        Args:
            model: The input PyTorch model.
            config: Configuration dictionary.

        Returns:
            The modified model with dropout layers replaced by Identity.
        """
        if config.get("remove_dropout", True):
            logger.info("Starting dropout removal pass")
            self._remove_dropout_recursive(model)
            logger.info("Dropout removal completed")
        else:
            logger.info("Dropout removal skipped (remove_dropout=False)")

        return model
