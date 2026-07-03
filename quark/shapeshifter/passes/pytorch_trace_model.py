#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Pass to trace a PyTorch model using torch.jit.trace.

This pass converts a PyTorch model to a TorchScript traced model, which can improve
inference performance and enable static analysis. Useful as pre-export preparation.
"""

from typing import Any

import torch

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_base import PytorchPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


@register_pass
class PytorchTraceModelPass(PytorchPass):
    """Trace a PyTorch model using torch.jit.trace.

    This pass converts a PyTorch model to a TorchScript traced model,
    which can improve inference performance and enable static analysis.
    Useful as pre-export preparation.

    Configuration:
        input_shapes (dict): Required. Input tensor shapes.
            Key: input name, Value: shape as list.
            Example: {'input': [1, 3, 224, 224]}

        input_dtypes (dict): Optional. Input tensor dtypes.
            Key: input name, Value: dtype string ('float32', 'float16', 'int64').
            Defaults to float32 if not specified.

    Example:
        >>> config = {
        ...     "input_shapes": {"input": [1, 3, 224, 224]},
        ...     "input_dtypes": {"input": "float32"}
        ... }
        >>> pass_instance = PytorchTraceModelPass(config)
        >>> traced_model = pass_instance.run(model)
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        config = {
            "input_shapes": PassConfigParam(
                type_=dict,
                default_value=None,
                required=True,
                description="Input tensor shapes. Key: input name, Value: shape as list. "
                "Example: {'input': [1, 3, 224, 224]}",
            ),
            "input_dtypes": PassConfigParam(
                type_=dict,
                default_value={},
                required=False,
                description="Input tensor dtypes. Key: input name, Value: dtype string. "
                "Supported: 'float32', 'float16', 'int64'. Defaults to float32 if not specified.",
            ),
        }
        config.update(self.config)
        return config

    def _run_for_config(self, model: Any, config: dict[str, Any]) -> Any:
        """Execute the pass to trace the model.

        Args:
            model: The input PyTorch model.
            config: Configuration dictionary.

        Returns:
            The traced TorchScript model.

        Raises:
            ValueError: If input_shapes is not provided.
        """
        input_shapes = config.get("input_shapes")
        if input_shapes is None:
            raise ValueError("input_shapes is required for pytorch_trace_model pass")

        input_dtypes = config.get("input_dtypes", {})

        # Dtype mapping
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "int64": torch.int64,
            "int32": torch.int32,
            "float64": torch.float64,
        }

        # Create dummy inputs
        dummy_inputs = []
        for name, shape in input_shapes.items():
            dtype_str = input_dtypes.get(name, "float32")
            dtype = dtype_map.get(dtype_str, torch.float32)
            dummy_input = torch.randn(shape, dtype=dtype)
            dummy_inputs.append(dummy_input)
            logger.info(f"Created dummy input '{name}' with shape {shape} and dtype {dtype_str}")

        # Trace model
        logger.info("Tracing PyTorch model with torch.jit.trace")
        traced_model = torch.jit.trace(model, tuple(dummy_inputs))

        logger.info("Model tracing completed successfully")
        return traced_model
