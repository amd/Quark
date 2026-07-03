#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from collections.abc import Callable
from typing import Any

from onnx import ModelProto

from .engine import Engine
from .run_config import RunConfig


def shapeshifter(
    run_config: RunConfig, model: ModelProto | Callable[..., Any] | None = None
) -> ModelProto | Callable[..., Any] | None:
    """Execute Shapeshifter workflow programmatically.

    Supports both ONNX models (onnx.ModelProto) and PyTorch models (any callable).
    Model type is automatically detected from input_model_config type or passes used.

    The workflow:
    1. Load input model (from file or use provided in-memory model)
    2. Execute configured passes in sequence
    3. Save output model (if no in-memory model provided) or return transformed model

    Args:
        run_config: RunConfig object specifying passes and optional file paths.
            Must include:
            - passes: Dict of pass configurations (pass_name -> pass_config)
            - Optional: input_model_config (ONNXModelConfig or PytorchModelConfig)
            - Optional: output_model_path (for file-based saving)

        model: Optional pre-loaded model. Can be:
            - onnx.ModelProto for ONNX models
            - Any callable (torch.nn.Module, function, etc.) for PyTorch models
            - None to load from run_config.input_model_config

    Returns:
        - If model is provided: Returns transformed model (same type as input)
        - If model is None: Saves to file and returns None

    Raises:
        ValueError: If pass types don't match or configuration is invalid.

    Example - ONNX Model (file-based):
        >>> from pathlib import Path
        >>> from quark.shapeshifter import shapeshifter
        >>> from quark.shapeshifter.run_config import RunConfig
        >>> from quark.shapeshifter.model_config import ONNXModelConfig
        >>>
        >>> model_config = ONNXModelConfig(input_model_path=Path("model.onnx"))
        >>> config = RunConfig(
        ...     input_model_config=model_config,
        ...     passes={"onnx_simplify": {"simplify": True}},
        ...     output_model_path="output.onnx"
        ... )
        >>> shapeshifter(config)

    Example - PyTorch Model (file-based):
        >>> from quark.shapeshifter import shapeshifter
        >>> from quark.shapeshifter.run_config import RunConfig
        >>> from quark.shapeshifter.model_config import PytorchModelConfig
        >>>
        >>> model_config = PytorchModelConfig(
        ...     input_model_path=Path("model.pt"),
        ...     map_location="cuda:0"
        ... )
        >>> config = RunConfig(
        ...     input_model_config=model_config,
        ...     passes={"pytorch_remove_dropout": {}},
        ...     output_model_path="output.pt"
        ... )
        >>> shapeshifter(config)

    Example - PyTorch Model (in-memory):
        >>> import torch.nn as nn
        >>> from quark.shapeshifter import shapeshifter
        >>> from quark.shapeshifter.run_config import RunConfig
        >>>
        >>> model = MyModel()  # Your PyTorch model
        >>> config = RunConfig(passes={"pytorch_remove_dropout": {}})
        >>> optimized_model = shapeshifter(config, model=model)

    Example - PyTorch Model (in-memory with file save):
        >>> model = MyModel()
        >>> config = RunConfig(
        ...     passes={"pytorch_remove_dropout": {}},
        ...     output_model_path="output.pt"
        ... )
        >>> # Will save to file even though model was passed in-memory
        >>> # Returns the transformed model
        >>> optimized_model = shapeshifter(config, model=model)
    """
    # Create and run engine with Pydantic object directly
    engine = Engine(config=run_config)
    engine.initialize()
    return engine.run(float_model=model)
