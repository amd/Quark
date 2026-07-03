#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Pass registration, discovery, and base classes for Shapeshifter.

This module provides all pass-related infrastructure:
- BasePass: Abstract base class with common pass logic
- ONNXPass, PytorchPass: Type-specific pass base classes
- REGISTRY: Unified registry for all passes

- @register_pass: Decorator for automatic pass registration
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import onnx

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

# Unified registry for both ONNX and PyTorch passes
REGISTRY: dict[str, type] = {}


# =============================================================================
# Base Pass Class (Common Logic)
# =============================================================================


class BasePass(ABC):
    """Abstract base class for all transformation passes.

    This class provides the common implementation for pass lifecycle management,
    configuration handling, and the template method pattern for pass execution.

    Both ONNXPass and PytorchPass inherit from this class and only override
    type signatures for model parameters.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the pass with the given configuration.

        Args:
            config: Configuration dictionary for the pass.
        """
        self.config = config
        self._initialized: bool = False

    @abstractmethod
    def _default_config(self) -> dict[str, Any]:
        """Return the default configuration dictionary for the pass."""
        raise NotImplementedError("New `Pass` must implement `_default_config` method.")

    def _initialize(self) -> None:
        """Optional initialization hook. Override in subclasses if needed."""
        self._initialized = True

    def validate_config(self) -> bool:
        """Optional configuration validation. Override in subclasses if needed."""
        return True

    def run(self, model: Any) -> Any:
        """Execute the pass on the given model."""
        if not self._initialized:
            self._initialize()
            self._initialized = True
        return self._run_for_config(model, self.config)

    @abstractmethod
    def _run_for_config(self, model: Any, config: dict[str, Any]) -> Any:
        """Run the pass logic. Must be implemented by subclasses."""
        raise NotImplementedError("New `Pass` must implement `_run_for_config` method.")


# =============================================================================
# ONNX Pass Classes
# =============================================================================


class ONNXPass(BasePass):
    """Base class for ONNX model transformation passes."""

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Execute the pass on an ONNX model."""
        return super().run(model)

    @abstractmethod
    def _run_for_config(self, model: onnx.ModelProto, config: dict[str, Any]) -> onnx.ModelProto:
        """Transform an ONNX model."""
        raise NotImplementedError("New `ONNXPass` must implement `_run_for_config` method.")


# =============================================================================
# PyTorch Pass Classes
# =============================================================================


class PytorchPass(BasePass):
    """Base class for PyTorch model transformation passes."""

    def run(self, model: Callable[..., Any]) -> Callable[..., Any]:
        """Execute the pass on a PyTorch model."""
        return super().run(model)

    @abstractmethod
    def _run_for_config(self, model: Callable[..., Any], config: dict[str, Any]) -> Callable[..., Any]:
        """Transform a PyTorch model."""
        raise NotImplementedError("New `PytorchPass` must implement `_run_for_config` method.")


# =============================================================================
# Pass Registration Decorator
# =============================================================================


def register_pass(cls: Any) -> Any:
    """Decorator for registering ONNX and PyTorch passes.

    Automatically registers a pass class in the global REGISTRY using the
    filename (without .py extension) as the pass name.

    Args:
        cls: Pass class (must subclass ONNXPass or PytorchPass).

    Returns:
        The same class, unchanged.
    """
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        raise RuntimeError("register_pass decorator could not determine the calling module")

    calling_frame = frame.f_back
    module_file = calling_frame.f_globals.get("__file__")

    if module_file is None:
        raise RuntimeError(
            "register_pass decorator could not find __file__ in the calling module. "
            "Make sure the pass class is defined in a file, not in an interactive session."
        )

    filename = Path(module_file).stem
    pass_name = filename

    # Validate the class
    valid_base_class = False
    pass_type = None

    try:
        if issubclass(cls, ONNXPass):
            valid_base_class = True
            pass_type = "ONNX"
    except (TypeError, NameError):
        pass

    if not valid_base_class:
        try:
            if issubclass(cls, PytorchPass):
                valid_base_class = True
                pass_type = "PyTorch"
        except (TypeError, NameError):
            pass

    if not valid_base_class:
        raise ValueError(f"Class '{cls.__name__}' must be a subclass of ONNXPass or PytorchPass")

    # Check for duplicate pass names
    if pass_name in REGISTRY:
        existing_cls = REGISTRY[pass_name]
        existing_module = existing_cls.__module__
        new_module = cls.__module__

        # Determine if this is a core vs community conflict
        is_core_existing = "shapeshifter.passes" in existing_module and "contrib" not in existing_module
        is_community_new = "contrib" in new_module and "shapeshifter_community_passes" in new_module

        conflict_type = ""
        if is_core_existing and is_community_new:
            conflict_type = " (community pass conflicts with core pass)"
        elif not is_core_existing and not is_community_new:
            conflict_type = " (core pass conflicts with core pass)"
        else:
            conflict_type = " (community pass conflicts with community pass)"

        raise ValueError(
            f"Pass name '{pass_name}' is already registered{conflict_type}. "
            f"Existing: {existing_cls.__name__} from {existing_module}, "
            f"New: {cls.__name__} from {new_module}. "
            f"Pass names must be unique across all passes."
        )

    REGISTRY[pass_name] = cls
    logger.debug(f"Registered {pass_type} pass '{pass_name}'")

    return cls


__all__ = [
    "REGISTRY",
    "register_pass",
    "BasePass",
    "ONNXPass",
    "PytorchPass",
]
