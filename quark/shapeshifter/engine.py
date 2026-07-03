#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import onnx
import torch
from onnx import ModelProto

from quark.common.utils.log import ScreenLogger

from .model_config import ONNXModelConfig, PytorchModelConfig
from .pass_base import REGISTRY, ONNXPass, PytorchPass
from .run_config import RunConfig

logger = ScreenLogger(__name__)


class Engine:
    """The engine executes the registered Passes.

    Supports both ONNX models (onnx.ModelProto) and PyTorch models (any callable).
    Model type is detected by checking input_model_config type (ONNXModelConfig vs PytorchModelConfig)
    or by inspecting pass inheritance.
    """

    def __init__(self, config: RunConfig | dict[str, Any]):
        """Initialize Engine with configuration.

        Args:
            config: Either a RunConfig Pydantic object or a dict (from YAML/JSON).
                   If dict is provided, it will be converted to RunConfig.
        """
        if isinstance(config, dict):
            # Legacy path: CLI provides dict from YAML
            logger.debug("Converting dict config to RunConfig")

            # Handle legacy format: top-level input_model_path/output_model_path
            if "input_model_path" in config and "input_model_config" not in config:
                logger.debug("Detected legacy config format, migrating to new format")
                # Detect model type from passes
                pass_names = list(config.get("passes", {}).keys())
                if pass_names:
                    first_pass = pass_names[0]
                    # Guess model type from pass name prefix
                    if first_pass.startswith("pytorch_"):
                        model_type = "pytorch"
                    else:
                        model_type = "onnx"  # Default to ONNX for legacy configs
                else:
                    model_type = "onnx"  # Default to ONNX

                config["input_model_config"] = {
                    "model_type": model_type,
                    "input_model_path": config.pop("input_model_path"),
                }
                if model_type == "pytorch":
                    config["input_model_config"]["weights_only"] = False

            self._config = RunConfig(**config)
        elif isinstance(config, RunConfig):
            # API path: already a Pydantic object
            self._config = config
        else:
            raise ValueError(f"Config must be RunConfig or dict, got {type(config)}")

        self._passes_registry: dict[str, type] = OrderedDict()
        self._initialized: bool = False

    def initialize(self) -> None:
        """Initialize engine state and register all passes from unified registry."""
        for name, cls in REGISTRY.items():
            self._passes_registry[name] = cls

        logger.debug(f"Initialized engine with {len(self._passes_registry)} registered passes")
        self._initialized = True

    def _detect_model_type_from_config(self) -> type[ONNXModelConfig | PytorchModelConfig]:
        """Detect model type from input_model_config type.

        Returns:
            The model config class (ONNXModelConfig or PytorchModelConfig).

        Raises:
            ValueError: If model type cannot be determined.
        """
        if self._config.input_model_config is not None:
            if isinstance(self._config.input_model_config, PytorchModelConfig):
                logger.info("Detected PyTorch model from PytorchModelConfig")
                return PytorchModelConfig
            elif isinstance(self._config.input_model_config, ONNXModelConfig):
                logger.info("Detected ONNX model from ONNXModelConfig")
                return ONNXModelConfig

        # No input_model_config, detect from passes
        return self._detect_model_type_from_passes()

    def _detect_model_type_from_passes(self) -> type[ONNXModelConfig | PytorchModelConfig]:
        """Detect model type by checking the first pass's inheritance.

        Returns:
            The model config class (ONNXModelConfig or PytorchModelConfig).

        Raises:
            ValueError: If no passes configured or pass type cannot be determined.
        """
        pass_names = list(self._config.passes.keys())

        if not pass_names:
            raise ValueError(
                "Cannot determine model type: no passes configured and no input_model_config provided. "
                "Either configure at least one pass, or provide input_model_config."
            )

        # Check first pass to determine model type
        first_pass_name = pass_names[0]
        first_pass_cls = self._passes_registry.get(first_pass_name)

        if first_pass_cls is None:
            raise ValueError(
                f"Pass '{first_pass_name}' is not registered. Available passes: {list(self._passes_registry.keys())}"
            )

        # Determine model type from first pass
        if issubclass(first_pass_cls, ONNXPass):
            logger.info("Detected ONNX model from pass inheritance")
            return ONNXModelConfig
        elif issubclass(first_pass_cls, PytorchPass):
            logger.info("Detected PyTorch model from pass inheritance")
            return PytorchModelConfig
        else:
            raise ValueError(
                f"Pass '{first_pass_name}' does not inherit from ONNXPass or PytorchPass. "
                f"It inherits from: {first_pass_cls.__bases__}"
            )

    def _validate_all_passes_match_type(self, model_config_type: type) -> None:
        """Validate that all configured passes match the model type.

        Args:
            model_config_type: The model config class (ONNXModelConfig or PytorchModelConfig).

        Raises:
            ValueError: If any pass does not match the model type.
        """
        pass_names = list(self._config.passes.keys())
        is_pytorch = model_config_type == PytorchModelConfig

        for pass_name in pass_names:
            pass_cls = self._passes_registry.get(pass_name)

            if pass_cls is None:
                raise ValueError(
                    f"Pass '{pass_name}' is not registered. Available passes: {list(self._passes_registry.keys())}"
                )

            if is_pytorch:
                if not issubclass(pass_cls, PytorchPass):
                    raise ValueError(
                        f"Pass '{pass_name}' is not a PyTorch pass (does not inherit from PytorchPass). "
                        f"Model type is PyTorch but pass inherits from {pass_cls.__bases__}. "
                        f"All passes must be PyTorch passes for PyTorch models."
                    )
            else:  # ONNX
                if not issubclass(pass_cls, ONNXPass):
                    raise ValueError(
                        f"Pass '{pass_name}' is not an ONNX pass (does not inherit from ONNXPass). "
                        f"Model type is ONNX but pass inherits from {pass_cls.__bases__}. "
                        f"All passes must be ONNX passes for ONNX models."
                    )

        model_type_name = "PyTorch" if is_pytorch else "ONNX"
        logger.info(f"Validated {len(pass_names)} passes for model type {model_type_name}")

    def _load_onnx_model(self, path: str) -> ModelProto:
        """Load ONNX model from file.

        Args:
            path: Path to the ONNX model file.

        Returns:
            Loaded ONNX ModelProto.
        """
        logger.info(f"Loading ONNX model from: {path}")
        return onnx.load(path)

    def _load_pytorch_model(self, config: PytorchModelConfig) -> Callable[..., Any]:
        """Load PyTorch model from file using PytorchModelConfig.

        Args:
            config: PytorchModelConfig with path and loading options.

        Returns:
            Loaded PyTorch callable (nn.Module, function, or any callable).

        Raises:
            ValueError: If the loaded checkpoint is not callable.
        """
        path = str(config.input_model_path)
        logger.info(f"Loading PyTorch model from: {path}")
        logger.info(f"Load config: weights_only={config.weights_only}, map_location={config.map_location}")

        model = torch.load(path, map_location=config.map_location, weights_only=config.weights_only)

        # Handle different checkpoint formats
        if callable(model):
            logger.info(f"Loaded PyTorch model: {type(model).__name__}")
            return model
        elif isinstance(model, dict) and "model" in model:
            logger.info("Loaded checkpoint dict with 'model' key")
            if callable(model["model"]):
                return model["model"]
            else:
                raise ValueError(
                    f"Model in checkpoint dict is not callable. Type: {type(model['model'])}. Cannot proceed."
                )
        else:
            raise ValueError(
                f"Loaded checkpoint is not callable or dict with 'model' key. "
                f"Type: {type(model)}. PyTorch models must be callable. Cannot proceed."
            )

    def _save_onnx_model(self, model: ModelProto, path: str) -> None:
        """Save ONNX model to file.

        Args:
            model: The ONNX ModelProto to save.
            path: Output file path.
        """
        logger.info(f"Saving ONNX model to: {path}")
        onnx.save(model, path)

    def _save_pytorch_model(self, model: Callable[..., Any], path: str) -> None:
        """Save PyTorch model to file.

        Handles both regular PyTorch models and TorchScript models.

        Args:
            model: The PyTorch callable to save.
            path: Output file path.
        """
        logger.info(f"Saving PyTorch model to: {path}")

        # Check if this is a TorchScript model
        if isinstance(model, torch.jit.ScriptModule | torch.jit.RecursiveScriptModule):
            logger.info("Detected TorchScript model, using torch.jit.save()")
            torch.jit.save(model, path)
        else:
            logger.info("Using torch.save() for regular PyTorch model")
            torch.save(model, path)

    def run(self, float_model: ModelProto | Callable[..., Any] | None = None) -> ModelProto | Callable[..., Any] | None:
        """Run all registered passes on the input model.

        Supports both ONNX models (ModelProto) and PyTorch models (any callable).

        Args:
            float_model: Optional pre-loaded model. Can be:
                - onnx.ModelProto for ONNX models
                - Any callable (nn.Module, function, etc.) for PyTorch models
                - None to load from config

        Returns:
            If float_model is provided, returns the transformed model.
            If float_model is None, saves to file and returns None.

        Raises:
            ValueError: If passes don't match model type or configuration is invalid.
        """
        # Detect model type
        model_config_type = self._detect_model_type_from_config()

        # Validate all passes match the model type
        self._validate_all_passes_match_type(model_config_type)

        # Load model if not provided
        model: ModelProto | Callable[..., Any]
        if float_model is None:
            if self._config.input_model_config is None:
                raise ValueError(
                    "input_model_config is required when model is not provided. "
                    "Either pass model parameter or provide input_model_config in RunConfig."
                )

            if model_config_type == ONNXModelConfig:
                model = self._load_onnx_model(str(self._config.input_model_config.input_model_path))
            else:  # PytorchModelConfig
                model = self._load_pytorch_model(self._config.input_model_config)
        else:
            model = float_model
            logger.info(f"Using pre-loaded model of type: {type(model).__name__}")

        # Execute passes
        for pass_name, pass_config in self._config.passes.items():
            pass_class = self._passes_registry[pass_name]
            pass_instance = pass_class(pass_config)
            logger.info(f"Running pass: {pass_name}")
            # TODO: Add code that if `Engine.save_intermediate_models is True`, save the model after each pass for debugging purposes
            model = pass_instance._run_for_config(model, pass_config)

        # Save model if path provided and no pre-loaded model
        if float_model is not None:
            logger.info("Returning transformed model (float_model was provided)")
            return model

        if self._config.output_model_path is None:
            raise ValueError("output_model_path is required when saving to file (model parameter not provided)")

        if model_config_type == ONNXModelConfig:
            self._save_onnx_model(model, self._config.output_model_path)
        else:  # PytorchModelConfig
            self._save_pytorch_model(model, self._config.output_model_path)

        return None
