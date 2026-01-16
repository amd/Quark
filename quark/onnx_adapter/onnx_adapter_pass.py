#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from abc import ABC, abstractmethod
from typing import Any

import onnx
from pydantic.dataclasses import dataclass

from quark.contrib.onnx_utils import ryzenai_onnx_utils


class ONNXAdapterPass(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the pass with the given configuration.

        This is the core implementation and cannot be modified by Pass implementations, but
        :py:meth:`._initialize` may be implemented to initialize the pass.

        A Pass is a unit of transformation that operates on an
        ONNX model. Each pass can be configured using a dictionary of parameters.

        Args:
            config (dict[str, Any]): Configuration dictionary that controls
                the behavior of the pass. The expected keys and values should
                be documented in the specific pass implementation.

        Attributes:
            config (dict[str, Any]): The configuration dictionary.
            _initialized (bool): Internal flag that indicates whether the pass
                has been initialized.
        """
        self.config = config
        self._initialized: bool = False

    @abstractmethod
    def _default_config(self) -> dict[str, Any]:
        """Return the default configuration dictionary for the pass.

        This method must be implemented by subclasses to provide the
        default values for all configurable parameters.

        Returns:
            dict[str, Any]: A dictionary containing default configuration
            options for this pass.
        """
        raise NotImplementedError("New `Pass` must implement `_default_config` method.")

    def _initialize(self) -> None:
        """Optional method to initialize internal state for the pass.

        Subclasses may override this method to perform custom initialization

        By default, this method only marks the pass as initialized.
        """
        self._initialized = True

    def validate_config(self) -> bool:
        """Optional method to validate the configuration dictionary for the pass.

        Subclasses may override this method to enforce custom validation

        Returns:
            bool: True if the configuration is valid. False or exception
            otherwise. The default implementation always returns True.
        """
        return True

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Execute the pass on the given ONNX model.

        This method ensures that the pass is initialized before running, and
        then delegates execution to :meth:`_run_for_config`.

        Args:
            model (ModelProto): The input ONNX model to transform or analyze.

        Returns:
            ModelProto: A new or modified ONNX model after applying the pass.

        Notes:
            - Subclasses must implement :meth:`_run_for_config`.
            - This method should not be overridden directly.
        """
        if not self._initialized:
            self._initialize()
            self._initialized = True

        # TODO: complete implementation
        return self._run_for_config(model, self.config)

    @abstractmethod
    def _run_for_config(self, model: onnx.ModelProto, config: dict[str, Any]) -> onnx.ModelProto:
        """Run the pass logic using the provided configuration.

        Subclasses must implement this method to perform the actual
        transformation or analysis.

        Args:
            model (ModelProto): The ONNX model to transform or analyze.
            config (dict[str, Any]): Configuration dictionary for this pass.
                Typically a merge of user-provided and default configuration.

        Returns:
            ModelProto: The transformed ONNX model.
        """
        raise NotImplementedError("New `Pass` must implement `_run_for_config` method.")


@dataclass
class PatternConfig:
    name: str
    pattern: list[str]

    def __len__(self) -> int:
        return len(self.pattern)


class PatternPass(ONNXAdapterPass):
    @abstractmethod
    def _pattern(self) -> list[PatternConfig]:
        """
        Return the list of patterns that this pass will match.
        """
        raise NotImplementedError("New `PatternPass` must implement `_pattern` method.")

    @abstractmethod
    def _run_for_pattern(
        self, extractor: onnx.utils.Extractor, pass_id: str, subgraph: list[onnx.NodeProto], config: dict[str, Any]
    ) -> ryzenai_onnx_utils.typing.PassOutputArgs:
        raise NotImplementedError("New `PatternPass` must implement `_run_for_pattern` method.")

    def _run_for_config(self, model: onnx.ModelProto, config: dict[str, Any]) -> onnx.ModelProto:
        extractor = ryzenai_onnx_utils.matcher.get_extractor(model)
        patterns = self._pattern()
        pass_id = "0"  # TODO: needs to be passed in
        for pass_idx, pattern_config in enumerate(patterns):
            pattern = pattern_config.pattern

            matcher = ryzenai_onnx_utils.matcher.Matcher(pattern, True)

            # replace as many instances of the pattern that are found
            max_to_replace = None

            pass_id += f"_{pass_idx}"

            matcher.replace(extractor, self._run_for_pattern, pass_id, config, max_to_replace)

        ryzenai_onnx_utils.matcher.save_external_data_with_extractor(extractor.model, extractor, "", "", False)

        return extractor.model
