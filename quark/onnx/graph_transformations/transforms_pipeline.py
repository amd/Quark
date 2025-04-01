#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Abstract Base Class for model transformations pipeline to a onnx model."""

import abc
import six
from typing import Any, Dict, Optional
from onnx import ModelProto


@six.add_metaclass(abc.ABCMeta)
class TransformsPipeline(object):
    """Wrapper of transforms to the model, apply in sequence.
    Transforms the original model to perform better during quantization.
    """

    def __init__(self, configs: Optional[Dict[str, Any]] = None) -> None:
        """Init.

        Args:
            configs: Dict objects containing the detailed configurations.
        """
        self._configs: Optional[Dict[str, Any]] = configs

    def get_configs(self) -> Optional[Dict[str, Any]]:
        """Get the configurations.

        Args:
            None
        Returns:
            Dict of configurations
        """
        return self._configs

    @abc.abstractmethod
    def apply(self, model: ModelProto, candidate_layers: Any, layer_metadata: Any) -> Any:
        """Apply list of transforms to onnx model.

        Args:
            model: onnx model to be quantized.
        Returns:
            New onnx model based on `model` which has been transformed.
        """
        raise NotImplementedError('Must be implemented in subclasses.')
