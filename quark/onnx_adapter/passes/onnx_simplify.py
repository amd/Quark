#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import onnxslim  # type: ignore
from onnx import ModelProto

from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.import_utils import _is_package_available
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ONNXSimplifyPass(ONNXAdapterPass):
    """A pass that simplifies ONNX models using the `onnxslim` library.

    This pass removes redundant nodes and performs graph-level optimizations
    like constant folding and eliminating Identities to make the model smaller
    and more efficient for quantization and inference.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Defines the default configuration for the ONNXSimplifyPass.

        Returns:
            dict[str, PassConfigParam]: A dictionary containing configuration
            parameters for this pass. Includes:
                - "simplify" (bool): Whether to simplify the input model.
        """
        config = {
            "simplify": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to simplify the input model.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_simplify(self, model: ModelProto) -> ModelProto:
        """Simplifies the given ONNX model using the `onnxslim` package.

        Args:
            model (ModelProto): The input ONNX model.

        Returns:
            ModelProto: The simplified ONNX model.

        Raises:
            ImportError: If the `onnxslim` package is not installed.
        """
        if not _is_package_available("onnxslim")[0]:
            raise ImportError(
                "The 'onnxslim' is required but not installed. Please install it via 'pip install onnxslim'."
            )

        simplified_model = onnxslim.slim(model)

        return simplified_model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """Executes the ONNX simplification pass according to the given configuration.

        Args:
            model (ModelProto): The ONNX model to simplify.
            config (dict[str, PassConfigParam]): A configuration dictionary
                specifying whether the simplification should be applied.

        Returns:
            ModelProto: The simplified ONNX model if `simplify=True` is set;
            otherwise, returns the original model and logs a warning.
        """
        if "simplify" in config and config["simplify"]:
            simplified_model = self._onnx_simplify(model)
        else:
            logger.warning("Please ensure that the onnx_simplify pass contains the simplify parameter and it is True.")
        return simplified_model
