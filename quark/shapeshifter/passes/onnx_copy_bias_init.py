#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from onnx import ModelProto

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_base import register_pass
from quark.shapeshifter.pass_config import PassConfigParam

from .onnx_copy_shared_init import ONNXCopySharedInitPass

logger = ScreenLogger(__name__)


@register_pass
class ONNXCopyBiasInitPass(ONNXCopySharedInitPass):
    """
    ONNX pass to duplicate shared initializers of Bias for specified node types.

    This pass allows separate quantization of bias initializers that are shared across
    multiple nodes.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Returns the default configuration for the pass.

        Returns:
            dict[str, PassConfigParam]: Configuration parameters including:
                - shared_bias_op_types: Node op_types to duplicate bias initializers for.
        """
        config = {
            "shared_bias_op_types": PassConfigParam(
                type_=list,
                default_value=None,
                required=True,
                description="Specifies the node op_types to run duplicating bias initializers in the model for separate quantization use across different nodes.",
            )
        }
        config.update(self.config)
        return config

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Executes the pass on the model according to the provided configuration.

        Args:
            model (ModelProto): The ONNX model to modify.
            config (dict[str, PassConfigParam]): Configuration dictionary for this pass.

        Returns:
            ModelProto: The modified ONNX model after applying the pass.
        """
        if "shared_bias_op_types" in config and config["shared_bias_op_types"]:
            model = self._onnx_copy_shared_init(model, config["shared_bias_op_types"], "duplicated", True)
        else:
            logger.warning(
                "Please ensure that the copy_bias_init pass contains the shared_bias_op_types parameter and it is not None."
            )
        return model
