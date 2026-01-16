#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


from onnx import ModelProto

from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ONNXRemoveInputInitPass(ONNXAdapterPass):
    """
    ONNX pass to remove initializers from the model's graph input.

    This pass ensures that any tensor present both in the model's initializer
    and graph input is removed from the graph input list, while preserving the
    initializer itself. This is useful for cleaning up inputs that have default
    values defined by initializers.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Returns the default configuration for the pass.

        Returns:
            dict[str, PassConfigParam]: A dictionary containing the configuration
            parameters for this pass, including whether to remove input initializers.
        """
        config = {
            "remove_input_init": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to remove initializers from the model input.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_remove_input_init(self, model: ModelProto) -> ModelProto:
        """
        Removes graph inputs that are also present as initializers.

        Args:
            model (ModelProto): The ONNX model to process.

        Returns:
            ModelProto: The ONNX model with duplicate input initializers removed.
        """
        if model.ir_version < 4:
            logger.warning(
                "Model with ir_version below 4 requires to include initializer in graph input, change ir_version to 7"
            )
            model.ir_version = 7

        inputs = model.graph.input
        name_to_input = {}
        for input in inputs:
            name_to_input[input.name] = input

        for initializer in model.graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])
        return model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Executes the pass on the model according to the provided configuration.

        Args:
            model (ModelProto): The ONNX model to process.
            config (dict[str, PassConfigParam]): Configuration dict for this pass.

        Returns:
            ModelProto: The processed ONNX model.
        """
        if "remove_input_init" in config and config["remove_input_init"]:
            model = self._onnx_remove_input_init(model)
        else:
            logger.warning(
                "Please ensure that the onnx_remove_input_init pass contains the remove_input_init parameter and it is True."
            )
        return model
