#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


from onnx import ModelProto, version_converter

from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ONNXConvertOpsetVersionPass(ONNXAdapterPass):
    """ "Convert an ONNX model from its original opset version to a target opset version."""

    def _default_config(self) -> dict[str, PassConfigParam]:
        config = {
            "target_opset_version": PassConfigParam(
                type_=int,
                default_value=None,
                required=True,
                description="Convert opset version of the input model to the target like 21.",
            )
        }
        config.update(self.config)
        return config

    def onnx_convert_opset_version(self, model: ModelProto, target_opset_version: int) -> ModelProto:
        current_opset_version = model.opset_import[0].version
        logger.info(f"The current opset version of model is {current_opset_version}.")
        converted_model = version_converter.convert_version(model, target_opset_version)
        opset_version = converted_model.opset_import[0].version
        if opset_version == target_opset_version:
            logger.info(f"Convert opset version of the model to {target_opset_version} successfully.")
        else:
            logger.warning(f"Failed to convert opset version of the model to {target_opset_version}.")
        return converted_model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        if "target_opset_version" in config:
            converted_model = self.onnx_convert_opset_version(model, config["target_opset_version"])
        else:
            logger.warning(
                "Please ensure that the onnx_convert_opset_version pass contains the target_opset_version parameter."
            )

        return converted_model
