#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

import onnx
from onnx import ModelProto
from onnxruntime.transformers.fusion_gelu import FusionGelu
from onnxruntime.transformers.onnx_model import OnnxModel

from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ONNXFuseGeluPass(ONNXAdapterPass):
    """
    An ONNX adapter pass that performs GELU fusion by identifying GELU-related
    operator patterns and replacing them with a single `Gelu` operator.

    Features:
    - Checks the model's opset version and fuses only if opset >= 20.
    - Controlled by a `fuse_gelu` config parameter.
    - Uses ONNX Runtime's `FusionGelu` transformer to perform graph fusion.
    - Returns a modified ONNX ModelProto with fused GELU operators.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Define the default configuration parameters for this pass.

        Returns:
            dict[str, PassConfigParam]: A dictionary containing the configuration
            for enabling or disabling GELU fusion.
        """
        config = {
            "fuse_gelu": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to fuse a bunch of separate Gelu operations into a single Gelu operation.",
            )
        }
        config.update(self.config)
        return config

    def _get_opset_version(self, model: onnx.ModelProto) -> Any:
        """
        Extract the opset version for the ai.onnx domain from the ONNX model.

        Args:
            model (onnx.ModelProto): The ONNX model.

        Returns:
            int: The opset version of the ai.onnx domain.

        Raises:
            ValueError: If a valid ai.onnx opset cannot be identified.
        """
        ai_onnx_domain = [opset for opset in model.opset_import if not opset.domain or opset.domain == "ai.onnx"]
        if len(ai_onnx_domain) != 1:
            raise ValueError("Failed to find proper ai.onnx domain")
        opset_version = ai_onnx_domain[0].version
        return opset_version

    def _onnx_fuse_gelu(self, model: ModelProto) -> ModelProto:
        """
        Run the GELU fusion transformation on the provided ONNX model.

        If the opset version is below 20, GELU fusion is skipped. Otherwise,
        the model is wrapped into an OnnxModel object and the FusionGelu pass
        is executed.

        Args:
            model (onnx.ModelProto): The ONNX model to be processed.

        Returns:
            onnx.ModelProto: The transformed model with fused GELU operators,
            or the input model unchanged if fusion is skipped.
        """
        opset_version = self._get_opset_version(model)
        if opset_version < 20:
            logger.warning(f"The opset version is {opset_version} < 20. Skipping fusing Gelu.")
            return model
        else:
            onnx_model = OnnxModel(model)
            fusion_gelu = FusionGelu(onnx_model)
            fusion_gelu.apply()
            return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Execute this pass based on the provided configuration.

        If the `fuse_gelu` configuration parameter is enabled, GELU fusion is
        executed. Otherwise, a warning is logged.

        Args:
            model (onnx.ModelProto): The input ONNX model.
            config (dict[str, PassConfigParam]): Configuration parameters.

        Returns:
            onnx.ModelProto: The resulting ONNX model after executing this pass.
        """
        if "fuse_gelu" in config and config["fuse_gelu"]:
            model = self._onnx_fuse_gelu(model)
        else:
            logger.warning(
                "Please ensure that the onnx_fuse_gelu pass contains the fuse_gelu parameter and it is True."
            )
        return model
