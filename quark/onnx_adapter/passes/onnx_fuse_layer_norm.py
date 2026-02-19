#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

import onnx
from onnx import ModelProto
from onnxruntime.transformers.fusion_layernorm import FusionLayerNormalization
from onnxruntime.transformers.onnx_model import OnnxModel

from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ONNXFuseLayerNormPass(ONNXAdapterPass):
    """
    An ONNX adapter pass that fuses LayerNormalization-related operator patterns
    into a single `LayerNormalization` operator.

    Features:
    - Checks the model's opset version and fuses only if opset >= 17.
    - Controlled by a `fuse_layer_norm` configuration parameter.
    - Uses ONNX Runtime's `FusionLayerNormalization` transformer to perform graph fusion.
    - Returns a modified ONNX ModelProto with fused LayerNormalization operators.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Define the default configuration parameters for this pass.

        Returns:
            dict[str, PassConfigParam]: A dictionary containing the configuration
            for enabling or disabling LayerNormalization fusion.
        """
        config = {
            "fuse_layer_norm": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to fuse a bunch of LayerNormalization operations into a single LayerNormalization operation.",
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

    def _onnx_fuse_layer_norm(self, model: ModelProto) -> ModelProto:
        """
        Run the LayerNormalization fusion transformation on the provided ONNX model.

        If the opset version is below 17, LayerNormalization fusion is skipped.
        Otherwise, the model is wrapped into an OnnxModel object and the
        FusionLayerNormalization pass is executed.

        Args:
            model (onnx.ModelProto): The ONNX model to be processed.

        Returns:
            onnx.ModelProto: The transformed model with fused LayerNormalization operators,
            or the input model unchanged if fusion is skipped.
        """
        opset_version = self._get_opset_version(model)
        if opset_version < 17:
            logger.warning(f"The opset version is {opset_version} < 17. Skipping fusing LayerNormalization.")
            return model
        else:
            onnx_model = OnnxModel(model)
            fusion_gelu = FusionLayerNormalization(onnx_model)
            fusion_gelu.apply()
            return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Execute this pass based on the provided configuration.

        If the `fuse_layer_norm` configuration parameter is enabled, LayerNormalization fusion is
        executed. Otherwise, a warning is logged.

        Args:
            model (onnx.ModelProto): The input ONNX model.
            config (dict[str, PassConfigParam]): Configuration parameters.

        Returns:
            onnx.ModelProto: The resulting ONNX model after executing this pass.
        """
        if "fuse_layer_norm" in config and config["fuse_layer_norm"]:
            model = self._onnx_fuse_layer_norm(model)
        else:
            logger.warning(
                "Please ensure that the onnx_fuse_layer_norm pass contains the fuse_layer_norm parameter and it is True."
            )
        return model
