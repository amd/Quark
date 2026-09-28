#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

from onnx import ModelProto

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


@register_pass
class ONNXCrossLayerEqualizationPass(ONNXPass):
    """Shapeshifter pass that applies Cross-Layer Equalization (CLE) to an ONNX model.

    CLE is a data-free, function-preserving weight-rebalancing technique that
    equalizes the weight ranges of adjacent ``Conv``/``Gemm`` layers (joined by a
    positive-scaling-invariant activation such as ReLU) so both layers quantize
    better. This pass is a thin wrapper over the existing Quark ONNX CLE
    implementation and reuses it unchanged.

    This algorithm is referenced from the cross-layer equalization proposed in:
    "Markus Nagel et al., Data-Free Quantization Through Weight Equalization and
    Bias Correction, arXiv:1906.04721, 2019."
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return the default configuration for this pass.

        Returns:
            dict[str, PassConfigParam]: Configuration parameters. The keys map onto
            the arguments / ``extra_options`` consumed by ``apply_CLE``.
        """
        config: dict[str, PassConfigParam] = {
            "cross_layer_equalization": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to apply Cross-Layer Equalization to the input ONNX model.",
            ),
            "op_types_to_quantize": PassConfigParam(
                type_=list,
                default_value=["Conv", "Gemm"],
                required=False,
                description="Operator types to equalize. Empty means the CLE-supported types (Conv, Gemm).",
            ),
            "nodes_to_quantize": PassConfigParam(
                type_=list,
                default_value=[],
                required=False,
                description="Node names to include. Empty means all supported nodes.",
            ),
            "nodes_to_exclude": PassConfigParam(
                type_=list,
                default_value=[],
                required=False,
                description="Node names to exclude from equalization.",
            ),
            "replace_clip6_relu": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description="Whether to replace Clip(0, 6) nodes with Relu before equalization "
                "to expose more equalizable patterns.",
            ),
            "cle_steps": PassConfigParam(
                type_=int,
                default_value=1,
                required=False,
                description="Number of equalization iterations. Use -1 for adaptive iteration until convergence.",
            ),
            "cle_balance_method": PassConfigParam(
                type_=str,
                default_value="max",
                required=False,
                description="The method used to balance weight ranges between layers.",
            ),
            "cle_weight_threshold": PassConfigParam(
                type_=float,
                default_value=0.5,
                required=False,
                description="Weight range threshold below which a channel is left unscaled.",
            ),
            "cle_scale_append_bias": PassConfigParam(
                type_=bool,
                default_value=True,
                required=False,
                description="Whether to fold the head bias into the range computation when calculating scales.",
            ),
            "cle_scale_use_threshold": PassConfigParam(
                type_=bool,
                default_value=True,
                required=False,
                description="Whether to apply the weight threshold when calculating scales.",
            ),
            "cle_total_layer_diff_threshold": PassConfigParam(
                type_=float,
                default_value=2e-7,
                required=False,
                description="Convergence threshold on the total weight change between iterations.",
            ),
        }
        config.update(self.config)
        return config

    def _run_for_config(self, model: ModelProto, config: dict[str, Any]) -> ModelProto:
        """Execute Cross-Layer Equalization on the supplied model.

        If ``cross_layer_equalization`` is disabled, the model is returned
        unchanged. Otherwise the pass builds the ``extra_options`` dict expected by
        ``apply_CLE`` and delegates to the existing CLE implementation.

        Args:
            model: The ONNX model to process.
            config: The resolved pass configuration parameters.

        Returns:
            The equalized model, or the original model if the pass is disabled.
        """
        if not config.get("cross_layer_equalization", True):
            logger.info("onnx_cross_layer_equalization is disabled (cross_layer_equalization=False); skipping.")
            return model

        # Imported lazily to avoid an import cycle
        # (quark.shapeshifter.__init__ -> passes -> quark.onnx).
        from quark.onnx.algorithm.interface import apply_CLE

        # Empty op_types would make CLE a no-op (see Optimizer.should_quantize_node),
        # so fall back to the CLE-supported operator types.
        op_types_to_quantize = config.get("op_types_to_quantize", []) or ["Conv", "Gemm"]
        nodes_to_quantize = config.get("nodes_to_quantize", [])
        nodes_to_exclude = config.get("nodes_to_exclude", [])

        extra_options = {
            "ReplaceClip6Relu": config.get("replace_clip6_relu", False),
            "CLESteps": config.get("cle_steps", 1),
            "CLEBalanceMethod": config.get("cle_balance_method", "max"),
            "CLEWeightThreshold": config.get("cle_weight_threshold", 0.5),
            "CLEScaleAppendBias": config.get("cle_scale_append_bias", True),
            "CLEScaleUseThreshold": config.get("cle_scale_use_threshold", True),
            "CLETotalLayerDiffThreshold": config.get("cle_total_layer_diff_threshold", 2e-7),
        }

        model = apply_CLE(
            model,
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            extra_options=extra_options,
        )
        return model
