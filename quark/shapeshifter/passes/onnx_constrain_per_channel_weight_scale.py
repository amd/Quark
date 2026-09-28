#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Per-channel weight scale constraint pass for QDQ quantized models.

For weight tensors quantized with per-channel granularity, this pass detects the
groups of channel scales whose spread is too large and lifts the smallest scales:

- A weight tensor is only touched when ``max_scale / min_scale > maxmin_scale_ratio``.
- For a violating tensor, the smallest scales are clamped up to an effective floor;
  the largest scale is never modified.
- When ``adaptive_min_w_scale`` is enabled, the floor is raised (per tensor) to
  ``max(min_w_scale, max_scale / maxmin_scale_ratio)`` so the ratio is satisfied.
  Otherwise the fixed ``min_w_scale`` floor is used as-is.

When any channel scale is lifted, the entire scale array for that weight
initializer is updated in-place so the model remains consistent.
"""

from typing import Any

import numpy as np
import onnx
from onnx import ModelProto

from quark.common.utils.log import ScreenLogger
from quark.onnx.quantization.quant_utils import DEQUANT_OP_TYPES, QUANT_OP_TYPES
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

# Op types whose weight input (index 1) may carry per-channel scales.
_WEIGHT_OP_TYPES = ["Conv", "ConvTranspose", "Gemm"]

# Q/DQ op types that can carry per-channel weight scales.
_QDQ_OP_TYPES = QUANT_OP_TYPES + DEQUANT_OP_TYPES

logger = ScreenLogger(__name__)


@register_pass
class ONNXConstrainPerChannelWeightScalePass(ONNXPass):
    """A pass that detects and constrains per-channel weight scales in QDQ quantized models.

    For each weight tensor of Conv, ConvTranspose, and Gemm nodes that uses per-channel
    quantization (scale initializer is a 1-D array with more than one element), this pass
    lifts the smallest channel scales when the spread within the tensor is too large:

    - **maxmin_scale_ratio**: The tensor is only modified when
      ``max(scales) / min(scales) > maxmin_scale_ratio``.
    - **min_w_scale**: The floor used to clamp up the smallest scales.
    - **adaptive_min_w_scale**: When ``True``, the floor for a violating tensor is
      raised to ``max(min_w_scale, max(scales) / maxmin_scale_ratio)`` so the ratio
      constraint is satisfied. When ``False``, the fixed ``min_w_scale`` is used.

    In all cases only the minimum side is clamped up (``scales = max(scales, floor)``);
    the largest scale of the tensor is never modified.

    The scale initializer of the matching DequantizeLinear node is updated in-place.
    The paired QuantizeLinear initializer (when it shares the same scale tensor name)
    is updated simultaneously.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return the default configuration for this pass.

        Returns
        -------
        dict[str, PassConfigParam]
            Config dict with the following parameters:

            - **constrain_per_channel_weight_scale** (bool): Master switch. Set to
              ``True`` to enable the pass.
            - **min_w_scale** (float): Lower bound used to clamp up the smallest channel
              scales of a violating weight tensor. Default: ``1e-7``.
            - **adaptive_min_w_scale** (bool): When ``True``, the floor for a violating
              weight tensor is raised to
              ``max(min_w_scale, max_scale / maxmin_scale_ratio)`` so the ratio
              constraint is satisfied; when ``False``, the fixed ``min_w_scale`` is
              used. Default: ``False``.
            - **maxmin_scale_ratio** (float): Maximum allowed ratio between the largest
              and smallest scale within one weight tensor
              (``max_scale / min_scale``). A weight tensor is only modified when this
              ratio is exceeded. Default: ``1e6``.
            - **adjust_bias** (bool): When ``True``, keep the quantized bias of every
              modified channel consistent with its new weight scale. Default: ``True``.
        """
        config = {
            "constrain_per_channel_weight_scale": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Enable per-channel weight scale constraint; accepts bool value.",
            ),
            "min_w_scale": PassConfigParam(
                type_=float,
                default_value=1e-7,
                required=False,
                description="Lower bound used to clamp up the smallest channel scales of a "
                "violating weight tensor. Scales below this floor are raised to min_w_scale.",
            ),
            "adaptive_min_w_scale": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description="When True, the floor for a violating weight tensor is raised to "
                "max(min_w_scale, max_scale / maxmin_scale_ratio) so the max/min ratio is "
                "satisfied. When False, the fixed min_w_scale is used as-is.",
            ),
            "maxmin_scale_ratio": PassConfigParam(
                type_=float,
                default_value=1e6,
                required=False,
                description="Maximum allowed ratio between the largest and smallest scale "
                "within one weight tensor (max_scale / min_scale). A weight tensor is only "
                "modified when this ratio is exceeded.",
            ),
            "adjust_bias": PassConfigParam(
                type_=bool,
                default_value=True,
                required=False,
                description="When True, and the node has a quantized bias whose scale tracks "
                "input_scale * weight_scale, update that bias for every channel whose weight "
                "scale changed: scale the bias scale by the same per-channel factor and "
                "re-quantize the int bias so the dequantized bias value is preserved.",
            ),
        }
        config.update(self.config)
        return config

    def _constrain_per_channel_weight_scale(
        self,
        model: ModelProto,
        min_w_scale: float,
        adaptive_min_w_scale: bool,
        maxmin_scale_ratio: float,
        adjust_bias: bool,
    ) -> ModelProto:
        """Apply per-channel weight scale constraints to the model.

        Scans all Conv, ConvTranspose, and Gemm nodes. For each node whose weight
        input is provided through a DequantizeLinear node with a multi-element (per-channel)
        scale initializer, the scale array is constrained as follows:

        1. Only act when ``max(scales) / min(scales) > maxmin_scale_ratio``.
        2. Pick the effective floor: when ``adaptive_min_w_scale`` is ``True`` use
           ``max(min_w_scale, max(scales) / maxmin_scale_ratio)``; otherwise use the
           fixed ``min_w_scale``.
        3. Clamp up the smallest scales only: ``scales = max(scales, floor)``. The
           largest scale of the tensor is never modified.

        When ``adjust_bias`` is ``True`` and the node has a quantized bias (its 3rd input
        is produced by a DequantizeLinear), the bias of every channel whose weight scale
        changed is updated so that (a) the invariant ``bias_scale = input_scale *
        weight_scale`` is preserved and (b) the dequantized bias value is unchanged.

        Parameters
        ----------
        model : ModelProto
            The ONNX model to modify (must have QDQ nodes for weights).
        min_w_scale : float
            Lower bound used to clamp up the smallest channel scales.
        adaptive_min_w_scale : bool
            When ``True``, raise the floor per tensor to satisfy the max/min ratio;
            when ``False``, use the fixed ``min_w_scale``.
        maxmin_scale_ratio : float
            Maximum allowed max/min scale ratio per weight tensor; tensors within this
            ratio are left untouched.
        adjust_bias : bool
            When ``True``, keep each modified channel's quantized bias consistent with
            the new weight scale.

        Returns
        -------
        ModelProto
            The model with constrained per-channel weight scale initializers.
        """
        # Build a lookup from tensor name -> initializer for fast access.
        init_by_name: dict[str, Any] = {init.name: init for init in model.graph.initializer}

        # Build a lookup from output tensor name -> node for DQ/Q nodes.
        output_to_node: dict[str, Any] = {}
        for node in model.graph.node:
            for out in node.output:
                output_to_node[out] = node

        for node in model.graph.node:
            if node.op_type not in _WEIGHT_OP_TYPES:
                continue

            # Weight input is always at index 1.
            if len(node.input) < 2 or not node.input[1]:
                continue

            weight_input_name = node.input[1]
            dq_node = output_to_node.get(weight_input_name)
            if dq_node is None or dq_node.op_type not in DEQUANT_OP_TYPES:
                continue

            # DequantizeLinear: input[0]=quantized_weight, input[1]=scale, input[2]=zero_point.
            if len(dq_node.input) < 2:
                continue

            scale_init_name = dq_node.input[1]
            scale_init = init_by_name.get(scale_init_name)
            if scale_init is None:
                continue

            scales = onnx.numpy_helper.to_array(scale_init).copy()

            # Per-channel quantization: scale tensor has more than one element.
            if scales.ndim == 0 or scales.size <= 1:
                continue

            original_scales = scales.copy()

            current_max = float(scales.max())
            current_min = float(scales.min())

            # Only act when the max/min ratio within this weight tensor is too large.
            if current_min <= 0 or (current_max / current_min) <= maxmin_scale_ratio:
                continue

            # Pick the effective floor for the smallest scales.
            if adaptive_min_w_scale:
                floor = max(min_w_scale, current_max / maxmin_scale_ratio)
            else:
                floor = min_w_scale

            # Clamp up the smallest scales only; never modify the largest scale.
            scales = np.maximum(scales, floor)

            if np.array_equal(scales, original_scales):
                continue

            # Write the constrained scales back into the initializer.
            new_init = onnx.numpy_helper.from_array(scales.astype(original_scales.dtype), name=scale_init_name)
            scale_init.CopyFrom(new_init)
            logger.info(
                f"Constrained per-channel weight scales for '{node.name}' "
                f"(DQ scale initializer '{scale_init_name}'): "
                f"floor={floor:.3e} (adaptive={adaptive_min_w_scale}), "
                f"original range [{float(original_scales.min()):.3e}, {float(original_scales.max()):.3e}], "
                f"new range [{float(scales.min()):.3e}, {float(scales.max()):.3e}]."
            )

            if adjust_bias:
                # Per-channel factor s_new / s_old (1.0 for channels left unchanged).
                factor = scales.astype(np.float64) / original_scales.astype(np.float64)
                self._adjust_channel_bias(node, init_by_name, output_to_node, factor)

        return model

    def _adjust_channel_bias(
        self,
        node: Any,
        init_by_name: dict[str, Any],
        output_to_node: dict[str, Any],
        factor: np.ndarray,
    ) -> None:
        """Keep a node's quantized bias consistent after its weight scale changed.

        For a node whose bias (input index 2) is produced by a DequantizeLinear with a
        per-channel int bias, this scales each channel's bias scale by ``factor`` (the
        per-channel ``s_new / s_old``) and re-quantizes the int bias so the dequantized
        bias value ``(q_bias - zero_point) * bias_scale`` is preserved. This restores the
        ``bias_scale = input_scale * weight_scale`` invariant on the modified channels.

        Parameters
        ----------
        node : NodeProto
            The Conv/ConvTranspose/Gemm node whose weight scale was changed.
        init_by_name : dict[str, Any]
            Lookup from initializer name to initializer proto.
        output_to_node : dict[str, Any]
            Lookup from output tensor name to producing node.
        factor : np.ndarray
            Per-channel ratio ``s_new / s_old`` aligned with the output channels.
        """
        # Bias is input index 2 and must be produced by a DequantizeLinear.
        if len(node.input) < 3 or not node.input[2]:
            return
        bias_dq = output_to_node.get(node.input[2])
        if bias_dq is None or bias_dq.op_type not in DEQUANT_OP_TYPES or len(bias_dq.input) < 2:
            return

        q_bias_init = init_by_name.get(bias_dq.input[0])
        bias_scale_init = init_by_name.get(bias_dq.input[1])
        if q_bias_init is None or bias_scale_init is None:
            return

        q_bias = onnx.numpy_helper.to_array(q_bias_init)
        bias_scale = onnx.numpy_helper.to_array(bias_scale_init)

        # Only per-channel bias aligned with the weight output channels is handled.
        if bias_scale.ndim == 0 or bias_scale.size != factor.size or q_bias.size != factor.size:
            logger.warning(
                f"Skipping bias adjustment for '{node.name}': bias is not per-channel or "
                f"does not align with the weight scale "
                f"(bias_scale size {bias_scale.size}, q_bias size {q_bias.size}, "
                f"channels {factor.size})."
            )
            return

        # Zero point is optional; default to 0 (the standard for int32 bias).
        zero_point = np.zeros_like(q_bias)
        if len(bias_dq.input) >= 3 and bias_dq.input[2]:
            zp_init = init_by_name.get(bias_dq.input[2])
            if zp_init is not None:
                zero_point = onnx.numpy_helper.to_array(zp_init).reshape(-1).astype(np.float64)

        factor = factor.reshape(-1)
        old_scale = bias_scale.astype(np.float64).reshape(-1)
        new_scale = old_scale * factor

        # Preserve the dequantized bias value B = (q_bias - zp) * old_scale.
        dequant_bias = (q_bias.astype(np.float64).reshape(-1) - zero_point) * old_scale
        # Re-quantize at the new scale and clip to the int bias range.
        new_q = np.round(dequant_bias / new_scale) + zero_point
        info = np.iinfo(q_bias.dtype)
        new_q = np.clip(new_q, info.min, info.max).astype(q_bias.dtype).reshape(q_bias.shape)

        new_scale_arr = new_scale.astype(bias_scale.dtype).reshape(bias_scale.shape)
        bias_scale_init.CopyFrom(onnx.numpy_helper.from_array(new_scale_arr, name=bias_scale_init.name))
        q_bias_init.CopyFrom(onnx.numpy_helper.from_array(new_q, name=q_bias_init.name))

        changed = int(np.count_nonzero(factor != 1.0))
        logger.info(
            f"Adjusted quantized bias for '{node.name}' "
            f"(bias scale '{bias_scale_init.name}', int bias '{q_bias_init.name}'): "
            f"updated {changed} channel(s) to keep bias_scale = input_scale * weight_scale "
            f"and preserve the dequantized bias value."
        )

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """Run the per-channel weight scale constraint pass on the given model.

        Parameters
        ----------
        model : ModelProto
            The ONNX model to modify (QDQ quantized with per-channel weight scales).
        config : dict[str, PassConfigParam]
            Runtime configuration; must contain ``constrain_per_channel_weight_scale``
            (bool). Optional keys: ``min_w_scale``, ``adaptive_min_w_scale``,
            ``maxmin_scale_ratio``, ``adjust_bias``.

        Returns
        -------
        ModelProto
            The model after constraining per-channel weight scales.

        Raises
        ------
        ValueError
            If ``min_w_scale`` is not positive, or ``maxmin_scale_ratio`` is below 1.0.
        """
        if not config.get("constrain_per_channel_weight_scale"):
            logger.warning(
                "constrain_per_channel_weight_scale is missing or falsy; "
                "enable constrain_per_channel_weight_scale to run this pass."
            )
            return model

        min_w_scale = float(config.get("min_w_scale", 1e-7))
        adaptive_min_w_scale = bool(config.get("adaptive_min_w_scale", False))
        maxmin_scale_ratio = float(config.get("maxmin_scale_ratio", 1e6))
        adjust_bias = bool(config.get("adjust_bias", True))

        if min_w_scale <= 0:
            raise ValueError(f"min_w_scale must be positive, got {min_w_scale}.")
        if maxmin_scale_ratio < 1.0:
            raise ValueError(f"maxmin_scale_ratio must be >= 1.0, got {maxmin_scale_ratio}.")

        model = self._constrain_per_channel_weight_scale(
            model,
            min_w_scale=min_w_scale,
            adaptive_min_w_scale=adaptive_min_w_scale,
            maxmin_scale_ratio=maxmin_scale_ratio,
            adjust_bias=adjust_bias,
        )
        return model
