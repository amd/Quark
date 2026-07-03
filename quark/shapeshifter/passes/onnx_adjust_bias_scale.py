#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


import numpy as np
import onnx
from onnx import ModelProto, helper

from quark.common.utils.log import ScreenLogger
from quark.onnx.quantization.quant_utils import DEQUANT_OP_TYPES, is_version_below
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


@register_pass
class ONNXAdjustBiasScalePass(ONNXPass):
    """A pass that adjusts bias scale in QDQ quantized models.

    This pass ensures that bias scale equals activation scale multiplied by weights scale
    for Conv, Gemm, and ConvTranspose operations. When the scales do not match, it adjusts
    the bias values and scale to maintain quantization correctness.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return the default configuration for the pass.

        Returns:
            dict[str, PassConfigParam]: A dictionary defining configuration options
            for enabling or disabling bias scale adjustment.
        """
        config = {
            "adjust_bias_scale": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to adjust bias scale to match activation scale * weights scale "
                "in QDQ quantized models. When enabled, the pass ensures bias scale "
                "equals the product of activation and weights scales for Conv, Gemm, "
                "and ConvTranspose operations.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_adjust_bias_scale(self, model: ModelProto) -> ModelProto:
        """Adjust bias scale to match activation scale * weights scale in QDQ quantized models.

        This function processes Conv, Gemm, and ConvTranspose nodes in quantized models and ensures
        that the bias scale equals the product of activation scale and weights scale. When a mismatch
        is detected, it adjusts the bias values and scale accordingly.

        The adjustment is only performed when:
        - The bias is quantized with int32 data type
        - All three inputs (activation, weights, bias) have associated QDQ nodes
        - The bias scale does not equal activation scale * weights scale

        Args:
            model (ModelProto): The ONNX model to process.

        Returns:
            ModelProto: The model with adjusted bias scales.
        """
        scale_values = {}
        output2node = {}

        for node in model.graph.node:
            if node.op_type in DEQUANT_OP_TYPES:
                scale_input_name = node.input[1]
                scale_initializer = next(
                    (init for init in model.graph.initializer if init.name == scale_input_name), None
                )
                if scale_initializer:
                    scale_value = onnx.numpy_helper.to_array(scale_initializer)
                    scale_values[node.output[0]] = scale_value
                    output2node[node.output[0]] = node

        for node in model.graph.node:
            if (
                node.op_type in ["Conv", "Gemm", "ConvTranspose"]
                and len(node.input) == 3
                and node.input[0] in scale_values
                and node.input[1] in scale_values
                and node.input[2] in scale_values
            ):
                act_scale = scale_values[node.input[0]]
                weights_scale = scale_values[node.input[1]]
                bias_scale = scale_values[node.input[2]]
                bias_node = output2node[node.input[2]]

                if (act_scale * weights_scale != bias_scale).all():
                    for initializer in model.graph.initializer:
                        if initializer.name == bias_node.input[2]:
                            if is_version_below(onnx, "1.19.0"):
                                data_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]  # type: ignore
                            else:
                                data_type = helper.tensor_dtype_to_np_dtype(initializer.data_type)
                            if data_type != np.int32:
                                logger.warning(
                                    f"The bias scale does not match activation scale * weights scale in QDQ "
                                    f"of node '{node.name}' because the bias quantization is not int32. "
                                    f"Skipping adjustment. Please verify the quantization configuration."
                                )
                                continue
                            else:
                                for initializer in model.graph.initializer:
                                    if initializer.name == bias_node.input[0]:
                                        array = onnx.numpy_helper.to_array(initializer)
                                        new_array = array / (act_scale * weights_scale / bias_scale)
                                        new_array = new_array.astype(np.int32)
                                        new_initializer = onnx.numpy_helper.from_array(
                                            new_array, name=bias_node.input[0]
                                        )
                                        initializer.CopyFrom(new_initializer)
                                    if initializer.name == bias_node.input[1]:
                                        array = onnx.numpy_helper.to_array(initializer)
                                        new_array = act_scale * weights_scale
                                        new_initializer = onnx.numpy_helper.from_array(
                                            new_array, name=bias_node.input[1]
                                        )
                                        initializer.CopyFrom(new_initializer)
                                logger.info(
                                    f"Adjusted bias scale to match activation scale * weights scale "
                                    f"in QDQ of node '{node.name}'."
                                )

        return model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """Run this pass using the provided configuration.

        Args:
            model (ModelProto): The ONNX model to modify.
            config (dict[str, PassConfigParam]): Runtime configuration parameters.

        Returns:
            ModelProto: The processed model after applying the pass.
        """
        if "adjust_bias_scale" in config and config["adjust_bias_scale"]:
            model = self._onnx_adjust_bias_scale(model)
        else:
            logger.warning(
                "The onnx_adjust_bias_scale pass requires the 'adjust_bias_scale' parameter "
                "to be set to True. Skipping bias scale adjustment."
            )
        return model
