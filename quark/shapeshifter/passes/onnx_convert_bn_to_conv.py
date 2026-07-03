#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import ModelProto, NodeProto
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


@register_pass
class ONNXConvertBNToConvPass(ONNXPass):
    """Pass that converts BatchNormalization nodes into equivalent Conv nodes."""

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return the default configuration for the pass.

        Returns:
            dict[str, PassConfigParam]: A dictionary defining configuration options
            for enabling or disabling BatchNormalization-to-Conv conversion.
        """
        config = {
            "convert_bn_to_conv": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to convert BatchNormalization operations to Conv operations.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_convert_bn_to_conv(self, model: ModelProto) -> ModelProto:
        """Convert eligible BatchNormalization nodes into depthwise Conv nodes.

        This method identifies BatchNormalization nodes whose parameters and
        input shapes allow folding into an equivalent 1×1 depthwise convolution.
        The folded weights and bias are computed and a new Conv node replaces
        the BatchNormalization node.

        Args:
            model (ModelProto): The ONNX model to process.

        Returns:
            ModelProto: The updated ONNX model with converted nodes applied.
        """

        def _get_folded_conv_weights(
            bn_gamma: NDArray[np.float32],
            bn_beta: NDArray[np.float32],
            bn_mm: NDArray[np.float32],
            bn_mv: NDArray[np.float32],
            bn_epsilon: float,
        ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
            """Fold BatchNormalization parameters into convolution weights.

            Args:
                bn_gamma (NDArray[np.float32]): Scale parameter (gamma).
                bn_beta (NDArray[np.float32]): Bias parameter (beta).
                bn_mm (NDArray[np.float32]): Running mean.
                bn_mv (NDArray[np.float32]): Running variance.
                bn_epsilon (float): Epsilon for numerical stability.

            Returns:
                tuple[NDArray[np.float32], NDArray[np.float32]]:
                    The folded convolution kernel and bias.
            """
            if bn_gamma is not None:
                multiplier = bn_gamma / np.sqrt(bn_mv + bn_epsilon)
            else:
                multiplier = 1 / np.sqrt(bn_mv + bn_epsilon)

            folded_conv_kernel = multiplier
            folded_conv_bias = bn_beta + (-bn_mm) * multiplier
            return folded_conv_kernel, folded_conv_bias

        nodes_to_remove: list[NodeProto] = []
        init_to_remove: list[str] = []
        onnx_model = ONNXModel(model)
        init_name = onnx_model.get_initializer_name_set()

        for node in onnx_model.model.graph.node:
            if node.op_type == "BatchNormalization":
                input_name = node.input[0]
                input_shape: list[str] = []

                # Resolve input shape
                for input_info in onnx_model.model.graph.value_info:
                    if input_info.name == input_name:
                        input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]

                # Only handle 4D inputs and complete BN parameters
                if len(node.input) == 5 and len(input_shape) == 4:
                    bn_epsilon = next(
                        (attr.f for attr in node.attribute if attr.name == "epsilon"),
                        1e-10,
                    )

                    missing_initializer_names = ", ".join(
                        f"{name}" for i, name in enumerate(node.input[1:]) if name not in init_name
                    )
                    if missing_initializer_names:
                        logger.warning(
                            f"Skip converting bn to conv for node '{node.name}': "
                            f"missing initializer(s): {missing_initializer_names}."
                        )
                        continue

                    gamma_init = onnx_model.get_initializer(node.input[1])
                    bn_gamma = onnx.numpy_helper.to_array(gamma_init)
                    beta_init = onnx_model.get_initializer(node.input[2])
                    bn_beta = onnx.numpy_helper.to_array(beta_init)
                    mm_init = onnx_model.get_initializer(node.input[3])
                    bn_mm = onnx.numpy_helper.to_array(mm_init)
                    mv_init = onnx_model.get_initializer(node.input[4])
                    bn_mv = onnx.numpy_helper.to_array(mv_init)

                    try:
                        weights, bias = _get_folded_conv_weights(bn_gamma, bn_beta, bn_mm, bn_mv, bn_epsilon)
                        num_channel = bn_mm.shape[0]
                        weights = weights.reshape([num_channel, 1, 1, 1])

                        weights_tensor = onnx.numpy_helper.from_array(weights, name=node.output[0] + "weights")
                        bias_tensor = onnx.numpy_helper.from_array(bias, name=node.output[0] + "bias")
                        onnx_model.model.graph.initializer.extend([weights_tensor, bias_tensor])

                        new_node = onnx.helper.make_node(
                            "Conv",
                            inputs=[
                                node.input[0],
                                node.output[0] + "weights",
                                node.output[0] + "bias",
                            ],
                            outputs=[node.output[0]],
                            group=num_channel,
                            kernel_shape=[1, 1],
                            strides=[1, 1],
                            name=node.name,
                        )

                        nodes_to_remove.append(node)
                        init_to_remove.extend([node.input[1], node.input[2], node.input[3], node.input[4]])
                        onnx_model.model.graph.node.append(new_node)

                        logger.info(f"Found BatchNormalization node {node.name}. Replacing with Conv.")
                    except Exception as e:
                        logger.warning(
                            f"Fail to generate conv's weights and bias beacuse of {e}, skip converting bn to conv"
                        )
                else:
                    logger.warning(
                        f"Fail to convert bn {node.name} to conv beacuse BatchNormalization's "
                        "input or shape does not meet the requirements"
                    )

        onnx_model.remove_nodes(nodes_to_remove)
        onnx_model.remove_initializers(init_to_remove)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()

        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """Run the conversion pass based on config settings.

        Args:
            model (ModelProto): The ONNX model to modify.
            config (dict[str, PassConfigParam]): Runtime configuration parameters.

        Returns:
            ModelProto: The processed model after applying the pass.
        """
        if "convert_bn_to_conv" in config and config["convert_bn_to_conv"]:
            model = self._onnx_convert_bn_to_conv(model)
        else:
            logger.warning(
                "Please ensure that the onnx_convert_bn_to_conv pass contains the "
                "convert_bn_to_conv parameter and it is True."
            )
        return model
