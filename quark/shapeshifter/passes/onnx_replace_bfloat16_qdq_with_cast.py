#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

import numpy as np
import onnx
from onnx import ModelProto, helper, numpy_helper
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


@register_pass
class ONNXReplaceBfloat16QDQWithCastPass(ONNXPass):
    """Pass that replaces Bfloat16 QuantizeLinear/DequantizeLinear operations with Cast operations."""

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return the default configuration for the pass.

        Returns:
            dict[str, PassConfigParam]: A dictionary defining configuration options
            for enabling or disabling Bfloat16 QDQ to Cast conversion.
        """
        config = {
            "replace_bfloat16_qdq_with_cast": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to replace Bfloat16 QuantizeLinear/DequantizeLinear operations with Cast operations.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_replace_bfloat16_qdq_with_cast(self, model: ModelProto) -> ModelProto:
        """Replace Bfloat16 QuantizeLinear/DequantizeLinear operations with Cast operations.
        This method converts ExtendedQuantizeLinear and ExtendedDequantizeLinear nodes
        that use Bfloat16 data type with zero-point of 0 into equivalent Cast operations.
        If the scale value is not 1, additional Mul nodes are inserted to handle scaling.
        Args:
            model (ModelProto): The ONNX model to process.
        Returns:
            ModelProto: The modified ONNX model with Bfloat16 Q/DQ nodes replaced.
        """
        graph = model.graph

        new_nodes = []

        def _check_second_third_input(graph: onnx.GraphProto, node: onnx.NodeProto) -> tuple[bool, Any]:
            """Check if the second and third inputs of a node meet conditions for bfloat16 QDQ replacement.
            Args:
                graph (onnx.GraphProto): The ONNX graph containing the node and initializers.
                node (onnx.NodeProto): The node whose inputs are being validated.
            Returns:
                tuple[bool, Any]: A tuple where the first element is True if the conditions are met
                                  (third input is bfloat16 type and zero_point is 0), and the second
                                  element is the scale value. Returns (False, None) if conditions are not met.
            """
            # Ensure the node has at least 3 inputs
            if len(node.input) < 3:
                return False, None

            second_input_name = node.input[1]
            third_input_name = node.input[2]

            # Get tensor values for second and third inputs
            scale_value = None
            zero_point_value = None
            third_input_type = None

            for initializer in graph.initializer:
                if initializer.name == second_input_name:
                    scale_value = numpy_helper.to_array(initializer)
                if initializer.name == third_input_name:
                    zero_point_value = numpy_helper.to_array(initializer)
                    third_input_type = initializer.data_type

            # Check that the third input is bfloat16 and zero_point is 0
            if third_input_type != onnx.TensorProto.BFLOAT16 or not np.all(zero_point_value == 0):
                return False, None

            # Return True and the scale value
            return True, scale_value

        def _create_scale_initializer(graph: onnx.GraphProto, node: onnx.NodeProto, scale_value: Any) -> str:
            """Create and add scale initializer to graph.
            Args:
                graph: The ONNX graph
                node: The current node being processed
                scale_value: The scale value to use
            Returns:
                str: The name of the created scale tensor
            """
            scale_tensor_name = f"{node.name}_scale"
            reciprocal_scale = 1.0 / scale_value if node.op_type == "ExtendedQuantizeLinear" else scale_value
            # Convert scale to ndarray and add to initializers
            scale_initializer = helper.make_tensor(
                name=scale_tensor_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=scale_value.shape,
                vals=reciprocal_scale.flatten() if node.op_type == "ExtendedQuantizeLinear" else scale_value.flatten(),
            )
            graph.initializer.append(scale_initializer)
            return scale_tensor_name

        try:
            onnx_model = ONNXModel(model)
            for node in graph.node:
                if node.op_type in ["ExtendedQuantizeLinear", "ExtendedDequantizeLinear"]:
                    # Check if second input (scale) and third input (zero_point) meet the conditions
                    is_valid, scale_value = _check_second_third_input(graph, node)

                    if is_valid:
                        # If scale is not 1, prepare the scale or reciprocal of scale for Mul node
                        if scale_value is not None and np.all(scale_value != 1):
                            scale_tensor_name = _create_scale_initializer(graph, node, scale_value)

                        # Replace node with Cast and Mul if scale != 1
                        if node.op_type == "ExtendedQuantizeLinear":
                            # Add Mul before the Cast with scale's reciprocal
                            if scale_value is not None and np.all(scale_value != 1):
                                mul_before_cast = helper.make_node(
                                    "Mul",
                                    inputs=[node.input[0], scale_tensor_name],
                                    outputs=[f"{node.name}_mul_out"],  # Intermediate output before Cast
                                )
                                new_nodes.append(mul_before_cast)
                                cast_input = f"{node.name}_mul_out"  # Mul output as input to Cast
                            else:
                                cast_input = node.input[0]  # Direct input to Cast if scale == 1

                            # Create Cast to Bfloat16
                            cast_to_bfloat16 = helper.make_node(
                                "Cast",
                                inputs=[cast_input],
                                outputs=node.output,  # Final output of the QuantizeLinear node
                                to=onnx.TensorProto.BFLOAT16,
                            )
                            new_nodes.append(cast_to_bfloat16)

                        elif node.op_type == "ExtendedDequantizeLinear":
                            # Create Cast to Float
                            cast_to_float = helper.make_node(
                                "Cast",
                                inputs=[node.input[0]],  # Only keep the first input
                                outputs=[f"{node.name}_cast_out"],  # Intermediate output after Cast
                                to=onnx.TensorProto.FLOAT,
                            )
                            new_nodes.append(cast_to_float)

                            # Add Mul after the Cast with the original scale value
                            if scale_value is not None and np.all(scale_value != 1):
                                mul_after_cast = helper.make_node(
                                    "Mul",
                                    inputs=[f"{node.name}_cast_out", scale_tensor_name],
                                    outputs=node.output,  # Final output of the DequantizeLinear node
                                )
                                new_nodes.append(mul_after_cast)
                            else:
                                cast_to_float.output[0] = node.output[0]  # Directly use the cast output if scale == 1
                    else:
                        # If the condition is not met, keep the original node
                        new_nodes.append(node)
                else:
                    # Keep other nodes unchanged
                    new_nodes.append(node)

            # Replace the graph's nodes with the new node list
            graph.ClearField("node")
            graph.node.extend(new_nodes)

            onnx_model.clean_initializers()
            onnx_model.topological_sort()

            logger.info("Replaced Bfloat16 Q/DQ to Cast with optional Mul for scale.")
        except Exception as e:
            logger.warning(f"Exception in replacing Bfloat16 Q/DQ to Cast: {e}")

        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """Run this pass using the provided configuration.

        Args:
            model (ModelProto): The ONNX model to modify.
            config (dict[str, PassConfigParam]): Runtime configuration parameters.

        Returns:
            ModelProto: The processed model after applying the pass.
        """
        if "replace_bfloat16_qdq_with_cast" in config and config["replace_bfloat16_qdq_with_cast"]:
            model = self._onnx_replace_bfloat16_qdq_with_cast(model)
        else:
            logger.warning(
                "Please ensure that the onnx_replace_bfloat16_qdq_with_cast pass contains the "
                "replace_bfloat16_qdq_with_cast parameter and it is True."
            )
        return model
