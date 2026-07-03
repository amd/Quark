#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

from onnx import ModelProto, NodeProto
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.common.utils.log import ScreenLogger
from quark.onnx.quantization.quant_utils import get_tensor_to_consumer
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


@register_pass
class ONNXRemoveQDQBetweenOpTypesPass(ONNXPass):
    """A pass that removes QuantizeLinear and DequantizeLinear nodes between specified operator pairs.

    This pass removes redundant Q/DQ (QuantizeLinear/DequantizeLinear) operations that appear
    between specific operator type pairs in quantized ONNX models. It traverses from lower
    operator nodes upward to find matching patterns and removes the intermediate Q/DQ nodes
    to optimize the model graph.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return the default configuration for the pass.

        Returns:
            dict[str, PassConfigParam]: A dictionary defining configuration options
            for specifying operator type pairs between which Q/DQ nodes should be removed.
        """
        config = {
            "remove_qdq_between_op_types": PassConfigParam(
                type_=list,
                default_value=[],
                required=True,
                description="List of operator type pairs between which Q/DQ nodes should be removed. "
                "Each pair is a list of two operator type names [upper_op, lower_op]. "
                "Example: [['Conv', 'Relu'], ['Conv', 'LeakyRelu'], ['Mul', 'Add']]. "
                "The pass will remove Q/DQ nodes between the upper and lower operators.",
            )
        }
        config.update(self.config)
        return config

    def _find_node_by_output(self, nodes: list[NodeProto], output_name: str) -> NodeProto | None:
        """Find a node that produces the specified output tensor.

        Args:
            nodes (list[NodeProto]): List of nodes to search through.
            output_name (str): The name of the output tensor to match.

        Returns:
            NodeProto | None: The node that produces the specified output, or None if not found.
        """
        for node in nodes:
            if output_name in node.output:
                return node
        return None

    def _onnx_remove_qdq_between_ops(self, model: ModelProto, between_ops: list[tuple[str, str]] | Any) -> ModelProto:
        """Remove QuantizeLinear and DequantizeLinear nodes between specified operator type pairs.

        This function processes an ONNX quantized model and removes redundant Q/DQ operations
        that appear between specific operator pairs. It starts from lower operator nodes and
        traverses upwards to find matching patterns.

        The removal is performed when:
        - A lower operator node is found
        - Its input is connected to a DequantizeLinear node
        - The DequantizeLinear node's input is connected to a QuantizeLinear node
        - The QuantizeLinear node's input is connected to an upper operator node
        - The DequantizeLinear output has only one consumer (to avoid breaking other connections)

        Args:
            model (ModelProto): The input ONNX model to be modified.
            between_ops (list[list[str, str]]): A list of operator type pairs, where each pair
                                                is [upper_op_type, lower_op_type]. The function
                                                will look for Q/DQ nodes between these pairs.

        Returns:
            ModelProto: The modified ONNX model with the specified Q/DQ nodes removed.

        Raises:
            Exception: If an error occurs during the removal process, the original model is
                      returned and a warning is logged.
        """
        try:
            tensor_to_consumer = get_tensor_to_consumer(model)
            nodes = model.graph.node
            nodes_to_remove = []
            edges_to_reconnect = []

            for upper_op_type, lower_op_type in between_ops:
                for lower_node in nodes:
                    if lower_node.op_type == lower_op_type:
                        lower_inputs = lower_node.input

                        for input_name in lower_inputs:
                            dq_node = self._find_node_by_output(nodes, input_name)
                            if dq_node and dq_node.op_type == "DequantizeLinear":
                                consumers = tensor_to_consumer[dq_node.output[0]]
                                if len(consumers) > 1:
                                    consumer_str = ", ".join(f"{n.op_type}('{n.name}')" for n in consumers)
                                    logger.debug(
                                        f"Skipping pattern match: output of DequantizeLinear('{dq_node.name}') "
                                        f"is connected to {len(consumers)} nodes: {consumer_str}. "
                                        f"Removal would break other connections."
                                    )
                                    continue

                                dq_input = dq_node.input[0]
                                q_node = self._find_node_by_output(nodes, dq_input)
                                if q_node and q_node.op_type == "QuantizeLinear":
                                    q_input = q_node.input[0]

                                    upper_node = self._find_node_by_output(nodes, q_input)
                                    if upper_node and upper_node.op_type == upper_op_type:
                                        nodes_to_remove.extend([q_node, dq_node])
                                        edges_to_reconnect.append((upper_node.output[0], lower_node, input_name))

            for node in nodes_to_remove:
                nodes.remove(node)

            for upper_node_output, lower_node, original_input in edges_to_reconnect:
                for i, lower_inp in enumerate(lower_node.input):
                    if lower_inp == original_input:
                        lower_node.input[i] = upper_node_output

            onnx_model = ONNXModel(model)
            onnx_model.clean_initializers()
            onnx_model.topological_sort()
            logger.info(
                f"Removed QuantizeLinear and DequantizeLinear operations between operator pairs: {between_ops}."
            )

            return onnx_model.model

        except Exception as e:
            logger.warning(
                f"Unable to remove QuantizeLinear and DequantizeLinear operations between operator pairs "
                f"{between_ops}. Exception: {e}. Returning original model."
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
        if "remove_qdq_between_op_types" in config and config["remove_qdq_between_op_types"]:
            between_ops = config["remove_qdq_between_op_types"]
            if isinstance(between_ops, list) and len(between_ops) > 0:
                model = self._onnx_remove_qdq_between_ops(model, between_ops)
            else:
                logger.warning(
                    "The 'remove_qdq_between_op_types' parameter must be a non-empty list of operator pairs. "
                    "Skipping Q/DQ removal."
                )
        else:
            logger.warning(
                "The onnx_remove_qdq_between_op_types pass requires the 'remove_qdq_between_op_types' "
                "parameter to be set. Skipping Q/DQ removal."
            )
        return model
