#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from onnx import GraphProto, ModelProto, NodeProto, TensorProto, helper
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


@register_pass
class ONNXInsertClipBeforeBfloat16QDQPass(ONNXPass):
    """Pass that insert clip before Bfloat16 QuantizeLinear/DequantizeLinear operations."""

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return the default configuration for the pass.

        Returns:
            dict[str, PassConfigParam]: A dictionary defining configuration options
            for inserting clip nodes before Bfloat16 QuantizeLinear/DequantizeLinear operations.
        """
        config = {
            "insert_clip_before_bfloat16_qdq": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to insert clip nodes before Bfloat16 QuantizeLinear/DequantizeLinear operations.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_insert_clip_before_bfloat16_qdq(self, model: ModelProto) -> ModelProto:
        """Insert clip nodes before Bfloat16 QuantizeLinear/DequantizeLinear operations.

        This method identifies ExtendedQuantizeLinear nodes that operate on bfloat16 data types
        and inserts Clip nodes before them to ensure input values are within the valid range
        for bfloat16 representation.

        Args:
            model (ModelProto): The ONNX model to modify.

        Returns:
            ModelProto: The modified ONNX model with clip nodes inserted.
        """
        graph = model.graph
        onnx_model = ONNXModel(model)

        def _check_bfloat16_activation_qdq(graph: GraphProto, node: NodeProto) -> bool:
            """Check if a node is a Bfloat16 activation QuantizeLinear operation.

            Args:
                graph (GraphProto): The ONNX graph.
                node (NodeProto): The node to check.

            Returns:
                bool: True if the node is a Bfloat16 activation QuantizeLinear operation.
            """
            return (
                node.op_type == "ExtendedQuantizeLinear"
                and onnx_model.get_initializer(node.input[0]) is None
                and onnx_model.get_initializer(node.input[2]).data_type == TensorProto.BFLOAT16
            )

        try:
            for node in graph.node:
                if _check_bfloat16_activation_qdq(graph, node):
                    bf16_max = 3.38953139e38
                    min_initializer = helper.make_tensor(
                        node.input[0] + "_clip_min", TensorProto.FLOAT, [], [-bf16_max]
                    )
                    max_initializer = helper.make_tensor(node.input[0] + "_clip_max", TensorProto.FLOAT, [], [bf16_max])
                    onnx_model.model.graph.initializer.extend([min_initializer, max_initializer])
                    clip_node = helper.make_node(
                        "Clip",
                        [node.input[0], min_initializer.name, max_initializer.name],
                        [node.input[0] + "_clip_output"],
                    )
                    node.input[0] = node.input[0] + "_clip_output"
                    print(node.input[0])
                    onnx_model.add_node(clip_node)

            onnx_model.clean_initializers()
            onnx_model.topological_sort()

            logger.info("Insert Clip before BFloat16 activition Q/DQ")
        except Exception as e:
            logger.warning(f"Exception in inserting Clip before BFloat16 activition Q/DQ: {e}")

        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """Run this pass using the provided configuration.

        Args:
            model (ModelProto): The ONNX model to modify.
            config (dict[str, PassConfigParam]): Runtime configuration parameters.

        Returns:
            ModelProto: The processed model after applying the pass.
        """
        if "insert_clip_before_bfloat16_qdq" in config and config["insert_clip_before_bfloat16_qdq"]:
            model = self._onnx_insert_clip_before_bfloat16_qdq(model)
        else:
            logger.warning(
                "Please ensure that the onnx_insert_clip_before_bfloat16_qdq pass contains the "
                "insert_clip_before_bfloat16_qdq parameter and it is True."
            )
        return model
