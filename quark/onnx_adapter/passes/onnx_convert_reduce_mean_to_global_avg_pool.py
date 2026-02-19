#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import onnx
from onnx import ModelProto, NodeProto
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ONNXConvertReduceMeanToGlobalAvgPoolPass(ONNXAdapterPass):
    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Args:
            None

        Returns:
            dict[str, PassConfigParam]: The default configuration for converting
            ReduceMean to GlobalAveragePool, merged with the existing config.
        """
        config = {
            "convert_reduce_mean_to_global_avg_pool": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to convert ReduceMean operations to GlobalAveragePool operations.",
            )
        }
        config.update(self.config)
        return config

    def _check_reduce_mean_condition(self, model: ModelProto, node: NodeProto) -> bool:
        """
        Args:
            model (ModelProto): The ONNX model to inspect.
            node (NodeProto): The ReduceMean node to check.

        Returns:
            bool: True if the ReduceMean node matches the conditions to be replaced
            by GlobalAveragePool, otherwise False.
        """
        has_axes_attr = any(attr.name == "axes" for attr in node.attribute)
        has_axes_2_3_attr = any(
            attr.name == "axes" and len(attr.ints) == 2 and attr.ints == [2, 3] for attr in node.attribute
        )
        has_keepdims_attr = any(attr.name == "keepdims" for attr in node.attribute)
        has_keepdims_1_attr = any(attr.name == "keepdims" and attr.i == 1 for attr in node.attribute)

        if has_axes_attr:
            if has_axes_2_3_attr and (not has_keepdims_attr or has_keepdims_1_attr):
                return True
        # Handling opset >= 18 for Reduce Mean
        elif (not has_keepdims_attr or has_keepdims_1_attr) and len(node.input) == 2:
            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    axes = onnx.numpy_helper.to_array(init).tolist()
                    if axes == [2, 3]:
                        return True

        return False

    def _onnx_convert_reduce_mean_to_global_avg_pool(self, model: ModelProto) -> ModelProto:
        """
        Args:
            model (ModelProto): The ONNX model containing ReduceMean nodes.

        Returns:
            ModelProto: The updated ONNX model with ReduceMean nodes replaced
            by GlobalAveragePool when conditions are met.
        """
        nodes_to_remove = []
        onnx_model = ONNXModel(model)
        for node in onnx_model.model.graph.node:
            if node.op_type == "ReduceMean" and self._check_reduce_mean_condition(onnx_model.model, node):
                if len(node.input) == 1:
                    new_node = onnx.helper.make_node(
                        "GlobalAveragePool", inputs=node.input, outputs=node.output, name=node.name
                    )

                    onnx_model.model.graph.node.append(new_node)

                    nodes_to_remove.append(node)
                    logger.info(
                        f"Found ReduceMean node {node.name} with axes=[2, 3]. Replacing with GlobalAveragePool."
                    )
                # Handling opset >= 18 for Reduce Mean
                elif len(node.input) == 2:
                    new_node = onnx.helper.make_node(
                        "GlobalAveragePool", inputs=[node.input[0]], outputs=node.output, name=node.name
                    )

                    nodes_to_remove.append(node)
                    onnx_model.model.graph.node.append(new_node)
                    logger.info(
                        f"Found ReduceMean node {node.name} with axes=[2, 3]. Replacing with GlobalAveragePool."
                    )
        onnx_model.remove_nodes(nodes_to_remove)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()
        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Args:
            model (ModelProto): The ONNX model to process.
            config (dict[str, PassConfigParam]): The configuration specifying
                whether the conversion pass should run.

        Returns:
            ModelProto: The processed ONNX model with conversion applied if enabled.
        """
        if "convert_reduce_mean_to_global_avg_pool" in config and config["convert_reduce_mean_to_global_avg_pool"]:
            model = self._onnx_convert_reduce_mean_to_global_avg_pool(model)
        else:
            logger.warning(
                "Please ensure that the onnx_convert_reduce_mean_to_global_avg_pool pass contains the convert_reduce_mean_to_global_avg_pool parameter and it is True."
            )
        return model
