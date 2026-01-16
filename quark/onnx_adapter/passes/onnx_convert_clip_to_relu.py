#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

import onnx
from onnx import ModelProto, NodeProto
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ONNXConvertClipToReluPass(ONNXAdapterPass):
    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Return the default configuration for this pass.

        This configuration defines whether Clip → Relu conversion is enabled.

        Returns:
            dict[str, PassConfigParam]: Configuration parameters controlling
            the Clip-to-Relu conversion behavior.
        """
        config = {
            "convert_clip_to_relu": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to convert Clip operations to Relu operations.",
            )
        }
        config.update(self.config)
        return config

    def _get_clip_min_max(
        self, model: ModelProto, clip_node: NodeProto
    ) -> tuple[float | None, float | None, int | None]:
        """
        Extract the min and max values used by a Clip node.

        This function attempts to obtain Clip bounds in the following order:
            1. From Clip node attributes (`min`, `max`).
            2. From initializers connected to the Clip inputs.
            3. From auxiliary nodes (Constant / Identity) connected upstream.

        Args:
            model (ModelProto): The ONNX model containing the Clip node.
            clip_node (NodeProto): The Clip node whose bounds are to be inspected.

        Returns:
            tuple[float | None, float | None, int | None]:
                A tuple `(min_value, max_value, source_type)`:
                    - `min_value` / `max_value`: extracted bounds, or None if unavailable.
                    - `source_type`: indicates where the values were found:
                        * 0 → attributes
                        * 1 → initializers
                        * 2 → other nodes (Constant / Identity)
                        * None → not found
        """

        def _get_from_initializer(model: ModelProto, name: str) -> Any:
            for init in model.graph.initializer:
                if init.name == name:
                    return onnx.numpy_helper.to_array(init).tolist()
            return None

        def _get_from_attribute(node: NodeProto) -> Any:
            for attr in node.attribute:
                if attr.name == "value":
                    if attr.t.data_type == 1:
                        return list(attr.t.float_data)[0]
                    else:
                        return list(attr.t.int32_data)[0]
            return None

        def _get_from_other_node(model: ModelProto, name: str) -> Any:
            for node in model.graph.node:
                if node.op_type == "Identity" and name in node.output:
                    return _get_from_initializer(model, node.input[0])
                if node.op_type == "Constant" and name in node.output:
                    return _get_from_attribute(node)
            return None

        min_value = None
        max_value = None
        if clip_node.op_type != "Clip":
            return min_value, max_value, None

        # Get from attributes
        for attr in clip_node.attribute:
            if attr.name == "min":
                min_value = attr.f
            if attr.name == "max":
                max_value = attr.f

        if min_value is not None or max_value is not None:
            return min_value, max_value, 0

        # Get from initializers
        if len(clip_node.input) > 1:
            min_value = _get_from_initializer(model, clip_node.input[1])
        if len(clip_node.input) > 2:
            max_value = _get_from_initializer(model, clip_node.input[2])

        if min_value is not None or max_value is not None:
            return min_value, max_value, 1

        # Get from supporting nodes
        if len(clip_node.input) > 1:
            min_value = _get_from_other_node(model, clip_node.input[1])
        if len(clip_node.input) > 2:
            max_value = _get_from_other_node(model, clip_node.input[2])

        if min_value is not None or max_value is not None:
            return min_value, max_value, 2

        return min_value, max_value, None

    def _onnx_convert_clip_to_relu(self, model: ModelProto) -> ModelProto:
        """
        Convert eligible Clip nodes in the model to Relu nodes.

        A Clip node is replaced with a Relu node only when:
            - Its minimum bound is found and is ≥ 0.
            - Its maximum bound does not prevent Relu equivalence.

        This function:
            - Inspects Clip bounds and determines convertibility.
            - Removes initializers or auxiliary nodes used to define Clip bounds.
            - Inserts a Relu node with identical input/output tensors.
            - Removes original Clip nodes and cleans up unused initializers.
            - Performs graph cleanup and topological sorting.

        Args:
            model (ModelProto): The ONNX model to process.

        Returns:
            ModelProto: The updated ONNX model with eligible Clip nodes
            replaced by Relu nodes.
        """
        onnx_model = ONNXModel(model)
        nodes_to_remove = []
        init_to_remove = []

        for node in onnx_model.model.graph.node:
            if node.op_type == "Clip":
                min_value, max_value, para_type = self._get_clip_min_max(onnx_model.model, node)

                if min_value is None or min_value < 0:
                    continue  # could not be replaced with Relu

                if para_type == 1:
                    # min/max from initializers
                    for init in onnx_model.model.graph.initializer:
                        if len(node.input) > 1 and init.name == node.input[1]:
                            init_to_remove.append(init)
                        if len(node.input) > 2 and init.name == node.input[2]:
                            init_to_remove.append(init)

                elif para_type == 2:
                    # min/max from connected nodes
                    for nd in onnx_model.model.graph.node:
                        if (
                            (len(node.input) > 1 and node.input[1] in nd.output)
                            or (len(node.input) > 2 and node.input[2] in nd.output)
                        ) is False:
                            continue

                        if nd.op_type == "Identity":
                            for init in onnx_model.model.graph.initializer:
                                if len(nd.input) > 1 and init.name == nd.input[1]:
                                    init_to_remove.append(init)
                                if len(nd.input) > 2 and init.name == nd.input[2]:
                                    init_to_remove.append(init)
                            nodes_to_remove.append(nd)

                        elif nd.op_type == "Constant":
                            nodes_to_remove.append(nd)

                logger.info(
                    f"Convert Clip node {node.name} to Relu, "
                    f"its min is {min_value}, max is {max_value} and type is {para_type}"
                )
                relu_node = onnx.helper.make_node("Relu", [node.input[0]], node.output, node.name)
                onnx_model.model.graph.node.extend([relu_node])
                nodes_to_remove.append(node)

        onnx_model.remove_nodes(nodes_to_remove)
        onnx_model.remove_initializers(init_to_remove)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()

        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Execute the Clip → Relu conversion pass based on configuration.

        If the `convert_clip_to_relu` option is set to True, this pass processes
        the model and replaces eligible Clip nodes with Relu nodes.
        Otherwise, a warning is logged and no transformation occurs.

        Args:
            model (ModelProto): The ONNX model to modify.
            config (dict[str, PassConfigParam]): Configuration parameters that
                determine whether Clip → Relu conversion is applied.

        Returns:
            ModelProto: The processed model (or the unchanged model if disabled).
        """
        if "convert_clip_to_relu" in config and config["convert_clip_to_relu"] is not None:
            model = self._onnx_convert_clip_to_relu(model)
        else:
            logger.warning(
                "Please ensure that the onnx_convert_clip_to_relu pass contains the convert_clip_to_relu parameter and it is True."
            )
        return model
