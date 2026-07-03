#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import numpy as np
import torch
from numpy.typing import NDArray
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


@register_pass
class ONNXRemoveBfloat16CastPass(ONNXPass):
    """Pass that remove Bfloat16 Cast operations."""

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return the default configuration for the pass.

        Returns:
            dict[str, PassConfigParam]: A dictionary defining configuration options
            for remove Bfloat16 Cast operations.
        """
        config = {
            "remove_bfloat16_cast": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to remove Bfloat16 Cast operations.",
            )
        }
        config.update(self.config)
        return config

    def _get_input_tensor_to_node_dict(self, model: ModelProto) -> dict[str, NodeProto]:
        """Build a dictionary mapping input tensor names to nodes that consume them.

        Args:
            model (ModelProto): The ONNX model to process.

        Returns:
            dict[str, NodeProto]: A dictionary where keys are input tensor names and
            values are lists of nodes that use those tensors as inputs.
        """
        input_tensor_to_node_dict: dict[str, NodeProto] = {}
        for node in model.graph.node:
            for input_ in node.input:
                if input_ not in input_tensor_to_node_dict:
                    input_tensor_to_node_dict[input_] = []  # type: ignore
                input_tensor_to_node_dict[input_].append(node)
        return input_tensor_to_node_dict

    def _remove_couples_of_cast(self, model: ModelProto, input_tensor_to_node_dict: dict[str, NodeProto]) -> ModelProto:
        """Remove consecutive pairs of Cast operations that convert to bfloat16 and back to float32.

        This method identifies and removes redundant Cast node pairs where data is first cast to
        bfloat16 (type 16) and then immediately cast back to float32 (type 1), reconnecting the
        edges to bypass these unnecessary operations.

        Args:
            model (ModelProto): The ONNX model to process.
            input_tensor_to_node_dict (dict[str, NodeProto]): A dictionary mapping input tensor
                names to their consuming nodes.

        Returns:
            ModelProto: The modified model with redundant Cast pairs removed.
        """
        input_list = []
        for input_ in model.graph.input:
            input_list.append(input_.name)

        output_list = []
        for output in model.graph.output:
            output_list.append(output.name)

        nodes_to_remove = []
        edges_to_reconnect: list[tuple[NodeProto, str, NodeProto]] = []

        for node in model.graph.node:
            first_node = node
            for i in range(len(first_node.output)):
                if (
                    first_node.output[i] in input_tensor_to_node_dict
                    and len(input_tensor_to_node_dict[first_node.output[i]]) == 1
                ):
                    second_node = input_tensor_to_node_dict[first_node.output[i]][0]
                    if (
                        second_node.op_type == "Cast"
                        and len(second_node.attribute) == 1
                        and second_node.attribute[0].name == "to"
                        and second_node.attribute[0].i == 16
                        and len(second_node.output) == 1
                        and second_node.output[0] in input_tensor_to_node_dict
                        and len(input_tensor_to_node_dict[second_node.output[0]]) == 1
                    ):
                        third_node = input_tensor_to_node_dict[second_node.output[0]][0]
                        if (
                            third_node.op_type == "Cast"
                            and len(third_node.attribute) == 1
                            and third_node.attribute[0].name == "to"
                            and third_node.attribute[0].i == 1
                            and third_node.output[0] not in output_list
                            and third_node.output[0] in input_tensor_to_node_dict
                        ):
                            fourth_nodes = input_tensor_to_node_dict[third_node.output[0]]
                            nodes_to_remove.extend([second_node, third_node])
                            edges_to_reconnect.extend(
                                (first_node, third_node.output[0], fourth_node) for fourth_node in fourth_nodes
                            )
        nodes = model.graph.node
        for node in nodes_to_remove:
            nodes.remove(node)
        for first_node, third_node_output, fourth_node in edges_to_reconnect:
            for i in range(len(fourth_node.input)):
                if fourth_node.input[i] == third_node_output:
                    for j in range(len(first_node.output)):
                        if first_node.output[j] in fourth_node.input[i]:
                            fourth_node.input[i] = first_node.output[j]
                            break

        return model

    def _float32_to_bfloat16(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert float32 array to bfloat16 and back to float32 to simulate precision loss.

        Args:
            x (NDArray[np.float32]): Input float32 numpy array.

        Returns:
            NDArray[np.float32]: Float32 array after bfloat16 conversion and back.
        """
        bfloat16_array = torch.tensor(x).to(torch.bfloat16)
        float32_back_array = bfloat16_array.to(torch.float32)
        new_x = float32_back_array.numpy()
        return new_x  # type: ignore

    def _convert_bf16_cast_to_fp32_weights(
        self, model: ModelProto, input_tensor_to_node_dict: dict[str, NodeProto]
    ) -> ModelProto:
        """Convert bfloat16 Cast operations to fp32 weights.

        This method identifies sequences of Cast nodes (fp32->bf16->fp32) connected to initializers,
        applies the bfloat16 conversion directly to the initializer weights, and removes the
        redundant Cast nodes from the graph.

        Args:
            model (ModelProto): The ONNX model to modify.
            input_tensor_to_node_dict (dict[str, NodeProto]): Dictionary mapping input tensor names to nodes.

        Returns:
            ModelProto: The modified model with bfloat16 Cast operations converted to fp32 weights.
        """
        cast_cast_node_list = []
        for node in model.graph.node:
            first_node = node
            if (
                first_node.op_type == "Cast"
                and len(first_node.attribute) == 1
                and first_node.attribute[0].name == "to"
                and first_node.attribute[0].i == 16
                and len(first_node.output) == 1
                and first_node.output[0] in input_tensor_to_node_dict
                and len(input_tensor_to_node_dict[first_node.output[0]]) == 1
            ):
                second_node = input_tensor_to_node_dict[first_node.output[0]][0]
                if (
                    second_node.op_type == "Cast"
                    and len(second_node.attribute) == 1
                    and second_node.attribute[0].name == "to"
                    and second_node.attribute[0].i == 1
                    and second_node.output[0] in input_tensor_to_node_dict
                ):
                    third_node_list = input_tensor_to_node_dict[second_node.output[0]]
                    cast_cast_node_list.append((first_node, second_node, third_node_list))

        for first_node, second_node, third_node_list in cast_cast_node_list:
            init_name = first_node.input[0]
            for init in model.graph.initializer:
                if init.name == init_name:
                    float32_init = numpy_helper.to_array(init)
                    bfloat16_init = self._float32_to_bfloat16(float32_init)
                    new_tensor = numpy_helper.from_array(bfloat16_init, name=init.name + "_bf16")
                    new_tensor.data_type = TensorProto.FLOAT
                    second_node_output = second_node.output[0]
                    for k in range(len(third_node_list)):
                        third_node = third_node_list[k]
                        for i in range(len(third_node.input)):
                            if third_node.input[i] == second_node_output:
                                third_node.input[i] = new_tensor.name
                                if k == 0:  # only execute once
                                    model.graph.initializer.append(new_tensor)
                                    model.graph.node.remove(first_node)
                                    model.graph.node.remove(second_node)
                                    model.graph.initializer.remove(init)

        return model

    def _remove_output_cast(self, model: ModelProto, input_tensor_to_node_dict: dict[str, NodeProto]) -> ModelProto:
        """Remove Cast operations at model outputs that convert to bfloat16 and back to float32.

        Args:
            model (ModelProto): The ONNX model to process.
            input_tensor_to_node_dict (dict[str, NodeProto]): Dictionary mapping input tensor names to nodes.

        Returns:
            ModelProto: The modified model with output cast operations removed.
        """
        output_list = []
        for output in model.graph.output:
            output_list.append(output.name)

        for node in model.graph.node:
            first_node = node
            if (
                len(first_node.output) == 1
                and first_node.output[0] in input_tensor_to_node_dict
                and len(input_tensor_to_node_dict[first_node.output[0]]) == 1
            ):
                second_node = input_tensor_to_node_dict[first_node.output[0]][0]
                if (
                    second_node.op_type == "Cast"
                    and len(second_node.attribute) == 1
                    and second_node.attribute[0].name == "to"
                    and second_node.attribute[0].i == 16
                    and len(second_node.output) == 1
                    and second_node.output[0] in input_tensor_to_node_dict
                    and len(input_tensor_to_node_dict[second_node.output[0]]) == 1
                ):
                    third_node = input_tensor_to_node_dict[second_node.output[0]][0]
                    if (
                        third_node.op_type == "Cast"
                        and len(third_node.attribute) == 1
                        and third_node.attribute[0].name == "to"
                        and third_node.attribute[0].i == 1
                        and third_node.output[0] in output_list
                    ):
                        model.graph.node.remove(second_node)
                        model.graph.node.remove(third_node)
                        first_node.output[0] = third_node.output[0]

        onnx_model = ONNXModel(model)
        onnx_model.topological_sort()
        return model

    def _onnx_remove_bf16_cast(self, model: ModelProto) -> ModelProto:
        """Remove bfloat16 cast operations from the model.

        This method removes bfloat16 cast operations by removing pairs of cast nodes,
        converting bfloat16 weights to float32, and removing output cast operations.

        Args:
            model (ModelProto): The ONNX model to process.

        Returns:
            ModelProto: The modified model with bfloat16 cast operations removed.
        """
        input_tensor_to_node_dict = self._get_input_tensor_to_node_dict(model)
        model = self._remove_couples_of_cast(model, input_tensor_to_node_dict)
        model = self._convert_bf16_cast_to_fp32_weights(model, input_tensor_to_node_dict)
        model = self._remove_output_cast(model, input_tensor_to_node_dict)
        return model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """Run this pass using the provided configuration.

        Args:
            model (ModelProto): The ONNX model to modify.
            config (dict[str, PassConfigParam]): Runtime configuration parameters.

        Returns:
            ModelProto: The processed model after applying the pass.
        """
        if "remove_bfloat16_cast" in config and config["remove_bfloat16_cast"]:
            model = self._onnx_remove_bf16_cast(model)
        else:
            logger.warning(
                "Please ensure that the onnx_remove_bfloat16_cast pass contains the "
                "remove_bfloat16_cast parameter and it is True."
            )
        return model
