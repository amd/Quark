#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
from typing import Any

from onnx import ModelProto

from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ONNXCopySharedInitPass(ONNXAdapterPass):
    """
    ONNX pass to duplicate shared initializers for specified node types.

    This pass allows separate quantization of initializers that are shared across
    multiple nodes.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Returns the default configuration for the pass.

        Returns:
            dict[str, PassConfigParam]: Configuration parameters including:
                - shared_init_op_types: Node op_types to duplicate initializers for.
        """
        config = {
            "shared_init_op_types": PassConfigParam(
                type_=list,
                default_value=None,
                required=True,
                description="Specifies the node op_types to run duplicating initializers in the model for separate quantization use across different nodes.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_copy_shared_init(
        self, model: ModelProto, support_op_types: list[str], prefix: str, only_bias: bool
    ) -> ModelProto:
        """
        Duplicates shared initializers for nodes of specified types.

        Args:
            model (ModelProto): The ONNX model to modify.
            support_op_types (list[str]): List of node op_types to process. If empty, all node types are considered.
            prefix (str): Prefix to add to duplicated initializer names.
            only_bias (bool): Whether to duplicate only bias initializers (input index 2).

        Returns:
            ModelProto: The modified ONNX model with duplicated initializers.
        """
        if support_op_types == []:
            support_op_types = []
            for node_idx in range(len(model.graph.node)):
                node_op_type = model.graph.node[node_idx].op_type
                if node_op_type not in support_op_types:
                    support_op_types.append(node_op_type)

        all_initializer_names = [item.name for item in model.graph.initializer]
        all_initializer_dict = {}
        for ini_item in model.graph.initializer:
            all_initializer_dict[ini_item.name] = ini_item

        ini_used_static: dict[str, Any] = {}

        for i in range(len(model.graph.node)):
            if model.graph.node[i].op_type in support_op_types:
                # get all input for one node
                inputs_name = model.graph.node[i].input
                for idx, input_name in enumerate(inputs_name):
                    # get the initializer from the input
                    if only_bias and (idx != 2):
                        continue
                    if input_name in all_initializer_names:
                        # copy initializer for shared initializer or pass
                        if input_name in list(ini_used_static.keys()):
                            ini_used_static[input_name] += 1
                            new_ini = copy.deepcopy(all_initializer_dict[input_name])
                            new_ini.name = prefix + new_ini.name + str(ini_used_static[input_name])
                            model.graph.node[i].input[idx] = new_ini.name
                            model.graph.initializer.append(new_ini)
                        else:
                            ini_used_static[input_name] = 1
        return model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Executes the pass on the model according to the provided configuration.

        Args:
            model (ModelProto): The ONNX model to modify.
            config (dict[str, PassConfigParam]): Configuration dictionary for this pass.

        Returns:
            ModelProto: The modified ONNX model after applying the pass.
        """
        if "shared_init_op_types" in config and config["shared_init_op_types"]:
            model = self._onnx_copy_shared_init(model, config["shared_init_op_types"], "duplicated", False)
        else:
            logger.warning(
                "Please ensure that the copy_shared_init pass contains the shared_init_op_types parameter and it is not None."
            )
        return model
