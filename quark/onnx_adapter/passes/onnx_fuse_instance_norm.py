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


class ONNXFuseInstanceNormPass(ONNXAdapterPass):
    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Returns the default configuration for the ONNXFuseInstanceNormPass.

        The configuration contains a single parameter:
        - fuse_instance_norm: A boolean flag indicating whether to fuse separate
          InstanceNormalization nodes into a single InstanceNormalization node.
        """
        config = {
            "fuse_instance_norm": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to fuse a bunch of separate InstanceNormalization operations into one single InstanceNormalization operation.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_fuse_instance_norm(self, model: ModelProto) -> ModelProto:
        """
        Performs fusion of InstanceNormalization patterns in the ONNX model.

        This method scans the model graph for a specific subgraph pattern
        corresponding to InstanceNormalization operations. When a matching
        pattern is found, the subgraph is replaced by a single InstanceNormalization
        node, and the original nodes and related initializers are removed.

        Args:
            model (ModelProto): The ONNX model to be optimized.

        Returns:
            ModelProto: The modified ONNX model with fused InstanceNormalization nodes.
        """
        onnx_model = ONNXModel(model)
        tensor_to_producer_dict = {}
        remove_nodes: list[NodeProto] = []
        remove_inits: list[onnx.TensorProto] = []
        for node in onnx_model.model.graph.node:
            for output in node.output:
                tensor_to_producer_dict[output] = node
        for init in onnx_model.model.graph.initializer:
            tensor_to_producer_dict[init.name] = init
        for node in onnx_model.model.graph.node:
            if node.op_type == "Add":
                try:
                    add0_i0 = node.input[0]
                    add0_i1 = node.input[1]
                    add0_i0_node = tensor_to_producer_dict[add0_i0]
                    add0_i1_node = tensor_to_producer_dict[add0_i1]
                    # TODO: Use different dictionaries to distinguish between node and init.
                    if add0_i0_node.op_type == "Mul" and add0_i1_node.op_type == "Sub":
                        sub0_node = add0_i1_node
                        sub0_i0 = sub0_node.input[0]
                        sub0_i1 = sub0_node.input[1]
                        sub0_i1_node = tensor_to_producer_dict[sub0_i1]
                        if sub0_i1_node.op_type == "Mul":
                            mul0_node = sub0_i1_node
                            mul0_i0 = mul0_node.input[0]
                            mul0_i1 = mul0_node.input[1]
                            mul0_i0_node = tensor_to_producer_dict[mul0_i0]
                            mul0_i1_node = tensor_to_producer_dict[mul0_i1]
                            if mul0_i0_node.op_type == "GlobalAveragePool" and mul0_i1_node.op_type == "Mul":
                                mul1_node = mul0_i1_node
                                mul1_i0 = mul1_node.input[0]
                                mul1_i1 = mul1_node.input[1]
                                mul1_i0_node = tensor_to_producer_dict[mul1_i0]
                                if mul1_i0_node.op_type == "Reciprocal":
                                    rec0_node = mul1_i0_node
                                    rec0_i0 = rec0_node.input[0]
                                    rec0_i0_node = tensor_to_producer_dict[rec0_i0]
                                    if rec0_i0_node.op_type == "Sqrt":
                                        sqr0_node = rec0_i0_node
                                        sqr0_i0 = sqr0_node.input[0]
                                        sqr0_i0_node = tensor_to_producer_dict[sqr0_i0]
                                        if sqr0_i0_node.op_type == "Add":
                                            add1_node = sqr0_i0_node
                                            add1_i0 = add1_node.input[0]
                                            add1_i1 = add1_node.input[1]
                                            add1_i0_node = tensor_to_producer_dict[add1_i0]
                                            if add1_i0_node.op_type == "GlobalAveragePool":
                                                gap0_node = add1_i0_node
                                                gap0_i0 = gap0_node.input[0]
                                                gap0_i0_node = tensor_to_producer_dict[gap0_i0]
                                                if gap0_i0_node.op_type == "Mul":
                                                    mul2_node = gap0_i0_node
                                                    mul2_i0 = mul2_node.input[0]
                                                    mul2_i0_node = tensor_to_producer_dict[mul2_i0]
                                                    if mul2_i0_node.op_type == "Sub":
                                                        sub1_node = mul2_i0_node
                                                        sub1_i0 = sub1_node.input[0]
                                                        sub1_i1 = sub1_node.input[1]
                                                        sub1_i1_node = tensor_to_producer_dict[sub1_i1]
                                                        if sub1_i1_node.op_type == "GlobalAveragePool":
                                                            # Remove nodes
                                                            remove_node_list = [
                                                                node,
                                                                add0_i0_node,
                                                                add0_i1_node,
                                                                sub0_i1_node,
                                                                mul0_i0_node,
                                                                mul0_i1_node,
                                                                mul1_i0_node,
                                                                rec0_i0_node,
                                                                sqr0_i0_node,
                                                                add1_i0_node,
                                                                gap0_i0_node,
                                                                mul2_i0_node,
                                                            ]

                                                            # Add InstanceNormalization
                                                            bias_init = onnx_model.get_initializer(sub0_i0)
                                                            bias_init.dims[:] = [bias_init.dims[1]]
                                                            weight_init = onnx_model.get_initializer(mul1_i1)
                                                            weight_init.dims[:] = [weight_init.dims[1]]
                                                            eps_init = onnx_model.get_initializer(add1_i1)

                                                            instance_norm_node = onnx.helper.make_node(
                                                                "InstanceNormalization",
                                                                [sub1_i0, mul1_i1, sub0_i0],
                                                                node.output,
                                                                node.name,
                                                                epsilon=onnx.numpy_helper.to_array(eps_init).item(),
                                                            )
                                                            logger.info(
                                                                f"Matched Instance Normalization, fuse it into InstanceNormalization {node.name}"
                                                            )
                                                            onnx_model.add_node(instance_norm_node)

                                                            remove_nodes.extend(remove_node_list)
                                                            remove_inits.append(eps_init)
                except Exception as e:
                    logger.debug(
                        f"FuseInstanceNorm is enabled, but {node.name} does not meet the matching rules:{e}, skipping this node"
                    )
        onnx_model.remove_nodes(remove_nodes)
        onnx_model.remove_initializers(remove_inits)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()

        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Executes the InstanceNormalization fusion pass based on the provided configuration.

        Args:
            model (ModelProto): The ONNX model to be processed.
            config (dict[str, PassConfigParam]): The configuration parameters for this pass.

        Returns:
            ModelProto: The ONNX model after applying the InstanceNormalization fusion, if enabled.
        """
        if "fuse_instance_norm" in config and config["fuse_instance_norm"] is not None:
            model = self._onnx_fuse_instance_norm(model)
        else:
            logger.warning(
                "Please ensure that the onnx_fuse_instance_norm pass contains the fuse_instance_norm parameter and it is True."
            )
        return model
