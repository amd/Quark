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


class ONNXFuseL2NormPass(ONNXAdapterPass):
    """Pass that fuses a sequence of ONNX nodes representing L2 normalization
    into a single LpNormalization(p=2) operator.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return default configuration parameters for this pass.

        Returns:
            dict[str, PassConfigParam]: Default pass configuration containing
            the fuse_l2_norm option.
        """
        config = {
            "fuse_l2_norm": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to fuse a bunch of separate L2Norm operations into one single LpNormalization operations with p=2.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_fuse_l2_norm(self, model: ModelProto) -> ModelProto:
        """Perform the L2Norm fusion on the provided ONNX model.

        Args:
            model (ModelProto): The input ONNX model.

        Returns:
            ModelProto: The fused ONNX model with L2Norm patterns replaced by
            LpNormalization(p=2) nodes.
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
            if node.op_type == "Mul":
                try:
                    inp_0 = node.input[0]
                    inp_1 = node.input[1]
                    inp_0_node = tensor_to_producer_dict[inp_0]
                    inp_1_node = tensor_to_producer_dict[inp_1]

                    if inp_0_node.op_type == "Unsqueeze" and inp_1_node.op_type == "Reciprocal":
                        rec_node = inp_1_node
                        rec_inp_0 = rec_node.input[0]
                        rec_inp_0_node = tensor_to_producer_dict[rec_inp_0]

                        if rec_inp_0_node.op_type == "Sqrt":
                            sqrt_node = rec_inp_0_node
                            sqrt_inp_0 = sqrt_node.input[0]
                            sqrt_inp_0_node = tensor_to_producer_dict[sqrt_inp_0]

                            if sqrt_inp_0_node.op_type == "Max":
                                max_node = sqrt_inp_0_node
                                max_inp_0 = max_node.input[0]
                                max_inp_1 = max_node.input[1]
                                max_inp_0_node = tensor_to_producer_dict[max_inp_0]

                                if max_inp_0_node.op_type == "ReduceSum":
                                    red_node = max_inp_0_node
                                    red_inp_0 = red_node.input[0]
                                    red_inp_0_node = tensor_to_producer_dict[red_inp_0]

                                if red_inp_0_node.op_type == "Mul":
                                    mul_node = red_inp_0_node
                                    mul_inp_0 = mul_node.input[0]
                                    mul_inp_0_node = tensor_to_producer_dict[mul_inp_0]

                                    if mul_inp_0_node.op_type == "Unsqueeze":
                                        uns_node = mul_inp_0_node

                                        logger.info(f"Found L2norm ops from {node.name}.")
                                        nodes_to_remove_list = [
                                            node,
                                            rec_node,
                                            sqrt_node,
                                            max_node,
                                            red_node,
                                            mul_node,
                                        ]
                                        remove_nodes.extend(nodes_to_remove_list)

                                        eps_init = onnx_model.get_initializer(max_inp_1)
                                        remove_inits.append(eps_init)

                                        inp = uns_node.output[0]
                                        out = node.output[0]
                                        l2norm_node = onnx.helper.make_node(
                                            "LpNormalization", [inp], [out], node.name, p=2
                                        )
                                        onnx_model.add_node(l2norm_node)
                                        logger.info("Converted L2norm ops from {node.name} to LpNormalization.")
                except Exception as e:
                    logger.debug(
                        f"FuseL2Norm is enabled, but {node.name} does not meet the matching rules:{e}, skipping this node"
                    )

        onnx_model.remove_nodes(remove_nodes)
        onnx_model.remove_initializers(remove_inits)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()

        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """Run the pass using the provided configuration.

        Args:
            model (ModelProto): Input ONNX model.
            config (dict[str, PassConfigParam]): Configuration dict.

        Returns:
            ModelProto: Updated ONNX model after applying L2Norm fusion.
        """
        if "fuse_l2_norm" in config and config["fuse_l2_norm"]:
            model = self._onnx_fuse_l2_norm(model)
        else:
            logger.warning(
                "Please ensure that the onnx_fuse_l2_norm pass contains the fuse_l2_norm parameter and it is True."
            )
        return model
