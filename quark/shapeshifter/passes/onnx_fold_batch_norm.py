#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import ModelProto
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


@register_pass
class ONNXFoldBatchNormPass(ONNXPass):
    """ONNX pass to fold BatchNormalization into preceding operators.

    This pass supports folding BatchNormalization into the following ops:
    - Gemm (when transB = 1)
    - ConvTranspose (when group = 1)

    Attributes:
        logger: Screen logger for debug and info messages.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return default pass configuration.

        Returns:
            dict[str, PassConfigParam]: Configuration for controlling
            whether BatchNormalization nodes should be folded.
        """
        config = {
            "fold_batch_norm": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to fold BatchNormalization operations into Conv, ConvTranspose and Gemm operations.",
            )
        }
        config.update(self.config)
        return config

    def _get_folded_weight_bias(
        self,
        target_type: str,
        target_weight: NDArray[np.float32],
        target_bias: NDArray[np.float32] | NDArray[np.float64],
        bn_gamma: NDArray[np.float32] | None,
        bn_beta: NDArray[np.float32] | None,
        bn_mean: NDArray[np.float32],
        bn_var: NDArray[np.float32],
        bn_epsilon: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Compute folded weight and bias after merging BatchNorm into a target node.

        Args:
            target_type (str): Operator type ("Gemm" or "ConvTranspose").
            target_weight (NDArray): Original weight tensor.
            target_bias (NDArray): Original bias tensor.
            bn_gamma (NDArray|None): BatchNorm scale.
            bn_beta (NDArray|None): BatchNorm bias.
            bn_mean (NDArray): BatchNorm running mean.
            bn_var (NDArray): BatchNorm running variance.
            bn_epsilon (float): BatchNorm epsilon.

        Returns:
            tuple[NDArray, NDArray]: Folded weight and folded bias.
        """
        if bn_gamma is not None:
            multiplier = bn_gamma / np.sqrt(bn_var + bn_epsilon)
        else:
            multiplier = 1 / np.sqrt(bn_var + bn_epsilon)

        if target_type == "Gemm":
            bn_weight = np.diag(multiplier)
        elif target_type == "ConvTranspose":
            bn_weight = multiplier.reshape(1, len(multiplier), 1, 1)

        if bn_beta is not None:
            bn_bias = bn_beta + (-bn_mean) * multiplier
        else:
            bn_bias = (-bn_mean) * multiplier

        if target_type == "Gemm":
            folded_weight = np.dot(bn_weight, target_weight)
            folded_bias = np.dot(bn_weight, target_bias) + bn_bias
        elif target_type == "ConvTranspose":
            folded_weight = bn_weight * target_weight
            folded_bias = bn_weight.reshape(1, -1) * target_bias + bn_bias
            folded_bias = folded_bias.reshape(-1)

        return folded_weight, folded_bias

    def _onnx_fold_batch_norm(self, model: ModelProto) -> ModelProto:
        """Fold BatchNormalization nodes inside an ONNX model.

        This modifies ConvTranspose/Gemm nodes' weight & bias,
        replaces their outputs, and removes BatchNorm nodes.

        Args:
            model (ModelProto): The ONNX model.

        Returns:
            ModelProto: Updated model with folded BatchNormalization nodes.
        """
        onnx_model = ONNXModel(model)

        TARGET_OPS = ("ConvTranspose", "Gemm")

        remove_nodes = []

        for node in onnx_model.model.graph.node:
            if node.op_type != "BatchNormalization":
                continue

            if len(node.input) != 5:
                logger.warning(f"BatchNorm {node.name} with {len(node.input)} inputs cannot be folded.")
                continue

            target_node = onnx_model.get_parent(node, 0)
            if target_node is None:
                logger.warning(f"BatchNorm {node.name} that is isolated node cannot be folded.")
                continue

            if target_node.op_type not in TARGET_OPS:
                logger.debug(f"BatchNorm {node.name} after node {target_node.name} cannot be folded.")
                continue

            bn_gamma_init = onnx_model.get_initializer(node.input[1])
            bn_gamma = None if bn_gamma_init is None else onnx.numpy_helper.to_array(bn_gamma_init)
            bn_beta_init = onnx_model.get_initializer(node.input[2])
            bn_beta = None if bn_beta_init is None else onnx.numpy_helper.to_array(bn_beta_init)
            bn_mean_init = onnx_model.get_initializer(node.input[3])
            bn_mean = None if bn_mean_init is None else onnx.numpy_helper.to_array(bn_mean_init)
            bn_var_init = onnx_model.get_initializer(node.input[4])
            bn_var = None if bn_var_init is None else onnx.numpy_helper.to_array(bn_var_init)
            bn_epsilon = next((attr.f for attr in node.attribute if attr.name == "epsilon"), 1e-10)

            if bn_mean is None or bn_var is None:
                logger.warning(f"BatchNorm {node.name} that is missing mean or variance cannot be folded.")
                continue

            target_weight_init = onnx_model.get_initializer(target_node.input[1])
            target_weight = None if target_weight_init is None else onnx.numpy_helper.to_array(target_weight_init)

            target_bias_init = onnx_model.get_initializer(target_node.input[2]) if len(target_node.input) > 2 else None
            target_bias = None if target_bias_init is None else onnx.numpy_helper.to_array(target_bias_init)

            if target_weight is None:
                logger.warning(f"BatchNorm {node.name}'s target node f{target_node.name} is not foldable.")
                continue

            target_type = target_node.op_type
            if target_type == "Gemm":
                transB = next((attr.i for attr in target_node.attribute if attr.name == "transB"), 0)
                # TODO: Support transB is 0
                if transB == 0:
                    logger.debug(f"Target node f{target_node.name}'s transB=0 is not supported.")
                    continue
            if target_type == "ConvTranspose":
                group = next((attr.i for attr in target_node.attribute if attr.name == "group"), 1)
                # TODO: Support ConvTranspose group != 1
                if group != 1:
                    logger.debug(f"Target node f{target_node.name}'s group !=1 is not supported.")
                    continue

            if target_bias is None:
                if target_type == "Gemm":
                    target_bias = np.zeros(target_weight.shape[0])
                else:  # target_type == "ConvTranspose":
                    target_bias = np.zeros(target_weight.shape[1])

                target_bias_name = target_node.name + "_bias_4bn"
                target_bias_init = onnx.numpy_helper.from_array(target_bias.astype(np.float32), name=target_bias_name)
                onnx_model.add_initializer(target_bias_init)
                target_node.input.append(target_bias_name)

            # Calculate the weight and bias after folded
            folded_weight, folded_bias = self._get_folded_weight_bias(
                target_type, target_weight, target_bias, bn_gamma, bn_beta, bn_mean, bn_var, bn_epsilon
            )

            # Update target node's weight and bias
            folded_weight_init = onnx.numpy_helper.from_array(
                folded_weight.astype(np.float32), name=target_weight_init.name
            )
            target_weight_init.CopyFrom(folded_weight_init)
            assert target_bias_init is not None
            folded_bias_init = onnx.numpy_helper.from_array(folded_bias.astype(np.float32), name=target_bias_init.name)
            target_bias_init = onnx_model.get_initializer(target_node.input[2])
            target_bias_init.CopyFrom(folded_bias_init)

            # Deal with the tensor name
            children = onnx_model.get_children(target_node)

            for child in children:
                if child is node:  # this node will be removed
                    continue

                for input_index, input_name in enumerate(child.input):
                    if input_name == target_node.output[0]:
                        child.input[input_index] = node.output[0]

            target_node.output[0] = node.output[0]

            # TODO: has shared initializers?
            remove_nodes.append(node)

            logger.info(f"Folded {node.op_type} {node.name} to {target_node.op_type} {target_node.name}.")

        onnx_model.remove_nodes(remove_nodes)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()

        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """Run this pass using the provided configuration.

        Args:
            model (ModelProto): The ONNX model to process.
            config (dict[str, PassConfigParam]): Configuration dict.

        Returns:
            ModelProto: Processed model with BatchNorm folded if enabled.
        """
        if "fold_batch_norm" in config and config["fold_batch_norm"]:
            model = self._onnx_fold_batch_norm(model)
        else:
            logger.warning(
                "Please ensure that the onnx_fold_batch_norm pass contains the fold_batch_norm parameter and it is True."
            )
        return model
