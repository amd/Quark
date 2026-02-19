#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import ModelProto
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ONNXFoldBatchNormAfterConcatPass(ONNXAdapterPass):
    """
    Pass that folds BatchNormalization nodes into their upstream Conv, ConvTranspose,
    or Gemm nodes when BatchNorm appears after a Concat. This improves inference
    efficiency by eliminating BatchNorm nodes and baking their effects into weights
    and biases.

    Main entry: `_run_for_config`
    Core logic: `_onnx_fold_batch_norm_after_concat`
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Return the default configuration for this pass.

        Returns:
            dict[str, PassConfigParam]: A dict containing the default boolean option
            `fold_batch_norm_after_concat`, which controls whether the folding pass
            should run.
        """
        config = {
            "fold_batch_norm_after_concat": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to fold BatchNormalization operations into Conv, ConvTranspose and Gemm operations before Concat operations.",
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
        start: int,
        end: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Compute the folded weights and biases for a Conv/Gemm/ConvTranspose operator
        when merging BatchNormalization parameters into them.

        Args:
            target_type (str): Op type of the target layer ("Conv", "Gemm", "ConvTranspose").
            target_weight (NDArray): The original weight tensor.
            target_bias (NDArray): The original bias tensor (created if missing).
            bn_gamma (NDArray | None): BN gamma scale.
            bn_beta (NDArray | None): BN beta shift.
            bn_mean (NDArray): BN running mean.
            bn_var (NDArray): BN running variance.
            bn_epsilon (float): BN epsilon.
            start (int): Channel slice start index.
            end (int): Channel slice end index.

        Returns:
            tuple: (folded_weight, folded_bias)
        """
        if bn_gamma is not None:
            multiplier = bn_gamma[start:end] / np.sqrt(bn_var[start:end] + bn_epsilon)
        else:
            multiplier = 1 / np.sqrt(bn_var[start:end] + bn_epsilon)

        if target_type == "Gemm":
            bn_weight = np.diag(multiplier)
        elif target_type == "ConvTranspose":
            bn_weight = multiplier.reshape(1, len(multiplier), 1, 1)
        elif target_type == "Conv":
            bn_weight = multiplier.reshape(len(multiplier), 1, 1, 1)

        if bn_beta is not None:
            bn_bias = bn_beta[start:end] + (-bn_mean[start:end]) * multiplier
        else:
            bn_bias = (-bn_mean[start:end]) * multiplier

        if target_type == "Gemm":
            folded_weight = np.dot(bn_weight, target_weight)
            folded_bias = np.dot(bn_weight, target_bias) + bn_bias
        elif target_type == "ConvTranspose":
            folded_weight = bn_weight * target_weight
            folded_bias = bn_weight.reshape(1, -1) * target_bias + bn_bias
            folded_bias = folded_bias.reshape(-1)
        elif target_type == "Conv":
            folded_weight = bn_weight * target_weight
            folded_bias = bn_weight.reshape(-1) * target_bias + bn_bias

        return folded_weight, folded_bias

    def _onnx_fold_batch_norm_after_concat(self, model: ModelProto) -> ModelProto:
        """
        Perform BatchNormalization folding for the pattern:
            Conv/ConvTranspose/Gemm → Concat → BatchNormalization

        Steps:
            1. Identify BatchNorm nodes whose input comes from a Concat.
            2. Verify all Concat parents are foldable target ops.
            3. Load BN parameters and fold them into each target op's weights/biases.
            4. Replace Concat outputs and remove the BatchNorm node.
            5. Clean and topologically sort the ONNX graph.

        Args:
            model (ModelProto): Input ONNX model.

        Returns:
            ModelProto: Modified ONNX model with BN nodes folded.
        """
        onnx_model = ONNXModel(model)

        TARGET_OPS = ("ConvTranspose", "Gemm", "Conv")

        remove_nodes = []

        for node in onnx_model.model.graph.node:
            if node.op_type != "BatchNormalization":
                continue

            if len(node.input) != 5:
                logger.warning(f"BatchNorm {node.name} with {len(node.input)} inputs cannot be folded.")
                continue

            parent_node = onnx_model.get_parent(node, 0)
            if parent_node is None:
                logger.warning(f"BatchNorm {node.name} that is isolated node cannot be folded.")
                continue

            if parent_node.op_type == "Concat":
                grandparent_nodes = onnx_model.get_parents(parent_node)
            else:
                continue

            is_foldable = True
            for target_node in grandparent_nodes:
                target_type = target_node.op_type
                if target_type not in TARGET_OPS:
                    logger.debug(
                        f"Not all parent nodes of Concat are in ['ConvTranspose', 'Gemm', 'Conv'], so BatchNorm {node.name} after Concat node cannot be folded."
                    )
                    is_foldable = False
                    break
                if target_type == "Gemm":
                    transB = next((attr.i for attr in target_node.attribute if attr.name == "transB"), 0)
                    # TODO: Support transB is 0
                    if transB == 0:
                        logger.debug(f"Target node f{target_node.name}'s transB=0 is not supported.")
                        is_foldable = False
                        if len(target_node.input) > 1:
                            target_weight_init = onnx_model.get_initializer(target_node.input[1])
                            if (
                                len(target_weight_init.dims) == 2
                                and target_weight_init.dims[0] == target_weight_init.dims[1]
                            ):
                                is_foldable = True
                        break
                if target_type == "ConvTranspose":
                    group = next((attr.i for attr in target_node.attribute if attr.name == "group"), 1)
                    # TODO: Support ConvTranspose group != 1
                    if group != 1:
                        logger.debug(f"Target node f{target_node.name}'s group !=1 is not supported.")
                        is_foldable = False
                        break

                target_weight_init = onnx_model.get_initializer(target_node.input[1])
                target_weight = None if target_weight_init is None else onnx.numpy_helper.to_array(target_weight_init)
                if target_weight is None:
                    logger.warning(f"BatchNorm {node.name}'s target node f{target_node.name} is not foldable.")
                    is_foldable = False
                    break

            if is_foldable is False:
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

            start_idx, end_idx = 0, 0
            for i in range(len(grandparent_nodes)):
                target_node = grandparent_nodes[i]

                target_weight_init = onnx_model.get_initializer(target_node.input[1])
                target_weight = onnx.numpy_helper.to_array(target_weight_init)

                target_bias_init = (
                    onnx_model.get_initializer(target_node.input[2]) if len(target_node.input) > 2 else None
                )
                target_bias = None if target_bias_init is None else onnx.numpy_helper.to_array(target_bias_init)

                if target_bias is None:
                    if target_type == "Conv" or target_type == "Gemm":
                        target_bias = np.zeros(target_weight.shape[0])
                    else:
                        target_bias = np.zeros(target_weight.shape[1])

                    target_bias_name = target_node.name + "_bias_4bn"
                    target_bias_init = onnx.numpy_helper.from_array(
                        target_bias.astype(np.float32), name=target_bias_name
                    )
                    onnx_model.add_initializer(target_bias_init)
                    target_node.input.append(target_bias_name)

                end_idx += target_bias.shape[0]

                folded_weight, folded_bias = self._get_folded_weight_bias(
                    target_type,
                    target_weight,
                    target_bias,
                    bn_gamma,
                    bn_beta,
                    bn_mean,
                    bn_var,
                    bn_epsilon,
                    start_idx,
                    end_idx,
                )

                start_idx += target_bias.shape[0]

                folded_weight_init = onnx.numpy_helper.from_array(
                    folded_weight.astype(np.float32), name=target_weight_init.name
                )
                target_weight_init.CopyFrom(folded_weight_init)
                assert target_bias_init is not None
                folded_bias_init = onnx.numpy_helper.from_array(
                    folded_bias.astype(np.float32), name=target_bias_init.name
                )
                target_bias_init = onnx_model.get_initializer(target_node.input[2])
                target_bias_init.CopyFrom(folded_bias_init)

            children = onnx_model.get_children(parent_node)

            for child in children:
                if child is node:
                    continue

                for input_index, input_name in enumerate(child.input):
                    if input_name == parent_node.output[0]:
                        child.input[input_index] = node.output[0]

            parent_node.output[0] = node.output[0]

            remove_nodes.append(node)

            logger.info(f"Folded {node.op_type} {node.name} to {target_node.op_type} {target_node.name}.")

        onnx_model.remove_nodes(remove_nodes)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()

        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Execute the pass using the provided configuration.

        Args:
            model (ModelProto): Input ONNX model.
            config (dict[str, PassConfigParam]): Configurations for the pass.

        Returns:
            ModelProto: Output model with BatchNorm folded if enabled.
        """
        if "fold_batch_norm_after_concat" in config and config["fold_batch_norm_after_concat"]:
            model = self._onnx_fold_batch_norm_after_concat(model)
        else:
            logger.warning(
                "Please ensure that the onnx_fold_batch_norm_after_concat pass contains the fold_batch_norm_after_concat parameter and it is True."
            )
        return model
