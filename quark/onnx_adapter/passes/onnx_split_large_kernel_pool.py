#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from math import sqrt

import onnx
from onnx import ModelProto
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ONNXSplitLargeKernelPoolPass(ONNXAdapterPass):
    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Return the default configuration for this pass.

        Returns:
            dict[str, PassConfigParam]: Configuration parameters including
            whether to enable splitting large-kernel GlobalAveragePool operations.
        """
        config = {
            "split_large_kernel_pool": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to split large kernel pooling operations into multiple smaller poolings.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_split_large_kernel_pool(self, model: ModelProto) -> ModelProto:
        """
        Split GlobalAveragePool nodes with excessively large kernels into
        smaller AveragePool operations when feasible.

        This function:
            - Identifies GlobalAveragePool nodes.
            - Extracts input tensor spatial dimensions (H, W) from value_info.
            - Determines whether the kernel area (H×W) exceeds the 512 threshold.
            - Factorizes H and W to obtain smaller kernel sizes.
            - Inserts an AveragePool node before the GlobalAveragePool to reduce kernel size.
            - Updates graph connections and logs actions.

        Args:
            model (ModelProto): The ONNX model to process.

        Returns:
            ModelProto: The updated ONNX model with large-kernel GAP nodes split
            into smaller pooling stages when possible.
        """
        onnx_model = ONNXModel(model)

        def _get_factors(num: int) -> tuple[int, int]:
            """
            Compute a pair of factors (a, b) such that a * b = num and
            a is as close to sqrt(num) as possible.

            Args:
                num (int): The number to factorize.

            Returns:
                tuple[int, int]: A pair of factors whose product equals `num`.
            """
            factor_1 = int(sqrt(num))
            while factor_1 > 1:
                if num % (factor_1) == 0:
                    factor_2 = num / factor_1
                    return int(factor_1), int(factor_2)
                factor_1 = factor_1 - 1
            factor_2 = num
            return int(factor_1), int(factor_2)

        for node in onnx_model.model.graph.node:
            if node.op_type == "GlobalAveragePool":
                input_name = node.input[0]
                kw = None
                kh = None

                # Locate input tensor shape from value_info
                for input_info in onnx_model.model.graph.value_info:
                    if input_info.name == input_name:
                        input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
                        if len(input_shape) == 4:
                            kh = input_shape[2]
                            kw = input_shape[3]
                        break

                if not kw or not kh:
                    logger.warning(f"Failed to get the input shape, skip optimizing for GlobalAveragePool {node.name}.")
                    continue

                # Kernel is large → attempt split
                elif kw * kh > 512:
                    kh1, kh2 = _get_factors(kh)
                    kw1, kw2 = _get_factors(kw)

                    # If splitting still produces large kernels → skip
                    if kh1 * kw1 > 512 or kh2 * kw2 > 512:
                        logger.warning(
                            "After split, the kernel size is still too large."
                            "Currently, only one split is supported. Skip optimization."
                        )
                    else:
                        split_tensor = node.input[0] + "_Split"
                        pool_node = onnx.helper.make_node(
                            "AveragePool",
                            inputs=[node.input[0]],
                            outputs=[split_tensor],
                            kernel_shape=[kh1, kw1],
                            strides=[kh1, kw1],
                            name=split_tensor,
                        )

                        if not node.name:
                            node.name = node.output[0]

                        node.input[0] = split_tensor
                        onnx_model.model.graph.node.extend([pool_node])

                        logger.info(
                            f"Found GlobalAveragePool node {node.name} with large kernel size. "
                            f"Split it into multiple AveragePools."
                        )

        onnx_model.clean_initializers()
        onnx_model.topological_sort()
        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Execute the pass according to the provided configuration.

        Args:
            model (ModelProto): The ONNX model to process.
            config (dict[str, PassConfigParam]): Configuration options for this pass.

        Returns:
            ModelProto: The processed model with large kernel pooling operations split
            when the corresponding configuration flag is enabled.
        """
        if "split_large_kernel_pool" in config and config["split_large_kernel_pool"] is not None:
            model = self._onnx_split_large_kernel_pool(model)
        else:
            logger.warning(
                "Please ensure that the onnx_split_large_kernel_pool pass contains the split_large_kernel_pool parameter and it is True."
            )
        return model
