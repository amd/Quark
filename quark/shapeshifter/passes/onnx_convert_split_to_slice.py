#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import onnx
from onnx import ModelProto, NodeProto
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


@register_pass
class ONNXConvertSplitToSlicePass(ONNXPass):
    """
    Convert Split operations into equivalent Slice node sequences.

    This pass detects Split nodes in the ONNX graph and rewrites them as
    multiple Slice nodes with appropriate starts/ends/axes/steps Constant
    initializers to preserve the original splitting behavior.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Return the default configuration for this pass.

        Returns:
            dict[str, PassConfigParam]: Configuration parameters including
            whether Split → Slice conversion is enabled.
        """
        config = {
            "convert_split_to_slice": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to convert Split operations to Slice operations.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_convert_split_to_slice(self, model: ModelProto) -> ModelProto:
        """
        Convert Split nodes in the model to an equivalent set of Slice nodes.

        This function:
            - Identifies Split nodes and reads their split sizes and axis.
            - Creates Constant nodes for starts, ends, axes, and steps.
            - Replaces each Split output with an appropriate Slice node.
            - Removes original Split nodes and unused initializers.

        Args:
            model (ModelProto): The ONNX model to process.

        Returns:
            ModelProto: The updated ONNX model with Split nodes replaced by Slice nodes.
        """
        onnx_model = ONNXModel(model)
        nodes_to_remove: list[NodeProto] = []
        init_to_remove: list[str] = []
        for node in onnx_model.model.graph.node:
            if node.op_type == "Split":
                num_input = len(node.input)
                axis_attr = next((attr for attr in node.attribute if attr.name == "axis"), None)
                assert axis_attr is not None, "No axis attribute founded in Split node"
                axis = axis_attr.i
                input_name = node.input[0]
                output_names = node.output
                if num_input == 2:
                    splits = None
                    for init in onnx_model.model.graph.initializer:
                        if init.name == node.input[1]:
                            splits = onnx.numpy_helper.to_array(init).tolist()
                    if splits is None:
                        logger.warning(
                            f"No split detected of {node.name}, "
                            "failed to convert split to slice, please check the input model."
                        )
                        break
                elif num_input == 1:
                    split_attr = next((attr for attr in node.attribute if attr.name == "split"), None)
                    if split_attr is None:
                        logger.warning(
                            f"No split detected of {node.name}, "
                            "failed to convert split to slice, please check the input model."
                        )
                        break
                    splits = split_attr.ints
                else:
                    logger.warning(
                        f"Failed to convert split of {node.name} to slice, the number of input nodes is not supported."
                    )
                    break
                starts = [sum(splits[:i]) for i in range(len(splits))]
                ends = [sum(splits[: i + 1]) for i in range(len(splits))]
                for i in range(len(output_names)):
                    starts_node = onnx.helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=[output_names[i] + "_starts_" + str(i)],
                        value=onnx.helper.make_tensor(
                            name=output_names[i] + "_starts_" + str(i),
                            data_type=onnx.TensorProto.INT64,
                            dims=[1],
                            vals=[starts[i]],
                        ),
                    )
                    ends_node = onnx.helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=[output_names[i] + "_ends_" + str(i)],
                        value=onnx.helper.make_tensor(
                            name=output_names[i] + "_ends_" + str(i),
                            data_type=onnx.TensorProto.INT64,
                            dims=[1],
                            vals=[ends[i]],
                        ),
                    )
                    axes_node = onnx.helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=[output_names[i] + "_axes_" + str(i)],
                        value=onnx.helper.make_tensor(
                            name=output_names[i] + "_axes_" + str(i),
                            data_type=onnx.TensorProto.INT64,
                            dims=[1],
                            vals=[axis],
                        ),
                    )
                    steps_node = onnx.helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=[output_names[i] + "_steps_" + str(i)],
                        value=onnx.helper.make_tensor(
                            name=output_names[i] + "_steps_" + str(i),
                            data_type=onnx.TensorProto.INT64,
                            dims=[1],
                            vals=[1],
                        ),
                    )
                    slice_node = onnx.helper.make_node(
                        "Slice",
                        inputs=[
                            input_name,
                            output_names[i] + "_starts_" + str(i),
                            output_names[i] + "_ends_" + str(i),
                            output_names[i] + "_axes_" + str(i),
                            output_names[i] + "_steps_" + str(i),
                        ],
                        outputs=[output_names[i]],
                        name=output_names[i] + "_" + str(i),
                    )
                    onnx_model.model.graph.node.extend([slice_node, starts_node, ends_node, axes_node, steps_node])
                nodes_to_remove.append(node)
                if len(node.input) > 1:
                    init_to_remove.append(node.input[1])
                logger.info(f"Found Split node {node.name}. Replacing with Slice.")
        onnx_model.remove_nodes(nodes_to_remove)
        onnx_model.remove_initializers(init_to_remove)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()

        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Execute the Split → Slice conversion if enabled in the configuration.

        Args:
            model (ModelProto): The ONNX model to process.
            config (dict[str, PassConfigParam]): The configuration for this pass.

        Returns:
            ModelProto: The updated model after conversion.
        """
        if "convert_split_to_slice" in config and config["convert_split_to_slice"]:
            model = self._onnx_convert_split_to_slice(model)
        else:
            logger.warning(
                "Please ensure that the onnx_convert_split_to_slice pass contains the convert_split_to_slice parameter and it is True."
            )
        return model
