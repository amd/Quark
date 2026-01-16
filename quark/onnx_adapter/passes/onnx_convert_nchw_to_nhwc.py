#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from typing import Any

import onnx
from onnx import ModelProto, NodeProto, TensorShapeProto
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ONNXConvertNCHWToNHWCPass(ONNXAdapterPass):
    """
    Convert an ONNX model from NCHW layout to NHWC layout.

    This pass modifies both input and output tensor shapes and inserts appropriate
    Transpose nodes to convert the model's data layout.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Return the default configuration for this pass.

        Returns:
            dict[str, PassConfigParam]: A dictionary defining configuration parameters
            such as whether to perform NCHW → NHWC conversion.
        """
        config = {
            "convert_nchw_to_nhwc": PassConfigParam(
                type_=bool | list,
                default_value=True,
                required=True,
                description="Whether to convert the input model from NCHW to NHWC.",
            )
        }
        config.update(self.config)
        return config

    def _clean_initializer_in_input(self, model: ModelProto) -> ModelProto:
        """
        Remove initializers that are incorrectly listed as inputs in the model graph.

        Args:
            model (ModelProto): The ONNX model to clean.

        Returns:
            ModelProto: The updated ONNX model with corrected inputs.
        """
        if model.ir_version < 4:
            logger.warning("Initilizer should be included in input domain if the model ir_version is below 4.")
            logger.warning("The mode ir_version will be set as 4")
            model.ir_version = 4

        inputs = model.graph.input
        input_name_dict = {}
        for inp in inputs:
            input_name_dict[inp.name] = inp

        for init in model.graph.initializer:
            if init.name in input_name_dict:
                model.graph.input.remove(input_name_dict[init.name])

        return model

    def _get_shape_list(self, shape: TensorShapeProto) -> list[int | str]:
        """
        Convert a TensorShapeProto object into a list of dimensions.

        Args:
            shape (TensorShapeProto): The shape proto to convert.

        Returns:
            list[Union[int, str]]: A list of integers or symbolic dimension strings.
        """
        shape_list = []
        for d in shape.dim:
            if d.HasField("dim_value"):
                shape_list.append(d.dim_value)
            elif d.HasField("dim_param"):
                shape_list.append(d.dim_param)
            else:
                shape_list.append("?")
        return shape_list

    def _onnx_convert_nchw_to_nhwc(self, model: ModelProto, specified_nodes: list[str] | None = None) -> Any:
        """
        Perform the actual NCHW → NHWC conversion on the ONNX model.

        This function modifies the model’s graph by:
            - Swapping channel and spatial dimensions in tensor shapes.
            - Inserting Transpose nodes before inputs and after outputs.
            - Handling quantized layers appropriately.

        Args:
            model (ModelProto): The ONNX model to convert.
            specified_nodes (List[str] | None): Only convert the nodes that are specified in the list.
        Returns:
            Any: The converted ONNX model with NHWC layout.
        """
        temp_model = self._clean_initializer_in_input(model)
        onnx_model = ONNXModel(temp_model)

        if specified_nodes:
            specified_set = set(specified_nodes)
        else:
            specified_set = None

        node_name_list = []
        for node in onnx_model.graph().node:
            node_name_list.append(node.name)

        for inp in onnx_model.graph().input:
            if specified_set is not None:
                if inp.name not in specified_set:
                    continue
                else:
                    specified_set.remove(inp.name)

            shape_list = self._get_shape_list(inp.type.tensor_type.shape)

            if len(shape_list) != 4:
                logger.warning(
                    f"Expected 4-dimension input shape but got {shape_list}, skip the nchw to nhwc conversion."
                )
                continue

            C, H, W = shape_list[1:]
            if not all(isinstance(_, int) for _ in [C, H, W]):
                logger.warning(
                    f"Expected integer input shape but got [{C}, {H}, {W}], skip the nchw to nhwc conversion."
                )
                continue

            if not (int(H) > int(C) and int(W) > int(C)):
                logger.warning(
                    f"Expected H,W > C but got [{C}, {H}, {W}]. Please confirm whether the input model is in NCHW format"
                )

            inp.type.tensor_type.shape.dim[1].dim_value = H
            inp.type.tensor_type.shape.dim[2].dim_value = W
            inp.type.tensor_type.shape.dim[3].dim_value = C

            transpose_name = inp.name + "_transpose"
            count = 1
            while transpose_name in node_name_list:
                transpose_name += "_" + str(count)
                count += 1
            inp_transpose_node = onnx.helper.make_node(
                "Transpose", [inp.name], [transpose_name], name=transpose_name, perm=[0, 3, 1, 2]
            )
            onnx_model.replace_input_of_all_nodes(inp.name, transpose_name)
            onnx_model.add_node(inp_transpose_node)

        for out in onnx_model.graph().output:
            if specified_set is not None:
                if out.name not in specified_set:
                    continue
                else:
                    specified_set.remove(out.name)

            shape_list = self._get_shape_list(out.type.tensor_type.shape)

            if len(shape_list) != 4:
                logger.info(
                    f"Expected 4-dimension output shape but got {shape_list}, skip the nchw to nhwc conversion for output {out}."
                )
                continue

            C, H, W = shape_list[1:]
            if not all(isinstance(_, int) for _ in [C, H, W]):
                logger.warning(
                    f"Expected integer output shape but got [{C}, {H}, {W}], skip the nchw to nhwc conversion for output {out}."
                )
                continue

            if not (int(H) > int(C) and int(W) > int(C)):
                logger.warning(
                    f"Expected H,W > C but got [{C}, {H}, {W}], Please confirm whether the output {out} is in NCHW format"
                )

            out.type.tensor_type.shape.dim[1].dim_value = H
            out.type.tensor_type.shape.dim[2].dim_value = W
            out.type.tensor_type.shape.dim[3].dim_value = C

            transpose_name = out.name + "_transpose"
            count = 1
            while transpose_name in node_name_list:
                transpose_name += "_" + str(count)
                count += 1
            out_transpose_node = onnx.helper.make_node(
                "Transpose", [out.name], [transpose_name], name=transpose_name, perm=[0, 2, 3, 1]
            )
            onnx_model.add_node(out_transpose_node)
            last_node: NodeProto | Any = None
            penultimate_node: NodeProto | Any = None
            for node in onnx_model.graph().node:
                if node.output[0] == out.name:
                    last_node = node
                    logger.debug(f"last_node name :`{last_node.name}` .")
            for node in onnx_model.graph().node:
                if node.output[0] == last_node.input[0]:
                    penultimate_node = node
                    logger.debug(f"penultimate_node name :`{penultimate_node.name}` .")
            if last_node.op_type == "DequantizeLinear" and penultimate_node.op_type == "QuantizeLinear":
                quantize_linear_name = out_transpose_node.name + "_QuantizeLinear"
                transpose_QuantizeLinear = onnx.helper.make_node(
                    op_type=penultimate_node.op_type,
                    inputs=[out_transpose_node.output[0], penultimate_node.input[1], penultimate_node.input[2]],
                    outputs=[quantize_linear_name],
                    name=out_transpose_node.name + "_QuantizeLinear",
                    domain=penultimate_node.domain,
                )
                out_transpose_node.output[0] = quantize_linear_name
                onnx_model.graph().node.extend([transpose_QuantizeLinear])
                dequantize_linear_name = out_transpose_node.name + "_DequantizeLinear"
                transpose_DequantizeLinear = onnx.helper.make_node(
                    op_type=last_node.op_type,
                    inputs=[transpose_QuantizeLinear.output[0], last_node.input[1], last_node.input[2]],
                    outputs=[dequantize_linear_name],
                    name=out_transpose_node.name + "_DequantizeLinear",
                    domain=last_node.domain,
                )
                onnx_model.graph().node.extend([transpose_DequantizeLinear])
                transpose_DequantizeLinear.output[0] = transpose_name
                out.name = dequantize_linear_name
            else:
                out.name = transpose_name
        onnx_model.topological_sort()
        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Execute the NCHW → NHWC conversion if enabled in the configuration.

        Args:
            model (ModelProto): The ONNX model to process.
            config (dict[str, PassConfigParam]): The configuration for this pass.

        Returns:
            ModelProto: The converted ONNX model.
        """
        if "convert_nchw_to_nhwc" in config and (
            isinstance(config["convert_nchw_to_nhwc"], list) or config["convert_nchw_to_nhwc"]
        ):
            converted_model = (
                self._onnx_convert_nchw_to_nhwc(model, config["convert_nchw_to_nhwc"])
                if isinstance(config["convert_nchw_to_nhwc"], list)
                else self._onnx_convert_nchw_to_nhwc(model)
            )
        else:
            logger.warning(
                "Please ensure that the onnx_convert_nchw_to_nhwc pass contains the convert_nchw_to_nhwc parameter and it is True."
            )
        return converted_model
