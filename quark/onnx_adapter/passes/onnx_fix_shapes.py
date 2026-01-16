#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import ModelProto, helper

from quark.onnx.utils.system_utils import create_tmp_dir
from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ONNXFixShapesPass(ONNXAdapterPass):
    """ONNX adapter pass that fixes model input/output shapes and infers all
    intermediate tensor shapes via ONNX Runtime execution.

    This pass:
      1. Rewrites user-specified input/output shapes.
      2. Runs ONNX Runtime inference with random data.
      3. Collects & updates all intermediate tensor shapes in value_info.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return default configuration for this pass.

        Returns:
            dict[str, PassConfigParam]: Configuration containing the
            ``input_output_name_shapes`` parameter that maps tensor names
            to their desired static shapes.
        """
        config = {
            "input_output_name_shapes": PassConfigParam(
                type_=dict[str, list[int]],
                default_value=None,
                required=True,
                description="Model input/output name & input/output shapes to replace shape of. For example: {'input_1': [1, 224, 224, 3], 'input_2': [1, 96, 96, 3], 'output_1' :[1, 1000], 'output_2':[1, 10]}",
            )
        }
        config.update(self.config)
        return config

    def _create_infer_session_for_onnx_model(
        self, model: ModelProto, sess_options: ort.SessionOptions | None = None
    ) -> ort.InferenceSession:
        """Create an ONNX Runtime InferenceSession for a given model.

        Args:
            model (ModelProto): The ONNX model.
            sess_options (ort.SessionOptions | None): Optional inference session options.

        Returns:
            ort.InferenceSession: A session created from the serialized temporary model.
        """
        temp_dir = create_tmp_dir(prefix="onnx_adapter.fix_shapes.")
        temp_path = Path(temp_dir.name).joinpath("infer_model.onnx").as_posix()
        model_to_save = copy.deepcopy(model)
        onnx.save(model_to_save, temp_path, save_as_external_data=True)
        return ort.InferenceSession(temp_path, sess_options)

    def _fix_input_and_output_shapes(
        self, model: ModelProto, input_output_name_shapes: dict[str, list[int]]
    ) -> ModelProto:
        """Rewrite input/output tensor shapes in the model graph.

        Args:
            model (ModelProto): The ONNX model to modify.
            input_output_name_shapes (dict[str, list[int]]): Mapping of input/output
                tensor names to new shapes.

        Returns:
            ModelProto: The updated ONNX model with modified input/output shapes.
        """
        for i in range(len(model.graph.input)):
            name = model.graph.input[i].name
            shapes = input_output_name_shapes[name]
            for j in range(len(shapes)):
                dim_value = shapes[j]
                model.graph.input[i].type.tensor_type.shape.dim[j].dim_value = dim_value
        for i in range(len(model.graph.output)):
            name = model.graph.output[i].name
            shapes = input_output_name_shapes[name]
            for j in range(len(shapes)):
                dim_value = shapes[j]
                model.graph.output[i].type.tensor_type.shape.dim[j].dim_value = dim_value
        return model

    def _generate_random_data(self, model: ModelProto) -> dict[str, np.ndarray[Any, Any]]:
        """Generate random input tensors matching model input signatures.

        Args:
            model (ModelProto): Model used to determine input names, dtypes, and shapes.

        Returns:
            dict[str, np.ndarray]: Randomly generated input data keyed by input name.

        Raises:
            ValueError: If an unsupported input dtype is encountered.
        """
        np.random.seed(42)
        sess = self._create_infer_session_for_onnx_model(model)
        input_info = sess.get_inputs()
        input_data = {}

        for inp in input_info:
            input_name = inp.name
            input_shape = inp.shape
            input_dtype = inp.type

            if input_dtype == "tensor(int8)":
                dtype = np.int8
            elif input_dtype == "tensor(uint8)":
                dtype = np.uint8  # type: ignore
            elif input_dtype == "tensor(int16)":
                dtype = np.int16  # type: ignore
            elif input_dtype == "tensor(uint16)":
                dtype = np.uint16  # type: ignore
            elif input_dtype == "tensor(int32)":
                dtype = np.int32  # type: ignore
            elif input_dtype == "tensor(uint32)":
                dtype = np.uint32  # type: ignore
            elif input_dtype == "tensor(int64)":
                dtype = np.int64  # type: ignore
            elif input_dtype == "tensor(uint64)":
                dtype = np.uint64  # type: ignore
            elif input_dtype == "tensor(float16)":
                dtype = np.float16  # type: ignore
            elif input_dtype == "tensor(float)":
                dtype = np.float32  # type: ignore
            elif input_dtype == "tensor(double)":
                dtype = np.float64  # type: ignore
            elif input_dtype == "tensor(bool)":
                dtype = np.bool_  # type: ignore
            else:
                raise ValueError(f"Unsupported dtype: {input_dtype}")

            random_input_data = np.random.random(input_shape).astype(dtype)
            input_data[input_name] = random_input_data

        return input_data

    def _infer_all_tensors_shape(self, model: ModelProto, save_as_external_data: bool = False) -> dict[str, tuple[int]]:
        """Infer shapes for all intermediate tensors by temporarily adding them
        as outputs and running inference.

        Args:
            model (ModelProto): The ONNX model.
            save_as_external_data (bool): Unused parameter, kept for API compatibility.

        Returns:
            dict[str, tuple[int]]: Mapping from tensor name to inferred shape.
        """
        output_list = []
        for node in model.graph.node:
            for tensor_name in node.input:
                if tensor_name in model.graph.input:
                    continue
                model.graph.output.extend([onnx.ValueInfoProto(name=tensor_name)])
                output_list.append(tensor_name)

        input_data = self._generate_random_data(model)
        ort_session = self._create_infer_session_for_onnx_model(model)
        output = ort_session.run(output_list, input_data)

        assert len(output_list) == len(output)
        tensor_name_shape_dict = {}
        for i in range(len(output_list)):
            tensor_name_shape_dict[output_list[i]] = output[i].shape
        return tensor_name_shape_dict

    def _save_all_tensors_shape(self, model: ModelProto, tensor_name_shape_dict: dict[str, tuple[int]]) -> ModelProto:
        """Write inferred tensor shapes back into the model's value_info.

        Args:
            model (ModelProto): The ONNX model to update.
            tensor_name_shape_dict (dict[str, tuple[int]]): Mapping of tensor names to shapes.

        Returns:
            ModelProto: The updated model containing full tensor shape metadata.
        """
        for tensor_name, new_shape in tensor_name_shape_dict.items():
            if len(new_shape) > 0:
                for value_info in model.graph.value_info:
                    if value_info.name == tensor_name:
                        new_dtype = value_info.type.tensor_type.elem_type
                        updated_value_info = helper.make_tensor_value_info(tensor_name, new_dtype, new_shape)
                        value_info.CopyFrom(updated_value_info)
        return model

    def _onnx_fix_shapes(self, model: ModelProto, input_output_name_shapes: dict[str, list[int]]) -> ModelProto:
        """Fix input/output shapes and infer all internal tensor shapes.

        Args:
            model (ModelProto): The ONNX model.
            input_output_name_shapes (dict[str, list[int]]): Updated shapes for inputs/outputs.

        Returns:
            ModelProto: Model with updated shapes. If inference fails, the original
            model is returned and a warning is logged.
        """
        try:
            temp_model = self._fix_input_and_output_shapes(model, input_output_name_shapes)
            tensor_name_shape_dict = self._infer_all_tensors_shape(temp_model)
            model = self._save_all_tensors_shape(temp_model, tensor_name_shape_dict)
        except Exception as e:
            logger.warning(f"Failed to fix shapes of the model beacuse of {e}.")

        return model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """Execute this pass using a resolved configuration.

        Args:
            model (ModelProto): The ONNX model.
            config (dict[str, PassConfigParam]): Resolved config parameters.

        Returns:
            ModelProto: The processed model.
        """
        if "input_output_name_shapes" in config and config["input_output_name_shapes"] is not None:
            model = self._onnx_fix_shapes(model, config["input_output_name_shapes"])
        else:
            logger.warning(
                "Please ensure that the onnx_fix_shapes pass contains the input_output_name_shapes parameter and it is not None."
            )
        return model
