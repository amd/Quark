#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import itertools

import numpy as np
import onnx
import packaging.version as pv
from numpy.typing import NDArray
from onnx import ModelProto, helper, numpy_helper
from onnx import onnx_pb as onnx_proto

from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

DEFAULT_OP_BLOCK_LIST_FP16 = [
    "ArrayFeatureExtractor",
    "Binarizer",
    "CastMap",
    "CategoryMapper",
    "DictVectorizer",
    "FeatureVectorizer",
    "Imputer",
    "LabelEncoder",
    "LinearClassifier",
    "LinearRegressor",
    "Normalizer",
    "OneHotEncoder",
    "RandomUniformLike",
    "SVMClassifier",
    "SVMRegressor",
    "Scaler",
    "TreeEnsembleClassifier",
    "TreeEnsembleRegressor",
    "ZipMap",
    "NonMaxSuppression",
    "TopK",
    "RoiAlign",
    "Resize",
    "Range",
    "CumSum",
    "Min",
    "Max",
    "Upsample",
]

DEFAULT_OP_BLOCK_LIST_FP32: list[str] = []


class ONNXConvertFP16ToFP32Pass(ONNXAdapterPass):
    """
    Pass to convert ONNX models from FP16 to FP32 for tensors and nodes.
    Provides methods to handle conversion for numpy arrays, TensorProto objects,
    and automatic graph-level transformations while respecting block lists.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """
        Define the default configuration for this pass.

        Returns:
            dict[str, PassConfigParam]: Dictionary containing configuration parameters.
        """
        config = {
            "convert_fp16_to_fp32": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to convert the input model from FP16 to FP32.",
            )
        }
        config.update(self.config)
        return config

    def _npfloat16_to_int(self, np_list: NDArray[np.float16]) -> list[int]:
        """
        Convert a numpy array of float16 values to a list of integers representing
        the raw binary format.

        Args:
            np_list (NDArray[np.float16]): Array of float16 values.

        Returns:
            list[int]: List of integers representing float16 binary data.
        """
        return [int(bin(_.view("H"))[2:].zfill(16), 2) for _ in np_list]

    def _npint_to_float(self, np_list: NDArray[np.int32]) -> list[float]:
        """
        Convert a numpy array of int32 values (representing float16) back to float32.

        Args:
            np_list (NDArray[np.int32]): Array of int32 values.

        Returns:
            list[float]: List of converted float32 values.
        """
        return [_.astype(np.uint16).view(np.float16).astype(np.float32).item() for _ in np_list]

    def _convert_np_to_float16(
        self, np_array: NDArray[np.float32], min_positive_val: float = 1e-7, max_finite_val: float = 1e4
    ) -> NDArray[np.float16]:
        """
        Safely convert a float32 numpy array to float16, clipping extreme positive
        and negative values, while preserving NaN, zero, and infinity values.

        Args:
            np_array (NDArray[np.float32]): Input float32 array.
            min_positive_val (float): Minimum positive value allowed.
            max_finite_val (float): Maximum finite value allowed.

        Returns:
            NDArray[np.float16]: Converted float16 array.
        """

        def between(a: float, b: NDArray[np.float32], c: float) -> NDArray[np.bool_]:
            return np.logical_and(a < b, b < c)

        if np_array[np.where(np_array > 0)].shape[0] > 0:
            pos_max = np_array[np.where(np_array > 0)].max()
            pos_min = np_array[np.where(np_array > 0)].min()
            if pos_max >= max_finite_val:
                logger.warning(f"the float32 number {pos_max} will be truncated to {max_finite_val}")
            if pos_min <= min_positive_val:
                logger.warning(f"the float32 number {pos_min} will be truncated to {min_positive_val}")

        if np_array[np.where(np_array < 0)].shape[0] > 0:
            neg_max = np_array[np.where(np_array < 0)].max()
            neg_min = np_array[np.where(np_array < 0)].min()
            if neg_min <= -max_finite_val:
                logger.warning(f"the float32 number {neg_min} will be truncated to {-max_finite_val}")
            if neg_max >= -min_positive_val:
                logger.warning(f"the float32 number {neg_max} will be truncated to {-min_positive_val}")

        np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
        np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
        np_array = np.where(between(max_finite_val, np_array, float("inf")), max_finite_val, np_array)
        np_array = np.where(between(float("-inf"), np_array, -max_finite_val), -max_finite_val, np_array)
        return np_array.astype(np.float16)

    def _convert_tensor_float_to_float16(
        self, tensor: onnx_proto.TensorProto, min_positive_val: float = 1e-7, max_finite_val: float = 1e4
    ) -> onnx_proto.TensorProto:
        """
        Convert an ONNX TensorProto from float32 to float16, handling float_data
        and raw_data attributes safely.

        Args:
            tensor (onnx_proto.TensorProto): Input tensor.
            min_positive_val (float): Minimum positive value allowed for clipping.
            max_finite_val (float): Maximum finite value allowed for clipping.

        Returns:
            onnx_proto.TensorProto: Converted tensor with float16 data.
        """
        if not isinstance(tensor, onnx_proto.TensorProto):
            raise ValueError(f"Expected input type is an ONNX TensorProto but got {type(tensor)}")

        if tensor.data_type == onnx_proto.TensorProto.FLOAT:
            tensor.data_type = onnx_proto.TensorProto.FLOAT16
            if tensor.float_data:
                float16_data = self._convert_np_to_float16(
                    np.array(tensor.float_data), min_positive_val, max_finite_val
                )
                tensor.int32_data[:] = self._npfloat16_to_int(float16_data)
                tensor.float_data[:] = []
            if tensor.raw_data:
                float32_list = np.fromstring(tensor.raw_data, dtype="float32")  # type: ignore
                float16_list = self._convert_np_to_float16(float32_list, min_positive_val, max_finite_val)
                tensor.raw_data = float16_list.tostring()  # type: ignore

        return tensor

    def _make_value_info_from_tensor(self, tensor: onnx_proto.TensorProto) -> onnx_proto.ValueInfoProto:
        """
        Generate a ValueInfoProto object from a given TensorProto.

        Args:
            tensor (onnx_proto.TensorProto): Input tensor.

        Returns:
            onnx_proto.ValueInfoProto: Generated value info.
        """
        shape = numpy_helper.to_array(tensor).shape
        return helper.make_tensor_value_info(tensor.name, tensor.data_type, shape)

    def _sort_graph_node(self, graph_proto: onnx_proto.GraphProto) -> None:
        """
        Topologically sort nodes in a graph so that inputs precede outputs.

        Args:
            graph_proto (onnx_proto.GraphProto): Input graph to sort.
        """

        def find_first_node(output2node_dict: dict[str, onnx_proto.NodeProto]) -> onnx_proto.NodeProto | None:
            for node in org_nodes:
                is_not_first_node = any(item in output2node_dict for item in node.input)
                if not is_not_first_node:
                    return node
            return None

        def remove_first_node_from_dict2(first_node: onnx_proto.NodeProto) -> None:
            for output in first_node.output:
                if output in output2node_dict:
                    del output2node_dict[output]

        org_nodes = graph_proto.node
        output2node_dict = {output: node for node in org_nodes for output in node.output}
        sorted_node = []

        while len(output2node_dict) > 0:
            first_node = find_first_node(output2node_dict)
            assert first_node is not None, "Cannot find the first node in the graph."
            sorted_node.append(first_node)
            remove_first_node_from_dict2(first_node)
            org_nodes.remove(first_node)

        for new_node in sorted_node:
            graph_proto.node.extend([new_node])

    def _sort_topology(self, graph_proto: onnx_proto.GraphProto) -> None:
        """
        Recursively sort the graph topology for the main graph and all sub-graphs.

        Args:
            graph_proto (onnx_proto.GraphProto): Input graph to sort.
        """
        assert isinstance(graph_proto, onnx_proto.GraphProto)
        self._sort_graph_node(graph_proto)
        for node in graph_proto.node:
            for attr in node.attribute:
                if isinstance(attr.g, onnx_proto.GraphProto) and len(attr.g.node) > 0:
                    self._sort_topology(attr.g)
                for g in attr.graphs:
                    if isinstance(g, onnx_proto.GraphProto):
                        self._sort_topology(g)

    def _convert_np_to_float(
        self, np_array: NDArray[np.float16], min_positive_val: float = 1e-7, max_finite_val: float = 1e4
    ) -> NDArray[np.float32]:
        """
        Convert float16 numpy array to float32 without changing sign or finiteness.

        Args:
            np_array (NDArray[np.float16]): Input float16 array.
            min_positive_val (float): Minimum positive value (unused).
            max_finite_val (float): Maximum finite value (unused).

        Returns:
            NDArray[np.float32]: Converted float32 array.
        """
        return np_array.astype(np.float32)

    def _convert_tensor_float16_to_float(self, tensor: onnx_proto.TensorProto) -> onnx_proto.TensorProto:
        """
        Convert a TensorProto from float16 to float32.

        Args:
            tensor (onnx_proto.TensorProto): Input tensor.

        Returns:
            onnx_proto.TensorProto: Converted tensor.
        """
        if not isinstance(tensor, onnx_proto.TensorProto):
            raise ValueError(f"Expected input type is an ONNX TensorProto but got {type(tensor)}")

        if tensor.data_type == onnx_proto.TensorProto.FLOAT16:
            tensor.data_type = onnx_proto.TensorProto.FLOAT
            if tensor.int32_data:
                float_list = self._npint_to_float(np.array(tensor.int32_data))
                tensor.int32_data[:] = []
                tensor.float_data[:] = float_list
            if tensor.raw_data:
                float16_list = np.fromstring(tensor.raw_data, dtype="float16")  # type: ignore
                float32_list = self._convert_np_to_float(float16_list)
                tensor.raw_data = float32_list.tostring()  # type: ignore
        return tensor

    def _onnx_convert_fp16_to_fp32(
        self,
        model: onnx_proto.ModelProto,
        disable_shape_infer: bool = False,
        op_block_list: list[str] | None = None,
        node_block_list: list[str] | None = None,
    ) -> onnx_proto.ModelProto:
        """
        Convert all tensor float16 types in the ONNX model to float32, respecting
        operator and node block lists.

        Args:
            model (onnx_proto.ModelProto): Input ONNX model.
            disable_shape_infer (bool): If True, skip shape/type inference.
            op_block_list (list[str] | None): List of operator types to exclude from conversion.
            node_block_list (list[str] | None): List of node names to exclude from conversion.

        Returns:
            onnx_proto.ModelProto: Converted ONNX model.
        """
        func_infer_shape = None
        if not disable_shape_infer and pv.Version(onnx.__version__) >= pv.Version("1.2"):  # type: ignore[attr-defined]
            try:
                from onnx.shape_inference import infer_shapes

                func_infer_shape = infer_shapes
            finally:
                pass

        if not isinstance(model, onnx_proto.ModelProto):
            raise ValueError(f"Expected model type is an ONNX ModelProto but got {type(model)}")

        if op_block_list is None:
            op_block_list = DEFAULT_OP_BLOCK_LIST_FP32
        if node_block_list is None:
            node_block_list = []
        op_block_list = set(op_block_list)
        node_block_list = set(node_block_list)

        queue = []
        value_info_list = []
        node_list = []
        node_dict = {}
        if func_infer_shape is not None:
            model = func_infer_shape(model)
        queue.append(model)
        name_mapping: dict[str, str] = {}
        graph_io_to_skip: set[str] = set()
        io_casts: set[str] = set()

        while queue:
            next_level = []
            for q in queue:
                if isinstance(q, onnx_proto.ModelProto):
                    next_level.append(q.graph)
                if isinstance(q, onnx_proto.GraphProto):
                    for n in q.node:
                        if n.name in io_casts:
                            continue
                        for i in range(len(n.input)):
                            if n.input[i] in name_mapping:
                                n.input[i] = name_mapping[n.input[i]]
                        for i in range(len(n.output)):
                            if n.output[i] in name_mapping:
                                n.output[i] = name_mapping[n.output[i]]
                        if n.op_type in op_block_list or n.name in node_block_list:
                            node_list.append(n)
                            node_dict[n.name] = q
                        else:
                            if n.op_type == "Cast":
                                for attr in n.attribute:
                                    if attr.name == "to" and attr.i == 10:
                                        attr.i = 1
                                        break
                            for attr in n.attribute:
                                next_level.append(attr)
                if isinstance(q, onnx_proto.AttributeProto):
                    next_level.append(q.g)
                    for n in q.graphs:
                        next_level.append(n)
                    q.t.CopyFrom(self._convert_tensor_float16_to_float(q.t))
                    for n in q.tensors:
                        n = self._convert_tensor_float16_to_float(n)
                if isinstance(q, onnx_proto.GraphProto):
                    for n in q.initializer:
                        if n.data_type == onnx_proto.TensorProto.FLOAT16:
                            n = self._convert_tensor_float16_to_float(n)
                            value_info_list.append(self._make_value_info_from_tensor(n))
                    for n in itertools.chain(q.input, q.output, q.value_info):
                        if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT16:
                            if n.name not in graph_io_to_skip:
                                n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                                value_info_list.append(n)
            queue = next_level

        # Handle Cast insertion for blocked nodes
        for node in node_list:
            for i in range(len(node.input)):
                input_name = node.input[i]
                for value_info in value_info_list:
                    if input_name == value_info.name:
                        graph = node_dict[node.name]
                        new_value_info = graph.value_info.add()
                        new_value_info.CopyFrom(value_info)
                        output_name = f"{node.name}_input_cast_{i}"
                        new_value_info.name = output_name
                        new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                        node_name = f"{node.name}_input_cast{i}"
                        new_node = [helper.make_node("Cast", [input_name], [output_name], to=10, name=node_name)]
                        graph.node.extend(new_node)
                        node.input[i] = output_name
                        break
            for i in range(len(node.output)):
                output_name = node.output[i]
                for value_info in value_info_list:
                    if output_name == value_info.name:
                        graph = node_dict[node.name]
                        new_value_info = graph.value_info.add()
                        new_value_info.CopyFrom(value_info)
                        input_name = f"{node.name}_output_cast_{i}"
                        new_value_info.name = input_name
                        new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                        node_name = f"{node.name}_output_cast{i}"
                        new_node = [helper.make_node("Cast", [input_name], [output_name], to=1, name=node_name)]
                        graph.node.extend(new_node)
                        node.output[i] = input_name
                        break

        self._sort_topology(model.graph)
        return model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """
        Execute the FP16 to FP32 conversion pass based on the provided configuration.

        Args:
            model (ModelProto): ONNX model to convert.
            config (dict[str, PassConfigParam]): Configuration parameters.

        Returns:
            ModelProto: Converted ONNX model.
        """
        if "convert_fp16_to_fp32" in config and config["convert_fp16_to_fp32"]:
            converted_model = self._onnx_convert_fp16_to_fp32(model)
        else:
            logger.warning(
                "Please ensure that the onnx_convert_fp16_to_fp32 pass contains the convert_fp16_to_fp32 parameter and it is True."
            )
            converted_model = model
        return converted_model
