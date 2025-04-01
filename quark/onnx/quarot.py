#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import onnx
import numpy as np
from onnx import numpy_helper, helper, onnx_pb
from onnxruntime.quantization.onnx_model import ONNXModel
from typing import Dict, Tuple, Any
import os
from quark.shares.utils.log import ScreenLogger
import json
from typing import Optional

logger = ScreenLogger(__name__)


class QuaRot():
    """
    A class for quarot
    Args:
        onnx_model_path (str): The ONNX model path to be rotated.
        input_model (onnx.ModelProto): The ONNX model to be rotated.
        r_matrixs (Dict[str, np.ndarray]): The dict of rotation matrix
        rotation_config_info (Dict): The dict to define which sub-structure need rotation.
        is_large (bool): True if the model size is larger than 2GB.
    """

    def __init__(self,
                 onnx_model_path: str,
                 input_model: onnx.ModelProto,
                 r_matrixs: Dict[str, np.ndarray[Any, Any]],
                 rotation_config_info: Dict[Any, Any],
                 is_large: bool = True) -> None:
        self.onnx_model_path = onnx_model_path

        # parse rotation matrixs by each stage (R1/R2/R3/R4)
        self.r1_matrix = r_matrixs.get("R1", None)
        self.r2_matrix = r_matrixs.get("R2", None)
        self.r3_matrix = r_matrixs.get("R3", None)
        self.r4_matrix = r_matrixs.get("R4", None)

        self.base_dir = os.path.dirname(self.onnx_model_path)
        self.is_large = is_large
        self.rotation_config_info = rotation_config_info
        self.rotated_model_path = os.path.join(self.base_dir, "decoder_model_rotated.onnx")
        self.verbose = True

        self.model = input_model

        self.onnx_model = ONNXModel(self.model)

    def r1_pair_rotation(self) -> None:

        assert self.r1_matrix is not None

        # Start to process r1 rotations for each pair
        r1_pair_nodes_list = self.rotation_config_info.get("R1_pairs", [])
        for pair_idx, r1_pair in enumerate(r1_pair_nodes_list):
            # Step 0: Fetch corresponding nodes
            prev_nodes = [self.get_node_by_name(pre_node_name) for pre_node_name in r1_pair.get("prev_nodes", [])]
            next_nodes = [self.get_node_by_name(next_node_name) for next_node_name in r1_pair.get("next_nodes", [])]
            # Step 1: Preprocess norm layer. Erase scale tensor and bias tensor from norm layer.
            norm_node_name = r1_pair.get("norm_node", None)
            if norm_node_name is not None:
                norm_node = self.get_node_by_name(norm_node_name)
                if self.verbose:
                    logger.info(f"Folding {norm_node_name}")
                self.transform_rms_norm_and_linear(norm_node, next_nodes)
            # Step 2: Process previous nodes by positive transformation
            for node_idx, prev_node in enumerate(prev_nodes):
                # Rot weight data
                if pair_idx == 0 and node_idx == 0 and prev_node.op_type == "Gather":
                    # Specially exec for embed_node [32000, 4096]
                    if self.verbose:
                        logger.info(f"{prev_node.name} with in rot")
                    self.rotate_weight(prev_node, self.r1_matrix, dim="in", transpose=False)

                else:
                    if self.verbose:
                        logger.info(f"{prev_node.name} with out rot")
                    self.rotate_weight(prev_node,
                                       self.r1_matrix,
                                       dim="out",
                                       transpose=self.is_w_trans_needed(prev_node))
                    # Rot bias data if has
                    bias_tensor_proto, bias_tensor = self.get_bias_data_from_node(prev_node)
                    if bias_tensor is not None and bias_tensor_proto is not None:
                        if self.verbose:
                            logger.info(f"{prev_node.name} has bias and with out rot")
                        bias_tensor = self.rotate_out_channels(bias_tensor, self.r1_matrix, transpose=False)
                        bias_tensor_proto.CopyFrom(numpy_helper.from_array(bias_tensor, bias_tensor_proto.name))

            # Step 3: Process next nodes by inverse transformation
            for next_node in next_nodes:
                # Rot weight data
                if self.verbose:
                    logger.info(f"{next_node.name} with in rot")
                self.rotate_weight(next_node, self.r1_matrix, dim="in", transpose=self.is_w_trans_needed(next_node))

    def is_w_trans_needed(self, node: onnx.NodeProto) -> bool:
        trans_flag = True
        if node.op_type == "Gemm":
            # TODO Here do not consider Gemm's alpha
            trans_flag = (node.attribute[1].f != 1)  # Beta is considered.

        return trans_flag

    def remove_initializer(self, node: onnx.NodeProto, initializer: onnx.TensorProto) -> None:
        # remove initializer
        self.onnx_model.remove_initializer(initializer)

        # Remove the reference of the initializer
        for node in self.onnx_model.model.graph.node:
            node.input[:] = [input_name for input_name in node.input if input_name != initializer.name]

    def transform_rms_norm_and_linear(self, norm_node: onnx.NodeProto, next_nodes: list[onnx.NodeProto]) -> None:
        ln_w_tensor_proto, raw_ln_w = self.get_weight_data_from_node(norm_node)
        ln_w_f = raw_ln_w.astype(np.float64)
        fake_ln_w = np.ones_like(raw_ln_w)
        # Replace ori norm weight
        ln_w_tensor_proto.CopyFrom(numpy_helper.from_array(fake_ln_w, ln_w_tensor_proto.name))

        # Process norm bias
        ln_b_proto, raw_ln_b = self.get_bias_data_from_node(norm_node)  # raw_ln_b may be None
        if ln_b_proto is not None:
            if self.verbose:
                logger.info(f"{norm_node.name} has bias. Folding!")
            self.remove_initializer(norm_node, ln_b_proto)

        # Iterate each next_node and process
        for next_node in next_nodes:
            weight_tensor_proto, raw_weight_tensor = self.get_weight_data_from_node(next_node)
            dtype = raw_weight_tensor.dtype
            weight_tensor = raw_weight_tensor.astype(np.float64)
            # Weight tensor replacement
            trans_flag = self.is_w_trans_needed(next_node)
            if trans_flag:
                weight_tensor = weight_tensor.transpose(*range(weight_tensor.ndim - 2), -1,
                                                        -2)  # Whatever shape is, the last two dim will be transposed.
            weight_tensor = (weight_tensor * ln_w_f).astype(dtype)
            if trans_flag:
                weight_tensor = weight_tensor.transpose(*range(weight_tensor.ndim - 2), -1, -2)
            weight_tensor_proto.CopyFrom(numpy_helper.from_array(weight_tensor, weight_tensor_proto.name))

            # Process norm bias. Here Node will be replaced by a Gemm Node
            if raw_ln_b is not None:
                if trans_flag:
                    fused_ln_b = np.matmul(raw_ln_b.astype(np.float64), weight_tensor)
                else:
                    fused_ln_b = np.matmul(weight_tensor, raw_ln_b.astype(np.float64))

                if next_node.op_type == "MatMul":
                    # Build a Gemm Node and replace
                    new_b_name = next_node.name + ".bias"
                    new_b_tensor_proto = helper.make_tensor(name=new_b_name,
                                                            data_type=onnx_pb.TensorProto.FLOAT,
                                                            dims=fused_ln_b.shape,
                                                            vals=fused_ln_b.astype(dtype).flatten().tolist())

                    new_node = helper.make_node("Gemm",
                                                inputs=[next_node.input[0], next_node.input[1], new_b_name],
                                                outputs=[next_node.output[0]],
                                                name=next_node.name,
                                                alpha=1.0,
                                                beta=1.0,
                                                transA=0,
                                                transB=0)

                    self.onnx_model.remove_node(next_node)
                    self.onnx_model.add_node(new_node)
                    self.onnx_model.add_initializer(new_b_tensor_proto)

                elif next_node.op_type == "Gemm":
                    raw_b_tensor_proto, raw_b_tensor = self.get_bias_data_from_node(
                        next_node)  # raw_b_tensor may be None
                    # Just to update bias
                    if raw_b_tensor is None:
                        raw_b_tensor = np.zeros_like(fused_ln_b)
                        raw_b_name = next_node.name + ".bias"
                        raw_b_tensor_proto = helper.make_tensor(name=raw_b_name,
                                                                data_type=onnx_pb.TensorProto.FLOAT,
                                                                dims=raw_b_tensor.shape,
                                                                vals=raw_b_tensor.flatten().tolist())

                        self.onnx_model.add_initializer(raw_b_tensor_proto)
                        next_node.input.append(raw_b_tensor_proto.name)

                    assert raw_b_tensor_proto is not None
                    fused_ln_b = (fused_ln_b + raw_b_tensor).astype(dtype)
                    raw_b_tensor_proto.CopyFrom(numpy_helper.from_array(fused_ln_b, raw_b_tensor_proto.name))

    def rotate_weight(self,
                      node: onnx.NodeProto,
                      rotation_matrix: np.ndarray[Any, Any],
                      dim: str = "out",
                      transpose: bool = True) -> None:
        weight_tensor_proto, weight_tensor = self.get_weight_data_from_node(node)
        # NOTE Here we must transpose weight_tensor, due to diffrence between MatMul and nn.Linear
        # In nn.Linear, define computation: Y = X \times W^T + B
        # In MatMul node, define computation: Y = X \times W + B
        # Therefore the transpose flag is necessary.
        if dim == "out":
            weight_tensor = self.rotate_out_channels(weight_tensor, rotation_matrix, transpose)
        elif dim == "in":
            weight_tensor = self.rotate_in_channels(weight_tensor, rotation_matrix, transpose)
        # Update tensor
        weight_tensor_proto.CopyFrom(numpy_helper.from_array(weight_tensor, weight_tensor_proto.name))

    def rotate_in_channels(self, data_tensor: np.ndarray[Any, Any], rotation_matrix: np.ndarray[Any, Any],
                           transpose: bool) -> np.ndarray[Any, Any]:
        """Rotate the input channels of a weight matrix, i.e., inverse transformation to origin field"""
        if transpose:
            data_tensor = data_tensor.transpose(*range(data_tensor.ndim - 2), -1,
                                                -2)  # Whatever shape is, the last two dim will be transposed.

        dtype = data_tensor.dtype
        data_tensor = np.matmul(data_tensor.astype(np.float64), rotation_matrix).astype(dtype)

        if transpose:
            data_tensor = data_tensor.transpose(*range(data_tensor.ndim - 2), -1, -2)
        return data_tensor  # type: ignore

    def rotate_out_channels(self, data_tensor: np.ndarray[Any, Any], rotation_matrix: np.ndarray[Any, Any],
                            transpose: bool) -> np.ndarray[Any, Any]:
        """Rotate the output channels of a weight matrix, i.e., transformation to orthogonal field"""
        if transpose:
            data_tensor = data_tensor.transpose(*range(data_tensor.ndim - 2), -1, -2)

        dtype = data_tensor.dtype
        data_tensor = np.matmul(rotation_matrix.T, data_tensor.astype(np.float64)).astype(dtype)

        if transpose:
            data_tensor = data_tensor.transpose(*range(data_tensor.ndim - 2), -1, -2)
        return data_tensor  # type: ignore

    def get_weight_data_from_node(
            self, node: onnx.NodeProto) -> Tuple[onnx.TensorProto, np.ndarray[Any, np.dtype[np.float32]]]:
        for input_name in node.input:
            if 'weight' in input_name:
                return self.get_initializer_tensor(input_name)

        raise ValueError(
            f"Node {node.name} do not have a input which has a name include \"weight\"! Is node name {node.name} correct?"
        )

    def get_bias_data_from_node(
            self,
            node: onnx.NodeProto) -> Tuple[Optional[onnx.TensorProto], Optional[np.ndarray[Any, np.dtype[np.float32]]]]:
        for input_name in node.input:
            if 'bias' in input_name:
                return self.get_initializer_tensor(input_name)

        return None, None

    def get_node_by_name(self, node_name: str) -> onnx.NodeProto:
        for node in self.onnx_model.nodes():
            if node.name == node_name:
                return node  # type: ignore

        raise ValueError(f"Can not get the corresponding node! Is node name {node_name} correct?")

    def get_initializer_tensor(self, init_name: str) -> Tuple[onnx.TensorProto, np.ndarray[Any, np.dtype[np.float32]]]:
        tensor_proto = [init for init in self.onnx_model.model.graph.initializer if init.name == init_name][0]
        tensor = numpy_helper.to_array(tensor_proto, self.base_dir)
        return tensor_proto, tensor

    def transform(self) -> None:
        if self.r1_matrix is not None:
            logger.info("Conduct R1 rotation...")
            self.r1_pair_rotation()
        # TODO Implement R2/R3/R4 rotation

        # self.onnx_model.save_model_to_file(self.rotated_model_path,
        #                                    use_external_data_format=self.is_large)  # Not neccessary

    def get_rotated_model(self) -> onnx.ModelProto:
        return self.onnx_model.model  # type:ignore


def rotation_transforms(onnx_model_path: str, input_model: onnx.ModelProto, r_matrixs: Dict[str, np.ndarray[Any, Any]],
                        rotation_config_file: str) -> onnx.ModelProto:

    # Load rot_config
    with open(rotation_config_file, 'r') as file:
        rotation_config_info = json.load(file)
    processpr_ = QuaRot(onnx_model_path, input_model, r_matrixs, rotation_config_info)
    processpr_.transform()
    return processpr_.get_rotated_model()
