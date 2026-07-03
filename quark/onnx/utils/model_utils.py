#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# pylint: disable=g-explicit-length-test
"""Utility functions."""

import copy
import enum
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
from google.protobuf import text_format
from onnx import ModelProto, NodeProto, TensorProto, TensorShapeProto, ValueInfoProto
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import add_infer_metadata

from quark.common.profiler import ProfileStep, profile_scope
from quark.common.utils.log import ScreenLogger, log_errors
from quark.onnx.operators.custom_ops import get_library_path
from quark.onnx.quantization.quant_utils import (
    DEQUANT_OP_TYPES,
    FN_OP_TYPES,
    QUANT_OP_TYPES,
    load_model_with_shape_infer,
)
from quark.onnx.utils.system_utils import create_tmp_dir

logger = ScreenLogger(__name__)

global USER_CUSTOM_OP_LIB_PATHS
USER_CUSTOM_OP_LIB_PATHS: list[str] = []


def get_tensor_value(initializer: TensorProto) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Convert TensorProto to numpy array."""
    return onnx.numpy_helper.to_array(initializer)


def generate_initializer(tensor_array: np.ndarray[Any, np.dtype[np.float32]], dtype: Any, name: str) -> TensorProto:
    """Generate initializers from numpy array."""
    tensor = tensor_array.astype(dtype)
    init = onnx.numpy_helper.from_array(tensor, name)
    return init


class SharedNodesHelper:
    class NodeType(enum.Enum):
        NODE = 1
        INITIALIZER = 2
        INPUT = 3

    @staticmethod
    def _node_type(node: NodeProto | TensorProto | ValueInfoProto) -> NodeType:
        """Returns whether the node is a node or initializer."""
        if isinstance(node, onnx.NodeProto):
            return SharedNodesHelper.NodeType.NODE
        elif isinstance(node, onnx.TensorProto):
            return SharedNodesHelper.NodeType.INITIALIZER
        elif isinstance(node, onnx.ValueInfoProto):
            return SharedNodesHelper.NodeType.INPUT
        else:
            raise ValueError(f"Unknown node type for node: {node}")

    @staticmethod
    def _add_node_init(model: ModelProto, node_init_to_add: NodeProto | TensorProto | ValueInfoProto) -> None:
        """Add the node or initializer/input to the model."""
        if SharedNodesHelper._node_type(node_init_to_add) == SharedNodesHelper.NodeType.NODE:
            for node in model.graph.node:
                if node.name == node_init_to_add.name:
                    logger.info(f"Node `{node.name}` is already in model, skip adding it.")
                    return
            new_node = model.graph.node.add()
            new_node.CopyFrom(node_init_to_add)
        elif SharedNodesHelper._node_type(node_init_to_add) == SharedNodesHelper.NodeType.INITIALIZER:
            for init in model.graph.initializer:
                if init.name == node_init_to_add.name:
                    logger.info(f"Initializer `{init.name}` is already in model, skip adding it.")
                    return
            new_init = model.graph.initializer.add()
            new_init.CopyFrom(node_init_to_add)
        elif SharedNodesHelper._node_type(node_init_to_add) == SharedNodesHelper.NodeType.INPUT:
            for inp in model.graph.input:
                if inp.name == node_init_to_add.name:
                    logger.info(f"Input `{inp.name}` is already in model, skip adding it.")
                    return
            new_input = model.graph.input.add()
            new_input.CopyFrom(node_init_to_add)

    @staticmethod
    def _map_name_to_node(model: ModelProto) -> dict[str, NodeProto]:
        """Returns a dict of name to node.

        Returns:
            {node.name: node}
        """
        name_to_node_map = {}
        for node in model.graph.node:
            name_to_node_map[SharedNodesHelper._get_node_name(node)] = node
        return name_to_node_map

    @staticmethod
    def _map_name_to_init(model: ModelProto) -> dict[str, TensorProto]:
        """Returns a dict of name to initializer.

        Returns:
            {initializer.name: initializer}
        """
        name_to_init_map = {}
        for init in model.graph.initializer:
            name_to_init_map[init.name] = init
        return name_to_init_map

    @staticmethod
    def _map_name_to_input(model: ModelProto) -> dict[str, ValueInfoProto]:
        """Returns a dict of name to input.

        Returns:
            {initializer.name: initializer}
        """
        name_to_input_map = {}
        for inp in model.graph.input:
            name_to_input_map[inp.name] = inp
        return name_to_input_map

    @staticmethod
    def _map_tensor_to_producer(model: ModelProto) -> dict[str, NodeProto]:
        """Returns a dict of tensor to its producer node.

        Returns:
            {tensor.name: producer_node}
        """
        tensor_to_producer_map = {}
        for node in model.graph.node:
            for output_tensor in node.output:
                tensor_to_producer_map[output_tensor] = node

        for init in model.graph.initializer:
            tensor_to_producer_map[init.name] = init

        for inp in model.graph.input:
            tensor_to_producer_map[inp.name] = inp
        return tensor_to_producer_map

    @staticmethod
    def _map_tensor_to_consumer(model: ModelProto) -> dict[str, list[NodeProto]]:
        """Returns a dict of tensor to its consumer nodes.

        Returns:
            {tensor.name: [consumer_nodes]}
        """
        tensor_to_consumer_map = {}
        for node in model.graph.node:
            for input_tensor in node.input:
                if input_tensor not in tensor_to_consumer_map:
                    tensor_to_consumer_map[input_tensor] = [node]
                else:
                    tensor_to_consumer_map[input_tensor].append(node)
        return tensor_to_consumer_map

    @staticmethod
    def _get_node_name(node: NodeProto) -> str:
        return node.name


def copy_shared_nodes(model: ModelProto) -> ModelProto:
    helper = SharedNodesHelper()

    # Rename all nodes
    type_idx = {}
    for node in model.graph.node:
        if node.op_type not in type_idx:
            type_idx[node.op_type] = 1
        node.name = node.op_type + "_" + str(type_idx[node.op_type])
        logger.info("Add node name: ", node.name)
        type_idx[node.op_type] += 1

    modified_flag = True
    while modified_flag:
        modified_flag = False
        name_to_node_map: dict[str, onnx.NodeProto] = helper._map_name_to_node(model)
        tensor_to_producer_map: dict[str, NodeProto] = helper._map_tensor_to_producer(model)

        tensor_to_consumer_map = {}
        new_nodes = {}
        node_inputs_to_rename = []
        for node in model.graph.node:
            for inp_id, input_tensor in enumerate(node.input):
                if input_tensor not in tensor_to_consumer_map:
                    tensor_to_consumer_map[input_tensor] = [node]
                else:
                    producer = tensor_to_producer_map[input_tensor]
                    new_node = copy.deepcopy(producer)
                    idx = len(tensor_to_consumer_map[input_tensor])
                    new_node.name = new_node.name + "_" + str(idx)

                    if isinstance(producer, onnx.TensorProto):
                        new_nodes[new_node.name] = new_node
                        logger.info("Need to update init: ", node.name, inp_id, new_node.name)
                        node_inputs_to_rename.append((node.name, inp_id, new_node.name))
                        tensor_to_consumer_map[input_tensor].append(node)
                    elif isinstance(producer, onnx.NodeProto):
                        if producer.op_type in ["DequantizeLinear"]:
                            new_node.output[0] = new_node.name + "_out"
                            new_nodes[new_node.name] = new_node
                            logger.info("Need to update: ", node.name, inp_id, new_node.output[0])
                            node_inputs_to_rename.append((node.name, inp_id, new_node.output[0]))
                            tensor_to_consumer_map[input_tensor].append(node)
                    else:
                        pass

        for name, new_node in new_nodes.items():
            helper._add_node_init(model, new_node)

        name_to_node_map = helper._map_name_to_node(model)

        for node_name, inp_id, new_node_name in node_inputs_to_rename:
            modified_flag = True
            node = name_to_node_map[node_name]
            logger.info("Update node input", node_name, inp_id, new_node_name)
            node.input[inp_id] = new_node_name

    return model


def clean_initializer_in_input(model: ModelProto) -> ModelProto:
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


def get_shape_list(shape: TensorShapeProto) -> list[int | str]:
    shape_list = []
    for d in shape.dim:
        if d.HasField("dim_value"):
            shape_list.append(d.dim_value)
        elif d.HasField("dim_param"):
            shape_list.append(d.dim_param)
        else:
            shape_list.append("?")
    return shape_list


def convert_nchw_to_nhwc(model: ModelProto) -> Any:
    temp_model = clean_initializer_in_input(model)
    onnx_model = ONNXModel(temp_model)

    node_name_list = []
    for node in onnx_model.graph().node:
        node_name_list.append(node.name)

    for inp in onnx_model.graph().input:
        shape_list = get_shape_list(inp.type.tensor_type.shape)

        if len(shape_list) != 4:
            logger.warning(f"Expected 4-dimension input shape but got {shape_list}, skip the nchw to nhwc conversion.")
            continue

        C, H, W = shape_list[1:]
        if not all(isinstance(_, int) for _ in [C, H, W]):
            logger.warning(f"Expected integer input shape but got [{C}, {H}, {W}], skip the nchw to nhwc conversion.")
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
        shape_list = get_shape_list(out.type.tensor_type.shape)

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


class ONNXQuantizedModel:
    """
    This class is used to gather all Q/DQ nodes of a target node's inputs and outputs.
    It is applicable to sub-models extracted from quantized models.

    :param onnx.ModelProto model: the quantized sub-model.
    """

    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        self.onnx_model = ONNXModel(model)

        self.in_name_to_nodes = self.onnx_model.input_name_to_nodes()
        self.out_name_to_node = self.onnx_model.output_name_to_node()

    def _find_node_input_qdq(self, node: NodeProto, tensor_name: str) -> tuple[NodeProto | None, NodeProto | None]:
        """
        Find qdq nodes on input tensor. Note that dq always exists, but q sometimes is folded.

        :param onnx.NodeProto node: The target node.
        :param str tensor_name: Name of the tensor, which should be one of the target node's inputs.

        :return: The DQ and Q nodes quantizing the tensor.
        """
        if tensor_name not in self.out_name_to_node:
            logger.debug(f"input {tensor_name} of {node.name} came from initializer")
            return None, None

        dq_candidate = self.out_name_to_node[tensor_name]
        if dq_candidate.op_type not in DEQUANT_OP_TYPES:
            logger.debug(f"input {tensor_name} of {node.name} was not quantized")
            return None, None
        elif dq_candidate.input[0] not in self.out_name_to_node:
            logger.debug(f"input {tensor_name} of {node.name} has a folded Q")
            return dq_candidate, None

        q_candidate = self.out_name_to_node[dq_candidate.input[0]]
        if q_candidate.op_type not in QUANT_OP_TYPES:
            logger.warning(f"input {tensor_name} of {node.name} lost a Q")
            return dq_candidate, None

        return dq_candidate, q_candidate  # Note that DQ came first

    def _find_node_output_qdq(self, node: NodeProto, tensor_name: str) -> tuple[NodeProto | None, NodeProto | None]:
        """
        Find qdq nodes on output tensor.

        :param onnx.NodeProto node: The target node.
        :param str tensor_name: Name of the tensor, which should be one of the target node's outputs.

        :return: The DQ and Q nodes quantizing the tensor.
        """
        if tensor_name not in self.in_name_to_nodes:
            logger.debug(f"output {tensor_name} of {node.name} was a isolate node")
            return None, None

        # this assertion maybe uncessary, in some special cases
        assert len(self.in_name_to_nodes[tensor_name]) == 1

        q_candidate = self.in_name_to_nodes[tensor_name][0]
        if q_candidate.op_type not in QUANT_OP_TYPES:
            logger.debug(f"output {tensor_name} of {node.name} was not quantized")
            return None, None
        elif q_candidate.output[0] not in self.in_name_to_nodes:
            logger.debug(f"input {tensor_name} of {node.name} lost a DQ")
            return q_candidate, None

        dq_candidate = self.in_name_to_nodes[q_candidate.output[0]][0]
        if dq_candidate.op_type not in DEQUANT_OP_TYPES:
            logger.warning(f"input {tensor_name} of {node.name} lost a DQ")
            return q_candidate, None

        return q_candidate, dq_candidate  # Note that Q came first

    def find_target_op_type_qdqs(self, target_op_type: list[str]) -> dict[str, Any]:
        """
        Get the qdqs on all inputs and outputs of the target node,
        which is the first node with a target op type.

        :param list[str] target_op_type: The target op types.

        :return: The extracted structure containing input and output QDQs.
        """
        node_struct: dict[str, Any] = {"node": None, "input_qdqs": [], "output_qdqs": []}

        for node in self.model.graph.node:
            if node.op_type in target_op_type:
                node_struct["node"] = node

                input_qdqs = []  # This contains weight/bias qdqs
                for tensor_name in node.input:
                    dq, q = self._find_node_input_qdq(node, tensor_name)
                    input_qdqs.append((dq, q))
                node_struct["input_qdqs"] = input_qdqs

                output_qdqs = []
                for tensor_name in node.output:
                    q, dq = self._find_node_output_qdq(node, tensor_name)
                    output_qdqs.append((dq, q))
                node_struct["output_qdqs"] = output_qdqs

                break  # Note that only the first node of specified op type

        return node_struct

    def find_target_node_qdqs(self, target_node: NodeProto) -> dict[str, Any]:
        """
        Get the qdqs on all inputs and outputs of the target node.

        :param NodeProto target_node: The target node.

        :return: The extracted structure containing input and output QDQs.
        """
        node_struct: dict[str, Any] = {
            "node": None,
            "input_qdqs": [],
            "output_qdqs": [],
        }

        for node in self.model.graph.node:
            if node is target_node:
                node_struct["node"] = node

                input_qdqs = []  # This contains weight/bias qdqs
                for tensor_name in node.input:
                    dq, q = self._find_node_input_qdq(node, tensor_name)
                    input_qdqs.append((dq, q))
                node_struct["input_qdqs"] = input_qdqs

                output_qdqs = []
                for tensor_name in node.output:
                    q, dq = self._find_node_output_qdq(node, tensor_name)
                    output_qdqs.append((dq, q))
                node_struct["output_qdqs"] = output_qdqs

                break  # Got the target node and break

        return node_struct

    def _find_node_input_fn(self, node: NodeProto, tensor_name: str) -> NodeProto | None:
        """
        Find the quantization node on the input tensor.

        :param onnx.NodeProto node: The target node.
        :param str tensor_name: Name of the tensor, which should be one of the target node's inputs.

        :return: The node that is quantizing the tensor.
        """
        if tensor_name in self.out_name_to_node and self.out_name_to_node[tensor_name].op_type in FN_OP_TYPES:
            return self.out_name_to_node[tensor_name]

        return None

    def _find_node_output_fn(self, node: NodeProto, tensor_name: str) -> NodeProto | None:
        """
        Find the quantization node on the output tensor.

        :param onnx.NodeProto node: The target node.
        :param str tensor_name: Name of the tensor, which should be one of the target node's outputs.

        :return: The node that is quantizing the tensor.
        """
        if tensor_name in self.in_name_to_nodes and self.in_name_to_nodes[tensor_name][0].op_type in FN_OP_TYPES:
            return self.in_name_to_nodes[tensor_name][0]

        return None

    def find_target_node_fns(self, target_node: NodeProto) -> dict[str, Any]:
        """
        Get the qdqs on all inputs and outputs of the target node.

        :param NodeProto target_node: The target node.

        :return: The extracted structure containing input and output quantization nodes.
        """
        node_struct: dict[str, Any] = {
            "node": None,
            "input_qdqs": [],
            "output_qdqs": [],
        }

        for node in self.model.graph.node:
            if node is target_node:
                node_struct["node"] = node

                input_qdqs = []  # This contains weight/bias qdqs
                for tensor_name in node.input:
                    fn = self._find_node_input_fn(node, tensor_name)
                    input_qdqs.append((fn,))  # Be consistent with QDQ
                node_struct["input_qdqs"] = input_qdqs

                output_qdqs = []
                for tensor_name in node.output:
                    fn = self._find_node_output_fn(node, tensor_name)
                    output_qdqs.append((fn,))  # Single output may have dedicate quant nodes
                node_struct["output_qdqs"] = output_qdqs

                break  # Got the target node and break

        return node_struct


@log_errors
def save_model(model: ModelProto, path: str, as_text: bool = False) -> None:
    """Save onnx model to disk."""
    if as_text:
        with open(path, "w") as f:
            f.write(text_format.MessageToString(model))
    else:
        onnx.save(model, path)


@log_errors
def run_onnx_model(model_input: str | Path | onnx.ModelProto, data_reader: Any) -> None:
    """
    Check if the input ONNX can run successfully
    :param model_input: the model path or a ModelProto
    :param data_reader: the data reader for feeding data
    """
    try:
        sess = create_infer_session_for_onnx_model(model_input)
        inputs = data_reader.get_next()
        output = sess.run(None, inputs)
        if output:
            logger.info("The input ONNX model can run inference successfully")
        else:
            logger.warning("Fail to run inference, please check the input model and the 'calibration_data_reader'.")
    except Exception as e:
        raise ValueError(
            f"Fail to run inference. Exception: {e}. Please check the input model and the 'calibration_data_reader'."
        ) from e


@log_errors
def check_onnx_model(model_input: str | Path | onnx.ModelProto) -> None:
    """
    Check if the input ONNX can create InferenceSession successfully
    :param model_input: the model path or a ModelProto
    """
    try:
        create_infer_session_for_onnx_model(model_input)
        logger.info("The input ONNX model can create InferenceSession successfully")

    except Exception as e:
        raise ValueError(f"Fail to create InferenceSession. Exception: {e}. Please check the model.") from e


def encrypt_data(unencrypted_data: bytes, iv: bytes, key: bytes) -> Any:
    """
    Encrypt data using AES-256 algorithm.
    :param unencrypted_data: the original data to be encrypted
    :param iv: initialization vector, 16 bytes
    :param key: the key, 32 bytes (256 bits)
    :return: the encrypted data
    """
    from cryptography.hazmat.backends import default_backend  # type: ignore
    from cryptography.hazmat.primitives import padding  # type: ignore
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # type: ignore

    # Apply PKCS7 padding
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(unencrypted_data) + padder.finalize()

    # Encrypt using AES-256-CBC
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()

    return iv + ciphertext  # Store or transmit iv securely alongside the encrypted content


def decrypt_data(encrypted_data: bytes, iv: bytes, key: bytes) -> Any:
    """
    Decrypt data using AES-256 algorithm.
    :param encrypted_data: the data to be decrypted
    :param iv: initialization vector, 16 bytes
    :param key: the key, 32 bytes (256 bits)
    :return: the decrypted data
    """
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    assert iv == encrypted_data[:16]
    ciphertext = encrypted_data[16:]

    # Decrypt using AES-256-CBC
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded_data = decryptor.update(ciphertext) + decryptor.finalize()

    # Remove PKCS7 padding
    unpadder = padding.PKCS7(128).unpadder()
    decrypted_data = unpadder.update(decrypted_padded_data) + unpadder.finalize()

    return decrypted_data


def onnx_save_model_with_encryption(model: ModelProto, path: str | Path, secret_key: bytes) -> None:
    """
    Encrypt model before saving to disk. Only supports <2GB models
    :param model: the onnx ModelProto to be decrypted
    :param path: the path for the saving
    :param secret_key: 48 bytes secret key, 16 bytes for iv and 32 bytes as key
    """
    assert len(secret_key) == 48 and "This is an invalid secret key"

    model_bytes = model.SerializeToString()

    assert isinstance(secret_key, bytes)
    encrypted_data = encrypt_data(model_bytes, secret_key[:16], secret_key[16:])

    with open(path, "wb") as f:
        f.write(encrypted_data)


def onnx_load_model_with_decryption(path: str | Path, secret_key: bytes) -> ModelProto:
    """
    Decrypt model before loading to memory. Only supports <2GB models
    :param path: the model path
    :param secret_key: 48 bytes secret key, 16 bytes for iv and 32 bytes as key
    :return the loaded and decrypted model
    """
    assert len(secret_key) == 48 and "This is an invalid secret key"

    with open(path, "rb") as f:
        encrypted_data = f.read()

    if encrypted_data[:16] != secret_key[:16]:  # Was not encrypted
        try:
            return onnx.load(path)
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Failed to load an unknown model file {path}") from e

    assert isinstance(secret_key, bytes)
    decrypted_data = decrypt_data(encrypted_data, secret_key[:16], secret_key[16:])

    model = ModelProto()
    model.ParseFromString(decrypted_data)
    return model


@profile_scope(ProfileStep.MODEL_CACHING)
def cache_onnx_model_and_infer_shapes(
    input_model: str | Path | ModelProto,
    path: str | Path,
    save_as_external_data: bool = False,
    encrypt_algo: str | None = None,
    secret_key: bytes | None = None,
) -> ModelProto:
    """
    Save the model and then load it with shape infer and cryption if secret key provided
    :param model: the onnx model path or ModelProto to be saved
    :param path: the path for the saving
    :param save_as_external_data: save external data for the models >2GB
    :param secret_key: 48 bytes secret key, 16 bytes for iv and 32 bytes as key
    :return the model proto
    """
    model = input_model if isinstance(input_model, onnx.ModelProto) else onnx.load(input_model)

    if secret_key is not None and len(secret_key) == 48:
        if encrypt_algo is not None and encrypt_algo == "AES-256":
            # TODO: support more algorithms
            onnx_save_model_with_encryption(model, path, secret_key)
            reloaded_model = onnx_load_model_with_decryption(path, secret_key)
        else:
            # If no encryption algorithm, just copying in memory
            assert save_as_external_data is False
            reloaded_model = copy.deepcopy(model)

        add_infer_metadata(reloaded_model)  # It's a crucial step
        return onnx.shape_inference.infer_shapes(reloaded_model)

    save_onnx_model_with_external_data(model, path, save_as_external_data=save_as_external_data)
    return load_model_with_shape_infer(Path(path))  # type: ignore


def save_onnx_model_with_external_data(
    model: ModelProto, path: str | Path, save_as_external_data: bool = False
) -> None:
    """
    Save model to external data, the .data has same name as .onnx
    :param model: the onnx ModelProto to be saved
    :param path: the path for the saving
    :param save_as_external_data: this option is for >2GB ModelProto
    """
    if save_as_external_data:
        directory = Path(path).parent  # This is the directory to save the model
        location = Path(path).name + ".data"  # Must be a relative path (to the model path)

        # To avoid appending due to a duplicate name, remove it in advance
        data_file_path = Path(directory).joinpath(location).as_posix()
        if os.path.exists(data_file_path):
            os.remove(data_file_path)

        onnx.external_data_helper.convert_model_to_external_data(
            model, all_tensors_to_one_file=True, location=location, convert_attribute=True
        )
    onnx.save(model, path)


def update_user_custom_op_lib_paths(lib_paths: list[str] | str | None) -> None:
    if lib_paths is not None:
        if isinstance(lib_paths, str):
            USER_CUSTOM_OP_LIB_PATHS.append(lib_paths)
        elif isinstance(lib_paths, list):
            for lib_path in lib_paths:
                if isinstance(lib_path, str):
                    USER_CUSTOM_OP_LIB_PATHS.append(lib_path)
                else:
                    logger.warning(f"The {lib_path} is not a string, will be skipped!")
        else:
            logger.warning(
                "The custom library paths defined by the user are neither a string nor a list. No custom ops library will be registered!"
            )


def create_infer_session_for_onnx_model(
    model_input: str | Path | ModelProto,
    sess_options: onnxruntime.SessionOptions | None = None,
    providers: list[str] | None = ["CPUExecutionProvider"],
    provider_options: list[dict[str, str]] | None = None,
    use_external_data_format: bool = False,
    **kwargs: dict[str, Any],
) -> onnxruntime.InferenceSession:
    """
    Create an Inference Session for onnx model
    :param model_input: the onnx model, can be a path or ModelProto
    :param session_options: session options
    """

    if USER_CUSTOM_OP_LIB_PATHS != []:
        if sess_options is None:
            sess_options = onnxruntime.SessionOptions()
        for lib_path in USER_CUSTOM_OP_LIB_PATHS:
            sess_options.register_custom_ops_library(lib_path)

    def create_inference_session(model: str | Path | ModelProto) -> onnxruntime.InferenceSession:
        try:
            return onnxruntime.InferenceSession(
                model, sess_options=sess_options, providers=providers, provider_options=provider_options, **kwargs
            )
        except onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException as e:
            raise RuntimeError(f"Failed to create inference session, likely cannot allocate memory: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to create inference session, due to an unexpected error: {e}") from e

    if isinstance(model_input, onnx.ModelProto) and (
        use_external_data_format or model_input.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF
    ):
        with create_tmp_dir(prefix="quark_onnx.utils.") as temp_dir:
            temp_path = Path(temp_dir).joinpath("infer_model.onnx").as_posix()
            save_onnx_model_with_external_data(copy.deepcopy(model_input), temp_path, True)
            session = create_inference_session(temp_path)
    else:
        model = model_input.SerializeToString() if isinstance(model_input, onnx.ModelProto) else model_input
        session = create_inference_session(model)

    return session


def collect_tensor_shapes_from_feed(
    model: onnx.ModelProto,
    feed_dict: dict[str, np.ndarray[Any, Any]],
) -> dict[str, tuple[int, tuple[int, ...]]]:
    """Collect shape and dtype for every non-input intermediate tensor in *model*.

    The model is mutated in-place during the call (intermediate tensors are
    temporarily appended as graph outputs) and restored before returning.

    :param model: ONNX model whose intermediate tensor shapes should be inferred.
        The graph is mutated in-place and restored after the ORT run.
    :param feed_dict: One sample dict mapping graph input names to numpy arrays.
    :return: Mapping ``tensor_name -> (onnx_elem_type_int, shape_tuple)``.
        Returns an empty dict if the ORT session cannot be created or run.
    """
    graph_input_names: set[str] = {inp.name for inp in model.graph.input} | {
        init.name for init in model.graph.initializer
    }

    # Collect unique non-empty intermediate tensor names: all node outputs plus
    # node inputs (to catch any tensor consumed but not produced by a graph node),
    # excluding graph inputs and initializers.
    seen: set[str] = set()
    output_list: list[str] = []
    for node in model.graph.node:
        for name in node.output:
            if name and name not in graph_input_names and name not in seen:
                seen.add(name)
                output_list.append(name)
        for name in node.input:
            if name and name not in graph_input_names and name not in seen:
                seen.add(name)
                output_list.append(name)

    if not output_list:
        return {}

    # Append tensors as temporary graph outputs (in-place; restored below).
    original_output_len = len(model.graph.output)
    for name in output_list:
        model.graph.output.append(onnx.ValueInfoProto(name=name))

    result_map: dict[str, tuple[int, tuple[int, ...]]] = {}
    try:
        session = create_infer_session_for_onnx_model(model)
        results: list[np.ndarray[Any, Any]] = session.run(output_list, feed_dict)
        for name, arr in zip(output_list, results, strict=True):
            if arr is None:
                continue
            try:
                elem_type: int = onnx.helper.np_dtype_to_tensor_dtype(arr.dtype)
            except Exception:
                continue
            result_map[name] = (elem_type, tuple(int(d) for d in arr.shape))
    except Exception as e:
        logger.warning(
            f"collect_tensor_shapes_from_feed: ORT inference failed, value_info will not be populated. Reason: {e}"
        )
    finally:
        # Always restore the model to its original output list.
        del model.graph.output[original_output_len:]

    return result_map


def fill_all_tensors_value_info(
    model: onnx.ModelProto,
    data_reader: "CalibrationDataReader | None",
) -> onnx.ModelProto:
    """Populate and synchronise ``value_info`` entries for every intermediate tensor.

    Runs one ORT inference pass with a single sample from *data_reader* and
    records the concrete shape and dtype of every intermediate tensor.

    - Tensors in ``graph.input``, ``graph.output``, or ``graph.initializer``
      are skipped (they carry type information separately).
    - Tensors already present in ``graph.value_info`` are checked for
      consistency with the observed shape and dtype.  If inconsistent, a
      warning is logged and the entry is updated to match ORT's observation.
    - Tensors missing from ``graph.value_info`` have a new entry added.

    :param onnx.ModelProto model: The model to be updated in-place.
    :param data_reader: Data reader with a ``get_next()`` method that returns a
        ``dict[str, np.ndarray]`` sample.  If ``None`` or ``get_next()`` returns
        ``None``, a warning is logged and the model is returned unchanged.

    :return: The model with updated ``graph.value_info``.
    """
    if data_reader is None:
        logger.warning("fill_all_tensors_value_info: data_reader is None; skipping value_info population.")
        return model

    # Reset then fetch exactly one sample.
    if hasattr(data_reader, "reset_iter"):
        data_reader.reset_iter()
    feed_dict = data_reader.get_next()
    if hasattr(data_reader, "reset_iter"):
        data_reader.reset_iter()

    if feed_dict is None:
        logger.warning("fill_all_tensors_value_info: data_reader returned no sample; skipping value_info population.")
        return model

    # Tensors whose type is already described outside graph.value_info.
    skip_names: set[str] = (
        {inp.name for inp in model.graph.input}
        | {out.name for out in model.graph.output}
        | {init.name for init in model.graph.initializer}
    )

    # Run ORT to collect concrete shapes and dtypes.
    shape_map = collect_tensor_shapes_from_feed(model, feed_dict)

    if not shape_map:
        logger.warning(
            "fill_all_tensors_value_info: ORT inference returned no tensor shapes; value_info will not be populated."
        )
        return model

    # Fix dynamic dims on graph inputs using the feed_dict arrays directly.
    for inp in model.graph.input:
        if inp.name in feed_dict:
            arr = feed_dict[inp.name]
            tt = inp.type.tensor_type
            if tt.HasField("shape") and any(d.HasField("dim_param") for d in tt.shape.dim):
                for i, d in enumerate(tt.shape.dim):
                    if d.HasField("dim_param"):
                        d.ClearField("dim_param")
                        d.dim_value = int(arr.shape[i])

    # Fix dynamic dims on graph outputs using shapes already collected in shape_map.
    for out in model.graph.output:
        tt = out.type.tensor_type
        if tt.HasField("shape") and any(d.HasField("dim_param") for d in tt.shape.dim):
            if out.name in shape_map:
                _, shape_tuple = shape_map[out.name]
                for i, d in enumerate(tt.shape.dim):
                    if d.HasField("dim_param"):
                        d.ClearField("dim_param")
                        d.dim_value = int(shape_tuple[i])

    # Build a mutable lookup of existing value_info entries.
    existing_vi_map: dict[str, onnx.ValueInfoProto] = {vi.name: vi for vi in model.graph.value_info}

    new_entries: list[onnx.ValueInfoProto] = []
    updated = 0
    for tensor_name, (elem_type, shape_tuple) in shape_map.items():
        if tensor_name in skip_names:
            continue
        try:
            new_vi = onnx.helper.make_tensor_value_info(tensor_name, elem_type, list(shape_tuple))
        except Exception as e:
            logger.warning(f"fill_all_tensors_value_info: could not create value_info for '{tensor_name}': {e}")
            continue

        if tensor_name in existing_vi_map:
            old_vi = existing_vi_map[tensor_name]
            old_tt = old_vi.type.tensor_type
            new_tt = new_vi.type.tensor_type
            # Check dtype and shape consistency against new_tt.
            # Any dim_param (symbolic/dynamic axis) in the existing entry is treated as a
            # mismatch: we always replace it with the concrete value observed by ORT.
            if old_tt.HasField("shape") and new_tt.HasField("shape"):
                has_dynamic = any(d.HasField("dim_param") for d in old_tt.shape.dim)
                old_shape = tuple(d.dim_value if not d.HasField("dim_param") else None for d in old_tt.shape.dim)
                new_shape = tuple(d.dim_value for d in new_tt.shape.dim)
                shapes_match = (
                    not has_dynamic
                    and old_tt.elem_type == new_tt.elem_type
                    and len(old_shape) == len(new_shape)
                    and all(o == n for o, n in zip(old_shape, new_shape, strict=False))
                )
            else:
                old_shape = None
                new_shape = shape_tuple
                shapes_match = False
            if not shapes_match:
                logger.warning(
                    f"fill_all_tensors_value_info: inconsistent value_info for '{tensor_name}': "
                    f"existing (dtype={old_tt.elem_type}, shape={old_shape}) != "
                    f"observed (dtype={new_tt.elem_type}, shape={new_shape}). Updating."
                )
                old_vi.CopyFrom(new_vi)
                updated += 1
        else:
            new_entries.append(new_vi)

    if new_entries:
        model.graph.value_info.extend(new_entries)
    logger.info(f"Filled value info for {len(new_entries)} tensor(s), updated {updated} inconsistent tensor(s).")

    return model


def register_custom_ops_library(session_options: onnxruntime.SessionOptions, device: str = "CPU") -> None:
    # ``_initialize_kernels`` runs at module import; a silent build failure surfaces here.
    try:
        session_options.register_custom_ops_library(get_library_path(device))
    except Exception as e:
        logger.warning(
            f"Failed to register custom op library {get_library_path(device)} to ORT with {e},"
            "please check if the library has been compiled successfully."
        )


def sanitize_model_outputs(outputs: list[np.ndarray[Any, Any]]) -> None:
    """
    Sanitize model outputs by replacing non-finite values (NaN / Inf).

    For each output tensor:
    - If all values are non-finite (NaN or Inf), all elements are replaced with 0.0.
    - If the tensor contains both finite and non-finite values:
        - NaN values are replaced with 0.0
        - +Inf values are replaced with the maximum finite value in the tensor
        - -Inf values are replaced with the minimum finite value in the tensor

    The replacement is done in-place. A warning is logged if any non-finite
    values are detected and replaced.

    :param outputs: List of numpy arrays to be sanitized
    """
    has_nan_or_inf = False
    for i, output in enumerate(outputs):
        if not np.isfinite(output).all():
            if np.isfinite(output).any():
                outputs[i] = np.nan_to_num(output, nan=0.0, posinf=np.nanmax(output), neginf=np.nanmin(output))
            else:
                outputs[i] = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
            has_nan_or_inf = True
    if has_nan_or_inf:
        logger.warning(
            "Non-finite values (NaN or Inf) were detected and replaced in model outputs. "
            "NaN values are replaced with 0.0, +Inf with the maximum finite value, and -Inf with the minimum finite value. "
            "This may affect quantization accuracy. Please verify the input model and the 'calibration_data_reader'."
        )


def check_shared_initializers(onnx_model: onnx.ModelProto) -> bool:
    """
    Check whether the ONNX model contains shared initializers.

    :param onnx_model: the ONNX ModelProto to be analyzed
    :return: True if shared initializers exist, otherwise False
    """

    exist_shared_initializers: bool = False
    all_initializer_names = [item.name for item in onnx_model.graph.initializer]
    ini_used_static: dict[str, Any] = {}

    for i in range(len(onnx_model.graph.node)):
        inputs_name = onnx_model.graph.node[i].input
        for input_name in inputs_name:
            if input_name in all_initializer_names:
                if input_name in ini_used_static:
                    ini_used_static[input_name] += 1
                    exist_shared_initializers = True
                    break
                else:
                    ini_used_static[input_name] = 1

    if exist_shared_initializers:
        logger.warning(
            "Shared initializers detected in the model. "
            "Some initializers are referenced by multiple nodes, which may "
            "cause failures or incorrect results in quantization or optimization "
            "passes (e.g., Cross-Layer Equalization). "
            "It is recommended to duplicate these initializers so each node "
            "has its own copy (e.g., enable 'CopySharedInit' [] in extra_options)."
            "For more details, see: https://quark.docs.amd.com/latest/onnx/appendix_full_quant_config_features.html"
        )

    return exist_shared_initializers
