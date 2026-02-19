# Copyright (c) 2023 Zyy
# Modification Copyright (c) 2024 Advanced Micro Devices, Inc.
# Licensed under the MIT License.

from __future__ import annotations

import collections.abc
import contextlib
import copy
import hashlib
import itertools
import logging
import os
import re
import shutil
import sys
from collections.abc import Iterable, MutableSequence, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import ml_dtypes
import numpy as np
import numpy.typing as npt
import onnx
import onnx.external_data_helper
import onnx.onnx_cpp2py_export.checker as c_checker

import ryzenai_onnx_utils.proto as proto
from ryzenai_onnx_utils.typing import DtypeType, PassFunction, ReplaceParams, ShapeType, is_sequence_of, is_static_shape

from .lexer import Lexer

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import argparse


def configure_parser(subparser: argparse._SubParsersAction[Any]) -> argparse.ArgumentParser:
    matcher_parser: argparse.ArgumentParser = subparser.add_parser("match")
    matcher_parser.add_argument("input_path", type=Path, help="Path to input ONNX model")
    matcher_parser.add_argument("--strategy", type=Path, help="Strategy to run before extracting pattern")
    matcher_parser.add_argument(
        "--extract-inputs",
        type=str,
        nargs="*",
        help="Optionally extract from these inputs",
    )
    matcher_parser.add_argument(
        "--extract-outputs",
        type=str,
        nargs="*",
        help="Optionally extract from these outputs",
    )
    matcher_parser.add_argument(
        "--load-external-data",
        action="store_true",
        help="Load all external data at startup",
    )
    matcher_parser.add_argument(
        "--hints-key",
        default=None,
        nargs="*",
        help="Key to use for dynamic passes",
    )

    return matcher_parser


def input_name_from_node_name(node_name: str, suffix: str) -> str:
    input_name = node_name[1:].replace("/", ".") if node_name.startswith(("/", ".")) else node_name.replace("/", ".")
    input_name += f".{suffix}"
    return input_name


def find_nodes_by_input(input_name: str, graph: onnx.GraphProto, reverse: bool = False) -> list[onnx.NodeProto]:
    nodes = []
    for node in graph.node:
        if input_name in node.input:
            nodes.append(node)
    if reverse:
        nodes.reverse()
    return nodes


def is_used_input(input_name: str, graph: onnx.GraphProto) -> bool:
    for node in graph.node:
        if input_name in node.input:
            return True

        if node.op_type in {"Scan", "If", "Loop"}:
            for attr in node.attribute:
                subgraphs: list[onnx.GraphProto] = []
                if attr.type == onnx.AttributeProto.GRAPH:
                    subgraphs = [attr.g]
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    subgraphs = list(attr.graphs)
                for subgraph in subgraphs:
                    if input_name in [input_tvi.name for input_tvi in graph.input]:
                        return True
                    if is_used_input(input_name, subgraph):
                        return True
    return False


def is_used_output(output_name: str, graph: onnx.GraphProto) -> bool:
    for node in graph.node:
        if output_name in node.output:
            return True

        if node.op_type in {"Scan", "If", "Loop"}:
            for attr in node.attribute:
                subgraphs: list[onnx.GraphProto] = []
                if attr.type == onnx.AttributeProto.GRAPH:
                    subgraphs = [attr.g]
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    subgraphs = list(attr.graphs)
                for subgraph in subgraphs:
                    if output_name in [output_tvi.name for output_tvi in graph.output]:
                        return True
                    if is_used_output(output_name, subgraph):
                        return True
    return False


def is_used_initializer(input_name: str, graph: onnx.GraphProto) -> bool:
    if is_used_input(input_name, graph):
        return True

    if input_name in [o.name for o in graph.output]:
        return True

    for node in graph.node:
        if node.op_type in {"Scan", "If", "Loop"}:
            for attr in node.attribute:
                subgraphs: list[onnx.GraphProto] = []
                if attr.type == onnx.AttributeProto.GRAPH:
                    subgraphs = [attr.g]
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    subgraphs = list(attr.graphs)
                for subgraph in subgraphs:
                    if is_used_initializer(input_name, subgraph):
                        return True
    return False


def find_nodes_by_output(output_name: str, graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if output_name in node.output:
            nodes.append(node)
    return nodes


def build_input_map(
    graph_or_extractor: onnx.GraphProto | onnx.utils.Extractor,
    exclude_initializers: bool = True,
) -> dict[str, list[int]]:
    input_map: dict[str, list[int]] = {}
    if isinstance(graph_or_extractor, onnx.GraphProto):
        nodes = graph_or_extractor.node
    else:
        nodes = graph_or_extractor.graph.node
    for index, node in enumerate(nodes):
        for input_name in node.input:
            if not exclude_initializers or (
                exclude_initializers and not is_initializer(input_name, graph_or_extractor)
            ):
                if input_name not in input_map:
                    input_map[input_name] = []
                input_map[input_name].append(index)
    return input_map


def build_output_map(graph: onnx.GraphProto) -> dict[str, list[int]]:
    output_map: dict[str, list[int]] = {}
    for index, node in enumerate(graph.node):
        for output_name in node.output:
            if output_name not in output_map:
                output_map[output_name] = []
            output_map[output_name].append(index)
    return output_map


def find_consts(name: str, graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    nodes = []
    for node in graph.node:
        if name in node.output and node.op_type == "Constant":
            nodes.append(node)
    return nodes


def get_integer_const_by_value(
    value: int, extractor: onnx.utils.Extractor
) -> tuple[onnx.NodeProto | None, onnx.ValueInfoProto | None]:
    const_list = find_consts(f"const_{value}", extractor.graph)
    if const_list:
        assert len(const_list) == 1
        return None, None
    const = onnx.helper.make_node("Constant", inputs=[], outputs=[f"const_{value}"], value_ints=[value])
    tvi = onnx.helper.make_tensor_value_info(f"const_{value}", onnx.TensorProto.INT64, (1,))

    return const, tvi


def find_input(name: str, graph: onnx.GraphProto) -> onnx.ValueInfoProto | None:
    for input_tensor in graph.input:
        if input_tensor.name == name:
            return cast(onnx.ValueInfoProto, input_tensor)
    return None


def find_output(name: str, graph: onnx.GraphProto) -> onnx.ValueInfoProto | None:
    for output_tensor in graph.output:
        if output_tensor.name == name:
            return cast(onnx.ValueInfoProto, output_tensor)
    return None


def is_input_edge(name: str, graph: onnx.GraphProto) -> bool:
    return any(input_tensor.name == name for input_tensor in graph.input)


def is_output_edge(name: str, graph: onnx.GraphProto) -> bool:
    return any(output_tensor.name == name for output_tensor in graph.output)


def is_output_edge_or_adjacent(name: str, graph: onnx.GraphProto) -> bool:
    """
    Check if an edge is a direct output edge or feeds into one of a set of
    operators whose output is an output edge

    Args:
        name (str): Name of the edge
        graph (onnx.GraphProto): ONNX graph object
    """

    output_edges = [x.name for x in graph.output]

    child_nodes = find_nodes_by_input(name, graph)
    for node in child_nodes:
        if node.op_type in ["Cast", "CastAvx"] and is_output_edge(node.output[0], graph):
            output_edges.append(node.input[0])

    return any(name == x for x in output_edges)


def get_shapes(objs: Sequence[Any], extractor: onnx.utils.Extractor | None = None) -> list[tuple[int | str, ...]]:
    shapes = []
    for obj in objs:
        shapes.append(get_shape(obj, extractor))
    return shapes


def get_shape(obj: Any, extractor: onnx.utils.Extractor | None = None) -> tuple[int | str, ...]:
    if isinstance(obj, onnx.ValueInfoProto):
        if not obj.type.tensor_type.shape.dim:
            return (1,)
        shape: list[str | int] = []
        for d in obj.type.tensor_type.shape.dim:
            if d.dim_value < 1:
                shape.append(d.dim_param)
            else:
                shape.append(d.dim_value)
        return tuple(shape)
    if isinstance(obj, onnx.TensorProto):
        if not obj.dims:
            return (1,)
        return tuple(obj.dims)
    if isinstance(obj, str):
        assert extractor is not None
        if is_initializer(obj, extractor):
            obj = get_initializer(obj, extractor, False)
        elif obj in extractor.vimap:
            obj = extractor.vimap[obj]
        elif find_input(obj, extractor.graph) is not None:
            obj = find_input(obj, extractor.graph)
        elif find_output(obj, extractor.graph) is not None:
            obj = find_output(obj, extractor.graph)
        else:
            raise ValueError(f"Cannot interpret {obj} as valid input for get_shape()")
        return get_shape(obj)
    raise ValueError(f"Unsupported type: {type(obj)}")


def get_dtype(name: str, extractor: onnx.utils.Extractor) -> onnx.TensorProto.DataType | int:
    """Extract the dtype of a named tensor/weight

    Args:
        name (str): Name of the tensor
        extractor (onnx.utils.Extractor): The extractor is used to query the dtype

    Raises:
        ValueError: Raised if nothing is found

    Returns:
        int: The type as an integer defined in ONNX
    """
    if name in extractor.vimap:
        return extractor.vimap[name].type.tensor_type.elem_type  # type: ignore[no-any-return]
    if name in extractor.wmap:
        return extractor.wmap[name].data_type  # type: ignore[no-any-return]
    found_input = [x for x in extractor.graph.input if x.name == name]
    if len(found_input) == 1:
        return cast(int, found_input[0].type.tensor_type.elem_type)
    found_output = [x for x in extractor.graph.output if x.name == name]
    if len(found_output) == 1:
        return cast(int, found_output[0].type.tensor_type.elem_type)
    raise ValueError(f"{name} not found in graph")


def get_dtype_str(name: str, extractor: onnx.utils.Extractor) -> str:
    """Extract the dtype of a named weight/tensor as a string

    Args:
        name (str): Name of the tensor
        extractor (onnx.utils.Extractor): The extractor is used to query the dtype

    Returns:
        str: The string representation of the type
    """
    return onnx.helper.tensor_dtype_to_string(get_dtype(name, extractor))


def set_external_data(
    tensor: onnx.TensorProto,
    location: str,
    offset: int | None = None,
    length: int | None = None,
    checksum: str | None = None,
    basepath: str | None = None,
) -> None:
    """
    This is a copy of the same named function in onnx.external_data_helper.py
    but it doesn't require raw_data to be set. For memory reasons, tensors may
    not have loaded weights when this function is called.
    """

    del tensor.external_data[:]
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for k, v in {
        "location": location,
        "offset": int(offset) if offset is not None else None,
        "length": int(length) if length is not None else None,
        "checksum": checksum,
        "basepath": basepath,
    }.items():
        if v is not None:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)


def load_external_data_for_tensor(
    tensor: onnx.TensorProto, external_data_file_path: os.PathLike[Any], offset: int, length: int
) -> None:
    """Loads data from an external file for tensor.
    Ideally TensorProto should not hold any raw data but if it does it will be ignored.

    Arguments:
        tensor: a TensorProto object.
        base_dir: directory that contains the external data.
    """
    with open(external_data_file_path, "rb") as data_file:
        if offset:
            data_file.seek(offset)

        if length:
            tensor.raw_data = data_file.read(length)
        else:
            tensor.raw_data = data_file.read()


def parse_external_data(tensor: onnx.TensorProto) -> dict[str, int | str]:
    external_data = {}
    for item in tensor.external_data:
        value = int(item.value) if item.key in ["offset", "length"] else item.value
        external_data[item.key] = value
    return external_data


def get_external_data_for_tensor(tensor: onnx.TensorProto, base_dir: str) -> bytes:
    """Loads data from an external file for tensor.
    Ideally TensorProto should not hold any raw data but if it does it will be ignored.

    This is a copy of a similar function in onnx.external_data_helper but it just
    returns the data rather than setting the tensor's raw data

    Arguments:
        tensor: a TensorProto object.
        base_dir: directory that contains the external data.
    """
    info = onnx.external_data_helper.ExternalDataInfo(tensor)
    external_data_file_path = c_checker._resolve_external_data_location(  # type: ignore[attr-defined]
        base_dir, info.location, tensor.name
    )

    with open(external_data_file_path, "rb") as data_file:
        if info.offset:
            data_file.seek(info.offset)

        data = data_file.read(info.length) if info.length else data_file.read()
    return data


def load_tensor(name: str, extractor: onnx.utils.Extractor) -> None:
    tensor = extractor.wmap[name]
    info = onnx.external_data_helper.ExternalDataInfo(tensor)
    if not info.location:
        # no external data
        return
    if name in extractor.wmap_loaded:  # type: ignore[attr-defined]
        return
    props = extractor.model.graph.metadata_props
    path = None
    for prop in props:
        if prop.key == "onnx_utils_load":
            path = prop.value
    if path:
        assert is_initializer(name, extractor)
        tensor.raw_data = get_external_data_for_tensor(tensor, path)
        extractor.wmap_loaded.add(name)  # type: ignore[attr-defined]


def add_attribute(node: onnx.NodeProto, key: str, value: Any, attr_type: int | None = None) -> None:
    # support dynamic symbolic shape
    if isinstance(value, collections.abc.Iterable):
        value_types = {type(v) for v in value}
        if len(value_types) != 1 and str in value_types:
            value = [str(v) for v in value]

    node.attribute.append(onnx.helper.make_attribute(key, value, attr_type=attr_type))


def get_attribute(node: onnx.NodeProto, key: str, default_value: Any = None) -> Any:
    try:
        return onnx.helper.get_node_attr_value(node, key)
    except ValueError:
        if default_value is not None:
            return default_value
        raise


def has_attribute(node: onnx.NodeProto, key: str) -> bool:
    try:
        onnx.helper.get_node_attr_value(node, key)
    except ValueError:
        return False
    else:
        return True


def delete_attribute(node: onnx.NodeProto, key: str) -> None:
    index = None
    for i, attr in enumerate(node.attribute):
        if attr.name == key:
            index = i
            break
    if index is not None:
        del node.attribute[index]


def set_attribute(node: onnx.NodeProto, key: str, value: Any) -> None:
    delete_attribute(node, key)
    add_attribute(node, key, value)


def copy_attributes(src_node: onnx.NodeProto, dst_node: onnx.NodeProto) -> None:
    dst_node.attribute.extend(src_node.attribute)


def set_value_in_attribute(node: onnx.NodeProto, attr_key: str, original_value: Any, new_value: Any) -> None:
    values = onnx.helper.get_node_attr_value(node, attr_key)
    assert isinstance(values, list), f"Attribute {attr_key} is not a list"
    if isinstance(values[0], bytes):
        values = [x.decode("utf-8") for x in values]
    index = None
    for i, value in enumerate(values):
        if value == original_value:
            index = i
            break
    if index is not None:
        values[index] = new_value
        set_attribute(node, attr_key, values)


def append_value_in_attribute(node: onnx.NodeProto, attr_key: str, value: Any) -> None:
    try:
        values = onnx.helper.get_node_attr_value(node, attr_key)
    except ValueError:
        set_attribute(node, attr_key, [value])
        return
    assert isinstance(values, list), f"Attribute {attr_key} is not a list"
    if isinstance(values[0], bytes):
        values = [x.decode("utf-8") for x in values]
    if isinstance(value, list):
        values.extend(value)
    else:
        values.append(value)
    set_attribute(node, attr_key, values)


def is_initializer(name: str, graph_or_extractor: onnx.utils.Extractor | onnx.GraphProto) -> bool:
    if isinstance(graph_or_extractor, onnx.utils.Extractor):
        return name in graph_or_extractor.wmap
    return any(initializer.name == name for initializer in graph_or_extractor.initializer)


def is_initializer_or_const(name: str, graph_or_extractor: onnx.utils.Extractor | onnx.Graph) -> str:
    if is_initializer(name, graph_or_extractor):
        return name

    graph = graph_or_extractor.graph if isinstance(graph_or_extractor, onnx.utils.Extractor) else graph_or_extractor
    parent_nodes = find_nodes_by_output(name, graph)
    if not parent_nodes:
        return ""
    parent_node = parent_nodes[0]
    if parent_node.op_type == "Constant":
        return parent_node.attribute[0].t.name
    elif len(parent_node.input) == 1 and is_initializer(parent_node.input[0], graph_or_extractor):
        return parent_node.input[0]
    return ""


def get_initializer(
    initializer_name: str,
    graph_or_extractor: onnx.GraphProto | onnx.utils.Extractor,
    load: bool = True,
) -> onnx.TensorProto:
    if isinstance(graph_or_extractor, onnx.utils.Extractor):
        if load:
            load_tensor(initializer_name, graph_or_extractor)
        return graph_or_extractor.wmap[initializer_name]  # type: ignore[no-any-return]
    else:
        for initializer in graph_or_extractor.initializer:
            if initializer.name == initializer_name:
                return cast(onnx.TensorProto, initializer)
        raise ValueError(f"Initializer {initializer_name} not found")


def get_external_base_dir(extractor: onnx.utils.Extractor) -> str:
    path = os.getcwd()
    try:
        props = extractor.model.graph.metadata_props
        for prop in props:
            if prop.key == "onnx_utils_load":
                path = prop.value
    except AttributeError:
        # if the extractor was loaded with just the graph, there's no model
        # so assume path is fine as the working directory
        if extractor.model is not None:
            raise
    return path


def tensor_dtype_to_np_dtype(dtype: int) -> npt.DTypeLike:
    np_dtype: npt.DTypeLike = onnx.helper.tensor_dtype_to_np_dtype(dtype)
    if dtype == onnx.TensorProto.BFLOAT16:
        np_dtype = ml_dtypes.bfloat16
    elif dtype == onnx.TensorProto.INT4:
        # INT4 data is read as UINT8
        np_dtype = np.uint8
    return np_dtype


def get_initializer_as_numpy(
    name_or_tensor: str | onnx.TensorProto, extractor: onnx.utils.Extractor, do_reshape: bool = True
) -> npt.NDArray[Any]:
    tensor = extractor.wmap[name_or_tensor] if isinstance(name_or_tensor, str) else name_or_tensor
    info = onnx.external_data_helper.ExternalDataInfo(tensor)
    if not info.location or tensor.name in extractor.wmap_loaded:  # type: ignore[attr-defined]
        if tensor.data_type == onnx.TensorProto.BFLOAT16:
            return np.frombuffer(tensor.raw_data, dtype=ml_dtypes.bfloat16)
        elif tensor.data_type == onnx.TensorProto.INT4:
            # For INT4, onnx.numpy_helper.to_array is very slow
            # and we only need the data returned as UINT8. So
            # np.frombuffer(tensor.raw_data, dtype=np.uint8) would be
            # enough.
            return np.frombuffer(tensor.raw_data, dtype=np.uint8)
        else:
            return onnx.numpy_helper.to_array(tensor)

    path = get_external_base_dir(extractor)
    assert is_initializer(tensor.name, extractor)
    data = get_external_data_for_tensor(tensor, path)
    np_dtype = tensor_dtype_to_np_dtype(tensor.data_type)

    if do_reshape:
        assert tensor.data_type != onnx.TensorProto.INT4
        shape = get_shape(tensor)
        assert is_static_shape(shape)
        return np.frombuffer(data, dtype=np_dtype).reshape(shape)
    else:
        return np.frombuffer(data, dtype=np_dtype)


def get_initializer_or_const(
    name_or_tensor: str | onnx.TensorProto, extractor: onnx.utils.Extractor, do_reshape: bool = True
) -> npt.NDArray[Any]:
    if isinstance(name_or_tensor, onnx.TensorProto) or is_initializer(name_or_tensor, extractor):
        return get_initializer_as_numpy(name_or_tensor, extractor, do_reshape)

    # we know it's a string name at this point and it might be from a Constant node
    name = name_or_tensor
    parent_nodes = find_nodes_by_output(name, extractor.graph)
    if not parent_nodes:
        raise ValueError(f"{name} is not an initializer and may be a graph input")
    parent_node = parent_nodes[0]
    if parent_node.op_type == "Constant":
        return get_initializer_as_numpy(parent_node.attribute[0].t, extractor, do_reshape)
    elif len(parent_node.input) == 1 and is_initializer(parent_node.input[0], extractor):
        return get_initializer_as_numpy(parent_node.input[0], extractor, do_reshape)
    raise ValueError(f"{name} is neither an initializer nor from a Constant/Constant-like node")


def convert_to_external(
    input_path: Path,
    output_path: Path,
    external_data_extension: str,
    size_threshold: int = 1024,
) -> None:
    stem = output_path.stem
    location = f"{stem}.{external_data_extension}"
    extractor = load_extractor(input_path, False)
    for init in extractor.graph.initializer:
        if onnx.external_data_helper.uses_external_data(init):
            onnx.external_data_helper.load_external_data_for_tensor(init, str(input_path.parent))
            if sys.getsizeof(init.raw_data) < size_threshold:
                del init.external_data[:]
                init.data_location = onnx.TensorProto.DEFAULT
        else:
            np_array = get_initializer_as_numpy(init.name, extractor, do_reshape=False)
            if np_array.size * np_array.itemsize >= size_threshold:
                init.raw_data = np_array.tobytes()
        # onnx.external_data_helper.set_external_data(extractor.graph.initializer[index], location)
    ((input_path.parent) / location).unlink(True)

    onnx.external_data_helper.convert_model_to_external_data(
        extractor.model, location=location, size_threshold=size_threshold
    )
    onnx.save_model(
        extractor.model,
        output_path,
        save_as_external_data=True,
        location=location,
        size_threshold=size_threshold,
    )


def get_external_data_name(graph: onnx.GraphProto) -> str:
    for tensor in graph.initializer:
        if onnx.external_data_helper.uses_external_data(tensor):
            return onnx.external_data_helper.ExternalDataInfo(tensor).location
    return ""


def save_external_data(tensor: onnx.TensorProto, base_path: str, data: bytes) -> None:
    """This is a copy of the same function from ORT but where it doesn't require
    raw data to exist in the tensor. The data to set is passed externally

    Arguments:
        tensor (TensorProto): Tensor object to be serialized
        base_path: System path of a folder where tensor data is to be stored
        data (bytes): Data to write
    """

    info = onnx.external_data_helper.ExternalDataInfo(tensor)
    external_data_file_path = Path(base_path) / info.location
    assert info.location

    external_data_file_path.touch(exist_ok=True)

    # Open file for reading and writing at random locations ('r+b')
    with open(external_data_file_path, "r+b") as data_file:
        data_file.seek(0, 2)
        if info.offset is not None:
            # Pad file to required offset if needed
            file_size = data_file.tell()
            if info.offset > file_size:
                data_file.write(b"\0" * (info.offset - file_size))

            data_file.seek(info.offset)
        offset = data_file.tell()
        data_file.write(data)
        set_external_data(tensor, info.location, offset, data_file.tell() - offset)


def save_initializers_with_extractor(
    extractor: onnx.utils.Extractor,
    output_path: Path,
    external_data_name: str,
    alias_shared_tensors: bool = False,
    size_threshold: int = 1024,
) -> None:
    if external_data_name:
        assert output_path.is_dir()
        (output_path / external_data_name).unlink(True)
    tensor_addresses: dict[str, tuple[int, int, npt.DTypeLike]] = {}

    def recurse(
        extractor: onnx.utils.Extractor,
        output_path: Path,
        external_data_name: str,
        alias_shared_tensors: bool,
        size_threshold: int,
        offset: int,
    ) -> int:
        for node in extractor.graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    offset = recurse(
                        get_extractor(attr.g),
                        output_path,
                        external_data_name,
                        alias_shared_tensors,
                        size_threshold,
                        offset,
                    )
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        offset = recurse(
                            get_extractor(subgraph),
                            output_path,
                            external_data_name,
                            alias_shared_tensors,
                            size_threshold,
                            offset,
                        )

        del extractor.graph.initializer[:]

        unused_inits = []
        inits = []
        for init_name, init in extractor.wmap.items():
            if is_used_initializer(init.name, extractor.graph):
                with contextlib.chdir(output_path):
                    np_data = get_initializer_as_numpy(init_name, extractor, do_reshape=False)
                length = np_data.size * np_data.itemsize
                if length > size_threshold and external_data_name:
                    init.ClearField("raw_data")
                    if alias_shared_tensors and init_name in tensor_addresses:
                        saved_length = tensor_addresses[init_name][1]
                        saved_dtype = tensor_addresses[init_name][2]

                        assert saved_length == length, f"{init_name}: {saved_length} != {length}"
                        assert saved_dtype == np_data.dtype
                        set_external_data(
                            init,
                            external_data_name,
                            tensor_addresses[init_name][0],
                            length,
                        )
                    else:
                        if alias_shared_tensors:
                            tensor_addresses[init_name] = (
                                offset,
                                length,
                                np_data.dtype,
                            )
                        set_external_data(
                            init,
                            external_data_name,
                            offset,
                            length,
                        )
                        offset += length
                    save_external_data(init, str(output_path), np_data.tobytes())
                    extractor.graph.initializer.append(init)
                    inits.append(init)
                else:
                    load_tensor(init_name, extractor)
                    del init.external_data[:]
                    init.data_location = onnx.TensorProto.DEFAULT
                    extractor.graph.initializer.append(init)
                    inits.append(init)
            else:
                unused_inits.append(init_name)
        for init_name in unused_inits:
            del extractor.wmap[init_name]
        return offset

    recurse(extractor, output_path, external_data_name, alias_shared_tensors, size_threshold, 0)


def delete_custom_metadata_props(model: onnx.ModelProto) -> None:
    for prop in reversed(model.graph.metadata_props):
        if prop.key == "onnx_utils_load":
            model.graph.metadata_props.remove(prop)


def save_model_without_external_data(model: onnx.ModelProto, f: Path) -> None:
    """
    This is a snippet of onnx.save_model but only the part that serializes the
    onnx model, assuming the external data has already been saved with
    save_initializers_with_extractor()
    Args:
        model (onnx.ModelProto): Model to save
        f (Path): Path to save it to
    """
    delete_custom_metadata_props(model)

    serialized = onnx._get_serializer(None, f).serialize_proto(model)
    onnx._save_bytes(serialized, f)


def delete_model(
    model_path: Path,
    external_data_extension: str,
    model: bool = True,
    onnx_external_data: bool = True,
    header: bool = True,
    external_data: bool = True,
) -> None:
    _logger.debug("Deleting model and associated files at %s", model_path)
    missing_ok = True
    if model:
        model_path.unlink(missing_ok)
    if onnx_external_data:
        model_path.with_suffix(f".{external_data_extension}").unlink(missing_ok)
    if header:
        model_path.with_suffix(".pb.bin").unlink(missing_ok)
    if external_data:
        model_path.with_suffix(".bin").unlink(missing_ok)


def rename_external_data_file(original_model: Path, final_model: Path) -> None:
    _logger.debug("Renaming external data file from %s to %s", original_model.stem, final_model.stem)
    original_header = original_model.with_suffix(".pb.bin")
    new_header = final_model.with_suffix(".pb.bin")
    new_external_data = final_model.with_suffix(".bin")
    proto.rename_external_data_file(original_header, new_external_data.name, new_header)
    external_data_file = original_model.with_suffix(".bin")
    if external_data_file.exists():
        shutil.copyfile(external_data_file, new_external_data)


def get_initializers(
    names: Iterable[str], extractor: onnx.utils.Extractor, load: bool = True
) -> list[onnx.TensorProto]:
    if isinstance(names, str):
        names = [names]
    nodes = []
    for name in names:
        if is_initializer(name, extractor):
            nodes.append(get_initializer(name, extractor, load))
    return nodes


def find_initializers_by_nodes(
    extractor: onnx.utils.Extractor, nodes: list[onnx.NodeProto] | onnx.NodeProto, load: bool = True
) -> list[onnx.TensorProto]:
    if not isinstance(nodes, list):
        nodes = [nodes]

    initializer_names = set()

    for node in nodes:
        for io in itertools.chain(node.input, node.output):
            if is_initializer(io, extractor):
                # if io in model.graph.initializer:
                initializer_names.add(io)
    return get_initializers(initializer_names, extractor, load)


def _find_io_by_nodes(nodes: Iterable[onnx.NodeProto]) -> tuple[set[str], set[str]]:
    inputs: set[str] = set()
    outputs: set[str] = set()

    for node in nodes:
        for io in node.input:
            inputs.add(io)
        for io in node.output:
            outputs.add(io)

    return inputs, outputs


def find_inputs_by_nodes(
    nodes: list[onnx.NodeProto],
    graph_or_extractor: onnx.utils.Extractor | onnx.GraphProto,
    include_global_inputs: bool = False,
    include_initializers: bool = True,
) -> list[str]:
    if (include_global_inputs or not include_initializers) and graph_or_extractor is None:
        raise ValueError("If including global inputs or excluding initializers, an extractor/graph must be provided")

    graph = graph_or_extractor.graph if isinstance(graph_or_extractor, onnx.utils.Extractor) else graph_or_extractor

    inputs, outputs = _find_io_by_nodes(nodes)
    if include_global_inputs:
        for io in graph.input:
            inputs.add(io.name)
    if not include_initializers:
        initializers: set[str] = set()
        for io_input in inputs:
            if is_initializer(io_input, graph_or_extractor):
                initializers.add(io_input)
        inputs = inputs - initializers
    real_inputs = inputs - outputs

    inputs_ordered = []
    for node in nodes:
        for io_name in node.input:
            if io_name in real_inputs:
                inputs_ordered.append(io_name)
                real_inputs.remove(io_name)
    return inputs_ordered


def find_outputs_by_nodes(
    nodes: Iterable[onnx.NodeProto],
    graph: onnx.GraphProto | None = None,
    include_global_outputs: bool = False,
    extra_outputs: Iterable[str] | None = None,
) -> list[str]:
    inputs, outputs = _find_io_by_nodes(nodes)
    real_outputs = outputs - inputs
    # have to add these after removing inputs
    if extra_outputs is not None:
        # extra outputs are added at this stage so that when we go to order it
        # below, the extra outputs are in the right order
        for output in extra_outputs:
            real_outputs.add(output)
    if include_global_outputs:
        assert graph is not None
        for out in graph.output:
            real_outputs.add(out.name)

    outputs_ordered = []
    for node in nodes:
        for io in node.output:
            if io in real_outputs:
                outputs_ordered.append(io)
                real_outputs.remove(io)
    return outputs_ordered


def find_intermediates_by_nodes(nodes: list[onnx.NodeProto]) -> list[str]:
    inputs, outputs = _find_io_by_nodes(nodes)
    return list(outputs.intersection(inputs))


def has_multiple_successors(edge: str, graph: onnx.GraphProto) -> bool:
    cast_0_successors = find_nodes_by_input(edge, graph)
    multiple_successors = False
    if len(cast_0_successors) > 1 or is_output_edge(edge, graph):
        multiple_successors = True
    return multiple_successors


def find_input_tvis_by_nodes(
    extractor: onnx.utils.Extractor, nodes: Iterable[onnx.NodeProto], names: Sequence[str] | None = None
) -> list[onnx.ValueInfoProto]:
    tvis = []

    for node in nodes:
        for io in node.input:
            if names is None or io in names:
                if io in extractor.vimap:
                    tvis.append(extractor.vimap[io])
                if is_input_edge(io, extractor.graph):
                    for input_tvi in extractor.graph.input:
                        if input_tvi.name == io:
                            tvis.append(input_tvi)
                            break
    return tvis


def find_output_tvis_by_nodes(
    extractor: onnx.utils.Extractor, nodes: Iterable[onnx.NodeProto], names: Sequence[str] | None = None
) -> list[onnx.ValueInfoProto]:
    tvis = []

    for node in nodes:
        for io in node.output:
            if names is None or io in names:
                if io in extractor.vimap:
                    tvis.append(extractor.vimap[io])
                if is_output_edge(io, extractor.graph):
                    for output_tvi in extractor.graph.output:
                        if output_tvi.name == io:
                            tvis.append(output_tvi)
                            break
    return tvis


def find_tvis_by_nodes(extractor: onnx.utils.Extractor, nodes: Iterable[onnx.NodeProto]) -> list[onnx.ValueInfoProto]:
    tvis = []

    for node in nodes:
        for io in itertools.chain(node.input, node.output):
            if io in extractor.vimap:
                tvis.append(extractor.vimap[io])
    return tvis


def get_tvi(io_name: str, extractor: onnx.utils.Extractor) -> onnx.ValueInfoProto:
    if io_name in extractor.vimap:
        return extractor.vimap[io_name]  # type: ignore[no-any-return]
    for input_tvi in extractor.graph.input:
        if input_tvi.name == io_name:
            return input_tvi  # type: ignore[no-any-return]
    for output_tvi in extractor.graph.output:
        if output_tvi.name == io_name:
            return output_tvi  # type: ignore[no-any-return]
    raise ValueError(f"Cannot find tvi with name {io_name}")


def build_tvi(
    io_name: str,
    extractor: onnx.utils.Extractor,
    name: str | None = None,
    dtype: DtypeType | None = None,
    shape: ShapeType | None = None,
) -> onnx.ValueInfoProto:
    new_name = name if name is not None else io_name
    new_dtype = dtype if dtype is not None else get_dtype(io_name, extractor)
    new_shape = shape if shape is not None else get_shape(io_name, extractor)

    return onnx.helper.make_tensor_value_info(new_name, new_dtype, new_shape)


def split_sequence(seq: Iterable[Any], n: int) -> collections.abc.Generator[list[onnx.NodeProto], None, None]:
    """Yield n number of sequential chunks from seq."""
    d, r = divmod(len(seq), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield seq[si : si + (d + 1 if i < r else d)]


def _split_subgraph_by_count(subgraph: list[onnx.NodeProto], split_count: int) -> list[list[onnx.NodeProto]]:
    return list(split_sequence(subgraph, split_count))


def split_subgraph(subgraph: list[onnx.NodeProto], split_count: int) -> list[list[onnx.NodeProto]]:
    """
    Split a list of nodes either by count or by layer
    Args:
        subgraph (list[onnx.NodeProto]): Nodes to split
        split_count (int): Positive integers to split by layer, negative integers to split by count
    Returns:
        list[list[onnx.NodeProto]]: Split subgraph
    """
    assert split_count != 0, "split_count cannot be zero"
    if split_count in {-1, 1}:
        return [subgraph]

    if split_count < 0:
        # in this case, split by number of nodes so that each chunk has roughly
        # equal number of nodes
        return _split_subgraph_by_count(subgraph, abs(split_count))

    # otherwise, try to match based on layers to split at a more predicatable
    # node boundary. Go through the nodes and track when the layer ID changes
    layer_count = -1
    cross_over = {}
    for idx, node in enumerate(subgraph):
        matches = re.findall(r"layer[s]*\.(\d+)", node.name)
        if matches:
            layer_id = int(matches[0])
            if layer_id != layer_count:
                layer_count = max(layer_count, layer_id)
                cross_over[layer_id] = max(idx, cross_over.get(layer_id, idx))

    if layer_count == -1:
        _logger.warning("Could not determine layer count for dynamic splitting, defaulting to numeric split")
        return _split_subgraph_by_count(subgraph, split_count)

    if not all(cross_over[x] < cross_over[x + 1] for x in range(layer_count - 1)):
        _logger.warning("Layer indices are not strictly increasing for dynamic splitting, defaulting to numeric split")
        return _split_subgraph_by_count(subgraph, split_count)

    layers_per_chunk, _leftover = divmod(layer_count, split_count)
    splits = [
        subgraph[cross_over[i * layers_per_chunk] : cross_over[(i + 1) * layers_per_chunk]]
        for i in range(0, split_count)
    ]
    if cross_over[split_count * layers_per_chunk] < len(subgraph):
        splits[-1].extend(subgraph[cross_over[split_count * layers_per_chunk] :])
    if cross_over[0] > 0:
        # add any leading nodes that were before the first layer
        splits[0][:0] = subgraph[0 : cross_over[0]]
    return splits


def make_slice_nodes(
    input_tensor_name: str,
    dims: list[tuple[str, int, int]],  # e.g. [("h", 0, 2), ("w", 0, 3)]
    slice_ends: dict[str, str],
    dtype: int,
    output_shape: ShapeType | None = None,
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto], list[onnx.ValueInfoProto], str]:
    """
    Build chained Slice nodes with starts + axes.
    Args:
      input_tensor_name: input to slice.
      dims: list of (label, start_value, axis_value).
      slice_ends: {label: slice end input name}.
      dtype: dtype of the output tensor.
      output_shape: shape of the output tensor, if known.
    Returns:
      nodes: Slice nodes.
      initializers: starts and axes tensors.
      tvis: starts and axes TVIs.
    """
    nodes = []
    initializers = []
    tvis = []

    prev_output = input_tensor_name
    for i, (label, start_val, axis_val) in enumerate(dims):
        start_name = f"{input_tensor_name}_start"
        axis_name = f"{input_tensor_name}_axis"
        if label:
            start_name += f"_{label}"
            axis_name += f"_{label}"

        start_tensor = onnx.numpy_helper.from_array(np.array([start_val], dtype=np.int64), name=start_name)
        axis_tensor = onnx.numpy_helper.from_array(np.array([axis_val], dtype=np.int64), name=axis_name)
        initializers.extend([start_tensor, axis_tensor])

        tvi_start = onnx.helper.make_tensor_value_info(start_name, onnx.TensorProto.INT64, start_tensor.dims)
        tvi_axis = onnx.helper.make_tensor_value_info(axis_name, onnx.TensorProto.INT64, axis_tensor.dims)
        tvis.extend([tvi_start, tvi_axis])

        if label:
            out_name = f"{input_tensor_name}_" + "_".join([label for label, _, _ in dims[: i + 1]]) + "_slice"
        else:
            out_name = f"{input_tensor_name}_slice"
        slice_name = f"{input_tensor_name}_Slice"
        if label:
            slice_name += f"_{label.upper()}"
        slice_node = onnx.helper.make_node(
            "Slice",
            inputs=[prev_output, start_name, slice_ends[label], axis_name],
            outputs=[out_name],
            name=slice_name,
        )
        tvi_slice = onnx.helper.make_tensor_value_info(out_name, dtype, output_shape)
        tvis.append(tvi_slice)
        nodes.append(slice_node)
        prev_output = out_name

    return nodes, initializers, tvis, prev_output


def set_opset(model: onnx.ModelProto, domain: str, version: int) -> None:
    found = False
    opsets = model.opset_import
    for opset in opsets:
        if opset.domain == domain:
            found = True
            # strictly increase the configured version so the highest value is kept
            if opset.version < version:
                opset.version = version
            break
    if not found:
        model.opset_import.add(domain=domain, version=version)


def _remove_graph_io(names: Sequence[str], all_io: MutableSequence[onnx.ValueInfoProto]) -> None:
    inputs_to_remove = []
    for index, input_tvi in enumerate(all_io):
        if input_tvi.name in names:
            inputs_to_remove.append(index)
    sorted_indices = sorted(inputs_to_remove, reverse=True)
    for index in sorted_indices:
        del all_io[index]


def remove_graph_inputs(input_names: Sequence[str], graph: onnx.GraphProto) -> None:
    _remove_graph_io(input_names, graph.input)


def remove_graph_outputs(output_names: Sequence[str], graph: onnx.GraphProto) -> None:
    _remove_graph_io(output_names, graph.output)


def remove_node_and_init_by_indices(
    graph: onnx.GraphProto, inodes: list[int], inits: list[int], tvis: list[int]
) -> None:
    inodes = sorted(inodes, reverse=True)
    inits = sorted(inits, reverse=True)
    # tvis = sorted(tvis, reverse=True)
    for i in inodes:
        del graph.node[i]

    for i in inits:
        del graph.initializer[i]

    # for i in tvis:
    #     del model.graph.value_info[i]

    # gc.collect()


def remove_node_and_info(extractor: onnx.utils.Extractor, node: onnx.NodeProto) -> None:
    model = extractor.model
    node_indices = []
    init_indices = []
    tvi_indices: list[int] = []
    nodes_list = list(model.graph.node)
    init_list = list(model.graph.initializer)
    # tvis_list = list(model.graph.value_info)
    for input in node.input:
        consts = find_consts(input, model.graph)

        for n in consts:
            users = find_nodes_by_input(n.name, extractor.graph)
            if not len(users):
                node_indices.append(nodes_list.index(n))

        inits = get_initializers(input, extractor, False)
        for n in inits:
            users = find_nodes_by_input(n.name, extractor.graph)
            if not len(users):
                try:
                    init_index = init_list.index(n)
                    init_indices.append(init_index)
                except ValueError:
                    pass

                del extractor.wmap[n.name]

    # tvis = find_tvis_by_nodes(extractor, node)

    # for n in tvis:
    #     try:
    #         tvi_indices.append(tvis_list.index(n))
    #         del extractor.vimap[n.name]
    #     except ValueError:
    #         pass

    remove_node_and_init_by_indices(model.graph, node_indices, init_indices, tvi_indices)
    # gc.collect()


def graph_topological_sort(extractor: onnx.utils.Extractor, is_deterministic: bool = False) -> None:
    """
    Perform a topological sort of the nodes in the ONNX graph. This is a copy of
    a similar named function in onnxruntime, modified to work the extractor.

    Args:
        extractor (onnx.utils.Extractor): Extractor
        is_deterministic (bool, optional): Sort nodes before sort. Defaults to False.

    Raises:
        RuntimeError: raised if graph is not a DAG
    """
    deps_set = set()  # dependency set of all node
    sorted_node_set: set[int] = set()  # sorted node set
    sorted_nodes = []  # initialize sorted_nodes

    graph = extractor.graph

    initializer_names = set(extractor.wmap.keys())
    graph_input_names = {input.name for input in graph.input}
    deps_set = initializer_names | graph_input_names

    # for nested subgraphs, there can be dangling inputs in ONNX that are
    # implicitly provided from the parent graph
    all_inputs = set()
    all_outputs = set()
    for node in extractor.graph.node:
        all_inputs.update(node.input)
        all_outputs.update(node.output)
    all_inputs.discard("")
    all_outputs.discard("")
    dangling_inputs = all_inputs - all_outputs - deps_set
    deps_set.update(dangling_inputs)

    sorted_node_set_len = -1
    graph_nodes = graph.node if not is_deterministic else sorted(graph.node, key=lambda x: x.name)

    last_node_name = None
    while len(sorted_node_set) != len(graph_nodes):
        if len(sorted_node_set) == sorted_node_set_len:
            break
        sorted_node_set_len = len(sorted_node_set)
        for node_idx, node in enumerate(graph_nodes):
            if node_idx in sorted_node_set:
                _logger.debug(f"Node {node.name} is already sorted")
                continue
            input_count = sum(1 for _ in node.input if _)
            if input_count == 0:
                sorted_nodes.append(node)
                sorted_node_set.add(node_idx)
                for output in node.output:
                    if output:
                        deps_set.add(output)
                continue
            failed = False
            for input_name in node.input:
                if input_name and input_name not in deps_set:
                    _logger.debug(f"Node {node.name} is waiting for input {input_name}")
                    failed = True
                    last_node_name = node.name
            if not failed:
                sorted_nodes.append(node)
                sorted_node_set.add(node_idx)
                for output in node.output:
                    if output:
                        deps_set.add(output)
            else:
                continue

    if len(sorted_node_set) != len(graph.node):
        raise RuntimeError(
            f"Graph is not a DAG: len(sorted_node_set)={len(sorted_node_set)}, len(graph.node)={len(graph.node)}, failed at node {last_node_name}"
        )

    graph.ClearField("node")
    graph.node.extend(sorted_nodes)


# def cleanup(model):
#     in_graph_tensors = set()
#     already_pass = set()
#     output_names = set([item.name for item in model.graph.output])
#     tensors = [[item.name] for item in model.graph.input]
#     for node in model.graph.node:
#         if len(node.input) == 0:
#             tensors.extend(list(node.output))

#     already_pass_tensors = []
#     while len(tensors) > 0:
#         names = tensors.pop()
#         tensor = names[-1]
#         if tensor in already_pass:
#             already_pass_tensors.append(names)
#             continue

#         already_pass.add(tensor)
#         if tensor in output_names:
#             in_graph_tensors.update(names)
#             continue

#         nodes = find_nodes_by_input(tensor, model.graph)
#         for node in nodes:
#             for output in node.output:
#                 tensors.append(names + list(node.input) + [output])

#     for names in already_pass_tensors:
#         tensor = names[-1]
#         if tensor in in_graph_tensors:
#             in_graph_tensors.update(names)

#     del_nodes = []
#     del_inits = []
#     for inode, node in enumerate(model.graph.node):
#         in_graph = any([output in in_graph_tensors for output in node.output])
#         if not in_graph:
#             # log(
#             #     f"Remove a floating node: {node.name}, the node output is: {node.output}"
#             # )
#             del_nodes.append(inode)

#     for i, init in enumerate(model.graph.initializer):
#         in_graph = init.name in in_graph_tensors
#         if not in_graph:
#             # log(f"Remove a unused initializer: {init.name}")
#             del_inits.append(i)

#     remove_node_and_init_by_indices(model.graph, del_nodes, del_inits, [])


def get_extractor(m: onnx.ModelProto | onnx.GraphProto) -> onnx.utils.Extractor:
    """Custom Extractor constructor to work with models >2GB which the default
    Extractor doesn't work with because infer_shapes doesn't work for it. This
    assumes that you've called infer_shapes_path() already on the model
    Args:
        m (onnx.ModelProto): Model to use for extractor
    Returns:
        onnx.utils.Extractor: Extractor
    """
    extractor = onnx.utils.Extractor(onnx.ModelProto())
    if isinstance(m, onnx.ModelProto):
        extractor.model = m
        # this is for compatibility with original Extractor
        extractor.graph = m.graph
    else:
        extractor.model = None  # type: ignore[assignment]
        extractor.graph = m
    extractor.wmap = extractor._build_name2obj_dict(extractor.graph.initializer)  # type: ignore[no-untyped-call]
    extractor.vimap = extractor._build_name2obj_dict(extractor.graph.value_info)  # type: ignore[no-untyped-call]
    with contextlib.suppress(AttributeError):
        # older onnx versions don't have this function so skip it
        extractor.outmap = extractor._build_output_dict(extractor.graph)

    extractor.wmap_loaded = set()  # type: ignore[attr-defined]
    return extractor


def load_model(path: Path | str, load_external_data: bool, infer_shapes: bool = True) -> onnx.ModelProto:
    if isinstance(path, str):
        path = Path(path)
    if infer_shapes:
        # this is needed explicitly to work for models > 2GB
        onnx.shape_inference.infer_shapes_path(path)
    model = onnx.load_model(path, load_external_data=load_external_data)

    delete_custom_metadata_props(model)
    if not load_external_data:
        model.graph.metadata_props.add(key="onnx_utils_load", value=str(path.parent))
    return model


def compute_model_md5(onnx_path: os.PathLike[Any]) -> str:
    model = onnx.load_model(onnx_path, load_external_data=True)
    md5 = hashlib.md5()

    # Initializers are stored in graph.initializer
    # Sort by name to ensure stable ordering
    for init in sorted(model.graph.initializer, key=lambda t: t.name):
        md5.update(init.raw_data)

    return md5.hexdigest()


def load_extractor(path: Path | str, load_external_data: bool, infer_shapes: bool = True) -> onnx.utils.Extractor:
    if isinstance(path, str):
        path = Path(path)
    model = load_model(path, load_external_data, infer_shapes)
    return get_extractor(model)


def replace_subgraph(
    old_subgraph: list[onnx.NodeProto],
    new_subgraph: list[onnx.NodeProto],
    graph: onnx.GraphProto,
    rewrite: bool,
) -> None:
    nodes_list = list(graph.node)
    indices = sorted([nodes_list.index(item) for item in old_subgraph], reverse=True)

    for i in indices:
        del graph.node[i]

    if len(new_subgraph) == 0 and rewrite:
        input_node = old_subgraph[0]
        output_node = old_subgraph[-1]
        # rewrite the child nodes of the replaced block to use the new input
        # from before the replaced block.
        # in some cases, the nodes to replace may have different numbers of inputs/
        # outputs, so we explicitly use strict=False here. Nodes that are left
        # dangling should be removed by later passes to clean up the graph.
        o2i = dict(zip(output_node.output, input_node.input, strict=False))
        for output_name in output_node.output:
            if is_output_edge(output_name, graph):
                # However, this doesn't work if the replaced block is at
                # the end of the graph (no child nodes), so rewrite parents
                # instead
                input_name = o2i[output_name]
                parents = find_nodes_by_output(input_name, graph)
                for p in parents:
                    p.output[list(p.output).index(input_name)] = output_name
            else:
                children = find_nodes_by_input(output_name, graph)
                for c in children:
                    # can't use index because of possible duplicate inputs
                    for cindex, cin in enumerate(c.input):
                        if cin == output_name:
                            c.input[cindex] = o2i[output_name]
    elif indices:
        insert_point = indices[-1]
        for node in new_subgraph:
            graph.node.insert(insert_point, node)
            insert_point += 1


def replace_tvis(new_tvis: list[onnx.ValueInfoProto], extractor: onnx.utils.Extractor) -> None:
    for tvi in new_tvis:
        if tvi.name in extractor.vimap:
            idx = -1
            for index, tvi_2 in enumerate(extractor.graph.value_info):
                if tvi_2.name == tvi.name:
                    idx = index
                    break
            assert idx != -1
            del extractor.graph.value_info[idx]
        extractor.graph.value_info.append(tvi)
        extractor.vimap[tvi.name] = tvi


class Matcher:
    def __init__(self, pattern: list[str] | str, quiet: bool = False) -> None:
        self.lexer = Lexer(pattern)
        self.quiet = quiet

    # TODO(varunsh): add more documentation
    def _try_to_match(
        self,
        model: onnx.ModelProto,
        anchor: onnx.NodeProto,
        pattern_index: int | None = None,
        matched_pattern: list[onnx.NodeProto | None] | None = None,
    ) -> list[onnx.NodeProto | None]:
        """This is an updated matching function to replace the original one that
        was present. The original matching function only considered a depth
        based pattern where all nodes had to be children of the previous nodes.

        This pattern matcher allows matching on general patterns.
        """
        if matched_pattern is None:
            matched_pattern = [None] * len(self.lexer.patterns)

        if len(matched_pattern) == 1 and self.lexer.op_in_pattern(anchor.op_type):
            return [anchor]

        if not self.lexer.op_in_pattern(anchor.op_type):
            return matched_pattern

        if None not in matched_pattern:
            return matched_pattern

        if anchor in matched_pattern:
            return matched_pattern

        if pattern_index is not None:
            patterns = self.lexer.get_pattern_by_index(pattern_index)
        else:
            patterns = self.lexer.get_patterns_by_name(anchor.op_type)
        assert patterns

        for pattern, pattern_index in patterns:
            if matched_pattern[pattern_index] is not None:
                continue

            named_inputs, named_outputs = self.lexer.get_named_io(pattern)

            for io in named_inputs:
                edges = self.lexer.edges[io]
                # assert len(edges["src"]) == 1
                dsts = edges["dst"]
                srcs = edges["src"]
                assert len(srcs) == 1
                for dst in dsts:
                    io_index = dst["io_index"]
                    src_nodes = []
                    if len(anchor.input) > io_index:
                        src_nodes = find_nodes_by_output(anchor.input[io_index], model.graph)

                    for item in src_nodes:
                        if (
                            srcs[0]["name"] == item.op_type
                            and item.output[srcs[0]["io_index"]] == anchor.input[io_index]
                        ):
                            matched_pattern[pattern_index] = anchor
                            if item not in matched_pattern:
                                matched_pattern = self._try_to_match(
                                    model,
                                    item,
                                    srcs[0]["pattern_index"],
                                    matched_pattern,
                                )
            for io in named_outputs:
                edges = self.lexer.edges[io]
                dsts = edges["dst"]
                srcs = edges["src"]
                assert len(srcs) == 1
                io_index = srcs[0]["io_index"]
                dst_nodes = []
                if len(anchor.output) > io_index:
                    dst_nodes = find_nodes_by_input(anchor.output[io_index], model.graph)
                for item in dst_nodes:
                    for dst in dsts:
                        if dst["name"] == item.op_type and item.input[dst["io_index"]] == anchor.output[io_index]:
                            matched_pattern[pattern_index] = anchor
                            if item not in matched_pattern:
                                matched_pattern = self._try_to_match(model, item, dst["pattern_index"], matched_pattern)

        return matched_pattern

    def match(self, model: onnx.ModelProto) -> list[list[onnx.NodeProto]]:
        all_matched_pairs = []
        all_matched_set = set()
        for index, node in enumerate(model.graph.node):
            if node.op_type in {"Scan", "Loop", "If"}:
                continue
            else:
                # since the matcher can now use any node as the anchor, reduce
                # the search space by the op determined to be the best anchor
                if self.lexer.anchor != "?" and node.op_type != self.lexer.anchor:
                    continue
                potential_match = self._try_to_match(model, node)  # match
                if None in potential_match:
                    continue
                assert is_sequence_of(potential_match, onnx.NodeProto)
                if not node.name:
                    node.name = node.op_type + f"_{index}"
                node_names = ",".join([node.name for node in potential_match])
                if node_names not in all_matched_set:
                    all_matched_set.add(node_names)
                    if (potential_match) not in all_matched_pairs:
                        all_matched_pairs.append(potential_match)
        return all_matched_pairs

    def print_match(self, model: onnx.ModelProto) -> None:
        print("=====================================================================")
        matched_subgraphs = self.match(model)
        _logger.info(f"Found {len(matched_subgraphs)} subgraphs:")
        for i, subgraph in enumerate(self.match(model)):
            subgraph_names = ", ".join([f"{item.name}({item.op_type})" for item in subgraph])
            print(f"\tSubgraph{i}: {subgraph_names}")

        pattern_text = "\n\t".join(self.lexer.lines)
        _logger.info(f"Pattern is:\n\t{pattern_text}")
        print("=====================================================================")

    # replace some subgraph to new
    def replace(
        self,
        extractor: onnx.utils.Extractor,
        new_graph_fn: PassFunction,
        pass_id: str,
        params: ReplaceParams,
        max_to_replace: int | None = None,
    ) -> int:
        matched_subgraphs = self.match(extractor.model)
        # -1 means this is a global pass so no pattern is used and no replacements made
        replacement_count = -1 if matched_subgraphs and all(not elem for elem in matched_subgraphs) else 0

        if max_to_replace is not None and max_to_replace < len(matched_subgraphs):
            _logger.info(f"Reducing {len(matched_subgraphs)} to first {max_to_replace}")
            matched_subgraphs = matched_subgraphs[:max_to_replace]

        # resolve main graph
        for i, subgraph in enumerate(matched_subgraphs):
            if not all(node in extractor.graph.node for node in subgraph):
                continue
            rewrite_nodes = True
            if new_graph_fn is not None:
                new_nodes, new_initializers, new_tvis = new_graph_fn(
                    extractor, f"{pass_id}_{i}", copy.deepcopy(subgraph), params
                )
            else:
                new_nodes, new_initializers = [], []

            if new_nodes is None:
                rewrite_nodes = False
                new_nodes = []

            if subgraph != new_nodes:
                assert replacement_count >= 0
                replacement_count += 1

            replace_subgraph(subgraph, new_nodes, extractor.graph, rewrite_nodes)

            # disable this in the interest of runtime
            # for n in subgraph:
            #     # Remove the node and its corresponding information if it is not in new_nodes
            #     remove_node_and_info(extractor, n)

            if new_tvis:
                replace_tvis(new_tvis, extractor)

            for init in new_initializers:
                # don't maintain this in the interim and set it at the end
                # extractor.graph.initializer.append(init)
                extractor.wmap[init.name] = init

        return replacement_count
