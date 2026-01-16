# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import onnx
import onnx.helper
import onnx.onnx_cpp2py_export
import onnx.onnx_cpp2py_export.shape_inference
import onnx.shape_inference
import onnx.utils
import onnxruntime as rt

try:
    import ryzenai_dynamic_dispatch as dd
except ImportError:
    dd = None

import ryzenai_onnx_utils.matcher


def configure_parser(subparser: argparse._SubParsersAction[Any]) -> argparse.ArgumentParser:
    extractor_parser: argparse.ArgumentParser = subparser.add_parser("extract")
    extractor_parser.add_argument("input_path", type=Path, help="Path to input ONNX model")
    extractor_parser.add_argument("output_path", type=Path, help="Path to create directories for output files")
    extractor_parser.add_argument(
        "model_name",
        type=str,
        help="Name of the original model that's used as a prefix in the tensor data",
    )
    extractor_parser.add_argument(
        "--output-prefix",
        default=None,
        type=str,
        help="Prefix for generated files. By default, use model_name",
    )
    extractor_parser.add_argument(
        "--tensor-data",
        type=str,
        default=None,
        help="Path to the dumped tensor data from the original model",
    )
    extractor_parser.add_argument("--blocks", default=50, type=int, help="Number of nodes to process at once")
    extractor_parser.add_argument("--dd-meta", action="store_true", help="Create DynamicDispatch meta.json files")
    extractor_parser.add_argument(
        "--extract-first",
        action="store_true",
        help="Only extract the first matching pattern",
    )

    action = extractor_parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--pattern-file", type=Path, help="Path to the pattern file to use")
    action.add_argument("--pattern", type=str, help="Name of the pre-defined pattern to use")

    extractor_parser.add_argument("--quiet", action="store_true", help="Don't print status messages")

    return extractor_parser


class Node:
    """Represents the combination of multiple NodeProto objects"""

    def __init__(self, nodes: list[onnx.NodeProto]) -> None:
        inputs = set()
        outputs = set()

        for node in nodes:
            for input_name in node.input:
                inputs.add(input_name)
            for output_name in node.output:
                outputs.add(output_name)

        # remove common values and convert to an ordered list for determinism
        self.input = list(inputs - outputs)
        self.output = list(outputs - inputs)


class PatternFunc(Protocol):
    def __call__(self) -> str: ...


class NodeFunc(Protocol):
    def __call__(self, node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> bool: ...


def get_nodes_to_extract(
    input_path: Path, pattern: Callable[[], str] | Callable[[onnx.NodeProto, onnx.utils.Extractor], bool]
) -> tuple[list[Node], onnx.utils.Extractor]:
    """Use the pattern to extract the matching nodes

    Args:
        input_path (Path): Path to the onnx model
        pattern (Callable): Either a callable that takes no arguments (indicates
        that the Matcher will be used) or one that accepts a node and extractor
        and operates on a given node.

    Returns:
        list: List of matching nodes
    """
    m = onnx.load(input_path)
    extractor = ryzenai_onnx_utils.matcher.get_extractor(m)

    try:
        node_func = cast(PatternFunc, pattern)
        node_func()
    except TypeError:
        pattern_func = cast(NodeFunc, pattern)
        nodes = m.graph.node
        node_list = []

        for node in nodes:
            if pattern_func(node, extractor):
                node_list.append(node)

        return node_list, extractor

    matcher = ryzenai_onnx_utils.matcher.Matcher(node_func())
    matches = matcher.match(m)
    node_list = []
    for match in matches:
        # the Node object isn't a NodeProto as is produced in the other path
        # but we only use the node.input and node.output so it mimics it
        # enough
        node_list.append(Node(match))

    return node_list, extractor


def extract_model(
    input_path: Path | onnx.utils.Extractor,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    check_model: bool = False,
    save_as_external_data: bool = False,
) -> None:
    """
    This is a copy of the extract_model function in onnx.utils to work around
    the Extractor limitation for models > 2GB
    """
    if not output_path:
        raise ValueError("Output model path shall not be empty!")
    if not output_names:
        raise ValueError("Output tensor names shall not be empty!")

    if isinstance(input_path, onnx.utils.Extractor):
        e = input_path
    else:
        onnx.checker.check_model(input_path)
        model = onnx.load(input_path)

        e = ryzenai_onnx_utils.matcher.get_extractor(model)
    extracted = e.extract_model(input_names, output_names)

    onnx.save_model(extracted, output_path, save_as_external_data=save_as_external_data)
    if check_model:
        onnx.checker.check_model(output_path)


def extract_supported_nodes(
    node_list: list[Node],
    extractor: onnx.utils.Extractor,
    output_path: Path,
    output_prefix: str,
    offset: int,
    dd_meta: bool,
    quiet: bool,
) -> tuple[list[str], list[str]]:
    """Given a list of nodes, extract them all to individual ONNX graphs

    Args:
        node_list (list): List of nodes (NodeProto)
        extractor (onnx.utils.Extractor): Extractor to use for analyzing the graph
        output_path (Path): Directory to save the new ONNX graphs
        output_prefix (str): Prefix to use for naming the new ONNX model
        offset (int): Index to use for naming the new ONNX model
        dd_meta (bool): Generate DD metadata files

    Returns:
        (original_inputs, new_outputs): A tuple with the original model's inputs
            and the new outputs (a collection of the inputs and outputs of all
            the extracted nodes)
    """
    original_inputs = [x.name for x in extractor.model.graph.input]
    # keep track of all intermediate tensors to dump as outputs
    new_outputs = []
    for i, node in enumerate(node_list):
        index = i + offset
        output_model_name = f"{output_prefix}_{index}.onnx"
        output_model_path = output_path / output_model_name
        if not quiet:
            print(f"Generating {str(output_model_name)} and its metadata.json")

        # exclude initializer inputs
        inputs = [x for x in node.input if x not in extractor.wmap]
        outputs = node.output

        # remove any global model inputs from the new outputs
        new_inputs = [x for x in inputs if x not in original_inputs]
        new_outputs.extend(new_inputs)
        new_outputs.extend(outputs)

        extract_model(extractor, output_model_path, inputs, outputs)

        if dd_meta:
            assert dd is not None, "DD Python library could not be imported"
            new_m = onnx.load(output_model_path.as_posix())
            meta_info = dd.fuse.prepare_metadata(
                dd.onnx_graph.ONNXGraph(new_m), output_path, f"{output_prefix}_{index}_"
            )
            meta_json_name = output_path / f"{output_prefix}_{index}_meta.json"
            dd.fuse.save_tensors_to_json(meta_json_name, *meta_info)
    return original_inputs, new_outputs


def parse_dumped_input_tensors(
    extractor: onnx.utils.Extractor, tensor_data: str, model_name: str
) -> dict[str, npt.NDArray[Any]]:
    """Convert the dumped tensors from the original model to a dictionary

    Args:
        extractor (onnx.utils.Extractor): Extractor for utility
        tensor_data (str): Path to the dumped tensor data JSON or "random"
        model_name (str): Name of the model

    Raises:
        ValueError: Raised if expected names aren't found

    Returns:
        dict: Dictionary containing a map between names and data
    """
    input_metadata: dict[str, dict[str, Any]] = {}
    for x in extractor.model.graph.input:
        shape_tensor = x.type.tensor_type.shape
        shape = []
        for dim in shape_tensor.dim:
            shape.append(dim.dim_value)
        input_metadata[x.name] = {}
        input_metadata[x.name]["shape"] = shape
        input_metadata[x.name]["dtype"] = onnx.helper.tensor_dtype_to_np_dtype(x.type.tensor_type.elem_type)

    assert tensor_data != "random"
    tensor_data_path = Path(tensor_data)
    with open(tensor_data_path) as f:
        raw_inputs = json.load(f)

    onnx_inputs = {}
    for input_name, value in input_metadata.items():
        # the data can be in the dumped tensors in a variety of names
        aggregate_name = f"0_{model_name}_{input_name}"
        aggregate_name_2 = f"0_{model_name.split('_')[0]}s_{input_name}"
        potential_names = [input_name, aggregate_name, aggregate_name_2]
        found = False
        for name in potential_names:
            if name in raw_inputs:
                found = True
                onnx_inputs[input_name] = np.array(raw_inputs[name], dtype=value["dtype"])
        if not found:
            raise ValueError(f"Input {input_name} not found in tensor data")
        input_shape = onnx_inputs[input_name].shape

        assert len(input_shape) == len(value["shape"])
        # assert all(a == b for a, b in zip(input_shape, value["shape"]))

    return onnx_inputs


def save_dd_meta(
    dd_meta: bool,
    output_path: Path,
    output_prefix: str,
    index: int,
    local_data: dict[str, tuple[str, int, tuple[int, ...], np.dtype[Any]]],
) -> None:
    if dd_meta:
        meta_json = output_path / f"{output_prefix}_{index}_meta.json"
        with open(meta_json) as f:
            meta = json.load(f)
        for key, (filename, filesz, shape, _dtype) in local_data.items():
            meta["tensor_map"][key].update({"file_name": filename, "file_size": filesz, "shape": shape})
        with open(meta_json, "w") as f:
            json.dump(meta, f, indent=2)
    else:
        meta_json = output_path / f"{output_prefix}_{index}_meta.json"
        meta = {}
        for key, (filename, filesz, shape, dtype) in local_data.items():
            meta[key] = {}
            meta[key].update(
                {
                    "file_name": Path(filename).absolute().as_posix(),
                    "file_size": filesz,
                    "shape": shape,
                    "dtype": dtype.str,
                }
            )
        with open(meta_json, "w") as f:
            json.dump(meta, f, indent=2)


def save_intermediate_tensors(
    output_path: Path,
    output_prefix: str,
    original_inputs: list[str],
    new_outputs: list[str],
    onnx_inputs: dict[str, npt.NDArray[Any]],
    node_list: list[Node],
    extractor: onnx.utils.Extractor,
    offset: int,
    dd_meta: bool,
    quiet: bool,
) -> None:
    """Save the intermediate tensor data and update the metadata with real shapes.
    The new_outputs (all the intermediate tensors we're interested in) are extracted
    from the original model. Then, we run this model and then examine the outputs
    and save the tensor data. The data and real shapes of the tensors is used to update
    the metadata.json file for DD. The extracted shapes from the original model
    may be undefined resulting in zeroes in the shape until this point.

    Args:
        output_path (Path): Path to save the tensor data to
        output_prefix (str): Prefix to use for the tensor data
        original_inputs (list): List of the models's original inputs
        new_outputs (list): List of all the outputs of the model
        onnx_inputs (dict): Map of the input data to the original model
        node_list (list): List of nodes that were extracted
        extractor (onnx.utils.Extractor): Extractor utility
        offset (int): Number to use as the index for the saved data
        dd_meta (bool): Update generated DD meta json files
        quiet (bool): Suppress print outs

    Raises:
        ValueError: Raised if names aren't found
    """
    sess_options = rt.SessionOptions()
    sess_options.enable_profiling = False
    tmp_dir = output_path / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_onnx_file = tmp_dir / f"{output_prefix}_tmp.onnx"
    extract_model(
        extractor,
        tmp_onnx_file,
        original_inputs,
        new_outputs,
        save_as_external_data=True,
    )
    providers = ["CPUExecutionProvider"]
    # TODO(varunsh): need to remove this
    # import os
    # providers = ["RyzenAIExecutionProvider"]
    # MODEL_DIR = 'C:/Users/varunsh/Documents/RyzenAI_diffusers/onnx/scripting/sdxl-turbo'
    # DIFFUSION_MODEL_UNET_SUBFOLDER = "unet"
    # DD_ROOT = "C:/Users/varunsh/workspace/onnxruntime_ipsp/build/Windows/Release/_deps/dynamicdispatch-src"
    # sess_options.add_session_config_entry("dd_root", DD_ROOT)
    # sess_options.add_session_config_entry("model_name", "UNET")
    # # custom_op_path = ort.__path__[0] + "/capi/dynamic_dispatch_custom_op.dll"
    # custom_op_path = "C:/Users/varunsh/Documents/onnx_utils/build/install/bin/onnx_custom_ops.dll"
    # if os.path.exists(custom_op_path):
    #     dd_cache_path = MODEL_DIR + "/" +  DIFFUSION_MODEL_UNET_SUBFOLDER + "/.cache"
    #     sess_options.add_session_config_entry("dd_cache", dd_cache_path)
    #     sess_options.register_custom_ops_library(custom_op_path)
    sess = rt.InferenceSession(tmp_onnx_file, sess_options=sess_options, providers=providers)
    output_array = sess.run(None, onnx_inputs)
    output_data = {}
    for i, tensor in enumerate(output_array):
        output_data[new_outputs[i]] = tensor

    for i, node in enumerate(node_list):
        index = i + offset
        output_json_name = f"{output_prefix}_{index}.json"
        if not quiet:
            print(f"Generating input/output data in {output_json_name}")
        # exclude initializer inputs
        inputs = [x for x in node.input if x not in extractor.wmap]
        outputs = node.output

        local_data: dict[str, tuple[str, int, tuple[int, ...], np.dtype[Any]]] = {}
        for j, x in enumerate(inputs):
            if x in output_data:
                data = output_data[x]
            elif x in onnx_inputs:
                data = onnx_inputs[x]
            else:
                raise ValueError(f"{x} not found")
            assert isinstance(data, np.ndarray)
            output_file_name = output_path / f"{output_prefix}_{index}_{j}.in"
            data.tofile(output_file_name)
            local_data[x] = (
                output_file_name.as_posix(),
                data.nbytes,
                data.shape,
                data.dtype,
            )

        for j, x in enumerate(outputs):
            np_data = output_data[x]
            assert isinstance(np_data, np.ndarray)
            output_file_name = output_path / f"{output_prefix}_{index}_{j}.out"
            np_data.tofile(output_file_name)
            local_data[x] = (
                output_file_name.as_posix(),
                np_data.nbytes,
                np_data.shape,
                np_data.dtype,
            )

        save_dd_meta(dd_meta, output_path, output_prefix, index, local_data)

    shutil.rmtree(tmp_dir)


def main(args: argparse.Namespace) -> None:
    input_path = args.input_path
    tensor_data = args.tensor_data
    model_name = args.model_name
    output_path = args.output_path / model_name
    output_prefix = args.output_prefix or model_name
    quiet = args.quiet

    output_path.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print("Finding matching patterns")
    # some models fail verification but this *shouldn't* interfere with extraction
    with contextlib.suppress(onnx.onnx_cpp2py_export.shape_inference.InferenceError):
        # this is needed explicitly to work for models > 2GB
        onnx.shape_inference.infer_shapes_path(input_path)
    node_list, extractor = get_nodes_to_extract(input_path, args.pattern)
    start_index = 0
    node_list = node_list[start_index:]
    if not node_list:
        print("No matching patterns found")
        sys.exit(0)
    if not quiet:
        print(f"Found {len(node_list)} matching patterns")

    if args.extract_first:
        if not quiet:
            print("Only extracting first matched pattern")
        node_list = node_list[:1]

    # arbitrarily do n nodes at a time. Sometimes it can fail and having partial
    # results makes it easier to iterate
    chunk_size = args.blocks
    offset = start_index
    while node_list:
        nodes_to_process = min(len(node_list), chunk_size)
        if not quiet:
            print(f"Processing the next {nodes_to_process} nodes")
        original_inputs, new_outputs = extract_supported_nodes(
            node_list[:nodes_to_process],
            extractor,
            output_path,
            output_prefix,
            offset,
            args.dd_meta,
            quiet,
        )

        if tensor_data is not None:
            if not quiet:
                print("Generating new ONNX graph and running to get intermediates")

            onnx_inputs = parse_dumped_input_tensors(extractor, tensor_data, model_name)

            save_intermediate_tensors(
                output_path,
                output_prefix,
                original_inputs,
                new_outputs,
                onnx_inputs,
                node_list[:nodes_to_process],
                extractor,
                offset,
                args.dd_meta,
                quiet,
            )

        node_list = node_list[nodes_to_process:]
        offset += nodes_to_process
