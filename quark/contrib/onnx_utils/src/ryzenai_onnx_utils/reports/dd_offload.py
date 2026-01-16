# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import bisect
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import rich.box
import rich.table
from rich.console import Console
from rich.table import Table

import ryzenai_onnx_utils.matcher


def _get_io_as_str(node: onnx.NodeProto, extractor: onnx.utils.Extractor, io_name: str) -> str:
    io = getattr(node, io_name)
    io_list = []
    for name in io:
        if not name:
            io_list.append("[] - N/A")
        else:
            shape = ryzenai_onnx_utils.matcher.get_shape(name, extractor)
            shape_str = f"[{','.join(map(str, shape))}]"
            dtype = ryzenai_onnx_utils.matcher.get_dtype_str(name, extractor).split(".")[1]
            io_list.append(f"{shape_str} - {dtype}")
    return "\n".join(io_list)


def get_inputs_as_str(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> str:
    return _get_io_as_str(node, extractor, "input")


def get_outputs_as_str(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> str:
    return _get_io_as_str(node, extractor, "output")


class NodeData:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, dict[str, Any]]] = {}
        self.sorted_keys: list[str] = []
        self.offloaded_ops = 0
        self.not_offloaded_ops = 0
        self.offloaded_op_types: set[str] = set()
        self.offloaded_op_dict: dict[str, int] = {"sum": 0}
        self.offload_ratio = 0.0

    def add(self, name: str, op_type: str, inputs: str, outputs: str) -> None:
        if op_type not in self.data:
            self.data[op_type] = {}
            bisect.insort(self.sorted_keys, op_type)
        if inputs not in self.data[op_type]:
            self.data[op_type][inputs] = {}
        if outputs not in self.data[op_type][inputs]:
            self.data[op_type][inputs][outputs] = []
        self.data[op_type][inputs][outputs].append(name)

        if op_type.startswith("DynamicDispatch"):
            self.offloaded_ops += 1
        else:
            self.not_offloaded_ops += 1

    def _unpack_value(self, value: dict[str, dict[str, list[str]]]) -> list[dict[str, Any]]:
        unpacked_values = []
        for input_str, input_data in value.items():
            for output_str, output_data in input_data.items():
                data = {
                    "names": output_data,
                    "inputs": input_str,
                    "outputs": output_str,
                }
                unpacked_values.append(data)
        return unpacked_values

    def __iter__(self) -> Iterator[tuple[str, list[dict[str, Any]]]]:
        for key in self.sorted_keys:
            yield key, self._unpack_value(self.data[key])


def parse_model(input_path: Path) -> NodeData:
    extractor = ryzenai_onnx_utils.matcher.load_extractor(input_path, False, False)

    data = NodeData()

    op_dict = {}
    meta_list = [
        str(path.resolve()) for path in input_path.parent.rglob("*") if path.suffix == ".json" and "meta" in path.stem
    ]
    for meta in meta_list:
        with open(meta) as f:
            json_file = json.load(f)
        if "op_list" not in json_file:
            for dd_node in json_file:
                for op in json_file[dd_node]["op_list"]:
                    op_dict[op["name"]] = op["type"]
        else:
            for _dd_node in json_file:
                for op in json_file["op_list"]:
                    op_dict[op["name"]] = op["type"]

    for node in extractor.model.graph.node:
        if node.op_type.startswith("DynamicDispatch"):
            replaced_list = onnx.helper.get_node_attr_value(node, "replaced")
            for replaced in replaced_list:
                if replaced.decode() in op_dict:
                    data.offloaded_op_types.add(op_dict[replaced.decode()])
                    if op_dict[replaced.decode()] not in data.offloaded_op_dict:
                        data.offloaded_op_dict[op_dict[replaced.decode()]] = 1
                    else:
                        data.offloaded_op_dict[op_dict[replaced.decode()]] += 1
                    data.offloaded_op_dict["sum"] += 1
        inputs = get_inputs_as_str(node, extractor)
        outputs = get_outputs_as_str(node, extractor)
        data.add(node.name, node.op_type, inputs, outputs)
    data.offload_ratio = data.offloaded_op_dict["sum"] / (data.offloaded_op_dict["sum"] + data.not_offloaded_ops) * 100
    return data


def create_table(title: str, summarize: bool) -> Table:
    table = Table(title=title, box=rich.box.ASCII_DOUBLE_HEAD)
    table.add_column("Op Type")
    if not summarize:
        table.add_column("Name")
    else:
        table.add_column("Count")
    table.add_column("Inputs")
    table.add_column("Outputs")
    return table


def add_row(table: rich.table.Table, values: list[dict[str, str]], op_type: str, summarize: bool) -> int:
    count = 0
    for value in values:
        if summarize:
            table.add_row(
                op_type,
                str(len(value["names"])),
                value["inputs"],
                value["outputs"],
            )
            count += 1
        else:
            sorted_names = sorted(value["names"])
            for name in sorted_names:
                table.add_row(op_type, name, value["inputs"], value["outputs"])
                count += 1
    return count


def report(input_path: Path, comparison_path: Path | None, summarize: bool) -> None:
    console = Console()
    data = parse_model(input_path)
    if comparison_path:
        comparison_data = parse_model(comparison_path)

    table = create_table("DynamicDispatch Offload - not offloaded", summarize)
    for op_type, values in data:
        if not op_type.startswith("DynamicDispatch"):
            add_row(table, values, op_type, summarize)
    table.add_section()
    if comparison_path:
        table.add_row(
            "Not offloaded sum",
            str(data.not_offloaded_ops),
            "In comparison model:",
            str(comparison_data.not_offloaded_ops),
        )
    else:
        table.add_row("Not offloaded sum", str(data.not_offloaded_ops))
    console.print(table)

    table = create_table("DynamicDispatch Offload - offloaded", summarize)
    for op_type, values in data:
        if op_type.startswith("DynamicDispatch"):
            add_row(table, values, op_type, summarize)
    table.add_section()
    if comparison_path:
        table.add_row(
            "Offloaded sum",
            str(data.offloaded_ops),
            "In comparison model:",
            str(comparison_data.offloaded_ops),
        )
    else:
        table.add_row("Offloaded sum", str(data.offloaded_ops))
    table.add_section()
    if comparison_path:
        table.add_row(
            "Offloaded Op Types",
            "\n".join(list(data.offloaded_op_types)),
            "In comparison model:",
            "\n".join(list(comparison_data.offloaded_op_types)),
        )
    else:
        table.add_row("Offloaded Op Types", "\n".join(list(data.offloaded_op_types)))
    table.add_section()
    if comparison_path:
        table.add_row(
            "Offloaded sum (dd fusion)",
            str(data.offloaded_op_dict["sum"]),
            "In comparison model:",
            str(comparison_data.offloaded_op_dict["sum"]),
        )
        table.add_row(
            "Offload Ratio (dd fusion)",
            str(np.round(data.offload_ratio, 2)) + "%",
            "In comparison model:",
            str(np.round(comparison_data.offload_ratio, 2)) + "%",
        )
    else:
        table.add_row("Offloaded sum (dd fusion)", str(data.offloaded_op_dict["sum"]))
        table.add_row("Offload Ratio (dd fusion)", str(np.round(data.offload_ratio, 2)) + "%")
    console.print(table)
