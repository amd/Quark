# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import copy
import functools
import importlib
import importlib.resources
import importlib.util
import json
import logging
import logging.handlers
import shutil
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

import onnx
import sympy as sp
import yaml

import ryzenai_onnx_utils.auto
import ryzenai_onnx_utils.extract
import ryzenai_onnx_utils.lora
import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.optimize
import ryzenai_onnx_utils.pattern_builder as pb
import ryzenai_onnx_utils.pattern_generator as pg
import ryzenai_onnx_utils.postprocess
import ryzenai_onnx_utils.preprocess
import ryzenai_onnx_utils.report
import ryzenai_onnx_utils.utils
import ryzenai_onnx_utils.vaiml
from ryzenai_onnx_utils.typing import PassFunction, PatternType, SubPass, is_sequence_of

dd: ModuleType | None
try:
    import ryzenai_onnx_utils.transform.dd as dd
except ImportError:
    dd = None

_logger = logging.getLogger(__name__)


def configure_parser(subparser: argparse._SubParsersAction[Any]) -> None:
    partition_parser: argparse.ArgumentParser = subparser.add_parser("partition")

    partition_parser.add_argument("input_path", type=Path, help="Path to input ONNX model")
    partition_parser.add_argument("output_path", type=Path, help="Path to create directories for output files")
    partition_parser.add_argument("strategy", type=Path, help="Strategy to run")
    partition_parser.add_argument(
        "--attributes",
        metavar="KEY=VALUE",
        nargs="+",
        action="extend",
        help="Specify and override any strategy attributes as key=value pairs",
    )
    partition_parser.add_argument(
        "--dd-files-path",
        type=Path,
        default=Path("cache"),
        help="Path to create DD files in, either absolute or relative to output_path",
    )
    partition_parser.add_argument("--model-name", default="replaced", help="Name of the new onnx model")
    partition_parser.add_argument(
        "--save-as-external",
        action="store_true",
        help="Save the model with external data",
    )
    partition_parser.add_argument(
        "--load-external-data",
        action="store_true",
        help="Load all external data at startup",
    )
    partition_parser.add_argument(
        "--passes-per-iteration",
        type=int,
        default=None,
        help="Run the passes in intervals, saving the model in between. Defaults to None",
    )
    partition_parser.add_argument(
        "--combine-dd",
        action="store_true",
        help="Combine generated DD files",
    )
    partition_parser.add_argument(
        "--force",
        action="store_true",
        help="Answer yes to any confirmations that may come up",
    )
    # alias for setting the console log level to debug
    partition_parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Print more messages",
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recursion-limit", default=10000, type=int, help="Recursion limit for Python")
    parser.add_argument(
        "--external-data-extension",
        default="onnx.data",
        help="File extension for external data file",
    )
    parser.add_argument(
        "--external-data-threshold",
        # in ORT 1.23, internal tensors seem to not use our custom allocator so
        # forcing everything to be external is the workaround. However, ORT shape
        # inference does not use external data so we set an arbitrary non-zero
        # threshold to allow nodes like Reshape to have internal tensors.
        default=1024,
        type=int,
        help="Size threshold for moving things to ONNX external data.",
    )

    console_log_group = parser.add_mutually_exclusive_group()
    console_log_group.add_argument(
        "--console-log-level",
        choices=["off", "debug", "info", "warning", "error", "critical"],
        default="warning",
        help="Set the log level for console logging",
    )
    console_log_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        # TODO(varunsh): once the -v flag in partitioner is removed, this can be removed
        dest="global_verbose",
        help="Alias for setting --console-log-level. Use -v for info, -vv for debug",
    )

    parser.add_argument(
        "--file-log-level",
        choices=["off", "debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the log level for file logging",
    )
    subparsers = parser.add_subparsers(dest="subparser")

    configure_parser(subparsers)

    ryzenai_onnx_utils.matcher.configure_parser(subparsers)
    ryzenai_onnx_utils.extract.configure_parser(subparsers)
    ryzenai_onnx_utils.preprocess.configure_parser(subparsers)
    ryzenai_onnx_utils.optimize.configure_parser(subparsers)
    ryzenai_onnx_utils.postprocess.configure_parser(subparsers)
    ryzenai_onnx_utils.report.configure_parser(subparsers)
    ryzenai_onnx_utils.auto.configure_parser(subparsers)
    ryzenai_onnx_utils.lora.configure_parser(subparsers)
    ryzenai_onnx_utils.vaiml.configure_parser(subparsers)

    return parser


def replace(
    pattern: list[str] | str,
    replacement: PassFunction,
    params: ryzenai_onnx_utils.ReplaceParams,
    extractor: onnx.utils.Extractor,
    pass_id: str,
    indent: str = "",
    pass_name: str | None = None,
) -> int:
    matcher = ryzenai_onnx_utils.matcher.Matcher(pattern, True)

    max_to_replace = None
    # Setting max_to_replace value applies to every pass, including cast passes which can cause inconsistencies.
    # Filter using pass_name to narrow down the scope:
    # if pass_name == "hybrid_llm_matmulnbits":
    #    max_to_replace = 17

    # Use a specific policy to build new subgraphs and replace matching subgraphs.
    replaced_num = matcher.replace(extractor, replacement, pass_id, params, max_to_replace)
    if replaced_num < 0:
        _logger.info(f"{indent}Finished pass")
    else:
        _logger.info(f"{indent}Replaced {replaced_num} nodes")
    return replaced_num


def _include_constructor(prefix_path: Path, loader: yaml.SafeLoader, node: yaml.SequenceNode) -> Any:
    yaml_files = loader.construct_sequence(node)
    name = Path(yaml_files.pop(0))
    if not name.is_absolute():
        name = prefix_path / name
    with open(name) as f:
        content = yaml.safe_load(f)
    for name in yaml_files:
        name = Path(name)
        if not name.is_absolute():
            name = prefix_path / name
        with open(name) as f:
            new_content = yaml.safe_load(f)
        for i in new_content:
            if isinstance(content[i], str):
                content[i] = new_content[i]
            elif isinstance(content[i], list):
                content[i].extend(new_content[i])
            else:
                raise ValueError("Unhandled type")
    return content


def _load_strategy(strategy_path: Path) -> dict[str, Any]:
    strategy_prefix = importlib.resources.files("ryzenai_onnx_utils") / "data/partition_strategies"
    with importlib.resources.as_file(strategy_prefix) as strategy_prefix_path:
        strategy_file = strategy_prefix_path / strategy_path if not strategy_path.is_absolute() else strategy_path
        include_constructor = functools.partial(_include_constructor, strategy_prefix_path)
    yaml.add_constructor("!include", include_constructor, Loader=yaml.SafeLoader)
    with open(strategy_file) as f:
        strategy: dict[str, Any] = yaml.safe_load(f)
    return strategy


def load_strategy(strategy_path: Path) -> dict[str, Any]:
    strategy = _load_strategy(strategy_path)
    if "passes" not in strategy:
        strategy["passes"] = []
    if "inherit_passes_before" in strategy:
        passes_after = copy.copy(strategy["passes"])
        strategy["passes"] = strategy["inherit_passes_before"]["passes"]
        strategy["passes"].extend(passes_after)
        del strategy["inherit_passes_before"]
    if "inherit_passes_after" in strategy:
        strategy["passes"].extend(strategy["inherit_passes_after"]["passes"])
        del strategy["inherit_passes_after"]
    if "postprocess" in strategy:
        strategy["passes"].extend(strategy["postprocess"])
        del strategy["postprocess"]
    return strategy


def make_hints(
    extractor: onnx.utils.Extractor,
    params: ryzenai_onnx_utils.ReplaceParams,
    hints_keys: Iterable[str],
) -> dict[str, list[onnx.NodeProto]]:
    xclbin_mapping = params.attributes["xclbins"]
    node_mapping: dict[str, list[onnx.NodeProto]] = {}
    graph = extractor.graph
    for node in graph.node:
        if all(hints_key not in node.op_type for hints_key in hints_keys):
            continue
        if node.op_type in node_mapping:
            node_mapping[node.op_type].append(node)
        else:
            node_mapping[node.op_type] = [node]
    hints: dict[str, list[onnx.NodeProto]] = {}
    if isinstance(xclbin_mapping, dict):
        for op_type, xclbin_path in xclbin_mapping.items():
            if op_type not in node_mapping:
                continue
            label = xclbin_path.split("/")[-1]
            if label in hints:
                hints[label] += node_mapping[op_type]
            else:
                hints[label] = node_mapping[op_type]
    elif isinstance(xclbin_mapping, str):
        hints[xclbin_mapping.split("/")[-1]] = []
        for node_list in node_mapping.values():
            hints[xclbin_mapping.split("/")[-1]] += node_list
    return hints


class PassModule(Protocol):
    PATTERN: PatternType
    REPLACEMENT: PassFunction | list[PassFunction]


def partition(
    extractor: onnx.utils.Extractor,
    passes: Sequence[str | dict[str, Any]],
    params: ryzenai_onnx_utils.ReplaceParams,
    runtime_attributes: dict[str, Any] | None = None,
) -> tuple[onnx.ModelProto, int]:
    def recurse(sub_graph: onnx.GraphProto) -> tuple[int, onnx.GraphProto]:
        sub_model = onnx.helper.make_model(sub_graph, producer_name="from_subgraph")
        sub_extractor = ryzenai_onnx_utils.matcher.get_extractor(sub_model)
        # some passes need to know if they are running on a nested graph
        nested_graph = params.attributes.get("_nested_graph", False)
        params.attributes["_nested_graph"] = True
        sub_model, sub_total_replace_num = partition(sub_extractor, passes, params, runtime_attributes)
        params.attributes["_nested_graph"] = nested_graph
        return sub_total_replace_num, sub_model.graph

    if runtime_attributes is None:
        runtime_attributes = {}
    _logger.info("Starting passes...")

    total_replaced_num = 0

    for node in extractor.graph.node:
        if node.op_type != "Constant":
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    _logger.info(f"Starting passes of {node.name, node.op_type}...")
                    replaced_num, new_subgraph = recurse(attr.g)
                    total_replaced_num += replaced_num
                    ryzenai_onnx_utils.matcher.set_attribute(node, attr.name, new_subgraph)
                    _logger.info(f"End passes of {node.name, node.op_type}.")
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    new_graphs = []
                    _logger.info(f"Starting passes of {node.name, node.op_type}...")
                    for subgraph in attr.graphs:
                        replaced_num, new_subgraph = recurse(subgraph)
                        total_replaced_num += replaced_num
                        new_graphs.append(new_subgraph)
                    ryzenai_onnx_utils.matcher.set_attribute(node, attr.name, new_graphs)
                    _logger.info(f"End passes of {node.name, node.op_type}.")

    # if any global passes are run, we must always save the graph because we don't know
    # if the graph has been modified or not
    any_global_passes = False

    for pass_index, run_pass in enumerate(passes):
        if isinstance(run_pass, str):
            pass_name = run_pass
            pass_attributes = {}
            pass_exclusions = {}
        else:
            pass_name = next(iter(run_pass))
            pass_attributes = run_pass[pass_name].get("attributes", {})
            pass_exclusions = run_pass[pass_name].get("exclude", {})
        pass_params = copy.deepcopy(params)
        pass_params.attributes = {
            **params.attributes,
            **pass_attributes,
            **runtime_attributes,
        }
        current_pass = cast(PassModule, importlib.import_module(f"ryzenai_onnx_utils.passes.{pass_name}"))
        _logger.info(f"Starting pass {pass_name} ({pass_index + 1} of {len(passes)})...")
        start_time = time.perf_counter()
        if (
            current_pass.PATTERN  # ensure that there's at least one pattern
            and not callable(current_pass.PATTERN)  # and it's not a callable
            and not isinstance(current_pass.PATTERN[0], str)  # it has a nested pattern
            and callable(current_pass.REPLACEMENT)  #  it is a function not a list
        ):
            assert callable(current_pass.REPLACEMENT)
            current_pass.REPLACEMENT = [current_pass.REPLACEMENT] * len(current_pass.PATTERN)
        if isinstance(current_pass.REPLACEMENT, list):
            assert len(current_pass.REPLACEMENT) == len(current_pass.PATTERN)
            for subpass_index, (pattern, replacement) in enumerate(
                zip(current_pass.PATTERN, current_pass.REPLACEMENT, strict=True)
            ):
                if isinstance(pattern, SubPass):
                    subpass_name = pattern.name
                    pattern = pattern.pattern
                else:
                    subpass_name = (replacement.__module__).split(".")[-1]
                if subpass_name in pass_exclusions:
                    _logger.info(
                        f"  Skipping subpass {subpass_name} ({subpass_index + 1} of {len(current_pass.REPLACEMENT)})..."
                    )
                    continue
                _logger.info(
                    f"  Starting subpass {subpass_name} ({subpass_index + 1} of {len(current_pass.REPLACEMENT)})..."
                )
                replaced_num = replace(
                    cast(str | list[str], pattern),
                    replacement,
                    pass_params,
                    extractor,
                    f"{pass_index}_{subpass_index}",
                    "  ",
                    pass_name,
                )
                if replaced_num < 0:
                    any_global_passes = True
                total_replaced_num += max(replaced_num, 0)
        elif callable(current_pass.PATTERN):
            # PATTERN(extractor, str, ReplaceParam) -> list[list[str]]
            # and REPLACEMENT(extractor, str, list[onnx.NodeProto], ReplaceParam)
            patterns = current_pass.PATTERN(extractor, pass_index, pass_params)
            assert isinstance(patterns, list)
            if patterns:
                assert all(
                    isinstance(sublist, list) and all(isinstance(item, str) for item in sublist) for sublist in patterns
                )
            replacements = [current_pass.REPLACEMENT] * len(patterns)
            for subpass_index, (pattern, replacement) in enumerate(zip(patterns, replacements, strict=True)):
                replaced_num = replace(
                    pattern,
                    replacement,
                    pass_params,
                    extractor,
                    f"{pass_index}_{subpass_index}",
                )
                if replaced_num < 0:
                    any_global_passes = True
                total_replaced_num += max(replaced_num, 0)
        else:
            assert isinstance(current_pass.PATTERN, str) or is_sequence_of(current_pass.PATTERN, str)
            replaced_num = replace(
                current_pass.PATTERN,
                current_pass.REPLACEMENT,
                pass_params,
                extractor,
                str(pass_index),
                "",
                pass_name,
            )
            if replaced_num < 0:
                any_global_passes = True
            total_replaced_num += max(replaced_num, 0)
        elapsed = time.perf_counter() - start_time
        _logger.info(f"Finished pass: {elapsed:.6f} s ({elapsed * 1000:.2f} ms)")

    if any_global_passes:
        # if any global passes ran, set the minimum node replacement count to 1
        total_replaced_num = max(total_replaced_num, 1)

    _logger.info(f"Replaced at least {total_replaced_num} node(s) in total across {len(passes)} passes")

    if total_replaced_num > 0:
        domains = params.get_domains()
        for domain in domains:
            ryzenai_onnx_utils.matcher.set_opset(extractor.model, domain, 1)

    try:
        ryzenai_onnx_utils.matcher.graph_topological_sort(extractor, True)
    except RuntimeError as e:
        _logger.warning("*** This graph is invalid. Examine the saved model: " + str(e))

    return extractor.model, total_replaced_num


def _run_manual_passes(
    input_path: Path,
    output_path: Path,
    save_as_external: bool,
    location: str,
    passes: Sequence[str | dict[str, Any]],
    size_threshold: int,
) -> None:
    if not passes:
        return
    onnx_model = onnx.load_model(input_path)
    extractor = ryzenai_onnx_utils.matcher.get_extractor(onnx_model)
    params = ryzenai_onnx_utils.ReplaceParams(
        {"xclbins": "", "domains": "ai.onnx", "op_namespaces": ""},
        output_path,
        Path(""),
        Path(""),
    )
    model, total_replaced_num = partition(extractor, passes, params, {})
    if total_replaced_num > 0:
        ryzenai_onnx_utils.matcher.save_initializers_with_extractor(
            extractor, output_path.parent, location, size_threshold=size_threshold
        )
        ryzenai_onnx_utils.matcher.save_model_without_external_data(model, output_path)


def confirm(prompt: str) -> bool:
    answer = ""
    while answer not in ["y", "n", "yes", "no"]:
        answer = input(prompt).lower()
    return answer in ["y", "yes"]


def confirm_or_exit(prompt: str, force: bool) -> None:
    if force:
        _logger.warning(f"{prompt} Y")
    elif not confirm(prompt):
        _logger.warning("Exiting")
        sys.exit(0)


def parse_runtime_attributes(attributes: list[str] | None) -> dict[str, Any]:
    attributes_dict: dict[str, Any] = {}
    if attributes is not None:
        for attribute in attributes:
            if "=" not in attribute:
                raise ValueError(f"Invalid attribute key-value provided: {attribute}")
            split_string = attribute.split("=")
            attributes_dict[split_string[0].strip()] = split_string[1].strip()
            if split_string[0].strip() == "dynamic_shape_list":
                value = json.loads(split_string[1].replace("'", '"').strip())
                if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
                    raise TypeError("invalid dynamic_shape_list attr, shoudl be list(dict)")
                attributes_dict[split_string[0].strip()] = value
    return attributes_dict


def expression_value_inference(expression: tuple[str | int, ...], shape_candidate: dict[str, Any]) -> list[Any]:
    symbolic_func = {
        "floor": sp.floor,
        "Min": sp.Min,
    }
    context = {**shape_candidate, **symbolic_func}
    new_expression = [elem.decode("utf-8") if isinstance(elem, bytes) else elem for elem in expression]
    value = [sp.sympify(elem, locals=context) for elem in new_expression]
    return value


def get_dynamic_shape_candidate(expression: list[tuple[str | int, ...]], attributes: dict[str, Any]) -> list[list[Any]]:
    if "dynamic_shape_list" not in attributes:
        return [expression]
    dynamic_shape_candidates = attributes["dynamic_shape_list"]
    candidates = []
    for dynamic_shape_candidate in dynamic_shape_candidates:
        candidate = [expression_value_inference(elem, dynamic_shape_candidate) for elem in expression]
        candidates.append(candidate)
    return sorted(candidates)


def unpack_candidates_shapes(candidates_shape_lists: dict[str, Any]) -> list[dict[str, Any]]:
    input_shapes = candidates_shape_lists["input_shape"]
    output_shapes = candidates_shape_lists["output_shape"]
    if len(input_shapes) == len(output_shapes):
        return [
            {"input_shape": input_shape, "output_shape": output_shape}
            for input_shape, output_shape in zip(input_shapes, output_shapes, strict=True)
        ]
    elif len(input_shapes) == 1:
        return [{"input_shape": input_shapes[0], "output_shape": output_shape} for output_shape in output_shapes]
    elif len(output_shapes) == 1:
        return [{"input_shape": input_shape, "output_shape": output_shapes[0]} for input_shape in input_shapes]
    else:
        raise ValueError("Cannot align input_shape and output_shape: incompatible lengths")


def get_check_shapes(
    shape_lists: dict[str, list[tuple[int | str, ...]]], attributes: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates_shape_lists = {}
    for key, shapes in shape_lists.items():
        if not isinstance(shapes, list | tuple):
            candidates_shape_lists[key] = shapes
            continue
        candidates = get_dynamic_shape_candidate(shapes, attributes)
        candidates_shape_lists[key] = candidates
    candidates_shapes = unpack_candidates_shapes(candidates_shape_lists)

    diff_keys = set(candidates_shape_lists.keys()) - {"input_shape", "output_shape"}
    for diff_key in diff_keys:
        if isinstance(candidates_shape_lists[diff_key], (str | int | float)):
            for candidates_shape in candidates_shapes:
                candidates_shape[diff_key] = candidates_shape_lists[diff_key]
        elif isinstance(candidates_shape_lists[diff_key], (list | tuple)):
            assert len(candidates_shape_lists[diff_key]) == len(candidates_shapes), (
                f"Mismatched lengths in candidates_shape_lists[{diff_key}]"
            )
            for i in range(len(candidates_shapes)):
                candidates_shapes[i][diff_key] = candidates_shape_lists[diff_key][i]
    return candidates_shapes


def partition_main(args: argparse.Namespace) -> None:
    input_path: Path = args.input_path
    output_path: Path = args.output_path
    dd_files_path: Path = args.dd_files_path
    strategy_path = args.strategy
    if args.verbose > 0:
        _logger.warning(
            "Using deprecated --verbose/-v flag. Use --console-log-level [level] or use --verbose/-v before 'partition' instead."
        )
        logger = logging.getLogger("ryzenai_onnx_utils")
        if args.verbose > 1:
            logger.handlers[0].setLevel(logging.DEBUG)
        else:
            logger.handlers[0].setLevel(logging.INFO)
    runtime_attributes = parse_runtime_attributes(args.attributes)

    input_path = input_path.absolute()
    output_path = output_path.absolute()

    output_model_path = output_path / f"{args.model_name}.onnx"
    if output_model_path.exists():
        prompt = f"{str(output_model_path)} exists. Okay to overwrite? (y|n) "
        confirm_or_exit(prompt, args.force)
    curr_output_model = output_path / "tmp_0.onnx"
    next_output_model = output_path / "tmp_1.onnx"

    dd_files_path_abs = dd_files_path if dd_files_path.is_absolute() else output_path / dd_files_path
    if dd_files_path_abs.exists() and any(dd_files_path_abs.iterdir()):
        prompt = f"{str(dd_files_path_abs)} exists and is non-empty. Delete directory and continue? (y|n) "
        confirm_or_exit(prompt, args.force)
        shutil.rmtree(dd_files_path_abs)

    # this directory gets deleted and created by the build_dd_node function
    # when creating the DD files. If DD can generate files in the output path
    # directly, we can delete this
    backup_path = None
    if not dd_files_path.is_absolute():
        tmp_dd_files_path = dd_files_path.resolve()
        if tmp_dd_files_path.exists() and any(tmp_dd_files_path.iterdir()):
            backup_path = tmp_dd_files_path.with_name(f"{tmp_dd_files_path.name}_backup")
            tmp_dd_files_path.rename(backup_path)
        tmp_dd_files_path.mkdir(parents=True, exist_ok=True)

    output_path.mkdir(parents=True, exist_ok=True)
    dd_files_path_abs.mkdir(parents=True, exist_ok=True)

    strategy = load_strategy(strategy_path)
    default_enable_model_hash = strategy.get("attributes", {}).get("enable_model_hash", False)
    enable_model_hash = runtime_attributes.get("enable_model_hash", default_enable_model_hash)
    if enable_model_hash:
        hash_val = ryzenai_onnx_utils.matcher.compute_model_md5(input_path)
        strategy["attributes"]["model_hash"] = hash_val

    params = ryzenai_onnx_utils.ReplaceParams(
        strategy["attributes"], output_model_path, dd_files_path, dd_files_path_abs
    )

    passes_per_iteration = args.passes_per_iteration
    if passes_per_iteration is None or passes_per_iteration < 1:
        passes_per_iteration = len(strategy["passes"])
        run_all_passes = True
    else:
        _logger.info(f"Running passes iteratively {passes_per_iteration} at a time...")
        run_all_passes = False

    any_replaced = False
    for i in range(0, len(strategy["passes"]), passes_per_iteration):
        stop_index = min(i + passes_per_iteration, len(strategy["passes"]))

        _logger.info("Loading model...")
        try:
            extractor = ryzenai_onnx_utils.matcher.load_extractor(input_path, args.load_external_data)

        except onnx.onnx_cpp2py_export.shape_inference.InferenceError:
            _logger.info("Shape inference failed, continuing...")
            extractor = ryzenai_onnx_utils.matcher.load_extractor(input_path, args.load_external_data, False)
        del extractor.model.graph.initializer[:]
        _logger.info("Loaded model")

        model, total_replaced_num = partition(
            extractor,
            strategy["passes"][i:stop_index],
            params,
            runtime_attributes,
        )

        if total_replaced_num > 0:
            any_replaced = True
            if not run_all_passes:
                _logger.info("Saving temporary model...")
                external_data_name = f"{curr_output_model.stem}.{args.external_data_extension}"
                ryzenai_onnx_utils.matcher.save_initializers_with_extractor(
                    extractor,
                    output_model_path.parent,
                    external_data_name,
                    size_threshold=args.external_data_threshold,
                )

                ryzenai_onnx_utils.matcher.save_model_without_external_data(model, curr_output_model)
                _logger.info("Saved temporary model")

                input_path = copy.copy(curr_output_model)
                curr_output_model = copy.copy(next_output_model)
                next_output_model = copy.copy(input_path)

    if any_replaced:
        _logger.info("Saving final model...")
        if not run_all_passes:
            try:
                extractor = ryzenai_onnx_utils.matcher.load_extractor(input_path, args.load_external_data)
            except onnx.onnx_cpp2py_export.shape_inference.InferenceError:
                _logger.info("Shape inference failed, continuing...")
                extractor = ryzenai_onnx_utils.matcher.load_extractor(input_path, args.load_external_data, False)
        else:
            # reset the model with the updated model returned from the partitioner
            extractor.model = model

        external_data_name = f"{args.model_name}.{args.external_data_extension}"
        ryzenai_onnx_utils.matcher.save_initializers_with_extractor(
            extractor,
            output_model_path.parent,
            external_data_name,
            size_threshold=args.external_data_threshold,
        )

        ryzenai_onnx_utils.matcher.save_model_without_external_data(extractor.model, output_model_path)

        _logger.info("Saved final model")

        if args.combine_dd:
            assert dd is not None, "DD Python library could not be imported"
            _logger.info("Combining DD files...")
            dd.combine_dd(dd_files_path, dd_files_path_abs, output_path)
            _logger.info("Combined DD files")

        # onnx.checker.check_model(output_path / "replaced.onnx", full_check=True)

    if not dd_files_path.is_absolute():
        shutil.rmtree(dd_files_path)
    if backup_path is not None:
        backup_path.rename(backup_path.with_name(dd_files_path.name))

    ryzenai_onnx_utils.matcher.delete_model(curr_output_model, args.external_data_extension)
    ryzenai_onnx_utils.matcher.delete_model(next_output_model, args.external_data_extension)


def pattern_match(args: argparse.Namespace) -> None:
    model = pb.get_model(
        args.input_path,
        args.extract_inputs,
        args.extract_outputs,
        args.load_external_data,
    )
    extractor = ryzenai_onnx_utils.matcher.get_extractor(model)
    original_pattern = pb.convert_model_to_pattern(model)
    if args.strategy:
        _logger.info("Pattern before partitioner:")
        _logger.info(original_pattern)

        strategy = load_strategy(args.strategy)
        passes = strategy["passes"]
        if any(pass_name.startswith("dd") for pass_name in passes):
            raise ValueError("DD passes are not supported to run here")
        params = ryzenai_onnx_utils.ReplaceParams(
            strategy["attributes"],
            Path(),
            Path(),
            Path(),
        )
        model, _ = partition(extractor, passes, params, None)
        new_patterns = []
        if args.hints_key:
            hints = make_hints(extractor, params, args.hints_key)
            engine = pg.PartitionGraph(extractor, hints, args.hints_key)
            engine.partition()
            pattern_ = pg.PatternGenerator(engine)
            new_patterns = pattern_.get_patterns()
        else:
            new_patterns = [pb.convert_model_to_pattern(model)]
        _logger.info("Pattern after partitioner:")
        _logger.info(new_patterns)
    else:
        _logger.info("Pattern:")
        _logger.info(original_pattern)


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    sys.setrecursionlimit(args.recursion_limit)

    # assuming console handler will be the first handler
    logger = logging.getLogger("ryzenai_onnx_utils")
    console_handler = logger.handlers[0]
    if args.console_log_level == "off":
        logger.removeHandler(console_handler)
    else:
        if args.global_verbose > 1:
            console_handler.setLevel(logging.DEBUG)
        elif args.global_verbose > 0:
            console_handler.setLevel(logging.INFO)
        else:
            console_handler.setLevel(args.console_log_level.upper())

    # assuming file handler will be the last handler
    file_handler = logger.handlers[-1]
    if args.file_log_level == "off":
        logger.removeHandler(file_handler)
    else:
        file_handler.setLevel(args.file_log_level.upper())
        if isinstance(file_handler, logging.handlers.RotatingFileHandler):
            # change rotation from x.log -> x.log.1 to x.log -> x.1.log
            file_handler.namer = lambda name: name.replace(".log", "") + ".log"
            # start a new log every time
            file_handler.doRollover()

    if args.subparser == "partition":
        partition_main(args)
    elif args.subparser == "match":
        pattern_match(args)
    elif args.subparser == "extract":
        if args.pattern_file:
            user_pattern = ryzenai_onnx_utils.utils.load_module_from_file(args.pattern_file, "user_pattern")
            args.pattern = user_pattern.user_pattern
        elif args.pattern.startswith("op-"):
            name = args.pattern.removeprefix("op-")
            spec = importlib.util.spec_from_file_location(
                "user_pattern",
                str(importlib.resources.files("ryzenai_onnx_utils") / "data/extract_patterns/op.py"),
            )
            assert spec is not None, f"Pattern op-{name} not found"
            user_pattern = importlib.util.module_from_spec(spec)
            sys.modules["user_pattern"] = user_pattern
            loader = spec.loader
            assert loader is not None, f"Loader for pattern {args.pattern} is None"
            loader.exec_module(user_pattern)
            args.pattern = functools.partial(user_pattern.op, name)
        else:
            assert args.pattern != "op", "Use op-<name> to match on a particular op"
            spec = importlib.util.spec_from_file_location(
                "user_pattern",
                str(importlib.resources.files("ryzenai_onnx_utils") / f"data/extract_patterns/{args.pattern}.py"),
            )
            assert spec is not None, f"Pattern {args.pattern} not found"
            user_pattern = importlib.util.module_from_spec(spec)
            sys.modules["user_pattern"] = user_pattern
            loader = spec.loader
            assert loader is not None, f"Loader for pattern {args.pattern} is None"
            loader.exec_module(user_pattern)
            args.pattern = user_pattern.user_pattern

        ryzenai_onnx_utils.extract.main(args)
    elif args.subparser == "optimize":
        ryzenai_onnx_utils.optimize.main(args)
    elif args.subparser == "postprocess":
        ryzenai_onnx_utils.postprocess.main(args)
    elif args.subparser == "preprocess":
        ryzenai_onnx_utils.preprocess.main(args)
    elif args.subparser == "report":
        ryzenai_onnx_utils.report.main(args)
    elif args.subparser == "auto":
        ryzenai_onnx_utils.auto.main(args)
    elif args.subparser == "vaiml":
        ryzenai_onnx_utils.vaiml.main(args)
    elif args.subparser == "lora-process":
        ryzenai_onnx_utils.lora.main(args)
    else:
        raise ValueError(f"Unknown subparser: {args.subparser}")
