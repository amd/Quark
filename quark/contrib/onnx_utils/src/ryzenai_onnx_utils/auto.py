# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import onnxruntime

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.partitioner
import ryzenai_onnx_utils.preprocess
import ryzenai_onnx_utils.reports as reports
import ryzenai_onnx_utils.utils


# TODO(varunsh): this should be updated to inherit parser arguments from the others somehow
def configure_parser(subparser: argparse._SubParsersAction[Any]) -> argparse.ArgumentParser:
    auto_parser: argparse.ArgumentParser = subparser.add_parser("auto")

    auto_parser.add_argument("input_path", type=Path, help="Path to input model")
    auto_parser.add_argument("output_path", type=Path, help="Path to create directories for output files")
    auto_parser.add_argument(
        "--optimize",
        type=Path,
        help="Run an preprocess optimize script",
    )
    auto_parser.add_argument(
        "--save-as-external",
        action="store_true",
        help="Save the model with external data",
    )
    auto_parser.add_argument(
        "--size-threshold",
        type=int,
        default=1024,  # default value for onnx.save_model()
        help="Threshold of the buffer to move to external data. Does nothing if --save-as-external is not specified",
    )
    auto_parser.add_argument(
        "--keep-dynamic",
        action="store_true",
        help="Skip fixing the input shapes to preserve dynamic input shapes",
    )
    auto_parser.add_argument("strategy", type=Path, help="Strategy to run")
    auto_parser.add_argument(
        "--dd-files-path",
        type=Path,
        default=Path("cache"),
        help="Path to create DD files in, either absolute or relative to output_path",
    )
    auto_parser.add_argument("--model-name", default="replaced", help="Name of the new onnx model")
    auto_parser.add_argument(
        "--load-external-data",
        action="store_true",
        help="Load all external data at startup",
    )
    auto_parser.add_argument(
        "--passes-per-iteration",
        type=int,
        default=None,
        help="Run the passes in intervals, saving the model in between. Defaults to None",
    )
    auto_parser.add_argument(
        "--combine-dd",
        action="store_true",
        help="Combine generated DD files",
    )
    auto_parser.add_argument(
        "--force",
        action="store_true",
        help="Answer yes to any confirmations that may come up",
    )
    auto_parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Print more messages",
    )
    auto_parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the generated model with random input data.",
    )
    auto_parser.add_argument(
        "--custom-op-path",
        type=str,
        help="Path to custom op dll",
    )
    auto_parser.add_argument(
        "--dd-root",
        type=str,
        help="Path to Dynamic Dispatch root",
    )

    return auto_parser


def preprocess(args: argparse.Namespace) -> None:
    dd_path = args.output_path
    args.output_path = args.output_path / "preprocessed.onnx"
    ryzenai_onnx_utils.preprocess.main(args)
    args.output_path = dd_path


def partition(args: argparse.Namespace) -> None:
    ryzenai_onnx_utils.partitioner.partition_main(args)


def report(args: argparse.Namespace) -> None:
    reports.dd_offload.report(
        input_path=args.output_path / "replaced.onnx",
        comparison_path=None,
        summarize=True,
    )


def execute(args: argparse.Namespace) -> None:
    session_options = onnxruntime.SessionOptions()
    session_options.add_session_config_entry("dd_root", args.dd_root)
    if os.path.exists(args.custom_op_path):
        session_options.add_session_config_entry("dd_cache", str(args.output_path / "cache"))
        session_options.add_session_config_entry("compile_fusion_rt", "True")
        session_options.register_custom_ops_library(args.custom_op_path)
    ryzenai_onnx_utils.utils.run_onnx_model_with_dummy_inputs(
        args.output_path / "replaced.onnx", "DmlExecutionProvider", session_options
    )


def auto(args: argparse.Namespace) -> None:
    print("==== PREPROCESSING ====")
    preprocess(args)
    print("\n==== PARTITIONING ====")
    args.input_path = args.output_path / "preprocessed.onnx"
    partition(args)
    print("\n==== REPORTING ====")
    report(args)
    if args.execute:
        print("\n==== EXECUTING ====")
        execute(args)


def main(args: argparse.Namespace) -> None:
    auto(args)
